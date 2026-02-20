package engine

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	SolverRegions = &SolverRegionsStruct{make([]SolverRegion, 0)}
)

func init() {
	DeclFunc("DefSolverRegion", DefSolverRegion, "")
	DeclFunc("DefSolverRegionX", DefSolverRegionX, "")
	DeclFunc("DefSolverRegionY", DefSolverRegionY, "")
	DeclFunc("DefSolverRegionZ", DefSolverRegionZ, "")
	DeclVar("SolverRegions", &SolverRegions, "")
}

type SolverRegionsStruct struct {
	reg []SolverRegion
}

type SolverRegion struct {
	StartX     int
	EndX       int
	StartY     int
	EndY       int
	StartZ     int
	EndZ       int
	Solver     int
	Dt_si      float64
	LastErr    float64
	LastTorque float64
	PBCx       int
	PBCy       int
	PBCz       int
}

func DefSolverRegion(startX, endX, startY, endY, startZ, endZ, solver int) SolverRegion {
	util.AssertMsg(startX >= 0 && endX <= Nx && startY >= 0 && endY <= Ny && startZ >= 0 && endZ <= Nz, "Regions dimensions have to be inside of the mesh.")
	return SolverRegion{startX, endX, startY, endY, startZ, endZ, solver, 0, 0, 0, 0, 0, 0}
}

func DefSolverRegionX(startX, endX, solver int) SolverRegion {
	return DefSolverRegion(startX, endX, 0, Ny, 0, Nz, solver)
}

func DefSolverRegionY(startY, endY, startZ, endZ, solver int) SolverRegion {
	return DefSolverRegion(0, Nx, startY, endY, 0, Nz, solver)
}

func DefSolverRegionZ(startZ, endZ, solver int) SolverRegion {
	return DefSolverRegion(0, Nx, 0, Ny, startZ, endZ, solver)
}

func (s *SolverRegionsStruct) GetStepperSlice() (stepperSlice []Stepper, regionsData [][3]*data.Slice, overlappingElastic []bool) {
	//var regionsData = make([][3]*data.Slice, 0)
	overlappingElastic = s.CompareAllOverlapsElastic()
	for _, regionSolver := range s.reg {
		var regStepper Stepper
		switch regionSolver.Solver {
		default:
			util.Fatalf("SetSolver: unknown solver type: %v", regionSolver.Solver)
		case BACKWARD_EULER:
			regStepper = new(BackwardEuler)
		case EULER:
			regStepper = new(Euler)
		case HEUN:
			regStepper = new(Heun)
		case BOGAKISHAMPINE:
			regStepper = new(RK23)
		case RUNGEKUTTA:
			regStepper = new(RK4)
		case DORMANDPRINCE:
			regStepper = new(RK45DP)
		case FEHLBERG:
			regStepper = new(RK56)
		case SECONDDERIV:
			regStepper = new(secondHeun)
		case ELAS_RUNGEKUTTA:
			regStepper = new(elasRK4)
		case MAGELAS_RUNGEKUTTA:
			regStepper = new(magelasRK4)
		case ELAS_LEAPFROG:
			regStepper = new(elasLF)
		case ELAS_YOSH:
			regStepper = new(elasYOSH)
		case MAGELAS_RUNGEKUTTA_VARY_TIME:
			regStepper = new(magelasRK4_vary_time)
		}
		stepperSlice = append(stepperSlice, regStepper)
		mBuf := cuda.Buffer(VECTOR, regionSolver.Size())
		uBuf := cuda.Buffer(VECTOR, regionSolver.Size())
		duBuf := cuda.Buffer(VECTOR, regionSolver.Size())
		regionData := [3]*data.Slice{mBuf, uBuf, duBuf}
		regionsData = append(regionsData, regionData)
	}
	return stepperSlice, regionsData, overlappingElastic
}

func (s *SolverRegionsStruct) Run(seconds float64) {
	if len(s.reg) == 0 {
		panic("No solver region defined.")
	}
	Pause = false
	stepperSlice, regionsData, overlappingElastic := s.GetStepperSlice()
	start := Time
	stop := Time + seconds
	FixDtPre := FixDt
	hideProgressBarBool := true
	UseExcitationPre := UseExcitation
	if !HideProgressBarManualSet {
		hideProgressBarBoolTmp, err := strconv.ParseBool(strings.ToLower(HideProgressBar))
		if err != nil {
			fmt.Println("Failed to parse HideProgressBar from build process. Using as false.")
			hideProgressBarBoolTmp = false
		}
		hideProgressBarBool = hideProgressBarBoolTmp
	} else {
		hideProgressBarBool = HideProgressBarManual
	}
	ProgressBar := NewProgressBar(start, stop, "🧲", hideProgressBarBool)
	DoOutput()
	DoFFT4D()
	mBuf := cuda.Buffer(VECTOR, M.Buffer().Size())
	uBuf := cuda.Buffer(VECTOR, U.Buffer().Size())
	duBuf := cuda.Buffer(VECTOR, DU.Buffer().Size())
	defer cuda.Recycle(mBuf)
	defer cuda.Recycle(uBuf)
	defer cuda.Recycle(duBuf)
	for (Time < stop) && !Pause {
		ProgressBar.Update(Time)
		data.Copy(mBuf, M.Buffer())
		data.Copy(uBuf, U.Buffer())
		data.Copy(duBuf, DU.Buffer())
		fmt.Printf("HEree\n")
		cuda.PrintBufLength()
		if FixDtPre == 0 {
			Time0 := Time
			for i := range stepperSlice {
				UseExcitation = overlappingElastic[i]
				for Time == Time0 {
					stepperSlice[i].StepRegion(&s.reg[i])
				}
				Time = Time0
				data.Copy(M.Buffer(), mBuf)
				data.Copy(U.Buffer(), uBuf)
				data.Copy(DU.Buffer(), duBuf)
			}
			minSiDt := math.Inf(1)
			for i := range s.reg {
				if minSiDt > s.reg[i].Dt_si {
					minSiDt = s.reg[i].Dt_si
				}
			}
			FixDt = minSiDt
		}
		cuda.PrintBufLength()
		//Do step for region, set store result of region, reset back to state from before, evolve next region, etc
		for i := range stepperSlice {
			UseExcitation = overlappingElastic[i]
			stepperSlice[i].StepRegion(&s.reg[i])
			cuda.Crop(regionsData[i][0], M.Buffer(), s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ)
			cuda.Crop(regionsData[i][1], U.Buffer(), s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ)
			cuda.Crop(regionsData[i][2], DU.Buffer(), s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ)
			data.Copy(M.Buffer(), mBuf)
			data.Copy(U.Buffer(), uBuf)
			data.Copy(DU.Buffer(), duBuf)
		}
		cuda.PrintBufLength()
		for i := range stepperSlice {
			data.CopyPart(M.Buffer(), regionsData[i][0], 0, regionsData[i][0].Size()[X], 0, regionsData[i][0].Size()[Y], 0, regionsData[i][0].Size()[Z], 0, 1, s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ, 0)
			data.CopyPart(U.Buffer(), regionsData[i][1], 0, regionsData[i][0].Size()[X], 0, regionsData[i][0].Size()[Y], 0, regionsData[i][0].Size()[Z], 0, 1, s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ, 0)
			data.CopyPart(DU.Buffer(), regionsData[i][2], 0, regionsData[i][0].Size()[X], 0, regionsData[i][0].Size()[Y], 0, regionsData[i][0].Size()[Z], 0, 1, s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ, 0)
		}
		FixDt = FixDtPre
		DoOutput()
		DoFFT4D()
	}
	UseExcitation = UseExcitationPre
	ProgressBar.Finish()
	Pause = true
}
func (s *SolverRegionsStruct) Append(reg SolverRegion) {
	s.reg = append(s.reg, reg)
}

func (s SolverRegion) Size() [3]int {
	return [3]int{s.EndX - s.StartX, s.EndY - s.StartY, s.EndZ - s.StartZ}
}

func (s *SolverRegion) PBC(x, y, z int) {
	s.PBCx = x
	s.PBCy = y
	s.PBCz = z
}

func (r *SolverRegion) Overlaps(other *SolverRegion) bool {
	return Overlaps(r.StartX, r.EndX, r.StartY, r.EndY, r.StartZ, r.EndZ, other.StartX, other.EndX, other.StartY, other.EndY, other.StartZ, other.EndZ)
}

func Overlaps(StartX1, EndX1, StartY1, EndY1, StartZ1, EndZ1,
	StartX2, EndX2, StartY2, EndY2, StartZ2, EndZ2 int) bool {
	if EndX1 <= StartX2 || EndX2 <= StartX1 {
		return false
	}
	if EndY1 <= StartY2 || EndY2 <= StartY1 {
		return false
	}
	if EndZ1 <= StartZ2 || EndZ2 <= StartZ1 {
		return false
	}
	return true
}

func IsElasticSolver(solv int) bool {
	switch solv {
	default:
		util.Fatalf("SetSolver: unknown solver type: %v", solv)
	case BACKWARD_EULER:
		return false
	case EULER:
		return false
	case HEUN:
		return false
	case BOGAKISHAMPINE:
		return false
	case RUNGEKUTTA:
		return false
	case DORMANDPRINCE:
		return false
	case FEHLBERG:
		return false
	case SECONDDERIV:
		return true
	case ELAS_RUNGEKUTTA:
		return true
	case MAGELAS_RUNGEKUTTA:
		return true
	case ELAS_LEAPFROG:
		return true
	case ELAS_YOSH:
		return true
	case MAGELAS_RUNGEKUTTA_VARY_TIME:
		return true
	}
	panic("Error during check if solver contains elastic solver.")
}

func (s *SolverRegionsStruct) CompareAllOverlapsElastic() []bool {
	var comparisons []bool
	for i := 0; i < len(s.reg); i++ {
		var overlapping = false
		if !IsElasticSolver(s.reg[i].Solver) {
			for j := i + 1; j < len(s.reg); j++ {
				if s.reg[i].Overlaps(&s.reg[j]) && !IsElasticSolver(s.reg[j].Solver) {
					overlapping = true
					break
				}
			}
		} else {
			overlapping = false
		}
		comparisons = append(comparisons, overlapping)
	}
	return comparisons
}

func GetOverlapIndex(s SetSlice, setSliceSlice []SetSlice) (overlappingIndices []int) {
	if s == nil {
		return overlappingIndices
	}
	for i, s2 := range setSliceSlice {
		if Overlaps(s.StartAt()[X], s.EndAt()[X], s.StartAt()[Y], s.EndAt()[Y], s.StartAt()[Z], s.EndAt()[Z], s2.StartAt()[X], s2.EndAt()[X], s2.StartAt()[Y], s2.EndAt()[Y], s2.StartAt()[Z], s2.EndAt()[Z]) {
			overlappingIndices = append(overlappingIndices, i)
		}
	}
	return overlappingIndices
}

func FreeMemoryIndices(slice []SetSlice, indices []int) {
	for _, i := range indices {
		if slice[i].Buffer().GPUAccess() {
			cuda.Recycle(slice[i].Buffer())
		} else {
			slice[i].Buffer().Free()
		}
	}
}

func RemoveIndices[T any](slice []T, indices []int) []T {
	toRemove := make(map[int]struct{}, len(indices))
	for _, i := range indices {
		toRemove[i] = struct{}{}
	}

	newSlice := make([]T, 0, len(slice)-len(indices))
	for i, elem := range slice {
		if _, remove := toRemove[i]; !remove {
			newSlice = append(newSlice, elem)
		}
	}
	return newSlice
}

type SetSlice interface {
	StartAt() [3]int
	EndAt() [3]int
	Buffer() *data.Slice
}
