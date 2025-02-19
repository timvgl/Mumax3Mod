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
	return DefSolverRegion(0, Ny, startY, endY, 0, Ny, solver)
}

func DefSolverRegionZ(startZ, endZ, solver int) SolverRegion {
	return DefSolverRegion(0, Nx, 0, Ny, startZ, endZ, solver)
}

func (s *SolverRegionsStruct) GetStepperSlice() (stepperSlice []Stepper, regionsData [][3]*data.Slice) {
	//var regionsData = make([][3]*data.Slice, 0)
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
	return stepperSlice, regionsData
}

func (s *SolverRegionsStruct) Run(seconds float64) {
	if len(s.reg) == 0 {
		panic("No solver region defined.")
	}
	Pause = false
	stepperSlice, regionsData := s.GetStepperSlice()
	start := Time
	stop := Time + seconds
	FixDtPre := FixDt
	hideProgressBarBool := true
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
	for (Time < stop) && !Pause {
		ProgressBar.Update(Time)
		data.Copy(mBuf, M.Buffer())
		data.Copy(uBuf, U.Buffer())
		data.Copy(duBuf, DU.Buffer())
		if FixDtPre == 0 {
			Time0 := Time
			for i := range stepperSlice {
				for Time == Time0 {
					stepperSlice[i].StepRegion(&s.reg[i])
				}
				Time = Time0
			}
			minSiDt := math.Inf(1)
			for i := range s.reg {
				if minSiDt > s.reg[i].Dt_si {
					minSiDt = s.reg[i].Dt_si
				}
			}
			FixDt = minSiDt
		}
		//Do step for region, set store result of region, reset back to state from before, evolve next region, etc
		for i := range stepperSlice {
			stepperSlice[i].StepRegion(&s.reg[i])
			cuda.Crop(regionsData[i][0], M.Buffer(), s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ)
			cuda.Crop(regionsData[i][1], U.Buffer(), s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ)
			cuda.Crop(regionsData[i][2], DU.Buffer(), s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ)
			data.Copy(M.Buffer(), mBuf)
			data.Copy(U.Buffer(), uBuf)
			data.Copy(DU.Buffer(), duBuf)
		}
		for i := range stepperSlice {
			data.CopyPart(M.Buffer(), regionsData[i][0], 0, regionsData[i][0].Size()[X], 0, regionsData[i][0].Size()[Y], 0, regionsData[i][0].Size()[Z], 0, 1, s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ, 0)
			data.CopyPart(U.Buffer(), regionsData[i][1], 0, regionsData[i][0].Size()[X], 0, regionsData[i][0].Size()[Y], 0, regionsData[i][0].Size()[Z], 0, 1, s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ, 0)
			data.CopyPart(DU.Buffer(), regionsData[i][2], 0, regionsData[i][0].Size()[X], 0, regionsData[i][0].Size()[Y], 0, regionsData[i][0].Size()[Z], 0, 1, s.reg[i].StartX, s.reg[i].StartY, s.reg[i].StartZ, 0)
		}
		FixDt = FixDtPre
		DoOutput()
		DoFFT4D()
	}
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
