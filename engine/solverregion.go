package engine

import (
	"fmt"
	"math"
	"strconv"
	"strings"

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
	return SolverRegion{startX, endX, startY, endY, startZ, endZ, solver, 0, 0, 0, 0, 0, 0}
}

func DefSolverRegionX(startX, endX, solver int) SolverRegion {
	return SolverRegion{startX, endX, 0, Ny, 0, Nz, solver, 0, 0, 0, 0, 0, 0}
}

func DefSolverRegionY(startY, endY, startZ, endZ, solver int) SolverRegion {
	return SolverRegion{0, Ny, startY, endY, 0, Ny, solver, 0, 0, 0, 0, 0, 0}
}

func DefSolverRegionZ(startZ, endZ, solver int) SolverRegion {
	return SolverRegion{0, Nx, 0, Ny, startZ, endZ, solver, 0, 0, 0, 0, 0, 0}
}

func (s *SolverRegionsStruct) GetStepperSlice() (stepperSlice []Stepper) {
	for _, regionSolver := range s.reg {
		var regStepper Stepper
		switch regionSolver.Solver {
		default:
			util.Fatalf("SetSolver: unknown solver type: %v", regionSolver.Solver)
		/*case BACKWARD_EULER:
			regStepper = new(BackwardEuler)
		case EULER:
			regStepper = new(Euler)
		case HEUN:
			regStepper = new(Heun)
		case BOGAKISHAMPINE:
			regStepper = new(RK23)*/
		case RUNGEKUTTA:
			regStepper = new(RK4)
		}
		/*case DORMANDPRINCE:
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
		}*/
		stepperSlice = append(stepperSlice, regStepper)
	}
	return stepperSlice
}

func (s *SolverRegionsStruct) Run(seconds float64) {
	if len(s.reg) == 0 {
		panic("No solver region defined.")
	}
	Pause = false
	stepperSlice := s.GetStepperSlice()
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
	for (Time < stop) && !Pause {
		ProgressBar.Update(Time)
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
		for i := range stepperSlice {
			stepperSlice[i].StepRegion(&s.reg[i])
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
