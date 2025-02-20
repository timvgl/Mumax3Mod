package engine

import (
	"math"

	"github.com/mumax/3/util"
)

var (
	SolverRegions = SolverRegionsStruct{make([]SolverRegion, 0)}
)

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
}

func DefSolverRegion(startX, endX, startY, endY, startZ, endZ, solver int) SolverRegion {
	return SolverRegion{startX, endX, startY, endY, startZ, endZ, solver, 0, 0, 0}
}

func (s SolverRegionsStruct) GetStepperSlice() (stepperSlice []Stepper) {
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
	}
	return stepperSlice
}

func (s SolverRegionsStruct) Run(seconds float64) {
	if len(s.reg) == 0 {
		panic("No solver region defined.")
	}
	stepperSlice := s.GetStepperSlice()
	stop := Time + seconds
	FixDtPre := FixDt
	for (Time < stop) && !Pause {
		if FixDtPre == 0 {
			Time0 := Time
			for i := range stepperSlice {
				stepperSlice[i].StepRegion(s.reg[i])
			}
			minSiDt := math.Inf(1)
			for i := range s.reg {
				if minSiDt > s.reg[i].Dt_si {
					minSiDt = s.reg[i].Dt_si
				}
			}
			Time = Time0
			FixDt = minSiDt
		}
		for i := range stepperSlice {
			stepperSlice[i].StepRegion(s.reg[i])
		}
		FixDt = FixDtPre
	}
}

func (s SolverRegion) Size() [3]int {
	return [3]int{s.EndX - s.StartX, s.EndY - s.StartY, s.EndZ - s.StartZ}
}
