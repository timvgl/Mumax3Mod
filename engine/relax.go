package engine

// Relax tries to find the minimum energy state.

import (
	"github.com/mumax/3/cuda"
	"math"
	"github.com/mumax/3/util"
	//"fmt"
	//"github.com/progressbar"
)

//Stopping relax Maxtorque in T. The user can check MaxTorque for sane values (e.g. 1e-3).
// If set to 0, relax() will stop when the average torque is steady or increasing.
var (
	RelaxTorqueThreshold 		float64 = -1.
	RelaxFullCoupled 			bool = false
	IterativeHalfing			int
	SlopeTresholdEnergyRelax	float64
)

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy")
	DeclFunc("RelaxOutput", RelaxOutput, "Try to minimize the total energy")
	DeclVar("RelaxFullCoupled", &RelaxFullCoupled, "")
	DeclVar("IterativeHalfing", &IterativeHalfing, "Half FixDtCoupledRelaxEnergyE n times an calculate minima for energy and force")
	DeclVar("SlopeTresholdEnergyRelax", &SlopeTresholdEnergyRelax, "")
	DeclVar("RelaxTorqueThreshold", &RelaxTorqueThreshold, "MaxTorque threshold for relax(). If set to -1 (default), relax() will stop when the average torque is steady or increasing.")
}

// are we relaxing?
var relaxing = false

func RelaxOutput() {
	SanityCheck()
	pause = false

	// Save the settings we are changing...
	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
	}()
	SetSolver(BOGAKISHAMPINE)

	FixDt = 0
	Precess = true
	relaxing = false


	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	relaxStepsOutput(N)
	E0 := GetTotalEnergy()
	relaxStepsOutput(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !pause {
		relaxStepsOutput(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	// Now we are already close to equilibrium, but energy is too noisy to be used any further.
	// So now we minimize the torque which is less noisy.
	solver := stepper.(*RK23)
	defer stepper.Free() // purge previous rk.k1 because FSAL will be dead wrong.

	maxTorque := func() float64 {
		
		return cuda.MaxVecNorm(solver.k1)
	}
	avgTorque := func() float32 {
		return cuda.Dot(solver.k1, solver.k1)
	}

	if RelaxTorqueThreshold > 0 {
		// run as long as the max torque is above threshold. Then increase the accuracy and step more.
		for !pause {
			for maxTorque() > RelaxTorqueThreshold && !pause {
				relaxStepsOutput(N)
			}
			MaxErr /= math.Sqrt2
			if MaxErr < 1e-9 {
				break
			}
		}
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		var T0, T1 float32 = 0, avgTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		for MaxErr > 1e-9 && !pause {
			MaxErr /= math.Sqrt2
			relaxStepsOutput(N) // TODO: Play with other values
			T0, T1 = T1, avgTorque()
			for T1 < T0 && !pause {
				relaxStepsOutput(N) // TODO: Play with other values
				T0, T1 = T1, avgTorque()
			}
		}
	}
	pause = true
}

func Relax() {
	if RelaxFullCoupled == false {
		RelaxMag()
	} else {
		RelaxCoupled()
	}
}

func RelaxCoupled() {
	SanityCheck()
	pause = false

	// Save the settings we are changing...
	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
	}()
	SetSolver(MAGELAS_RUNGEKUTTA)

	//FixDt = 0
	util.AssertMsg(FixDt != 0, "FixDt has to be defined before Relax() if RelaxFullCoupled = true.")
	Precess = false
	relaxing = true


	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	FixDtOrg := FixDt
	FixDt *= 2
	for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
		FixDt /= 2
		relaxSteps(N)
		E0 := GetTotalEnergySystem()
		relaxSteps(N)
		E1 := GetTotalEnergySystem()
		relaxSteps(N)
		E2 := GetTotalEnergySystem()
		slope0 := (2*E1-E2-E0) / (2*N*FixDt)
		slope1 := math.NaN()
		slope2 := math.NaN()
		slopesLoaded := false
		//check if energy is extreme point and if it is a minimum
		//check if average of slope is also lower than treshold -> noise leads to problems otherwise
		for (math.Abs(2*E1-E2-E0) / (2*N*FixDt) > 2* SlopeTresholdEnergyRelax / float64(((IterativeHalfingIt +1)*2)) || (math.Abs(slope0) + math.Abs(slope1) + math.Abs(slope2)) / 3 > 2*SlopeTresholdEnergyRelax / float64(((IterativeHalfingIt +1)*2)) || (2*slope1-slope2-slope0) / (2*N*FixDt) < 0|| math.IsNaN(slope1) || math.IsNaN(slope2)) && !pause {
			relaxSteps(N)
			E0, E1, E2 = E1, E2, GetTotalEnergySystem()
			//fmt.Println("slope", math.Abs(2*E1-E2-E0) / (2*N*FixDt))
			if (math.IsNaN(slope1) && !slopesLoaded) {
				slope1 = (2*E1-E2-E0) / (2*N*FixDt)
			} else if (math.IsNaN(slope2) && !slopesLoaded) {
				slope2 = (2*E1-E2-E0) / (2*N*FixDt)
				slopesLoaded = true
			} else if (math.IsNaN(slope1) && slopesLoaded) {
				slope2 = math.NaN()
				slopesLoaded = false
			} else if (math.IsNaN(slope2) && slopesLoaded) {
				slope1 = math.NaN()
				slopesLoaded = false
			} else {
				slope0, slope1, slope2 = slope1, slope2, (2*E1-E2-E0) / (2*N*FixDt)
			}
			//fmt.Println("slopeAv", (math.Abs(slope0) + math.Abs(slope1) + math.Abs(slope2)) / 3)
			//fmt.Println("slope", math.Abs(slope2))
			//fmt.Println("slope2", (2*slope1-slope2-slope0) / (2*N*FixDt))
		}
	}

	FixDt = FixDtOrg
	// Now we are already close to equilibrium, but energy is too noisy to be used any further.
	// So now we minimize the torque which is less noisy.
	defer stepper.Free() // purge previous rk.k1 because FSAL will be dead wrong.

	if RelaxTorqueThreshold > 0 {
		// run as long as the max torque is above threshold. Then increase the accuracy and step more.
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			for !pause {
				for GetMaxTorque() > RelaxTorqueThreshold && !pause {
					relaxSteps(N)
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		FixDt = FixDtOrg
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		var T0, T1 float32 = 0, GetAverageTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !pause {
				MaxErr /= math.Sqrt2
				relaxSteps(N) // TODO: Play with other values
				T0, T1 = T1, GetAverageTorque()
				for T1 < T0 && !pause {
					relaxSteps(N) // TODO: Play with other values
					T0, T1 = T1, GetAverageTorque()
				}
			}
		}
		FixDt = FixDtOrg
	}
	pause = true
}

func RelaxMag() {
	SanityCheck()
	pause = false

	// Save the settings we are changing...
	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
	}()
	SetSolver(BOGAKISHAMPINE)

	FixDt = 0
	Precess = false
	relaxing = true


	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	relaxSteps(N)
	E0 := GetTotalEnergy()
	relaxSteps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !pause {
		relaxSteps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	// Now we are already close to equilibrium, but energy is too noisy to be used any further.
	// So now we minimize the torque which is less noisy.
	solver := stepper.(*RK23)
	defer stepper.Free() // purge previous rk.k1 because FSAL will be dead wrong.

	maxTorque := func() float64 {
		
		return cuda.MaxVecNorm(solver.k1)
	}
	avgTorque := func() float32 {
		return cuda.Dot(solver.k1, solver.k1)
	}

	if RelaxTorqueThreshold > 0 {
		// run as long as the max torque is above threshold. Then increase the accuracy and step more.
		for !pause {
			for maxTorque() > RelaxTorqueThreshold && !pause {
				relaxSteps(N)
			}
			MaxErr /= math.Sqrt2
			if MaxErr < 1e-9 {
				break
			}
		}
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		var T0, T1 float32 = 0, avgTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		for MaxErr > 1e-9 && !pause {
			MaxErr /= math.Sqrt2
			relaxSteps(N) // TODO: Play with other values
			T0, T1 = T1, avgTorque()
			for T1 < T0 && !pause {
				relaxSteps(N) // TODO: Play with other values
				T0, T1 = T1, avgTorque()
			}
		}
	}
	pause = true
}

func relaxStepsOutput(n int) {
	t0 := Time
	stop := NSteps + n
	cond := func() bool { return NSteps < stop }
	const output = true
	runWhile(cond, output)
	Time = t0
}

// take n steps without setting pause when done or advancing time
func relaxSteps(n int) {
	t0 := Time
	stop := NSteps + n
	cond := func() bool { return NSteps < stop }
	const output = false
	runWhile(cond, output)
	Time = t0
}
