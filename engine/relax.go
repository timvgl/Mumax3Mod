package engine

// Relax tries to find the minimum energy state.

import (
	"github.com/mumax/3/cuda"
	"math"
	"github.com/mumax/3/util"
	"fmt"
	//"github.com/progressbar"
)

//Stopping relax Maxtorque in T. The user can check MaxTorque for sane values (e.g. 1e-3).
// If set to 0, relax() will stop when the average torque is steady or increasing.
var (
	RelaxTorqueThreshold 		float64 = -1.
	RelaxFullCoupled 			bool = false
	outputRelax					bool = false
	useHighEta					bool = false
	IterativeHalfing			int = 3
	SlopeTresholdEnergyRelax	float64 = 1e-12
	prefix						string = "relax"
	printSlope					bool = false
	RelaxDDUThreshold			float64 = -1.
	RelaxDUThreshold			float64 = -1.
	precessRelax				bool = false
	relaxTime					float64 = 0.
	
)

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy")
	DeclVar("outputRelax", &outputRelax, "")
	DeclVar("useHighEta", &useHighEta, "")
	DeclVar("__prefix_relax__", &prefix, "")
	DeclVar("__precessRelax__", &precessRelax, "")
	DeclVar("__relaxing__", &relaxing, "")
	DeclVar("__printSlope__", &printSlope, "")
	DeclVar("RelaxDDUThreshold", &RelaxDDUThreshold, "")
	DeclVar("RelaxDUThreshold", &RelaxDUThreshold, "")
	DeclVar("RelaxFullCoupled", &RelaxFullCoupled, "")
	DeclVar("IterativeHalfing", &IterativeHalfing, "Half FixDtCoupledRelaxEnergyE n times an calculate minima for energy and force")
	DeclVar("SlopeTresholdEnergyRelax", &SlopeTresholdEnergyRelax, "")
	DeclVar("RelaxTorqueThreshold", &RelaxTorqueThreshold, "MaxTorque threshold for relax(). If set to -1 (default), relax() will stop when the average torque is steady or increasing.")
}

// are we relaxing?
var relaxing = false

func RelaxMagOutput() {
	SanityCheck()
	Pause = false

	// Save the settings we are changing...
	prevType := Solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	var (
		countOutputOvf = make(map[Quantity]int)
		startOutputOvf = make(map[Quantity]float64)

		countOutputTable int
		startOutputTable float64
	)

	for q, outputElement := range output {
		countOutputOvf[q] = outputElement.count
		startOutputOvf[q] = outputElement.start
	}
	countOutputTable = Table.autosave.count
	startOutputTable = Table.autosave.start
	t_1 := Time

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
		for q, outputElement := range output {
			outputElement.count = countOutputOvf[q]
			outputElement.start = startOutputOvf[q]
		}
		Table.autosave.count = countOutputTable
		Table.autosave.start = startOutputTable
		Time = t_1
	}()
	if (relaxTime != 0.0) {
		Time = relaxTime
	}
	SetSolver(BOGAKISHAMPINE)

	FixDt = 0
	Precess = precessRelax
	relaxing = true


	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	relaxStepsOutput(N, prefix)
	E0 := GetTotalEnergy()
	relaxStepsOutput(N, prefix)
	E1 := GetTotalEnergy()
	for E1 < E0 && !Pause {
		relaxStepsOutput(N, prefix)
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
		for !Pause {
			for maxTorque() > RelaxTorqueThreshold && !Pause {
				relaxStepsOutput(N, prefix)
			}
			MaxErr /= math.Sqrt2
			if MaxErr < 1e-9 {
				break
			}
		}
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		var T0, T1 float32 = 0, GetAverageTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		for MaxErr > 1e-9 && !Pause {
			MaxErr /= math.Sqrt2
			relaxStepsOutput(N, prefix) // TODO: Play with other values
			T0, T1 = T1, avgTorque()
			for T1 < T0 && !Pause {
				relaxStepsOutput(N, prefix) // TODO: Play with other values
				T0, T1 = T1, avgTorque()
			}
		}
	}
	relaxTime = Time
	Pause = true
}

func RelaxCoupledOutput() {
	SanityCheck()
	Pause = false

	// Save the settings we are changing...
	prevType := Solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess
	var (
		countOutputOvf = make(map[Quantity]int)
		startOutputOvf = make(map[Quantity]float64)

		countOutputTable int
		startOutputTable float64
	)

	for q, outputElement := range output {
		countOutputOvf[q] = outputElement.count
		startOutputOvf[q] = outputElement.start
	}
	countOutputTable = Table.autosave.count
	startOutputTable = Table.autosave.start
	
	prevEta := Eta
	t_1 := Time

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		Eta = prevEta
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
		for q, outputElement := range output {
			outputElement.count = countOutputOvf[q]
			outputElement.start = startOutputOvf[q]
		}
		Table.autosave.count = countOutputTable
		Table.autosave.start = startOutputTable
		Time = t_1
	}()

	if (useHighEta == true) {
		Eta.Set(1e6)
	}

	if (relaxTime != 0.0) {
		Time = relaxTime
	}
	
	SetSolver(MAGELAS_RUNGEKUTTA)

	util.AssertMsg(FixDt != 0, "FixDt has to be defined before Relax() if RelaxFullCoupled = true.")
	Precess = precessRelax
	relaxing = true


	// Minimize energy: evolve until SlopeTresholdEnergyRelax for average energy slope of three evaluations is reached
	// then half timesteps and slope, also check if we are in a minimum
	const N = 3 // evaluate energy (expensive) every N steps
	FixDtOrg := FixDt
	FixDt *= 2
	valuePrinted := 0
	for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
		FixDt /= 2
		relaxStepsOutput(N, prefix)
		E0 := GetTotalEnergySystem()
		relaxStepsOutput(N, prefix)
		E1 := GetTotalEnergySystem()
		relaxStepsOutput(N, prefix)
		E2 := GetTotalEnergySystem()
		slope0 := (2*E1-E2-E0) / (2*N*FixDt)
		slope1 := math.NaN()
		slope2 := math.NaN()
		slopesLoaded := false
		//check if energy is extreme point and if it is a minimum
		//check if average of slope is also lower than treshold -> noise leads to problems otherwise
		for (math.Abs(2*E1-E2-E0) / (2*N*FixDt) > 2* SlopeTresholdEnergyRelax / float64(((IterativeHalfingIt +1)*2)) || (math.Abs(slope0) + math.Abs(slope1) + math.Abs(slope2)) / 3 > 2*SlopeTresholdEnergyRelax / float64(((IterativeHalfingIt +1)*2)) || (2*slope1-slope2-slope0) / (2*N*FixDt) < 0 || math.IsNaN(slope1) || math.IsNaN(slope2)) && !Pause {
			relaxStepsOutput(N, prefix)
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
			if printSlope == true {
				fmt.Printf("\rEnergyslope: %e", slope2)
				valuePrinted += 1
			}
		}
	}
	if printSlope == true && valuePrinted > 0 {
		fmt.Printf("\n")
	}
	valuePrinted = 0

	// reset FixDt so that we can half it again later
	FixDt = FixDtOrg
	
	defer stepper.Free() 

	if RelaxTorqueThreshold > 0 {
		// run as long as the max torque is above threshold. Then increase the accuracy and step more.
		// redo that with halfing FixDt again
		var RelaxTorqueThresholdOrg = RelaxTorqueThreshold
		FixDt *= 2
		RelaxTorqueThreshold *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			RelaxTorqueThreshold /= 2
			for !Pause {
				T0 := GetMaxTorque()
				for T0 > RelaxTorqueThreshold && !Pause {
					relaxStepsOutput(N, prefix)
					if printSlope == true {
						fmt.Printf("\rMaximal torque: %e", T0)
						valuePrinted += 1
					}
					T0 = GetMaxTorque()
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		FixDt = FixDtOrg
		RelaxTorqueThreshold = RelaxTorqueThresholdOrg
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		// redo that with halfing FixDt again
		var T0, T1 float32 = 0, GetAverageTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !Pause {
				MaxErr /= math.Sqrt2
				relaxStepsOutput(N, prefix) // TODO: Play with other values
				T0, T1 = T1, GetAverageTorque()
				for T1 < T0 && !Pause {
					relaxStepsOutput(N, prefix) // TODO: Play with other values
					T0, T1 = T1, GetAverageTorque()
					if printSlope == true {
						fmt.Printf("\rAverage torque: %e", T1)
						valuePrinted += 1
					}
				}
			}
		}
		FixDt = FixDtOrg
	}

	if printSlope == true && valuePrinted > 0 {
		fmt.Printf("\n")
	}
	// Now we do the same we did for the torque for the elastic variables: here the second derivative in time of u
	valuePrinted = 0
	if RelaxDDUThreshold > 0 {
		var RelaxDDUThresholdOrg = RelaxDDUThreshold
		var DDU0 float64 = GetMaxDisplacementAcceleration()
		relaxStepsOutput(N, prefix)
		var DDU1 float64 = GetMaxDisplacementAcceleration()
		relaxStepsOutput(N, prefix)
		var DDU2 = GetMaxDisplacementAcceleration()
		slope0DDU := (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
		slope1DDU := math.NaN()
		slope2DDU := math.NaN()
		slopesLoadedDDU := false
		FixDt *= 2
		fmt.Println("hereee")
		RelaxDDUThreshold *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < 1; IterativeHalfingIt++ {
			FixDt /= 2
			RelaxDDUThreshold /= 2
			for !Pause {
				for (math.Abs(2*DDU1-DDU2-DDU0) / (2*N*FixDt) > 2* RelaxDDUThreshold / float64(((IterativeHalfingIt +1)*2)) || (math.Abs(slope0DDU) + math.Abs(slope1DDU) + math.Abs(slope2DDU)) / 3 > 2*RelaxDDUThreshold / float64(((IterativeHalfingIt +1)*2)) || (2*slope1DDU-slope2DDU-slope0DDU) / (2*N*FixDt) < 0 || math.IsNaN(slope1DDU) || math.IsNaN(slope2DDU)) && !Pause {
					relaxStepsOutput(N, prefix)
					if printSlope == true {
						fmt.Printf("\rMaximal acceleration displacement slope: %e", slope0DDU)
						valuePrinted += 1
					}
					if (math.IsNaN(slope1DDU) && !slopesLoadedDDU) {
						slope1DDU = (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
					} else if (math.IsNaN(slope2DDU) && !slopesLoadedDDU) {
						slope2DDU = (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
						slopesLoadedDDU = true
					} else if (math.IsNaN(slope1DDU) && slopesLoadedDDU) {
						slope2DDU = math.NaN()
						slopesLoadedDDU = false
					} else if (math.IsNaN(slope2DDU) && slopesLoadedDDU) {
						slope1DDU = math.NaN()
						slopesLoadedDDU = false
					} else {
						slope0DDU, slope1DDU, slope2DDU = slope1DDU, slope2DDU, (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
					}
					DDU0 = GetMaxDisplacementAcceleration()
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		RelaxDDUThreshold = RelaxDDUThresholdOrg
		FixDt = FixDtOrg
	} else {
		var DDU0, DDU1 float64 = 0, GetAverageDisplacementAcceleration()
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < 1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !Pause {
				MaxErr /= math.Sqrt2
				relaxStepsOutput(N, prefix) // TODO: Play with other values
				DDU0, DDU1 = DDU1, GetAverageDisplacementAcceleration()
				for DDU1 < DDU0 && !Pause {
					relaxStepsOutput(N, prefix) // TODO: Play with other values
					DDU0, DDU1 = DDU1, GetAverageDisplacementAcceleration()
					if printSlope == true {
						fmt.Printf("\rMean acceleration displacement: %e", DDU1)
						valuePrinted += 1
					}
				}
			}
		}
		FixDt = FixDtOrg
	}

	if printSlope == true && valuePrinted > 0 {
		fmt.Printf("\n")
	}

	valuePrinted = 0
	// Now we do the same we did for the torque for the elastic variables: here the first derivative in time of u
	if RelaxDUThreshold > 0 {
		var RelaxDUThresholdOrg = RelaxDUThreshold
		FixDt *= 2
		RelaxDUThreshold *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < 1; IterativeHalfingIt++ {
			FixDt /= 2
			RelaxDUThreshold /= 2
			for !Pause {
				var DU0 = GetMaxDU()
				for DU0 > RelaxDDUThreshold && !Pause {
					relaxStepsOutput(N, prefix)
					if printSlope == true {
						fmt.Printf("\rMaximal velocity displacement: %e", DU0)
						valuePrinted += 1
					}
					DU0 = GetMaxDU()
					
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		FixDt = FixDtOrg
		RelaxDUThreshold = RelaxDUThresholdOrg
	} else {
		var DU0, DU1 float64 = 0, GetAverageDU()
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < 1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !Pause {
				MaxErr /= math.Sqrt2
				relaxStepsOutput(N, prefix) // TODO: Play with other values
				DU0, DU1 = DU1, GetAverageDU()
				for DU1 < DU0 && !Pause {
					relaxStepsOutput(N, prefix) // TODO: Play with other values
					DU0, DU1 = DU1, GetAverageDU()
					if printSlope == true {
						fmt.Printf("\rMean velocity displacement: %e", DU1)
						valuePrinted += 1
					}
				}
			}
		}
		FixDt = FixDtOrg
	}

	if printSlope == true && valuePrinted > 0 {
		fmt.Printf("\n")
	}

	valuePrinted = 0
	relaxTime = Time
	Pause = true
}

func Relax() {
	if outputRelax == false {
		if RelaxFullCoupled == false {
			RelaxMag()
		} else {
			RelaxCoupled()
		}
	} else {
		if RelaxFullCoupled == false {
			RelaxMagOutput()
		} else {
			RelaxCoupledOutput()
		}
	}
}

func RelaxCoupled() {
	SanityCheck()
	Pause = false

	// Save the settings we are changing...
	prevType := Solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess
	prevEta := Eta

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		Eta = prevEta
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
	}()

	if (useHighEta == true) {
		Eta.Set(1e6)
	}
	
	SetSolver(MAGELAS_RUNGEKUTTA)

	//FixDt = 0
	util.AssertMsg(FixDt != 0, "FixDt has to be defined before Relax() if RelaxFullCoupled = true.")
	Precess = precessRelax
	relaxing = true

	// Minimize energy: evolve until SlopeTresholdEnergyRelax for average energy slope of three evaluations is reached
	// then half timesteps and slope, also check if we are in a minimum
	const N = 3 // evaluate energy (expensive) every N steps
	FixDtOrg := FixDt
	FixDt *= 2
	valuePrinted := 0
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
		for (math.Abs(2*E1-E2-E0) / (2*N*FixDt) > 2* SlopeTresholdEnergyRelax / float64(((IterativeHalfingIt +1)*2)) || (math.Abs(slope0) + math.Abs(slope1) + math.Abs(slope2)) / 3 > 2*SlopeTresholdEnergyRelax / float64(((IterativeHalfingIt +1)*2)) || (2*slope1-slope2-slope0) / (2*N*FixDt) < 0 || math.IsNaN(slope1) || math.IsNaN(slope2)) && !Pause {
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
			if printSlope == true {
				fmt.Printf("\rEnergyslope: %e", slope2)
				valuePrinted += 1
			}
		}
	}
	if printSlope == true && valuePrinted > 1 {
		fmt.Printf("\n")
	}
	valuePrinted = 0
	
	// reset FixDt so that we can half it again later
	FixDt = FixDtOrg
	defer stepper.Free() 

	if RelaxTorqueThreshold > 0 {
		// run as long as the max torque is above threshold. Then increase the accuracy and step more.
		// redo that with halfing FixDt and RelaxTorqueThreshold again
		var RelaxTorqueThresholdOrg = RelaxTorqueThreshold
		FixDt *= 2
		RelaxTorqueThreshold *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			RelaxTorqueThreshold /= 2
			for !Pause {
				T0 := GetMaxTorque()
				for T0 > RelaxTorqueThreshold && !Pause {
					relaxSteps(N)
					if printSlope == true {
						fmt.Printf("\rMaximal torque: %e", T0)
						valuePrinted += 1
					}
					T0 = GetMaxTorque()
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		FixDt = FixDtOrg
		RelaxTorqueThreshold = RelaxTorqueThresholdOrg
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		var T0, T1 float32 = 0, GetAverageTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		// redo that with halfing FixDt again
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !Pause {
				MaxErr /= math.Sqrt2
				relaxSteps(N) // TODO: Play with other values
				T0, T1 = T1, GetAverageTorque()
				for T1 < T0 && !Pause {
					relaxSteps(N) // TODO: Play with other values
					T0, T1 = T1, GetAverageTorque()
					if printSlope == true {
						fmt.Printf("\rAverage torque: %e", T1)
						valuePrinted += 1
					}
				}
			}
		}
		FixDt = FixDtOrg
	}

	if printSlope == true && valuePrinted > 1 {
		fmt.Printf("\n")
	}

	valuePrinted = 0
	if RelaxDDUThreshold > 0 {
		var RelaxDDUThresholdOrg = RelaxDDUThreshold
		var DDU0 float64 = GetMaxDisplacementAcceleration()
		relaxSteps(N)
		var DDU1 float64 = GetMaxDisplacementAcceleration()
		relaxSteps(N)
		var DDU2 = GetMaxDisplacementAcceleration()
		slope0DDU := (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
		slope1DDU := math.NaN()
		slope2DDU := math.NaN()
		slopesLoadedDDU := false
		FixDt *= 2
		RelaxDDUThreshold *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < 1; IterativeHalfingIt++ {
			FixDt /= 2
			RelaxDDUThreshold /= 2
			for !Pause {
				for (math.Abs(2*DDU1-DDU2-DDU0) / (2*N*FixDt) > 2* RelaxDDUThreshold / float64(((IterativeHalfingIt +1)*2)) || (math.Abs(slope0DDU) + math.Abs(slope1DDU) + math.Abs(slope2DDU)) / 3 > 2*RelaxDDUThreshold / float64(((IterativeHalfingIt +1)*2)) || (2*slope1DDU-slope2DDU-slope0DDU) / (2*N*FixDt) < 0 || math.IsNaN(slope1DDU) || math.IsNaN(slope2DDU)) && !Pause {
					relaxSteps(N)
					DDU0, DDU1, DDU2 = DDU1, DDU2, GetMaxDisplacementAcceleration()

					if printSlope == true {
						fmt.Printf("\rMaximal acceleration displacement slope: %e", slope2DDU)
						valuePrinted += 1
					}
					if (math.IsNaN(slope1DDU) && !slopesLoadedDDU) {
						slope1DDU = (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
					} else if (math.IsNaN(slope2DDU) && !slopesLoadedDDU) {
						slope2DDU = (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
						slopesLoadedDDU = true
					} else if (math.IsNaN(slope1DDU) && slopesLoadedDDU) {
						slope2DDU = math.NaN()
						slopesLoadedDDU = false
					} else if (math.IsNaN(slope2DDU) && slopesLoadedDDU) {
						slope1DDU = math.NaN()
						slopesLoadedDDU = false
					} else {
						slope0DDU, slope1DDU, slope2DDU = slope1DDU, slope2DDU, (2*DDU1-DDU2-DDU0) / (2*N*FixDt)
					}
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		RelaxDDUThreshold = RelaxDDUThresholdOrg
		FixDt = FixDtOrg
	} else {
		var DDU0, DDU1 float64 = 0, GetAverageDisplacementAcceleration()
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < 1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !Pause {
				MaxErr /= math.Sqrt2
				relaxSteps(N) // TODO: Play with other values
				DDU0, DDU1 = DDU1, GetAverageDisplacementAcceleration()
				for DDU1 < DDU0 && !Pause {
					relaxSteps(N) // TODO: Play with other values
					DDU0, DDU1 = DDU1, GetAverageDisplacementAcceleration()
					if printSlope == true {
						fmt.Printf("\rMean acceleration displacement: %e", DDU1)
						valuePrinted += 1
					}
				}
			}
		}
		FixDt = FixDtOrg
	}

	if printSlope == true && valuePrinted > 1 {
		fmt.Printf("\n")
	}

	valuePrinted = 0
	// Now we do the same we did for the torque for the elastic variables: here the first derivative in time of u
	if RelaxDUThreshold > 0 {
		var RelaxDUThresholdOrg = RelaxDUThreshold
		FixDt *= 2
		RelaxDUThreshold *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			RelaxDUThreshold /= 2
			for !Pause {
				var DU0 = GetMaxDU()
				for DU0 > RelaxDDUThreshold && !Pause {
					relaxSteps(N)
					if printSlope == true {
						fmt.Printf("\rMaximal velocity displacement: %e", DU0)
						valuePrinted += 1
					}
					DU0 = GetMaxDU()
					
				}
				MaxErr /= math.Sqrt2
				if MaxErr < 1e-9 {
					break
				}
			}
		}
		FixDt = FixDtOrg
		RelaxDUThreshold = RelaxDUThresholdOrg
	} else {
		var DU0, DU1 float64 = 0, GetAverageDU()
		FixDt *= 2
		for IterativeHalfingIt := 0; IterativeHalfingIt < IterativeHalfing +1; IterativeHalfingIt++ {
			FixDt /= 2
			for MaxErr > 1e-9 && !Pause {
				MaxErr /= math.Sqrt2
				relaxSteps(N) // TODO: Play with other values
				DU0, DU1 = DU1, GetAverageDU()
				for DU1 < DU0 && !Pause {
					relaxSteps(N) // TODO: Play with other values
					DU0, DU1 = DU1, GetAverageDU()
					if printSlope == true {
						fmt.Printf("\rMean velocity displacement: %e", DU1)
						valuePrinted += 1
					}
				}
			}
		}
		FixDt = FixDtOrg
	}

	if printSlope == true && valuePrinted > 1 {
		fmt.Printf("\n")
	}

	valuePrinted = 0

	Pause = true
}

func RelaxMag() {
	SanityCheck()
	Pause = false

	// Save the settings we are changing...
	prevType := Solvertype
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
	Precess = precessRelax
	relaxing = true


	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	relaxSteps(N)
	E0 := GetTotalEnergy()
	relaxSteps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !Pause {
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
		for !Pause {
			for maxTorque() > RelaxTorqueThreshold && !Pause {
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
		for MaxErr > 1e-9 && !Pause {
			MaxErr /= math.Sqrt2
			relaxSteps(N) // TODO: Play with other values
			T0, T1 = T1, avgTorque()
			for T1 < T0 && !Pause {
				relaxSteps(N) // TODO: Play with other values
				T0, T1 = T1, avgTorque()
			}
		}
	}
	Pause = true
}

func relaxStepsOutput(n int, prefix string) {
	//t0 := Time
	stop := NSteps + n
	cond := func() bool { return NSteps < stop }
	const output = true
	RunWhileRelax(cond, prefix)
	//Time = t0
}

// take n steps without setting Pause when done or advancing time
func relaxSteps(n int) {
	t0 := Time
	stop := NSteps + n
	cond := func() bool { return NSteps < stop }
	const output = false
	runWhile(cond, output)
	Time = t0
}
