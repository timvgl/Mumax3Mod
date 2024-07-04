package engine

import (
	"fmt"
	"math"
	"os"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Solver globals
var (
	Time                    			float64                      // time in seconds
	alarm                   			float64                      // alarm clock marks end time of run, dt adaptation must not cross it!
	pause                   			= true                       // set pause at any time to stop running after the current step
	postStep                			[]func()                     // called on after every full time step
	Inject                           		 	 = make(chan func()) // injects code in between time steps. Used by web interface.
	Dt_si                   			float64  = 1e-15             // time step = dt_si (seconds) *dt_mul, which should be nice float32
	MinDt, MaxDt            			float64                      // minimum and maximum time step
	MaxErr                  			float64  = 1e-5              // maximum error/step
	Headroom                			float64  = 0.8               // solver headroom, (Gustafsson, 1992, Control of Error and Convergence in ODE Solvers)
	LastErr, PeakErr        			float64                      // error of last step, highest error ever
	LastTorque              			float64                      // maxTorque of last time step
	NSteps, NUndone, NEvals 			int                          // number of good steps, undone steps
	FixDt                   			float64                      // fixed time step?
	stepper                 			Stepper                      // generic step, can be EulerStep, HeunStep, etc
	solvertype              			int
	CorePosRT						  			= make([]float64, 3) //core Position of Vortex
	CorePosRTR							float64
	BoolAllowInhomogeniousMECoupling	bool 	= false
	useBoundaries						bool 	= false
	//InsertTimeDepDisplacement 			int		= 0					 //1 for True, 0 for False
	//InsertTimeDepDisplacementFunc 		func(arg1, arg2, arg3, arg4, arg5 float64) Config //func for calc displacement that is supposed to be added
	//InsertTimeDepDisplacementFuncArgs	[]func(t float64) float64	 //slices of funcs that are going to be used as args for InsertTimeDepDisplacementFunc
)

func init() {
	DeclFunc("Run", Run, "Run the simulation for a time in seconds")
	DeclFunc("Steps", Steps, "Run the simulation for a number of time steps")
	DeclFunc("RunWhile", RunWhile, "Run while condition function is true")
	DeclFunc("SetSolver", SetSolver, "Set solver type. 1:Euler, 2:Heun, 3:Bogaki-Shampine, 4: Runge-Kutta (RK45), 5: Dormand-Prince, 6: Fehlberg, -1: Backward Euler")
	DeclFunc("Activate_corePosScriptAccess", Activate_corePosScriptAccess, "Activates the availability of the vortex core position as live data")
	DeclFunc("Activate_corePosScriptAccessR", Activate_corePosScriptAccessR, "")
	//DeclFunc("Set_InsertTimeDepDisplacement_to", Set_InsertTimeDepDisplacement_to, "Activates the insertion of timedep displacement into MAGELAS_RUNGEKUTTA")
	DeclFunc("AllowInhomogeniousMECoupling", AllowInhomogeniousMECoupling, "Bypasses an error that is going to be raised if B1 or B2 is inhomogenious")
	DeclVar("CorePosRT", &CorePosRT, "Vortex core position in real time")
	DeclVar("CorePosRTR", &CorePosRTR, "Radius of CorePosRT")
	NewScalarValue("CorePosR", "m", "Radius of CorePosRT", func() float64 { return getRadiusVortexCore() })
	DeclVar("useBoundaries", &useBoundaries, "")
	DeclTVar("t", &Time, "Total simulated time (s)")
	DeclVar("step", &NSteps, "Total number of time steps taken")
	DeclVar("MinDt", &MinDt, "Minimum time step the solver can take (s)")
	DeclVar("MaxDt", &MaxDt, "Maximum time step the solver can take (s)")
	DeclVar("MaxErr", &MaxErr, "Maximum error per step the solver can tolerate (default = 1e-5)")
	DeclVar("Headroom", &Headroom, "Solver headroom (default = 0.8)")
	DeclVar("FixDt", &FixDt, "Set a fixed time step, 0 disables fixed step (which is the default)")

	DeclFunc("Exit", Exit, "Exit from the program")
	//DeclVar("BoolAllowInhomogeniousMECoupling", BoolAllowInhomogeniousMECoupling, "Bypasses an error that is going to be raised if B1 or B2 is inhomogenious, bool")
	SetSolver(DORMANDPRINCE)
	_ = NewScalarValue("dt", "s", "Time Step", func() float64 { return Dt_si })
	_ = NewScalarValue("LastErr", "", "Error of last step", func() float64 { return LastErr })
	_ = NewScalarValue("PeakErr", "", "Overall maxium error per step", func() float64 { return PeakErr })
	_ = NewScalarValue("NEval", "", "Total number of torque evaluations", func() float64 { return float64(NEvals) })
	//_ = NewScalarValue("InsertTimeDepDisplacement", "", "Wherether time and space dependent displacement should be inserted", func() float64 { return float64(InsertTimeDepDisplacement)})
}

// Time stepper like Euler, Heun, RK23
type Stepper interface {
	Step() // take time step using solver globals
	Free() // free resources, if any (e.g.: RK23 previous torque)
}

// Arguments for SetSolver
const (
	BACKWARD_EULER     = -1
	EULER              = 1
	HEUN               = 2
	BOGAKISHAMPINE     = 3
	RUNGEKUTTA         = 4
	DORMANDPRINCE      = 5
	FEHLBERG           = 6
	SECONDDERIV        = 7
	ELAS_RUNGEKUTTA    = 8
	MAGELAS_RUNGEKUTTA = 9
	ELAS_LEAPFROG      = 10
	ELAS_YOSH          = 11
)

func SetSolver(typ int) {
	// free previous solver, if any
	if stepper != nil {
		stepper.Free()
	}
	switch typ {
	default:
		util.Fatalf("SetSolver: unknown solver type: %v", typ)
	case BACKWARD_EULER:
		stepper = new(BackwardEuler)
	case EULER:
		stepper = new(Euler)
	case HEUN:
		stepper = new(Heun)
	case BOGAKISHAMPINE:
		stepper = new(RK23)
	case RUNGEKUTTA:
		stepper = new(RK4)
	case DORMANDPRINCE:
		stepper = new(RK45DP)
	case FEHLBERG:
		stepper = new(RK56)
	case SECONDDERIV:
		stepper = new(secondHeun)
	case ELAS_RUNGEKUTTA:
		stepper = new(elasRK4)
	case MAGELAS_RUNGEKUTTA:
		stepper = new(magelasRK4)
	case ELAS_LEAPFROG:
		stepper = new(elasLF)
	case ELAS_YOSH:
		stepper = new(elasYOSH)
	}
	solvertype = typ
}

// write torque to dst and increment NEvals
func torqueFn(dst *data.Slice) {
	SetTorque(dst)
	NEvals++
}

// returns number of torque evaluations
func getNEval() int {
	return NEvals
}

// update lastErr and peakErr
func setLastErr(err float64) {
	LastErr = err
	if err > PeakErr {
		PeakErr = err
	}
}

func setMaxTorque(τ *data.Slice) {
	LastTorque = cuda.MaxVecNorm(τ)
}

// adapt time step: dt *= corr, but limited to sensible values.
func adaptDt(corr float64) {
	if FixDt != 0 {
		Dt_si = FixDt
		return
	}

	// corner case triggered by err = 0: just keep time step.
	// see test/regression017.mx3
	if math.IsNaN(corr) {
		corr = 1
	}

	util.AssertMsg(corr != 0, "Time step too small, check if parameters are sensible: ")
	corr *= Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	Dt_si *= corr
	if MinDt != 0 && Dt_si < MinDt {
		Dt_si = MinDt
	}
	if MaxDt != 0 && Dt_si > MaxDt {
		Dt_si = MaxDt
	}
	if Dt_si == 0 {
		util.Fatal("time step too small")
	}

	// do not cross alarm time
	if Time < alarm && Time+Dt_si > alarm {
		Dt_si = alarm - Time
	}

	util.AssertMsg(Dt_si > 0, fmt.Sprint("Time step too small: ", Dt_si))
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	stop := Time + seconds
	alarm = stop // don't have dt adapt to go over alarm
	RunWhile(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	stop := NSteps + n
	RunWhile(func() bool { return NSteps < stop })
}

// Runs as long as condition returns true, saves output.
func RunWhile(condition func() bool) {
	SanityCheck()
	pause = false // may be set by <-Inject
	const output = true
	stepper.Free() // start from a clean state
	runWhile(condition, output)
	pause = true
}

func runWhile(condition func() bool, output bool) {
	DoOutput() // allow t=0 output
	for condition() && !pause {
		select {
		default:
			step(output)
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
}

func RunWhileRelax(condition func() bool, prefix string) {
	const output = true
	DoOutputPrefix(prefix) // allow t=0 output
	for condition() && !pause {
		select {
		default:
			stepPrefix(output, prefix)
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
}

func stepPrefix(output bool, prefix string) {
	stepper.Step()
	for _, f := range postStep {
		f()
	}
	if output {
		DoOutputPrefix(prefix)
	}
}

// Runs as long as browser is connected to gui.
func RunInteractive() {
	gui_.RunInteractive()
}

// take one time step
func step(output bool) {
	stepper.Step()
	for _, f := range postStep {
		f()
	}
	if output {
		DoOutput()
	}
}

// Register function f to be called after every time step.
// Typically used, e.g., to manipulate the magnetization.
func PostStep(f func()) {
	postStep = append(postStep, f)
}

// inject code into engine and wait for it to complete.
func InjectAndWait(task func()) {
	ready := make(chan int)
	Inject <- func() { task(); ready <- 1 }
	<-ready
}

func SanityCheck() {
	if Msat.isZero() {
		util.Log("Note: Msat = 0")
	}
	if Aex.isZero() {
		util.Log("Note: Aex = 0")
	}
}

func Activate_corePosScriptAccess() {
	PostStep(func() {CorePosRT = corePos()})
}

func getRadiusVortexCore() float64 {
	CorePosRT = corePos()
	return math.Sqrt(math.Pow(CorePosRT[0], 2) + math.Pow(CorePosRT[1], 2))
}

func Activate_corePosScriptAccessR() {
	PostStep(func() {
		CorePosRTR = getRadiusVortexCore()	
	})
}

func AllowInhomogeniousMECoupling() {
	BoolAllowInhomogeniousMECoupling = !BoolAllowInhomogeniousMECoupling
}

/*
func Set_InsertTimeDepDisplacement_to(stat int, cfg func(arg1, arg2, arg3, arg4, arg5 float64) Config, args []func(t float64) float64) {
	InsertTimeDepDisplacement = stat
	InsertTimeDepDisplacementFunc = cfg
	InsertTimeDepDisplacementFuncArgs = args
}*/

func Exit() {
	Close()
	os.Exit(0)
}
