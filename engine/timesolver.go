package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/odeint"
)

var Slvr *odeint.Solver

func init() {
	DeclFunc("InitTimeSolver", InitTimeSolver, "Initialize time solver")
	DeclVar("TimeSolver", &Slvr, "The time solver")
	DeclVar("Time", &Time, "")
}

// Returns the right hand side of the LLG equation.
// This is the torque -m x H multiplied by GammaLL.
// TODO: check why mumax3 original solver does not
//       multiply the torque with GammaLL
func LLGrhs(dst *data.Slice) {
	torqueFn(dst)
	cuda.Madd2(dst, dst, dst, float32(GammaLL), 0) // TODO: do this more efficiently
}

func InitTimeSolver() {
	Time = 0
	Slvr = odeint.NewSolver(&Time, FixDt)
	//Slvr.AddODE(M.Buffer(), LLGrhs) // dm/dt = torqueFn
	//Slvr.AddPostChangeFunction(M.normalize)
	//Slvr.AddODE(U.Buffer(), calcRhs) // du/dt = torqueFn
	Slvr.SetInjectChannel(Inject)
}

// --- HARD CODED EXAMPLE BELOW ----- // TODO: remove this example

var UU *varVectorField

func init() {
	//DeclLValue("uu", UU, `examplatory variable UU`)
	DeclFunc("AddCustomEquationForU", AddCustomEquationForU, "Add custom equation for u")

	// make a variable field to play with
	UU = new(varVectorField)
	UU.name = "uu"
	UU.unit = ""
	UU.normalized = false
	DeclLValue("uu", UU, `field to play with`)
}

func AddCustomEquationForU(rhs Quantity) {
	UU.alloc()
	Slvr.AddODE(UU.Buffer(), rhs.EvalTo)
}
