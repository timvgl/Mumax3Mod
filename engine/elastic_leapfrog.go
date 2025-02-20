package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Leap frog method.
type elasLF struct{}

func (_ *elasLF) Step() {

	//################
	// Differential equation:
	// du/dt = v(t)
	// dv(t)/dt = [f(t) + bf(t) - eta*g(t)]/rho
	// dv(t)/dt = right
	// with f(t) = nabla sigma
	//#################################
	//Initialisation:
	u := U.Buffer()
	size := u.Size()

	//Set fixed displacement
	//SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)

	v := DU.Buffer()
	v0 := cuda.Buffer(3, size)
	defer cuda.Recycle(v0)
	data.Copy(v0, v)

	//ai = nabla sigma
	ai := cuda.Buffer(3, size)
	defer cuda.Recycle(ai)

	aii := cuda.Buffer(3, size)
	defer cuda.Recycle(aii)

	f := cuda.Buffer(3, size)
	defer cuda.Recycle(f)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	}
	//t0 := Time
	dt := float32(Dt_si)
	util.Assert(dt > 0)
	Time += Dt_si

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv
	//Stage 1:
	calcRhs(ai, f, v)
	cuda.Madd3(u, u0, v0, ai, 1, dt, 0.5*dt*dt)
	//calcBndry()
	calcRhs(aii, f, v)
	cuda.Madd3(v, v0, ai, aii, 1, 0.5*dt, 0.5*dt)

	NSteps++

}

func (_ *elasLF) StepRegion(region *SolverRegion) {
	//################
	// Differential equation:
	// du/dt = v(t)
	// dv(t)/dt = [f(t) + bf(t) - eta*g(t)]/rho
	// dv(t)/dt = right
	// with f(t) = nabla sigma
	//#################################
	//Initialisation:
	//u := U.Buffer()
	u := cuda.Buffer(U.NComp(), region.Size())
	u.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(u)
	U.EvalRegionTo(u)
	size := u.Size()

	m := cuda.Buffer(M.NComp(), region.Size())
	m.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(m)
	M.EvalRegionTo(m)

	//Set fixed displacement
	//SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	u0.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)

	//v := DU.Buffer()
	v := cuda.Buffer(DU.NComp(), region.Size())
	v.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(v)
	DU.EvalRegionTo(v)
	v0 := cuda.Buffer(3, size)
	v0.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(v0)
	data.Copy(v0, v)

	//ai = nabla sigma
	ai := cuda.Buffer(3, size)
	ai.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(ai)

	aii := cuda.Buffer(3, size)
	aii.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(aii)

	f := cuda.Buffer(3, size)
	f.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(f)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	}
	//t0 := Time
	dt := float32(Dt_si)
	util.Assert(dt > 0)
	Time += Dt_si

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv
	//Stage 1:
	calcRhsRegion(ai, m, u, v, f, v)
	cuda.Madd3(u, u0, v0, ai, 1, dt, 0.5*dt*dt)
	//calcBndry()
	calcRhsRegion(aii, m, u, v, f, v)
	cuda.Madd3(v, v0, ai, aii, 1, 0.5*dt, 0.5*dt)

	NSteps++

}

func (_ *elasLF) Free() {}
