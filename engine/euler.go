package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type Euler struct{}

// Euler method, can be used as solver.Step.
func (_ *Euler) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	torqueFn(dy0)
	setMaxTorque(dy0)

	// Adaptive time stepping: treat MaxErr as the maximum magnetization delta
	// (proportional to the error, but an overestimation for sure)
	var dt float32
	if FixDt != 0 {
		Dt_si = FixDt
		dt = float32(Dt_si * GammaLL)
	} else {
		dt = float32(MaxErr / LastTorque)
		Dt_si = float64(dt) / GammaLL
	}
	util.AssertMsg(dt > 0, "Euler solver requires fixed time step > 0")
	setLastErr(float64(dt) * LastTorque)

	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
	M.normalize()
	Time += Dt_si
	NSteps++
}

func (_ *Euler) StepRegion(region SolverRegion) {
	//y := M.Buffer()
	y := cuda.Buffer(M.NComp(), region.Size())
	y.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(y)
	M.EvalRegionTo(y)

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	dy0.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	torqueFnRegion(dy0)
	setMaxTorque(dy0)
	region.LastTorque = LastTorque

	// Adaptive time stepping: treat MaxErr as the maximum magnetization delta
	// (proportional to the error, but an overestimation for sure)
	var dt float32
	if FixDt != 0 {
		Dt_si = FixDt
		dt = float32(Dt_si * GammaLL)
	} else {
		dt = float32(MaxErr / LastTorque)
		Dt_si = float64(dt) / GammaLL
	}
	util.AssertMsg(dt > 0, "Euler solver requires fixed time step > 0")
	setLastErr(float64(dt) * LastTorque)
	region.LastErr = LastErr

	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
	geom := cuda.Buffer(Geometry.Gpu().NComp(), region.Size())
	cuda.Crop(geom, Geometry.Gpu(), region.StartX, region.StartY, region.StartZ)
	defer cuda.Recycle(geom)
	cuda.Normalize(y, geom)
	if FixDt != 0 {
		size := y.Size()
		data.CopyPart(M.Buffer(), y, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
	}
	region.Dt_si = Dt_si
	Time += Dt_si
	NSteps++
}

func (_ *Euler) Free() {}
