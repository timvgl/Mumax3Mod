package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Implicit midpoint solver.
type BackwardEuler struct {
	dy1 *data.Slice
}

// Euler method, can be used as solver.Step.
func (s *BackwardEuler) Step() {
	util.AssertMsg(MaxErr > 0, "Backward euler solver requires MaxErr > 0")

	t0 := Time

	y := M.Buffer()

	y0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(y0)
	data.Copy(y0, y)

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	if s.dy1 == nil {
		s.dy1 = cuda.Buffer(VECTOR, y.Size())
	}
	dy1 := s.dy1

	Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)
	util.AssertMsg(dt > 0, "Backward Euler solver requires fixed time step > 0")

	// Fist guess
	Time = t0 + 0.5*Dt_si // 0.5 dt makes it implicit midpoint method

	// with temperature, previous torque cannot be used as predictor
	if Temp.isZero() {
		cuda.Madd2(y, y0, dy1, 1, dt) // predictor euler step with previous torque
		M.normalize()
	}

	torqueFn(dy0)
	cuda.Madd2(y, y0, dy0, 1, dt) // y = y0 + dt * dy
	M.normalize()

	// One iteration
	torqueFn(dy1)
	cuda.Madd2(y, y0, dy1, 1, dt) // y = y0 + dt * dy1
	M.normalize()

	Time = t0 + Dt_si

	err := cuda.MaxVecDiff(dy0, dy1) * float64(dt)

	NSteps++
	setLastErr(err)
	setMaxTorque(dy1)
}

func (s *BackwardEuler) StepRegion(region *SolverRegion) {
	util.AssertMsg(MaxErr > 0, "Backward euler solver requires MaxErr > 0")

	t0 := Time

	//y := M.Buffer()
	y := cuda.Buffer(M.NComp(), region.Size())
	y.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(y)
	M.EvalRegionTo(y)

	geom := cuda.Buffer(Geometry.Gpu().NComp(), region.Size())
	GeomBig, _ := Geometry.Slice()
	cuda.Crop(geom, GeomBig, region.StartX, region.StartY, region.StartZ)
	defer cuda.Recycle(geom)
	cuda.Recycle(GeomBig)

	y0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(y0)
	data.Copy(y0, y)

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	if s.dy1 == nil {
		s.dy1 = cuda.Buffer(VECTOR, y.Size())
	}
	dy1Pre := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy1Pre)
	data.Copy(dy1Pre, s.dy1)
	dy1 := s.dy1

	if FixDt != 0 {
		Dt_si = FixDt
	} else {
		Dt_si = region.Dt_si
	}
	dt := float32(Dt_si * GammaLL)
	util.AssertMsg(dt > 0, "Backward Euler solver requires fixed time step > 0")

	// Fist guess
	Time = t0 + 0.5*Dt_si // 0.5 dt makes it implicit midpoint method

	// with temperature, previous torque cannot be used as predictor
	if Temp.isZero() {
		cuda.Madd2(y, y0, dy1, 1, dt) // predictor euler step with previous torque
		cuda.Normalize(y, geom)
	}
	dy0.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	torqueFnRegionNEW(dy0, y, region.PBCx, region.PBCy, region.PBCz)
	cuda.Madd2(y, y0, dy0, 1, dt) // y = y0 + dt * dy
	cuda.Normalize(y, geom)

	// One iteration
	dy1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	torqueFnRegionNEW(dy1, y, region.PBCx, region.PBCy, region.PBCz)
	cuda.Madd2(y, y0, dy1, 1, dt) // y = y0 + dt * dy1
	cuda.Normalize(y, geom)
	if FixDt != 0 {
		size := y.Size()
		data.CopyPart(M.Buffer(), y, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
	} else {
		data.Copy(s.dy1, dy1Pre)
	}
	Time = t0 + Dt_si

	err := cuda.MaxVecDiff(dy0, dy1) * float64(dt)

	NSteps++
	setLastErr(err)
	region.LastErr = LastErr
	setMaxTorque(dy1)
	region.LastTorque = LastTorque
	region.Dt_si = Dt_si
}

func (s *BackwardEuler) Free() {
	s.dy1.Free()
	s.dy1 = nil
}
