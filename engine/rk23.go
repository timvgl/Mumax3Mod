package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Bogacki-Shampine solver. 3rd order, 3 evaluations per step, adaptive step.
//
//	http://en.wikipedia.org/wiki/Bogacki-Shampine_method
//
//	k1 = f(tn, yn)
//	k2 = f(tn + 1/2 h, yn + 1/2 h k1)
//	k3 = f(tn + 3/4 h, yn + 3/4 h k2)
//	y{n+1}  = yn + 2/9 h k1 + 1/3 h k2 + 4/9 h k3            // 3rd order
//	k4 = f(tn + h, y{n+1})
//	z{n+1} = yn + 7/24 h k1 + 1/4 h k2 + 1/3 h k3 + 1/8 h k4 // 2nd order
type RK23 struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *RK23) Step() {
	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}

	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		torqueFn(rk.k1)
	}

	// FSAL cannot be used with temperature
	if !Temp.isZero() {
		torqueFn(rk.k1)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (3./4.)*Dt_si
	cuda.Madd2(m, m0, k2, 1, (3./4.)*h) // m = m0*1 + k2*3/4
	M.normalize()
	torqueFn(k3)

	// 3rd order solution
	cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (2./9.)*h, (1./3.)*h, (4./9.)*h)
	M.normalize()

	// error estimate
	Time = t0 + Dt_si
	torqueFn(k4)
	Err := k2 // re-use k2 as error
	// difference of 3rd and 2nd order torque without explicitly storing them first
	cuda.Madd4(Err, rk.k1, k2, k3, k4, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k4)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./3.))
		data.Copy(rk.k1, k4) // FSAL
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
	}
}

func (rk *RK23) StepRegion(region SolverRegion) {
	m := cuda.Buffer(M.NComp(), region.Size())
	m.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(m)
	M.EvalRegionTo(m)
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}

	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		rk.k1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
		torqueFnRegion(rk.k1)
	}

	// FSAL cannot be used with temperature
	if !Temp.isZero() {
		rk.k1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
		torqueFnRegion(rk.k1)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	k2.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k3.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k4.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)

	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	geom := cuda.Buffer(Geometry.Gpu().NComp(), region.Size())
	cuda.Crop(geom, Geometry.Gpu(), region.StartX, region.StartY, region.StartZ)
	defer cuda.Recycle(geom)

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	cuda.Normalize(m, geom)
	torqueFnRegion(k2)

	// stage 3
	Time = t0 + (3./4.)*Dt_si
	cuda.Madd2(m, m0, k2, 1, (3./4.)*h) // m = m0*1 + k2*3/4
	cuda.Normalize(m, geom)
	torqueFnRegion(k3)

	// 3rd order solution
	cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (2./9.)*h, (1./3.)*h, (4./9.)*h)
	cuda.Normalize(m, geom)

	// error estimate
	Time = t0 + Dt_si
	torqueFnRegion(k4)
	Err := k2 // re-use k2 as error
	// difference of 3rd and 2nd order torque without explicitly storing them first
	cuda.Madd4(Err, rk.k1, k2, k3, k4, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		if FixDt != 0 {
			size := m.Size()
			data.CopyPart(M.Buffer(), m, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
			data.Copy(rk.k1, k4) // FSAL
		}
		setLastErr(err)
		region.LastErr = LastErr
		setMaxTorque(k4)
		region.LastTorque = LastTorque
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
	}
	region.Dt_si = Dt_si
}

func (rk *RK23) Free() {
	rk.k1.Free()
	rk.k1 = nil
}
