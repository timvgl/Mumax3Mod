package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type RK45DP struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *RK45DP) Step() {
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

	// FSAL cannot be used with finite temperature
	if !Temp.isZero() {
		torqueFn(rk.k1)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k2, k3, k4, k5, k6 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	// k2 will be re-used as k7

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./5.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./5.)*h) // m = m*1 + k1*h/5
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (3./10.)*Dt_si
	cuda.Madd3(m, m0, rk.k1, k2, 1, (3./40.)*h, (9./40.)*h)
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + (4./5.)*Dt_si
	cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (44./45.)*h, (-56./15.)*h, (32./9.)*h)
	M.normalize()
	torqueFn(k4)

	// stage 5
	Time = t0 + (8./9.)*Dt_si
	cuda.Madd5(m, m0, rk.k1, k2, k3, k4, 1, (19372./6561.)*h, (-25360./2187.)*h, (64448./6561.)*h, (-212./729.)*h)
	M.normalize()
	torqueFn(k5)

	// stage 6
	Time = t0 + (1.)*Dt_si
	cuda.Madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (9017./3168.)*h, (-355./33.)*h, (46732./5247.)*h, (49./176.)*h, (-5103./18656.)*h)
	M.normalize()
	torqueFn(k6)

	// stage 7: 5th order solution
	Time = t0 + (1.)*Dt_si
	// no k2
	cuda.Madd6(m, m0, rk.k1, k3, k4, k5, k6, 1, (35./384.)*h, (500./1113.)*h, (125./192.)*h, (-2187./6784.)*h, (11./84.)*h) // 5th
	M.normalize()
	k7 := k2     // re-use k2
	torqueFn(k7) // next torque if OK

	// error estimate
	Err := cuda.Buffer(3, size) //k3 // re-use k3 as error estimate
	defer cuda.Recycle(Err)
	cuda.Madd6(Err, rk.k1, k3, k4, k5, k6, k7, (35./384.)-(5179./57600.), (500./1113.)-(7571./16695.), (125./192.)-(393./640.), (-2187./6784.)-(-92097./339200.), (11./84.)-(187./2100.), (0.)-(1./40.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k7)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./5.))
		data.Copy(rk.k1, k7) // FSAL
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	}
}

func (rk *RK45DP) StepRegion(region *SolverRegion) {
	m := cuda.Buffer(M.NComp(), region.Size())
	m.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(m)
	M.EvalRegionTo(m)
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	} else if region.Dt_si != 0 {
		Dt_si = region.Dt_si
	}

	geom := cuda.Buffer(Geometry.Gpu().NComp(), region.Size())
	GeomBig, _ := Geometry.Slice()
	cuda.Crop(geom, GeomBig, region.StartX, region.StartY, region.StartZ)
	defer cuda.Recycle(geom)
	cuda.Recycle(GeomBig)

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}

	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		rk.k1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
		torqueFnRegionNEW(rk.k1, m, region.PBCx, region.PBCy, region.PBCz)
	}

	// FSAL cannot be used with finite temperature
	if !Temp.isZero() {
		rk.k1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
		torqueFnRegionNEW(rk.k1, m, region.PBCx, region.PBCy, region.PBCz)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k2, k3, k4, k5, k6 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	k2.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k3.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k4.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k5.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k6.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)

	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	// k2 will be re-used as k7

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./5.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./5.)*h) // m = m*1 + k1*h/5
	cuda.Normalize(m, geom)
	torqueFnRegionNEW(k2, m, region.PBCx, region.PBCy, region.PBCz)

	// stage 3
	Time = t0 + (3./10.)*Dt_si
	cuda.Madd3(m, m0, rk.k1, k2, 1, (3./40.)*h, (9./40.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegionNEW(k3, m, region.PBCx, region.PBCy, region.PBCz)

	// stage 4
	Time = t0 + (4./5.)*Dt_si
	cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (44./45.)*h, (-56./15.)*h, (32./9.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegionNEW(k4, m, region.PBCx, region.PBCy, region.PBCz)

	// stage 5
	Time = t0 + (8./9.)*Dt_si
	cuda.Madd5(m, m0, rk.k1, k2, k3, k4, 1, (19372./6561.)*h, (-25360./2187.)*h, (64448./6561.)*h, (-212./729.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegionNEW(k5, m, region.PBCx, region.PBCy, region.PBCz)

	// stage 6
	Time = t0 + (1.)*Dt_si
	cuda.Madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (9017./3168.)*h, (-355./33.)*h, (46732./5247.)*h, (49./176.)*h, (-5103./18656.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegionNEW(k6, m, region.PBCx, region.PBCy, region.PBCz)

	// stage 7: 5th order solution
	Time = t0 + (1.)*Dt_si
	// no k2
	cuda.Madd6(m, m0, rk.k1, k3, k4, k5, k6, 1, (35./384.)*h, (500./1113.)*h, (125./192.)*h, (-2187./6784.)*h, (11./84.)*h) // 5th
	cuda.Normalize(m, geom)
	k7 := k2                                                        // re-use k2
	torqueFnRegionNEW(k7, m, region.PBCx, region.PBCy, region.PBCz) // next torque if OK

	// error estimate
	Err := cuda.Buffer(3, size) //k3 // re-use k3 as error estimate
	defer cuda.Recycle(Err)
	cuda.Madd6(Err, rk.k1, k3, k4, k5, k6, k7, (35./384.)-(5179./57600.), (500./1113.)-(7571./16695.), (125./192.)-(393./640.), (-2187./6784.)-(-92097./339200.), (11./84.)-(187./2100.), (0.)-(1./40.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		if FixDt != 0 {
			size := m.Size()
			data.CopyPart(M.Buffer(), m, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
			data.Copy(rk.k1, k7) // FSAL
		}
		setLastErr(err)
		region.LastErr = LastErr
		setMaxTorque(k7)
		region.LastTorque = LastTorque
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./5.))
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	}
	region.Dt_si = Dt_si
}

func (rk *RK45DP) Free() {
	rk.k1.Free()
	rk.k1 = nil
}
