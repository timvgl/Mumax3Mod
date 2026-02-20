package engine

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type RK56 struct {
}

func (rk *RK56) Step() {

	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k1, k2, k3, k4, k5, k6, k7, k8 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	defer cuda.Recycle(k7)
	defer cuda.Recycle(k8)
	//k2 will be recyled as k9

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	torqueFn(k1)

	// stage 2
	Time = t0 + (1./6.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./6.)*h) // m = m*1 + k1*h/6
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (4./15.)*Dt_si
	cuda.Madd3(m, m0, k1, k2, 1, (4./75.)*h, (16./75.)*h)
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + (2./3.)*Dt_si
	cuda.Madd4(m, m0, k1, k2, k3, 1, (5./6.)*h, (-8./3.)*h, (5./2.)*h)
	M.normalize()
	torqueFn(k4)

	// stage 5
	Time = t0 + (4./5.)*Dt_si
	cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (-8./5.)*h, (144./25.)*h, (-4.)*h, (16./25.)*h)
	M.normalize()
	torqueFn(k5)

	// stage 6
	Time = t0 + (1.)*Dt_si
	cuda.Madd6(m, m0, k1, k2, k3, k4, k5, 1, (361./320.)*h, (-18./5.)*h, (407./128.)*h, (-11./80.)*h, (55./128.)*h)
	M.normalize()
	torqueFn(k6)

	// stage 7
	Time = t0
	cuda.Madd5(m, m0, k1, k3, k4, k5, 1, (-11./640.)*h, (11./256.)*h, (-11/160.)*h, (11./256.)*h)
	M.normalize()
	torqueFn(k7)

	// stage 8
	Time = t0 + (1.)*Dt_si
	cuda.Madd7(m, m0, k1, k2, k3, k4, k5, k7, 1, (93./640.)*h, (-18./5.)*h, (803./256.)*h, (-11./160.)*h, (99./256.)*h, (1.)*h)
	M.normalize()
	torqueFn(k8)

	// stage 9: 6th order solution
	Time = t0 + (1.)*Dt_si
	//madd6(m, m0, k1, k3, k4, k5, k6, 1, (31./384.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h)
	cuda.Madd7(m, m0, k1, k3, k4, k5, k7, k8, 1, (7./1408.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h, (5./66.)*h)
	M.normalize()
	torqueFn(k2) // re-use k2

	// error estimate
	Err := cuda.Buffer(3, size)
	defer cuda.Recycle(Err)
	cuda.Madd4(Err, k1, k6, k7, k8, (-5. / 66.), (-5. / 66.), (5. / 66.), (5. / 66.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k2)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./7.))
	}
}

func (rk *RK56) StepRegion(region *SolverRegion) {
	fmt.Println("RK56 StepRegion called")
	cuda.PrintBufLength()
	u := cuda.Buffer(M.NComp(), region.Size())
	cuda.Zero(u)
	cuda.Recycle(u)

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

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	geom := cuda.Buffer(Geometry.Gpu().NComp(), region.Size())
	GeomBig, _ := Geometry.Slice()
	cuda.Crop(geom, GeomBig, region.StartX, region.StartY, region.StartZ)
	defer cuda.Recycle(geom)
	cuda.Recycle(GeomBig)

	k1, k2, k3, k4, k5, k6, k7, k8 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	k1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k2.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k3.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k4.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k5.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k6.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k7.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	k8.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)

	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	defer cuda.Recycle(k7)
	defer cuda.Recycle(k8)
	//k2 will be recyled as k9

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	cuda.PrintBufLength()
	torqueFnRegion(k1, m, u, region.PBCx, region.PBCy, region.PBCz)
	cuda.PrintBufLength()

	// stage 2
	Time = t0 + (1./6.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./6.)*h) // m = m*1 + k1*h/6
	cuda.Normalize(m, geom)
	torqueFnRegion(k2, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 3
	Time = t0 + (4./15.)*Dt_si
	cuda.Madd3(m, m0, k1, k2, 1, (4./75.)*h, (16./75.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k3, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 4
	Time = t0 + (2./3.)*Dt_si
	cuda.Madd4(m, m0, k1, k2, k3, 1, (5./6.)*h, (-8./3.)*h, (5./2.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k4, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 5
	Time = t0 + (4./5.)*Dt_si
	cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (-8./5.)*h, (144./25.)*h, (-4.)*h, (16./25.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k5, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 6
	Time = t0 + (1.)*Dt_si
	cuda.Madd6(m, m0, k1, k2, k3, k4, k5, 1, (361./320.)*h, (-18./5.)*h, (407./128.)*h, (-11./80.)*h, (55./128.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k6, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 7
	Time = t0
	cuda.Madd5(m, m0, k1, k3, k4, k5, 1, (-11./640.)*h, (11./256.)*h, (-11/160.)*h, (11./256.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k7, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 8
	Time = t0 + (1.)*Dt_si
	cuda.Madd7(m, m0, k1, k2, k3, k4, k5, k7, 1, (93./640.)*h, (-18./5.)*h, (803./256.)*h, (-11./160.)*h, (99./256.)*h, (1.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k8, m, u, region.PBCx, region.PBCy, region.PBCz)

	// stage 9: 6th order solution
	Time = t0 + (1.)*Dt_si
	//madd6(m, m0, k1, k3, k4, k5, k6, 1, (31./384.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h)
	cuda.Madd7(m, m0, k1, k3, k4, k5, k7, k8, 1, (7./1408.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h, (5./66.)*h)
	cuda.Normalize(m, geom)
	torqueFnRegion(k2, m, u, region.PBCx, region.PBCy, region.PBCz) // re-use k2

	// error estimate
	Err := cuda.Buffer(3, size)
	defer cuda.Recycle(Err)
	cuda.Madd4(Err, k1, k6, k7, k8, (-5. / 66.), (-5. / 66.), (5. / 66.), (5. / 66.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)
	fmt.Println("End")

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		if FixDt != 0 {
			size := m.Size()
			data.CopyPart(M.Buffer(), m, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
		}
		setLastErr(err)
		region.LastErr = LastErr
		setMaxTorque(k2)
		region.LastTorque = LastTorque
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./7.))
	}
	region.Dt_si = Dt_si
}

func (rk *RK56) Free() {
}
