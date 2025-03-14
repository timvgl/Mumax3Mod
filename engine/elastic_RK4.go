package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Classical 4th order RK solver.
type elasRK4 struct {
	ku1 *data.Slice
	ku2 *data.Slice
	ku3 *data.Slice
	ku4 *data.Slice
	kv1 *data.Slice
	kv2 *data.Slice
	kv3 *data.Slice
	kv4 *data.Slice
}

func (_ *elasRK4) Step() {

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
	SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)

	v := DU.Buffer()

	v0 := cuda.Buffer(3, size)
	defer cuda.Recycle(v0)
	data.Copy(v0, v)

	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	kv1, kv2, kv3, kv4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)
	defer cuda.Recycle(kv1)
	defer cuda.Recycle(kv2)
	defer cuda.Recycle(kv3)
	defer cuda.Recycle(kv4)

	//f(t) = nabla sigma
	f := cuda.Buffer(3, size)
	defer cuda.Recycle(f)

	right := cuda.Buffer(3, size)
	defer cuda.Recycle(right)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	}
	t0 := Time
	dt := float32(Dt_si)
	util.Assert(dt > 0)

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv

	//Stage 1:
	calcRhs(kv1, f, v)
	ku1 = v0

	//Stage 2:
	//u = u0*1 + k1*dt/2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(u, u0, ku1, 1, (1./2.)*dt)
	cuda.Madd2(v, v0, kv1, 1, (1./2.)*dt)
	//calcBndry()
	calcRhs(kv2, f, v)
	cuda.Madd2(ku2, v0, kv1, 1, (1./2.)*dt)

	//Stage 3:
	//u = u0*1 + k2*dt/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*dt)
	cuda.Madd2(v, v0, kv2, 1, (1./2.)*dt)
	//calcBndry()
	calcRhs(kv3, f, v)
	cuda.Madd2(ku3, v0, kv2, 1, (1./2.)*dt)

	//Stage 4:
	//u = u0*1 + k3*dt
	Time = t0 + Dt_si
	cuda.Madd2(u, u0, ku3, 1, 1.*dt)
	cuda.Madd2(v, v0, kv3, 1, 1.*dt)
	//calcBndry()
	calcRhs(kv4, f, v)
	cuda.Madd2(ku4, v0, kv3, 1, 1.*dt)

	//###############################
	//Error calculation
	err := cuda.MaxVecDiff(ku1, ku4)
	err2 := cuda.MaxVecDiff(kv1, kv4)

	if err != 0.0 {
		err = err * float64(dt) / cuda.MaxVecNorm(ku4)
	}
	if err2 != 0.0 {
		err2 = err2 * float64(dt) / cuda.MaxVecNorm(kv4)
	}

	// //################################
	// //Prints
	// fmt.Println("Max vector norm ku1:", cuda.MaxVecNorm(ku1))
	// fmt.Println("Max vector norm ku2:", cuda.MaxVecNorm(ku2))
	// fmt.Println("Max vector norm ku3:", cuda.MaxVecNorm(ku3))
	// fmt.Println("Max vector norm ku4:", cuda.MaxVecNorm(ku4))

	// //fmt.Println("Max vector norm kv1:", cuda.MaxVecNorm(kv1))
	// //fmt.Println("Max vector norm kv4:", cuda.MaxVecNorm(kv4))

	// fmt.Println("err = maxVecDiff * dt /MaxVexNorm", err)
	// fmt.Println("err2 = maxVecDiff * dt /MaxVexNorm", err2)

	//##########################
	// adjust next time step
	if (err < MaxErr && err2 < MaxErr) || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution

		cuda.Madd5(u, u0, ku1, ku2, ku3, ku4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		cuda.Madd5(v, v0, kv1, kv2, kv3, kv4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		//calcBndry()
		//If you run second derivative together with LLG, then remove NSteps++
		NSteps++

		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./2.))
			setLastErr(err)
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./2.))
			setLastErr(err2)
		}

	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(u, u0)
		data.Copy(v, v0)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
}

func (_ *elasRK4) StepRegion(region *SolverRegion) {
	//################
	// Differential equation:
	// du/dt = v(t)
	// dv(t)/dt = [f(t) + bf(t) - eta*g(t)]/rho
	// dv(t)/dt = right
	// with f(t) = nabla sigma
	//#################################
	m := cuda.Buffer(M.NComp(), region.Size())
	m.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(m)
	M.EvalRegionTo(m)
	//Initialisation:
	//u := U.Buffer()
	u := cuda.Buffer(U.NComp(), region.Size())
	u.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(u)
	U.EvalRegionTo(u)
	size := u.Size()

	//Set fixed displacement
	SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	u0.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	data.Copy(u0, u)

	//v := DU.Buffer()
	v := cuda.Buffer(DU.NComp(), region.Size())
	v.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(v)
	DU.EvalRegionTo(v)

	v0 := cuda.Buffer(3, size)
	defer cuda.Recycle(v0)
	v0.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	data.Copy(v0, v)

	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	kv1, kv2, kv3, kv4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)
	defer cuda.Recycle(kv1)
	defer cuda.Recycle(kv2)
	defer cuda.Recycle(kv3)
	defer cuda.Recycle(kv4)

	ku1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	ku2.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	ku3.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	ku4.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	kv1.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	kv2.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	kv3.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	kv4.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)

	//f(t) = nabla sigma
	f := cuda.Buffer(3, size)
	f.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(f)

	right := cuda.Buffer(3, size)
	right.SetSolverRegion(region.StartX, region.EndX, region.StartY, region.EndY, region.StartZ, region.EndZ)
	defer cuda.Recycle(right)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	} else if region.Dt_si != 0 {
		Dt_si = region.Dt_si
	}
	t0 := Time
	dt := float32(Dt_si)
	util.Assert(dt > 0)

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv

	//Stage 1:
	calcRhsRegion(kv1, m, u, v, f, v)
	ku1 = v0

	//Stage 2:
	//u = u0*1 + k1*dt/2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(u, u0, ku1, 1, (1./2.)*dt)
	cuda.Madd2(v, v0, kv1, 1, (1./2.)*dt)
	//calcBndry()
	calcRhsRegion(kv2, m, u, v, f, v)
	cuda.Madd2(ku2, v0, kv1, 1, (1./2.)*dt)

	//Stage 3:
	//u = u0*1 + k2*dt/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*dt)
	cuda.Madd2(v, v0, kv2, 1, (1./2.)*dt)
	//calcBndry()
	calcRhsRegion(kv3, m, u, v, f, v)
	cuda.Madd2(ku3, v0, kv2, 1, (1./2.)*dt)

	//Stage 4:
	//u = u0*1 + k3*dt
	Time = t0 + Dt_si
	cuda.Madd2(u, u0, ku3, 1, 1.*dt)
	cuda.Madd2(v, v0, kv3, 1, 1.*dt)
	//calcBndry()
	calcRhsRegion(kv4, m, u, v, f, v)
	cuda.Madd2(ku4, v0, kv3, 1, 1.*dt)

	//###############################
	//Error calculation
	err := cuda.MaxVecDiff(ku1, ku4)
	err2 := cuda.MaxVecDiff(kv1, kv4)

	if err != 0.0 {
		err = err * float64(dt) / cuda.MaxVecNorm(ku4)
	}
	if err2 != 0.0 {
		err2 = err2 * float64(dt) / cuda.MaxVecNorm(kv4)
	}

	// //################################
	// //Prints
	// fmt.Println("Max vector norm ku1:", cuda.MaxVecNorm(ku1))
	// fmt.Println("Max vector norm ku2:", cuda.MaxVecNorm(ku2))
	// fmt.Println("Max vector norm ku3:", cuda.MaxVecNorm(ku3))
	// fmt.Println("Max vector norm ku4:", cuda.MaxVecNorm(ku4))

	// //fmt.Println("Max vector norm kv1:", cuda.MaxVecNorm(kv1))
	// //fmt.Println("Max vector norm kv4:", cuda.MaxVecNorm(kv4))

	// fmt.Println("err = maxVecDiff * dt /MaxVexNorm", err)
	// fmt.Println("err2 = maxVecDiff * dt /MaxVexNorm", err2)

	//##########################
	// adjust next time step
	if (err < MaxErr && err2 < MaxErr) || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution

		cuda.Madd5(u, u0, ku1, ku2, ku3, ku4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		cuda.Madd5(v, v0, kv1, kv2, kv3, kv4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		//calcBndry()
		//If you run second derivative together with LLG, then remove NSteps++
		NSteps++

		if FixDt != 0 {
			size := u.Size()
			data.CopyPart(U.Buffer(), u, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
			data.CopyPart(DU.Buffer(), v, 0, size[X], 0, size[Y], 0, size[Z], 0, 1, region.StartX, region.StartY, region.StartZ, 0)
		}

		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./2.))
			setLastErr(err)
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./2.))
			setLastErr(err2)
		}
		region.LastErr = LastErr

	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(u, u0)
		data.Copy(v, v0)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
	region.Dt_si = Dt_si
}

func (elas *elasRK4) Free() {
	elas.kv1.Free()
	elas.kv1 = nil
	elas.kv2.Free()
	elas.kv2 = nil
	elas.kv3.Free()
	elas.kv3 = nil
	elas.kv4.Free()
	elas.kv4 = nil
	elas.ku1.Free()
	elas.ku1 = nil
	elas.ku2.Free()
	elas.ku2 = nil
	elas.ku3.Free()
	elas.ku3 = nil
	elas.ku4.Free()
	elas.ku4 = nil
}
