package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Classical 4th order RK solver.
type magelasRK4 struct {
	kv1 *data.Slice
	kv2 *data.Slice
	kv3 *data.Slice
	kv4 *data.Slice
}

func (_ *magelasRK4) Step() {

	//################
	// Differential equation:
	// du/dt = v(t)
	// dv(t)/dt = [f(t) + bf(t) - eta*g(t)]/rho
	// dv(t)/dt = right
	// with f(t) = nabla sigma

	//#################################
	//Initialisation:
	/*
		if InsertTimeDepDisplacement == 1 {
			U.Set(Uniform(0, 0, 0))
		}*/
	u := U.Buffer()
	size := u.Size()

	//Set fixed displacement
	SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)

	/*if InsertTimeDepDisplacement == 1 {
		var funcResults []float64
		for _, function := range InsertTimeDepDisplacementFuncArgs {
			funcResults = append(funcResults, function(Time))
		}
		fmt.Println(funcResults[0], funcResults[1], funcResults[2], funcResults[3], funcResults[4], "\n")
		UOVERLAY.Set(InsertTimeDepDisplacementFunc(funcResults[0], funcResults[1], funcResults[2], funcResults[3], funcResults[4]))
		cuda.Add(u0, u0, UOVERLAY.Buffer())
	}*/

	m := M.Buffer()
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	v := DU.Buffer()

	v0 := cuda.Buffer(3, size)
	defer cuda.Recycle(v0)
	data.Copy(v0, v)

	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	kv1, kv2, kv3, kv4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	km1, km2, km3, km4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)
	defer cuda.Recycle(kv1)
	defer cuda.Recycle(kv2)
	defer cuda.Recycle(kv3)
	defer cuda.Recycle(kv4)
	defer cuda.Recycle(km1)
	defer cuda.Recycle(km2)
	defer cuda.Recycle(km3)
	defer cuda.Recycle(km4)

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

	//h := float32(Dt_si * GammaLL)

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv
	//Stage 1:
	calcRhs(kv1, f, v)
	ku1 = v0
	if fixM == false {
		torqueFn(km1)
	}

	//Stage 2:
	//u = u0*1 + k1*dt/2
	Time = t0 + (1./2.)*Dt_si

	cuda.Madd2(u, u0, ku1, 1, (1./2.)*dt)
	if useBoundaries == true {
		calcBndry()
	}
	cuda.Madd2(v, v0, kv1, 1, (1./2.)*dt)
	if fixM == false {
		cuda.Madd2(m, m, km1, 1, (1./2.)*dt*float32(GammaLL))
		M.normalize()
	}

	calcRhs(kv2, f, v)
	cuda.Madd2(ku2, v0, kv1, 1, (1./2.)*dt)
	if fixM == false {
		torqueFn(km2)
	}

	//Stage 3:
	//u = u0*1 + k2*dt/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*dt)
	if useBoundaries == true {
		calcBndry()
	}
	cuda.Madd2(v, v0, kv2, 1, (1./2.)*dt)
	if fixM == false {
		cuda.Madd2(m, m0, km2, 1, (1./2.)*dt*float32(GammaLL))
		M.normalize()
	}

	calcRhs(kv3, f, v)
	cuda.Madd2(ku3, v0, kv2, 1, (1./2.)*dt)
	if fixM == false {
		torqueFn(km3)
	}

	//Stage 4:
	//u = u0*1 + k3*dt
	Time = t0 + Dt_si
	cuda.Madd2(u, u0, ku3, 1, 1.*dt)
	if useBoundaries == true {
		calcBndry()
	}
	cuda.Madd2(v, v0, kv3, 1, 1.*dt)
	if fixM == false {
		cuda.Madd2(m, m0, km3, 1, 1.*dt*float32(GammaLL))
		M.normalize()
	}

	calcRhs(kv4, f, v)
	cuda.Madd2(ku4, v0, kv3, 1, 1.*dt)
	if fixM == false {
		torqueFn(km4)
	}

	//###############################
	//Error calculation
	err := cuda.MaxVecDiff(ku1, ku4)
	err2 := cuda.MaxVecDiff(kv1, kv4)

	//err3 := cuda.MaxVecDiff(km1, km4) * float64(dt) * GammaLL

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
		if useBoundaries == true {
			calcBndry()
		}
		cuda.Madd5(v, v0, kv1, kv2, kv3, kv4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		if fixM == false {
			cuda.Madd5(m, m0, km1, km2, km3, km4, 1, (1./6.)*dt*float32(GammaLL), (1./3.)*dt*float32(GammaLL), (1./3.)*dt*float32(GammaLL), (1./6.)*dt*float32(GammaLL))
			//Post handlings
			M.normalize()
		} else {
			data.Copy(m, m0)
			M.normalize()
		}
		for i := 0; i < 3; i++ {
			cuda.Scale(u, 1, U.average())
		}

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
		data.Copy(m, m0)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
}

func (_ *magelasRK4) StepRegion(region *SolverRegion) {
	panic("Not implimented yet")
	//################
	// Differential equation:
	// du/dt = v(t)
	// dv(t)/dt = [f(t) + bf(t) - eta*g(t)]/rho
	// dv(t)/dt = right
	// with f(t) = nabla sigma

	//#################################
	//Initialisation:
	/*
		if InsertTimeDepDisplacement == 1 {
			U.Set(Uniform(0, 0, 0))
		}*/
	u := U.Buffer()
	size := u.Size()

	//Set fixed displacement
	SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)

	/*if InsertTimeDepDisplacement == 1 {
		var funcResults []float64
		for _, function := range InsertTimeDepDisplacementFuncArgs {
			funcResults = append(funcResults, function(Time))
		}
		fmt.Println(funcResults[0], funcResults[1], funcResults[2], funcResults[3], funcResults[4], "\n")
		UOVERLAY.Set(InsertTimeDepDisplacementFunc(funcResults[0], funcResults[1], funcResults[2], funcResults[3], funcResults[4]))
		cuda.Add(u0, u0, UOVERLAY.Buffer())
	}*/

	m := M.Buffer()
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	v := DU.Buffer()

	v0 := cuda.Buffer(3, size)
	defer cuda.Recycle(v0)
	data.Copy(v0, v)

	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	kv1, kv2, kv3, kv4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	km1, km2, km3, km4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)
	defer cuda.Recycle(kv1)
	defer cuda.Recycle(kv2)
	defer cuda.Recycle(kv3)
	defer cuda.Recycle(kv4)
	defer cuda.Recycle(km1)
	defer cuda.Recycle(km2)
	defer cuda.Recycle(km3)
	defer cuda.Recycle(km4)

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

	//h := float32(Dt_si * GammaLL)

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv
	//Stage 1:
	calcRhs(kv1, f, v)
	ku1 = v0
	if fixM == false {
		torqueFn(km1)
	}

	//Stage 2:
	//u = u0*1 + k1*dt/2
	Time = t0 + (1./2.)*Dt_si

	cuda.Madd2(u, u0, ku1, 1, (1./2.)*dt)
	if useBoundaries == true {
		calcBndry()
	}
	cuda.Madd2(v, v0, kv1, 1, (1./2.)*dt)
	if fixM == false {
		cuda.Madd2(m, m, km1, 1, (1./2.)*dt*float32(GammaLL))
		M.normalize()
	}

	calcRhs(kv2, f, v)
	cuda.Madd2(ku2, v0, kv1, 1, (1./2.)*dt)
	if fixM == false {
		torqueFn(km2)
	}

	//Stage 3:
	//u = u0*1 + k2*dt/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*dt)
	if useBoundaries == true {
		calcBndry()
	}
	cuda.Madd2(v, v0, kv2, 1, (1./2.)*dt)
	if fixM == false {
		cuda.Madd2(m, m0, km2, 1, (1./2.)*dt*float32(GammaLL))
		M.normalize()
	}

	calcRhs(kv3, f, v)
	cuda.Madd2(ku3, v0, kv2, 1, (1./2.)*dt)
	if fixM == false {
		torqueFn(km3)
	}

	//Stage 4:
	//u = u0*1 + k3*dt
	Time = t0 + Dt_si
	cuda.Madd2(u, u0, ku3, 1, 1.*dt)
	if useBoundaries == true {
		calcBndry()
	}
	cuda.Madd2(v, v0, kv3, 1, 1.*dt)
	if fixM == false {
		cuda.Madd2(m, m0, km3, 1, 1.*dt*float32(GammaLL))
		M.normalize()
	}

	calcRhs(kv4, f, v)
	cuda.Madd2(ku4, v0, kv3, 1, 1.*dt)
	if fixM == false {
		torqueFn(km4)
	}

	//###############################
	//Error calculation
	err := cuda.MaxVecDiff(ku1, ku4)
	err2 := cuda.MaxVecDiff(kv1, kv4)

	//err3 := cuda.MaxVecDiff(km1, km4) * float64(dt) * GammaLL

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
		if useBoundaries == true {
			calcBndry()
		}
		cuda.Madd5(v, v0, kv1, kv2, kv3, kv4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		if fixM == false {
			cuda.Madd5(m, m0, km1, km2, km3, km4, 1, (1./6.)*dt*float32(GammaLL), (1./3.)*dt*float32(GammaLL), (1./3.)*dt*float32(GammaLL), (1./6.)*dt*float32(GammaLL))
			//Post handlings
			M.normalize()
		} else {
			data.Copy(m, m0)
			M.normalize()
		}
		for i := 0; i < 3; i++ {
			cuda.Scale(u, 1, U.average())
		}

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
		data.Copy(m, m0)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
}

func (magelasRK4 *magelasRK4) Free() {
	//magelasRK4.kv1.Free()
	//magelasRK4.kv1 = nil
	//magelasRK4.kv2.Free()
	//magelasRK4.kv2 = nil
	//magelasRK4.kv3.Free()
	//magelasRK4.kv3 = nil
	//magelasRK4.kv4.Free()
	//magelasRK4.kv4 = nil
}
