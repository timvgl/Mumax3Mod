package engine

// RKF45 (Runge-Kutta-Fehlberg) stepper for the fully coupled state (m, u, du),
// ported from magnum.np (solvers/ode_solvers/rkf45.py + LLGWithLESolver.dv).
//
// Per-variable error control: the error of each variable class (m, u, du) is normalized by
// atol_c + rtol*max|x_c| and the step is accepted when the largest norm is <= 1
// (magnum.np uses the same atol/rtol structure with a pointwise norm).
// |m| = 1 is restored by normalization after every stage and step (mumax convention).

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	RtolNP   float64 = 1e-5  // relative tolerance of the NP solver
	AtolMNP  float64 = 1e-5  // absolute tolerance for m
	AtolUNP  float64 = 1e-15 // absolute tolerance for u [m]
	AtolDUNP float64 = 1e-6  // absolute tolerance for du [m/s] (magnum.np atol_pd/rho)
)

func init() {
	DeclVar("RtolNP", &RtolNP, "Relative tolerance of the NP magnetoelastic solver (default 1e-5)")
	DeclVar("AtolMNP", &AtolMNP, "Absolute tolerance for m of the NP solver (default 1e-5)")
	DeclVar("AtolUNP", &AtolUNP, "Absolute tolerance for u of the NP solver (default 1e-15 m)")
	DeclVar("AtolDUNP", &AtolDUNP, "Absolute tolerance for du of the NP solver (default 1e-6 m/s)")
}

type magelasNPRKF45 struct{}

// one RHS evaluation of the coupled system: km = torque [T], ku = du/dt, kv = d(du)/dt
func melasNPRhs(km, ku, kv *data.Slice) {
	m, u, du := M.Buffer(), U.Buffer(), DU.Buffer()

	flds := melasNPCompute(m, u)

	// elastic force / acceleration
	melasNPForce(kv, flds, m, u, du, 0)

	// du/dt (masked)
	mask := ElasticMaskNP.MSlice()
	frozen := FrozenDispLoc.MSlice()
	cuda.MelasDudNP(ku, du, mask, frozen, Mesh())
	mask.Recycle()
	frozen.Recycle()

	// dm/dt: torque in T units; the magnetoelastic field term reuses the cached stress
	melasNPCache.sd, melasNPCache.so = flds.sd, flds.so
	melasNPCache.valid = true
	if !fixM {
		torqueFn(km)
	} else {
		cuda.Zero(km)
	}
	melasNPCache.valid = false

	flds.free()
}

func (s *magelasNPRKF45) Step() {
	m, u, v := M.Buffer(), U.Buffer(), DU.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// CFL guard for elastodynamics: c_max*dt/min(dx) <= 1
	if cmax := melasNPMaxWaveSpeed(); cmax > 0 {
		cs := Mesh().CellSize()
		minDx := math.Min(cs[X], math.Min(cs[Y], cs[Z]))
		dtCFL := CFLFactorNP * minDx / cmax
		if Dt_si > dtCFL {
			if EnforceCFLNP {
				Dt_si = dtCFL
			} else if NSteps == 0 {
				util.Log("NP solver: dt exceeds the elastic CFL estimate", dtCFL, "s; consider FixDt/MaxDt or EnforceCFLNP=true")
			}
		}
	}

	dt := float32(Dt_si)
	util.Assert(dt > 0)
	hm := dt * float32(GammaLL) // m advances with dt*gamma*torque
	t0 := Time

	// enforce Dirichlet displacement values before evaluating anything
	melasNPApplyDirichlet()

	// backups
	m0, u0, v0 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	defer cuda.Recycle(u0)
	defer cuda.Recycle(v0)
	data.Copy(m0, m)
	data.Copy(u0, u)
	data.Copy(v0, v)

	// stage buffers
	var km, ku, kv [7]*data.Slice
	for i := 1; i <= 6; i++ {
		km[i], ku[i], kv[i] = cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
		defer cuda.Recycle(km[i])
		defer cuda.Recycle(ku[i])
		defer cuda.Recycle(kv[i])
	}

	// helper: after a stage update, advance time and re-apply constraints
	// (Time first: time-dependent Dirichlet values must be evaluated at the stage time)
	setStage := func(t float64) {
		Time = t
		if !fixM {
			M.normalize()
		}
		melasNPApplyDirichlet()
	}

	// stage 1
	melasNPRhs(km[1], ku[1], kv[1])

	// stage 2: y + 1/4 k1
	cuda.Madd2(m, m0, km[1], 1, (1./4.)*hm)
	cuda.Madd2(u, u0, ku[1], 1, (1./4.)*dt)
	cuda.Madd2(v, v0, kv[1], 1, (1./4.)*dt)
	setStage(t0 + (1./4.)*Dt_si)
	melasNPRhs(km[2], ku[2], kv[2])

	// stage 3: y + 3/32 k1 + 9/32 k2
	cuda.Madd3(m, m0, km[1], km[2], 1, (3./32.)*hm, (9./32.)*hm)
	cuda.Madd3(u, u0, ku[1], ku[2], 1, (3./32.)*dt, (9./32.)*dt)
	cuda.Madd3(v, v0, kv[1], kv[2], 1, (3./32.)*dt, (9./32.)*dt)
	setStage(t0 + (3./8.)*Dt_si)
	melasNPRhs(km[3], ku[3], kv[3])

	// stage 4: y + 1932/2197 k1 - 7200/2197 k2 + 7296/2197 k3
	cuda.Madd4(m, m0, km[1], km[2], km[3], 1, (1932./2197.)*hm, (-7200./2197.)*hm, (7296./2197.)*hm)
	cuda.Madd4(u, u0, ku[1], ku[2], ku[3], 1, (1932./2197.)*dt, (-7200./2197.)*dt, (7296./2197.)*dt)
	cuda.Madd4(v, v0, kv[1], kv[2], kv[3], 1, (1932./2197.)*dt, (-7200./2197.)*dt, (7296./2197.)*dt)
	setStage(t0 + (12./13.)*Dt_si)
	melasNPRhs(km[4], ku[4], kv[4])

	// stage 5: y + 439/216 k1 - 8 k2 + 3680/513 k3 - 845/4104 k4
	cuda.Madd5(m, m0, km[1], km[2], km[3], km[4], 1, (439./216.)*hm, (-8.)*hm, (3680./513.)*hm, (-845./4104.)*hm)
	cuda.Madd5(u, u0, ku[1], ku[2], ku[3], ku[4], 1, (439./216.)*dt, (-8.)*dt, (3680./513.)*dt, (-845./4104.)*dt)
	cuda.Madd5(v, v0, kv[1], kv[2], kv[3], kv[4], 1, (439./216.)*dt, (-8.)*dt, (3680./513.)*dt, (-845./4104.)*dt)
	setStage(t0 + Dt_si)
	melasNPRhs(km[5], ku[5], kv[5])

	// stage 6: y - 8/27 k1 + 2 k2 - 3544/2565 k3 + 1859/4104 k4 - 11/40 k5
	cuda.Madd6(m, m0, km[1], km[2], km[3], km[4], km[5], 1, (-8./27.)*hm, (2.)*hm, (-3544./2565.)*hm, (1859./4104.)*hm, (-11./40.)*hm)
	cuda.Madd6(u, u0, ku[1], ku[2], ku[3], ku[4], ku[5], 1, (-8./27.)*dt, (2.)*dt, (-3544./2565.)*dt, (1859./4104.)*dt, (-11./40.)*dt)
	cuda.Madd6(v, v0, kv[1], kv[2], kv[3], kv[4], kv[5], 1, (-8./27.)*dt, (2.)*dt, (-3544./2565.)*dt, (1859./4104.)*dt, (-11./40.)*dt)
	setStage(t0 + (1./2.)*Dt_si)
	melasNPRhs(km[6], ku[6], kv[6])

	// error estimate: (5th order) - (4th order) coefficients on k1,k3,k4,k5,k6
	const (
		e1 = 16./135. - 25./216.
		e3 = 6656./12825. - 1408./2565.
		e4 = 28561./56430. - 2197./4104.
		e5 = -9./50. + 1./5.
		e6 = 2. / 55.
	)
	errBuf := cuda.Buffer(3, size)
	defer cuda.Recycle(errBuf)

	cuda.Madd5(errBuf, km[1], km[3], km[4], km[5], km[6], e1, e3, e4, e5, e6)
	errM := cuda.MaxVecNorm(errBuf) * float64(hm)
	cuda.Madd5(errBuf, ku[1], ku[3], ku[4], ku[5], ku[6], e1, e3, e4, e5, e6)
	errU := cuda.MaxVecNorm(errBuf) * float64(dt)
	cuda.Madd5(errBuf, kv[1], kv[3], kv[4], kv[5], kv[6], e1, e3, e4, e5, e6)
	errV := cuda.MaxVecNorm(errBuf) * float64(dt)

	normM := errM / (AtolMNP + RtolNP*1.0)
	normU := errU / (AtolUNP + RtolNP*cuda.MaxVecNorm(u0))
	normV := errV / (AtolDUNP + RtolNP*cuda.MaxVecNorm(v0))
	nrm := math.Max(normM, math.Max(normU, normV))

	if nrm <= 1 || Dt_si <= MinDt || FixDt != 0 {
		// accept: 5th-order solution
		cuda.Madd6(m, m0, km[1], km[3], km[4], km[5], km[6], 1, (16./135.)*hm, (6656./12825.)*hm, (28561./56430.)*hm, (-9./50.)*hm, (2./55.)*hm)
		cuda.Madd6(u, u0, ku[1], ku[3], ku[4], ku[5], ku[6], 1, (16./135.)*dt, (6656./12825.)*dt, (28561./56430.)*dt, (-9./50.)*dt, (2./55.)*dt)
		cuda.Madd6(v, v0, kv[1], kv[3], kv[4], kv[5], kv[6], 1, (16./135.)*dt, (6656./12825.)*dt, (28561./56430.)*dt, (-9./50.)*dt, (2./55.)*dt)
		Time = t0 + Dt_si
		if !fixM {
			M.normalize()
		}
		melasNPApplyDirichlet()
		NSteps++
		setLastErr(nrm)
		adaptDt(math.Pow(1./math.Max(nrm, 1e-30), 1./5.))
	} else {
		// reject: undo
		util.Assert(FixDt == 0)
		data.Copy(m, m0)
		data.Copy(u, u0)
		data.Copy(v, v0)
		Time = t0
		NUndone++
		adaptDt(math.Pow(1./nrm, 1./6.))
	}
}

func (s *magelasNPRKF45) StepRegion(region *SolverRegion) {
	util.Fatal("The NP magnetoelastic solver (SetSolver(13)) does not support solver regions")
}

func (s *magelasNPRKF45) Free() {}
