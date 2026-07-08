package engine

// Right-hand-side evaluation of the NP fully coupled magnetoelastic solver:
// jump-aware gradients, iterative B jump conditions, stress, force and field
// (magnum.np: LLGWithLESolver._update_diff_data / dpd / dud, MagnetoElasticField.h).

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// melasNPFields holds the intermediate fields of one RHS evaluation (pooled buffers).
type melasNPFields struct {
	gmX, gmY, gmZ                *data.Slice   // exchange-jump-aware gradients of m components
	guX, guY, guZ                *data.Slice   // jump-aware gradients of u components
	Blx, Brx, Bly, Bry, Blz, Brz *data.Slice   // total B jump data (per direction, comp = u comp)
	sd, so                       *data.Slice   // sigma_el = sigma - sigma_m: diag (xx,yy,zz), offdiag (yz,xz,xy)
}

func (f *melasNPFields) free() {
	for _, s := range []*data.Slice{f.gmX, f.gmY, f.gmZ, f.guX, f.guY, f.guZ,
		f.Blx, f.Brx, f.Bly, f.Bry, f.Blz, f.Brz, f.sd, f.so} {
		if s != nil {
			cuda.Recycle(s)
		}
	}
}

// melasNPCompute runs the full gradient/jump-condition pipeline for the given m and u
// (magnum.np _update_diff_data + stress evaluation):
//  1. exchange-jump-aware gradients of m inside the magnetic window,
//  2. magnetic-stress B parameters from interface values of m,
//  3. jump-aware gradients of u with B = B_sigM,
//  4. IterationDepthNP x: B = B_sigM + harmonic-mean displacement part, recompute grad u,
//  5. sigma_el = C:(eps - eps_m).
func melasNPCompute(m, u *data.Slice) *melasNPFields {
	mesh := Mesh()
	size := mesh.Size()
	so2nd := BoundaryNodesNP > 2
	winM := melasNPWindow()
	winFull := melasNPFullWindow()

	f := new(melasNPFields)

	// ---- 1. exchange-jump-aware gradients of m ----
	aex := Aex.MSlice()
	f.gmX, f.gmY, f.gmZ = cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	cuda.MelasGradJumpNP(f.gmX, m, X, aex, aex, aex, nil, nil, nil, nil, nil, nil, 0, winM, mesh, so2nd)
	cuda.MelasGradJumpNP(f.gmY, m, Y, aex, aex, aex, nil, nil, nil, nil, nil, nil, 0, winM, mesh, so2nd)
	cuda.MelasGradJumpNP(f.gmZ, m, Z, aex, aex, aex, nil, nil, nil, nil, nil, nil, 0, winM, mesh, so2nd)
	aex.Recycle()

	C := melasNPStiffness()
	defer melasNPRecycleC(C)
	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()

	// ---- 2. magnetic-stress B parameters ----
	sBlx, sBrx := cuda.Buffer(3, size), cuda.Buffer(3, size)
	sBly, sBry := cuda.Buffer(3, size), cuda.Buffer(3, size)
	sBlz, sBrz := cuda.Buffer(3, size), cuda.Buffer(3, size)
	cuda.MelasBSigMNP(sBlx, sBrx, sBly, sBry, sBlz, sBrz, m, f.gmX, f.gmY, f.gmZ, l100, l111, C, mesh)

	// ---- 3. jump-aware gradients of u with B = B_sigM ----
	f.guX, f.guY, f.guZ = cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	gradU := func(Blx, Brx, Bly, Bry, Blz, Brz *data.Slice) {
		// C rows per displacement component (paper Table I)
		cuda.MelasGradJumpNP(f.guX, u, X, C.C11, C.C66, C.C55, Blx, Brx, Bly, Bry, Blz, Brz, X, winFull, mesh, so2nd)
		cuda.MelasGradJumpNP(f.guY, u, Y, C.C66, C.C22, C.C44, Blx, Brx, Bly, Bry, Blz, Brz, Y, winFull, mesh, so2nd)
		cuda.MelasGradJumpNP(f.guZ, u, Z, C.C55, C.C44, C.C33, Blx, Brx, Bly, Bry, Blz, Brz, Z, winFull, mesh, so2nd)
	}
	gradU(sBlx, sBrx, sBly, sBry, sBlz, sBrz)

	// ---- 4. iterative B construction ----
	if IterationDepthNP > 0 {
		f.Blx, f.Brx = cuda.Buffer(3, size), cuda.Buffer(3, size)
		f.Bly, f.Bry = cuda.Buffer(3, size), cuda.Buffer(3, size)
		f.Blz, f.Brz = cuda.Buffer(3, size), cuda.Buffer(3, size)
		for it := 0; it < IterationDepthNP; it++ {
			cuda.MelasBEpsNP(f.Blx, f.Brx, f.Bly, f.Bry, f.Blz, f.Brz,
				sBlx, sBrx, sBly, sBry, sBlz, sBrz,
				f.guX, f.guY, f.guZ, C, mesh)
			gradU(f.Blx, f.Brx, f.Bly, f.Bry, f.Blz, f.Brz)
		}
		cuda.Recycle(sBlx)
		cuda.Recycle(sBrx)
		cuda.Recycle(sBly)
		cuda.Recycle(sBry)
		cuda.Recycle(sBlz)
		cuda.Recycle(sBrz)
	} else {
		f.Blx, f.Brx, f.Bly, f.Bry, f.Blz, f.Brz = sBlx, sBrx, sBly, sBry, sBlz, sBrz
	}

	// ---- 5. elastic stress ----
	f.sd, f.so = cuda.Buffer(3, size), cuda.Buffer(3, size)
	cuda.MelasStressNP(f.sd, f.so, f.guX, f.guY, f.guZ, m, l100, l111, C, mesh)

	return f
}

// melasNPForce computes dst according to outMode (0: d(du)/dt, 1: f_el) from the fields.
func melasNPForce(dst *data.Slice, flds *melasNPFields, m, u, du *data.Slice, outMode int) {
	mesh := Mesh()
	size := mesh.Size()
	so2nd := BoundaryNodesNP > 2

	C := melasNPStiffness()
	defer melasNPRecycleC(C)
	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()

	fcx, fcy, fcz := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(fcx)
	defer cuda.Recycle(fcy)
	defer cuda.Recycle(fcz)

	cuda.MelasForceNP(fcx, fcy, fcz, u, flds.guX, flds.guY, flds.guZ,
		flds.gmX, flds.gmY, flds.gmZ, m,
		flds.Blx, flds.Brx, flds.Bly, flds.Bry, flds.Blz, flds.Brz,
		l100, l111, C, mesh, so2nd)

	traction, rec := TractionNP.Slice()
	if rec {
		defer cuda.Recycle(traction)
	}

	eta := Eta.MSlice()
	defer eta.Recycle()
	rho := Rho.MSlice()
	defer rho.Recycle()
	mask := ElasticMaskNP.MSlice()
	defer mask.Recycle()
	frozen := FrozenDispLoc.MSlice()
	defer frozen.Recycle()

	cuda.MelasAssembleNP(dst, fcx, fcy, fcz, flds.sd, flds.so, traction, du,
		eta, rho, mask, frozen, BoundaryNodesNP, outMode, mesh)
}

// stress cache: during one RHS evaluation of the stepper, the magnetoelastic field term
// reuses the stress computed for the elastic force instead of recomputing the pipeline.
var melasNPCache struct {
	valid  bool
	sd, so *data.Slice
}

// AddMagnetoelasticFieldNP adds the nonlinear magnetoelastic field (T) to dst.
// Active only when the NP solver is selected and lambda parameters are set.
func AddMagnetoelasticFieldNP(dst *data.Slice) {
	if !UsingMagelasNP() || !haveMelNP() {
		return
	}
	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()
	ms := Msat.MSlice()
	defer ms.Recycle()

	if melasNPCache.valid {
		cuda.MelasHFieldNP(dst, M.Buffer(), melasNPCache.sd, melasNPCache.so, l100, l111, ms, Mesh())
		return
	}
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()
	cuda.MelasHFieldNP(dst, M.Buffer(), flds.sd, flds.so, l100, l111, ms, Mesh())
}

// SetMagnetoelasticFieldNP writes the NP magnetoelastic field (T) to dst (quantity B_melNP).
// Computed regardless of the selected solver.
func SetMagnetoelasticFieldNP(dst *data.Slice) {
	cuda.Zero(dst)
	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()
	ms := Msat.MSlice()
	defer ms.Recycle()

	if melasNPCache.valid {
		cuda.MelasHFieldNP(dst, M.Buffer(), melasNPCache.sd, melasNPCache.so, l100, l111, ms, Mesh())
		return
	}
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()
	cuda.MelasHFieldNP(dst, M.Buffer(), flds.sd, flds.so, l100, l111, ms, Mesh())
}

// SetElasticForceNP writes f_el = div(sigma - sigma_m) incl. boundary conditions to dst.
func SetElasticForceNP(dst *data.Slice) {
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()
	melasNPForce(dst, flds, M.Buffer(), U.Buffer(), DU.Buffer(), 1)
}

// SetAccelerationNP writes d2u/dt2 = (f_el - eta*du)/rho (masked) to dst.
func SetAccelerationNP(dst *data.Slice) {
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()
	melasNPForce(dst, flds, M.Buffer(), U.Buffer(), DU.Buffer(), 0)
}

// SetMomentumNP writes p = rho*du to dst.
func SetMomentumNP(dst *data.Slice) {
	rho := Rho.MSlice()
	defer rho.Recycle()
	cuda.MulMSlice(dst, cuda.ToMSlice(DU.Buffer()), rho)
}

// setStrainNP writes strain diagnostics: mode 0 = eps, 1 = eps - eps_m; norm selects
// (xx,yy,zz) vs the engineering shear components (yz,xz,xy).
func setStrainNP(dst *data.Slice, mode int, norm bool) {
	size := Mesh().Size()
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()

	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()

	nrm, shr := cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(nrm)
	defer cuda.Recycle(shr)
	cuda.MelasStrainNP(nrm, shr, flds.guX, flds.guY, flds.guZ, M.Buffer(), l100, l111, mode, Mesh())
	if norm {
		data.Copy(dst, nrm)
	} else {
		data.Copy(dst, shr)
	}
}

// setStressNP writes the elastic stress sigma - sigma_m diagnostics.
func setStressNP(dst *data.Slice, norm bool) {
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()
	if norm {
		data.Copy(dst, flds.sd)
	} else {
		data.Copy(dst, flds.so)
	}
}
