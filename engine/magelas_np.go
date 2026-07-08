package engine

// Fully coupled magnetoelastic solver ported from magnum.np (LLGWithLESolver),
// following "Modeling magnetoelastic wave interactions in magnetic films and
// heterostructures: A finite-difference approach", PRApplied 25, 034050 (2026).
//
// This file: user-facing parameters, switches and quantities of the NP solver.
// The old fully coupled solver (SetSolver(9)/SetSolver(12), B1/B2-based) is untouched;
// the new solver must be selected explicitly with UseMagnumNPSolver(true) or SetSolver(13).
//
// State variables: m (M), u (U, displacement [m]), du (DU, velocity [m/s]).
// The momentum density of magnum.np is p = rho*du; the phenomenological damping keeps the
// existing mumax-ME convention f_damp = -eta*du with eta in kg/(s m3)
// (eta_magnumnp[1/s] = eta_mumax/rho).

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	// Additional stiffness components (C11, C12, C44 already exist in elastic_kernel.go).
	// Together they form the block-diagonal Voigt stiffness matrix of paper Eq. (24):
	// isotropic, cubic, tetragonal (4/mmm), hexagonal and orthorhombic materials.
	C13 = NewScalarParam("C13", "N/m2", "Stiffness constant C13 (NP solver)")
	C22 = NewScalarParam("C22", "N/m2", "Stiffness constant C22 (NP solver)")
	C23 = NewScalarParam("C23", "N/m2", "Stiffness constant C23 (NP solver)")
	C33 = NewScalarParam("C33", "N/m2", "Stiffness constant C33 (NP solver)")
	C55 = NewScalarParam("C55", "N/m2", "Stiffness constant C55 (NP solver)")
	C66 = NewScalarParam("C66", "N/m2", "Stiffness constant C66 (NP solver)")

	// Magnetostriction constants (used instead of B1/B2 by the NP solver).
	Lambda100 = NewScalarParam("lambda100", "", "Magnetostriction constant lambda_100 (NP solver)")
	Lambda111 = NewScalarParam("lambda111", "", "Magnetostriction constant lambda_111 (NP solver)")

	// Elastic mask: cells with 0 do not update u/du (magnum.np mask_elastic). Default 1.
	ElasticMaskNP = NewScalarParam("elasticMaskNP", "", "Elastic mask of the NP solver: 0 freezes u/du updates (default 1)")

	// Traction (sigma - sigma_m).n at the outer non-periodic faces [N/m2]; zero = natural
	// (homogeneous Neumann / free surface). Sampled at the boundary cells.
	TractionNP = NewExcitation("tractionNP", "N/m2", "Traction (sigma-sigma_m).n at outer non-periodic boundaries (NP solver)")

	// boundary_nodes of magnum.np: 1 = forward, 2 = midpoint, 3 = three-point (default,
	// favors correctness for SAW problems).
	BoundaryNodesNP int = 3

	// iteration_depth of magnum.np: number of B-jump-condition iterations (default 1).
	// Required >= 1 for the airbox method.
	IterationDepthNP int = 1

	// magnetic window (magnum.np magnetic_x/y/z_limits), in cells; -1 = full mesh
	magneticWindowNP = [6]int{-1, -1, -1, -1, -1, -1}

	// clamp Dt to the elastic CFL estimate (advisory log if false)
	EnforceCFLNP bool = false
	CFLFactorNP  float64 = 1.0

	// quantities
	F_elNP        = NewVectorField("F_elNP", "N/m3", "Elastic force density div(sigma-sigma_m) incl. boundary conditions (NP solver)", SetElasticForceNP)
	B_melNP       = NewVectorField("B_melNP", "T", "Nonlinear magnetoelastic field (NP solver)", SetMagnetoelasticFieldNP)
	DDUNP         = NewVectorField("dduNP", "m/s2", "Acceleration d2u/dt2 (NP solver)", SetAccelerationNP)
	PNP           = NewVectorField("pNP", "kg/(m2 s)", "Momentum density rho*du (NP solver)", SetMomentumNP)
	NormStrainNP  = NewVectorField("normStrainNP", "", "Normal strain (xx,yy,zz), jump-aware (NP solver)", func(dst *data.Slice) { setStrainNP(dst, 0, true) })
	ShearStrainNP = NewVectorField("shearStrainNP", "", "Engineering shear strain (yz,xz,xy), jump-aware (NP solver)", func(dst *data.Slice) { setStrainNP(dst, 0, false) })
	NormStrainElNP  = NewVectorField("normStrainElNP", "", "Elastic normal strain eps-eps_m (xx,yy,zz) (NP solver)", func(dst *data.Slice) { setStrainNP(dst, 1, true) })
	ShearStrainElNP = NewVectorField("shearStrainElNP", "", "Elastic engineering shear strain (yz,xz,xy) (NP solver)", func(dst *data.Slice) { setStrainNP(dst, 1, false) })
	NormStressNP  = NewVectorField("normStressNP", "N/m2", "Elastic stress sigma-sigma_m (xx,yy,zz) (NP solver)", func(dst *data.Slice) { setStressNP(dst, true) })
	ShearStressNP = NewVectorField("shearStressNP", "N/m2", "Elastic stress sigma-sigma_m (yz,xz,xy) (NP solver)", func(dst *data.Slice) { setStressNP(dst, false) })

	U_elNP        = NewScalarValue("U_elNP", "J", "Elastic potential energy 1/2 (eps-eps_m):C:(eps-eps_m) (NP solver)", GetUelNP)
	U_totNP       = NewScalarValue("U_totNP", "J", "Total strain energy 1/2 eps:C:eps (NP solver)", GetUtotNP)
	T_elNP        = NewScalarValue("T_elNP", "J", "Elastic kinetic energy 1/2 rho du^2 (NP solver)", GetTelNP)
	E_melNP       = NewScalarValue("E_melNP", "J", "Magnetoelastic energy (LLG budget, NP solver)", GetMelEnergyNP)
	GilbertDissNP = NewScalarValue("GilbertDissNP", "W", "Gilbert dissipation rate (positive = loss) (NP solver)", GetGilbertDissNP)
)

func init() {
	// sensible defaults: elastic everywhere
	ElasticMaskNP.Set(1)

	DeclFunc("UseMagnumNPSolver", UseMagnumNPSolver, "Enable (true) or disable (false) the magnum.np-style fully coupled magnetoelastic solver")
	DeclVar("BoundaryNodesNP", &BoundaryNodesNP, "Boundary node scheme of the NP solver: 1 forward, 2 midpoint, 3 three-point (default 3)")
	DeclVar("IterationDepthNP", &IterationDepthNP, "Iteration depth for the strain jump conditions of the NP solver (default 1)")
	DeclVar("EnforceCFLNP", &EnforceCFLNP, "Clamp the time step of the NP solver to the elastic CFL estimate")
	DeclVar("CFLFactorNP", &CFLFactorNP, "Safety factor for the CFL clamp of the NP solver (default 1)")
	DeclFunc("SetMagneticWindowNP", SetMagneticWindowNP, "Restrict the magnetic domain of the NP solver: x0, x1, y0, y1, z0, z1 in cells (end exclusive, -1 = full)")
	DeclFunc("SetStiffnessCubicNP", SetStiffnessCubicNP, "Set all stiffness components for a cubic material: C11, C12, C44")
	DeclFunc("SetStiffnessCubicRegionNP", SetStiffnessCubicRegionNP, "Set all stiffness components for a cubic material in a region: region, C11, C12, C44")
	DeclFunc("SetStiffnessIsotropicNP", SetStiffnessIsotropicNP, "Set all stiffness components for an isotropic material: E, nu")
	DeclFunc("SetStiffnessIsotropicRegionNP", SetStiffnessIsotropicRegionNP, "Set all stiffness components for an isotropic material in a region: region, E, nu")

	registerEnergy(GetMelEnergyNP, AddMelEnergyDensityNP)
}

// The NP solver is selected explicitly (old solvers remain available).
func UsingMagelasNP() bool {
	return Solvertype == MAGELAS_RKF45_NP
}

func UseMagnumNPSolver(b bool) {
	if b {
		SetSolver(MAGELAS_RKF45_NP)
	} else {
		SetSolver(DORMANDPRINCE)
	}
}

func SetMagneticWindowNP(x0, x1, y0, y1, z0, z1 int) {
	magneticWindowNP = [6]int{x0, x1, y0, y1, z0, z1}
}

// resolved magnetic window (cells), -1 mapped to the mesh bounds
func melasNPWindow() [6]int {
	n := Mesh().Size()
	w := magneticWindowNP
	full := [3]int{n[X], n[Y], n[Z]}
	out := [6]int{}
	for d := 0; d < 3; d++ {
		lo, hi := w[2*d], w[2*d+1]
		if lo < 0 {
			lo = 0
		}
		if hi < 0 {
			hi = full[d]
		}
		out[2*d], out[2*d+1] = lo, hi
	}
	return out
}

func melasNPFullWindow() [6]int {
	n := Mesh().Size()
	return [6]int{0, n[X], 0, n[Y], 0, n[Z]}
}

func SetStiffnessCubicNP(c11, c12, c44 float64) {
	C11.Set(c11)
	C22.Set(c11)
	C33.Set(c11)
	C12.Set(c12)
	C13.Set(c12)
	C23.Set(c12)
	C44.Set(c44)
	C55.Set(c44)
	C66.Set(c44)
}

func SetStiffnessCubicRegionNP(region int, c11, c12, c44 float64) {
	C11.SetRegionValueGo(region, c11)
	C22.SetRegionValueGo(region, c11)
	C33.SetRegionValueGo(region, c11)
	C12.SetRegionValueGo(region, c12)
	C13.SetRegionValueGo(region, c12)
	C23.SetRegionValueGo(region, c12)
	C44.SetRegionValueGo(region, c44)
	C55.SetRegionValueGo(region, c44)
	C66.SetRegionValueGo(region, c44)
}

func isotropicC(E, nu float64) (c11, c12, c44 float64) {
	c11 = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
	c12 = E * nu / ((1 + nu) * (1 - 2*nu))
	c44 = 0.5 * (c11 - c12)
	return
}

func SetStiffnessIsotropicNP(E, nu float64) {
	c11, c12, c44 := isotropicC(E, nu)
	SetStiffnessCubicNP(c11, c12, c44)
}

func SetStiffnessIsotropicRegionNP(region int, E, nu float64) {
	c11, c12, c44 := isotropicC(E, nu)
	SetStiffnessCubicRegionNP(region, c11, c12, c44)
}

// stiffness MSlice bundle; every returned MSlice must be recycled via melasNPRecycleC
func melasNPStiffness() cuda.MelasStiffness {
	return cuda.MelasStiffness{
		C11: C11.MSlice(), C12: C12.MSlice(), C13: C13.MSlice(),
		C22: C22.MSlice(), C23: C23.MSlice(), C33: C33.MSlice(),
		C44: C44.MSlice(), C55: C55.MSlice(), C66: C66.MSlice(),
	}
}

func melasNPRecycleC(C cuda.MelasStiffness) {
	C.C11.Recycle()
	C.C12.Recycle()
	C.C13.Recycle()
	C.C22.Recycle()
	C.C23.Recycle()
	C.C33.Recycle()
	C.C44.Recycle()
	C.C55.Recycle()
	C.C66.Recycle()
}

// haveMelNP reports whether the NP magnetoelastic coupling is active (lambda set).
func haveMelNP() bool {
	return Lambda100.nonZero() || Lambda111.nonZero()
}

// elastic wave speed estimate for the CFL condition, from the region values of C and rho
func melasNPMaxWaveSpeed() float64 {
	maxC := 0.0
	for _, p := range []*RegionwiseScalar{C11, C22, C33, C44, C55, C66} {
		for r := 0; r < NREGION; r++ {
			v := p.GetRegion(r)
			if v > maxC {
				maxC = v
			}
		}
	}
	minRho := math.Inf(1)
	for r := 0; r < NREGION; r++ {
		v := Rho.GetRegion(r)
		if v > 0 && v < minRho {
			minRho = v
		}
	}
	if maxC == 0 || math.IsInf(minRho, 1) {
		return 0
	}
	return math.Sqrt(maxC / minRho)
}
