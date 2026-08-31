package engine

// Aharonov-Casher (AC) effect: a general micromagnetic contribution to the effective
// field of the LLG equation.
//
// Derivation : ac_gauge_v5_general_effective_field_DERIVATION.md
// Numerics   : ac_gauge_v5_general_effective_field_NUMERICS.md
// Reference  : V. Basso, P. Ansalone, A. Di Pietro, Physica B 671 (2023) 415422.
//
// A magnetic moment transported through an electric field precesses in the motional
// field B' = -(v x E)/c^2 at a rate that is independent of |v| per unit length. That
// fixed rotation-per-length is an SO(3) gauge connection
//
//	A_alpha = ACxi * (GammaLL/c^2) * (e_alpha x E)      [1/m]
//
// and minimal coupling d_alpha m -> d_alpha m + A_alpha x m in the exchange energy
// produces the additive effective field implemented here. It shifts the exchange magnon
// parabola to k = k_AC = ACxi*(GammaLL/c^2)*(q x E), is structurally an interfacial DMI
// with axis E and strength D_AC = -2*Aex*ACxi*(GammaLL/c^2)*|E|, and adds a hard axis
// along E at order E^2.
//
// This term is ADDITIVE: it does not replace the exchange kernel, so it composes with
// Dind, Dbulk and inter-region exchange. It is disabled (zero cost) while ACxi == 0.
//
// See also cuda/aharonov_casher.cu. Superseeds the ACExchange flag in exchange.go.

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Speed of light in vacuum, m/s (CODATA, exact).
const cLight = 299792458.0

var (
	// Dimensionless screening / enhancement of the electric field seen by the spins.
	// 0 disables the term. 1 is the bare relativistic AC coupling. Basso et al. fit
	// alpha ~ 1.2e-2 for the AC effect in YIG and beta ~ 1e4 for the electric-field
	// induced DMI -- the same operator, so both are reachable from this one knob.
	ACxi = NewScalarParam("ACxi", "", "Aharonov-Casher coupling: dimensionless screening/enhancement of E (0 = off)")
	E    = NewExcitation("E", "V/m", "Electric field")

	// If true, keep only the component of the gauge connection along
	// quantizationAxisMagnon, i.e. the U(1) linear-spin-wave form with
	// k_AC = ACxi*(GammaLL/c^2)*(q x E). The default (false) uses the full SO(3)
	// connection, which needs no axis and is valid for arbitrary textures.
	ACProjectOnAxis = false

	B_AC     = NewVectorField("B_AC", "T", "Aharonov-Casher effective field", AddAharonovCasherField)
	Edens_AC = NewScalarField("Edens_AC", "J/m3", "Aharonov-Casher energy density", AddAharonovCasherEnergyDensity)
	E_AC     = NewScalarValue("E_AC", "J", "Total Aharonov-Casher energy", GetAharonovCasherEnergy)
)

// u_AC is homogeneous of degree 2 in m, so the standard exchange-type accounting
// eps = -1/2 M.B is exact (see DERIVATION section 6, NUMERICS section 5).
var AddAharonovCasherEnergyDensity = makeEdensAdder(&B_AC, -0.5)

func init() {
	registerEnergy(GetAharonovCasherEnergy, AddAharonovCasherEnergyDensity)
	DeclVar("ACProjectOnAxis", &ACProjectOnAxis,
		"Aharonov-Casher: project the gauge connection on quantizationAxisMagnon (default=false, full SO(3))")
	DeclFunc("ext_ACWavevector", ACWavevector,
		"Aharonov-Casher wavevector k_AC = ACxi*(GammaLL/c^2)*(q x E) in rad/m, for the given quantization axis")
}

// acConst returns gamma/c^2 = 1.9577e-6 m^-1 per V/m.
func acConst() float32 {
	return float32(GammaLL / (cLight * cLight))
}

// acEnabled reports whether the AC term contributes at all.
func acEnabled() bool {
	if !ACxi.nonZero() {
		return false
	}
	if ACExchange {
		util.Fatal("ACExchange = true and ACxi != 0 both add an Aharonov-Casher term, " +
			"which would double count. ACExchange (the old exchange-replacing kernel) is " +
			"superseded by ACxi: set ACExchange = false.")
	}
	return true
}

// ACWavevector returns k_AC = ACxi*(GammaLL/c^2)*(q x E) in rad/m, using region-0
// values of ACxi and E. Intended for setting up and checking benchmarks: the exchange
// magnon parabola has its minimum at k = k_AC.
func ACWavevector(q data.Vector) data.Vector {
	e := E.average()
	xi := ACxi.GetRegion(0)
	pref := xi * GammaLL / (cLight * cLight)
	qn := q.Div(q.Len())
	return data.Vector{
		pref * (qn[Y]*e[Z] - qn[Z]*e[Y]),
		pref * (qn[Z]*e[X] - qn[X]*e[Z]),
		pref * (qn[X]*e[Y] - qn[Y]*e[X]),
	}
}

// AddAharonovCasherField adds the AC effective field to dst.
func AddAharonovCasherField(dst *data.Slice) {
	if !acEnabled() {
		return
	}
	checkACAxis()

	ms := Msat.MSlice()
	defer ms.Recycle()
	xi := ACxi.MSlice()
	defer xi.Recycle()
	q := quantizationAxisMagnon.MSlice()
	defer q.Recycle()
	eField, _ := E.Slice()
	defer cuda.Recycle(eField)

	cuda.AddAharonovCasherField(dst, M.Buffer(), eField, xi, q, ms, lex2.Gpu(),
		regions.Gpu(), M.Mesh(), acConst(), ACProjectOnAxis, OpenBC)
}

// AddAharonovCasherFieldRegion adds the AC effective field to a cropped region.
func AddAharonovCasherFieldRegion(dst, m *data.Slice, useFullSample bool) {
	if !acEnabled() {
		return
	}
	if useFullSample {
		B := cuda.Buffer(M.NComp(), M.Buffer().Size())
		defer cuda.Recycle(B)
		cuda.Zero(B)
		AddAharonovCasherField(B)
		BCropped := cuda.Buffer(dst.NComp(), dst.Size())
		defer cuda.Recycle(BCropped)
		cuda.Crop(BCropped, B, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, BCropped)
		return
	}
	checkACAxis()

	size := dst.RegionSize()
	ms := Msat.MSliceRegion(size, dst.StartX, dst.StartY, dst.StartZ)
	defer ms.Recycle()
	xi := ACxi.MSliceRegion(size, dst.StartX, dst.StartY, dst.StartZ)
	defer xi.Recycle()
	q := quantizationAxisMagnon.MSliceRegion(size, dst.StartX, dst.StartY, dst.StartZ)
	defer q.Recycle()
	eField, _ := E.SliceRegion(size, dst.StartX, dst.StartY, dst.StartZ)
	defer cuda.Recycle(eField)

	mesh := Crop(&M.varVectorField, dst.StartX, dst.EndX, dst.StartY, dst.EndY, dst.StartZ, dst.EndZ).Mesh()
	cuda.AddAharonovCasherField(dst, m, eField, xi, q, ms, lex2.Gpu(),
		regions.Gpu(), mesh, acConst(), ACProjectOnAxis, OpenBC)
}

// GetAharonovCasherEnergy returns the total AC energy in joules.
func GetAharonovCasherEnergy() float64 {
	if !acEnabled() {
		return 0
	}
	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	AddAharonovCasherEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}

// checkACAxis guards the projected mode against an unset quantization axis, which would
// silently zero the whole term.
func checkACAxis() {
	if !ACProjectOnAxis {
		return
	}
	v := quantizationAxisMagnon.GetRegion(0)
	if math.Abs(v[X])+math.Abs(v[Y])+math.Abs(v[Z]) == 0 {
		util.Fatal("ACProjectOnAxis = true requires a non-zero quantizationAxisMagnon " +
			"(set it to the equilibrium magnetization direction), or use the default " +
			"ACProjectOnAxis = false for the axis-free SO(3) coupling.")
	}
}
