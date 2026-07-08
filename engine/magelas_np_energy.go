package engine

// Energy and dissipation diagnostics of the NP fully coupled magnetoelastic solver
// (magnum.np: LLGWithLESolver.U_el / U / T_el, MagnetoElasticField.E, paper Eqs. 60/67).

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// integral of an energy density computed by the MelasEnergyNP kernel [J]
func melasNPEnergyIntegral(mode int) float64 {
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()

	C := melasNPStiffness()
	defer melasNPRecycleC(C)
	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()

	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.MelasEnergyNP(buf, flds.guX, flds.guY, flds.guZ, M.Buffer(), l100, l111, C, mode, Mesh())
	return cellVolume() * float64(cuda.Sum(buf))
}

// GetUelNP returns U_el = 1/2 integral (eps-eps_m):C:(eps-eps_m) dx [J].
func GetUelNP() float64 {
	return melasNPEnergyIntegral(0)
}

// GetUtotNP returns U = 1/2 integral eps:C:eps dx [J].
func GetUtotNP() float64 {
	return melasNPEnergyIntegral(2)
}

// GetMelEnergyNP returns the magnetoelastic energy integral (0.5 eps_m - eps):C:eps_m dx [J]
// used in the LLG energy budget (magnum.np MagnetoElasticField.E).
func GetMelEnergyNP() float64 {
	if !haveMelNP() {
		return 0
	}
	return melasNPEnergyIntegral(1)
}

// AddMelEnergyDensityNP adds the magnetoelastic energy density [J/m3] to dst
// (registered in the energy density framework).
func AddMelEnergyDensityNP(dst *data.Slice) {
	if !haveMelNP() {
		return
	}
	flds := melasNPCompute(M.Buffer(), U.Buffer())
	defer flds.free()

	C := melasNPStiffness()
	defer melasNPRecycleC(C)
	l100 := Lambda100.MSlice()
	defer l100.Recycle()
	l111 := Lambda111.MSlice()
	defer l111.Recycle()

	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.MelasEnergyNP(buf, flds.guX, flds.guY, flds.guZ, M.Buffer(), l100, l111, C, 1, Mesh())
	cuda.Add(dst, dst, buf)
}

// GetTelNP returns the elastic kinetic energy 1/2 integral rho du^2 dx [J]
// (equal to integral p^2/(2 rho), with p = rho du), masked by the elastic mask.
func GetTelNP() float64 {
	size := Mesh().Size()
	du := DU.Buffer()

	buf := cuda.Buffer(1, size)
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	cuda.AddDotProduct(buf, 0.5, du, du) // 1/2 du.du per cell

	rho := Rho.MSlice()
	defer rho.Recycle()
	cuda.MulMSlice(buf, cuda.ToMSlice(buf), rho)

	mask := ElasticMaskNP.MSlice()
	defer mask.Recycle()
	cuda.MulMSlice(buf, cuda.ToMSlice(buf), mask)

	return cellVolume() * float64(cuda.Sum(buf))
}

// GetGilbertDissNP returns the instantaneous Gilbert dissipation rate [W] (positive = loss):
//   P = integral mu0 Ms alpha gamma/(1+alpha^2) |m x H|^2 dx
//     = integral Ms alpha gamma/((1+alpha^2) mu0) |m x B|^2 dx.
func GetGilbertDissNP() float64 {
	size := Mesh().Size()

	beff := cuda.Buffer(3, size)
	defer cuda.Recycle(beff)
	SetEffectiveField(beff)

	buf := cuda.Buffer(1, size)
	defer cuda.Recycle(buf)

	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	ms := Msat.MSlice()
	defer ms.Recycle()
	cuda.MelasGilbertNP(buf, M.Buffer(), beff, alpha, ms, Mesh())

	return cellVolume() * float64(cuda.Sum(buf)) * GammaLL / mag.Mu0
}
