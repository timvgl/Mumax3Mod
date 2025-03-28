package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Edens_kin = NewScalarField("Edens_kin", "J/m3", "Kinetic energy density", GetKineticEnergy)
	E_kin     = NewScalarValue("E_kin", "J", "Kinetic energy", GetTotKineticEnergy)
)

func init() {
	registerEnergyElastic(GetTotKineticEnergy, GetKineticEnergy)
}

func GetKineticEnergy(dst *data.Slice) {
	KineticEnergyDens(dst, DU, Rho)
}

func KineticEnergyDens(dst *data.Slice, DU firstDerivative, Rho *RegionwiseScalar) {
	rho := Rho.MSlice()
	defer rho.Recycle()
	cuda.KineticEnergy(dst, DU.Buffer(), rho, U.Mesh())
}

func GetTotKineticEnergy() float64 {
	kinetic_energy := ValueOf(Edens_kin.Quantity)
	defer cuda.Recycle(kinetic_energy)
	return cellVolume() * float64(cuda.Sum(kinetic_energy))
}
