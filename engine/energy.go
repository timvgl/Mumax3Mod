package engine

// Total energy calculation

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// TODO: Integrate(Edens)
// TODO: consistent naming SetEdensTotal, ...

var (
	energyTerms 		[]func() float64        // all contributions to total energy
	edensTerms  		[]func(dst *data.Slice) // all contributions to total energy density (add to dst)
	Edens_total 		= NewScalarField("Edens_total", "J/m3", "Total energy density", SetTotalEdens)
	E_total     		= NewScalarValue("E_total", "J", "total energy", GetTotalEnergy)
	energyTermsElastic 	[]func() float64        // all contributions to total energy of system - elastic and magnetic
	edensTermsElastic	[]func(dst *data.Slice) // all contributions to total energy density (add to dst)
	E_elastic 			= NewScalarValue("E_elastic", "J", "total energy of elstic system", GetTotalEnergyElastic)
	E_System 			= NewScalarValue("E_System", "J", "total energy of magnetic and elstic system", GetTotalEnergySystem)

)

// add energy term to global energy
func registerEnergy(term func() float64, dens func(*data.Slice)) {
	energyTerms = append(energyTerms, term)
	edensTerms = append(edensTerms, dens)
}

func registerEnergyElastic(term func() float64, dens func(*data.Slice)) {
	energyTermsElastic = append(energyTermsElastic, term)
	edensTermsElastic = append(edensTermsElastic, dens)
}

func GetTotalEnergySystem() float64 {
	return GetTotalEnergyElastic() + GetTotalEnergy() - GetMagnetoelasticEnergy() //remove ME energy because it would be in there twice otherwise.
}

func GetTotalEnergyElastic() float64 {
	E := 0.
	for _, f := range energyTermsElastic {
		E += f()
	}
	checkNaN1(E)
	return E
}

// Returns the total energy in J.
func GetTotalEnergy() float64 {
	E := 0.
	for _, f := range energyTerms {
		E += f()
	}
	checkNaN1(E)
	return E
}

// Set dst to total energy density in J/m3
func SetTotalEdens(dst *data.Slice) {
	cuda.Zero(dst)
	for _, addTerm := range edensTerms {
		addTerm(dst)
	}
}

// volume of one cell in m3
func cellVolume() float64 {
	c := Mesh().CellSize()
	return c[0] * c[1] * c[2]
}

// returns a function that adds to dst the energy density:
// 	prefactor * dot (M_full, field)
func makeEdensAdder(field Quantity, prefactor float64) func(*data.Slice) {
	return func(dst *data.Slice) {
		B := ValueOf(field)
		defer cuda.Recycle(B)
		m := ValueOf(M_full)
		defer cuda.Recycle(m)
		factor := float32(prefactor)
		cuda.AddDotProduct(dst, factor, B, m)
	}
}

// vector dot product
func dot(a, b Quantity) float64 {
	A := ValueOf(a)
	defer cuda.Recycle(A)
	B := ValueOf(b)
	defer cuda.Recycle(B)
	return float64(cuda.Dot(A, B))
}
