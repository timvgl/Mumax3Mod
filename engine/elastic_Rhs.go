package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Eta = NewScalarParam("eta", "kg/(sm3)", "Damping constant")
	Rho = NewScalarParam("rho", "kg/m3", "Density")
	Bf  = NewExcitation("force_density", "N/m3", "Defines force density [N/m3]")
)

func calcRhs(dst, f, g *data.Slice) {
	RightSide(dst, f, g, Eta, Rho, Bf)
}

func calcRhsRegion(dst, f, g *data.Slice) {
	RightSideRegion(dst, f, g, Eta, Rho, Bf)
}

func RightSide(dst, f, g *data.Slice, Eta, Rho *RegionwiseScalar, Bf *Excitation) {
	//No elastodynamics is calculated if density is zero
	if Rho.nonZero() {
		rho, _ := Rho.Slice()
		defer cuda.Recycle(rho)

		eta, _ := Eta.Slice()
		defer cuda.Recycle(eta)

		bf, _ := Bf.Slice()
		defer cuda.Recycle(bf)

		//Elastic part of wave equation
		calcSecondDerivDisp(f)

		size := f.Size()
		melForce := cuda.Buffer(3, size)
		defer cuda.Recycle(melForce)
		// cuda.Zero(melForce)
		GetMagnetoelasticForceDensity(melForce)
		thermalElasticNoise := cuda.Buffer(melForce.NComp(), melForce.Size())
		defer cuda.Recycle(thermalElasticNoise)
		F_therm.EvalTo(thermalElasticNoise)

		cuda.RightSide(dst, f, g, eta, rho, bf, melForce, thermalElasticNoise)
		//Sufficient to only set right to zero because udot2 = udot+right
		//If initial udot!=0, then do also FreezeDisp(udot2)
		FreezeDisp(dst)
	}
}

func RightSideRegion(dst, f, g *data.Slice, Eta, Rho *RegionwiseScalar, Bf *Excitation) {
	//No elastodynamics is calculated if density is zero
	if Rho.nonZero() {
		rho, _ := Rho.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(rho)

		eta, _ := Eta.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(eta)

		bf, _ := Bf.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(bf)

		//Elastic part of wave equation
		calcSecondDerivDispRegion(f)

		size := f.Size()
		melForce := cuda.Buffer(3, size)
		defer cuda.Recycle(melForce)
		// cuda.Zero(melForce)
		GetMagnetoelasticForceDensityRegion(melForce)
		thermalElasticNoise := cuda.Buffer(melForce.NComp(), melForce.Size())
		defer cuda.Recycle(thermalElasticNoise)
		F_therm.EvalTo(thermalElasticNoise)

		cuda.RightSide(dst, f, g, eta, rho, bf, melForce, thermalElasticNoise)
		//Sufficient to only set right to zero because udot2 = udot+right
		//If initial udot!=0, then do also FreezeDisp(udot2)
		FreezeDisp(dst)
	}
}
