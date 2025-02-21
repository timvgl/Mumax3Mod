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

func calcRhsRegion(dst, m, u, du, f, g *data.Slice) {
	if useFullSample {
		backupM := cuda.Buffer(M.Buffer().NComp(), M.Buffer().Size())
		backupU := cuda.Buffer(U.Buffer().NComp(), U.Buffer().Size())
		backupDU := cuda.Buffer(DU.Buffer().NComp(), DU.Buffer().Size())
		data.Copy(backupM, M.Buffer())
		data.Copy(backupU, U.Buffer())
		data.Copy(backupDU, DU.Buffer())
		data.CopyPart(M.Buffer(), m, 0, m.Size()[X], 0, m.Size()[Y], 0, m.Size()[Z], 0, 1, dst.StartX, dst.StartY, dst.StartZ, 0)
		data.CopyPart(U.Buffer(), u, 0, u.Size()[X], 0, u.Size()[Y], 0, u.Size()[Z], 0, 1, dst.StartX, dst.StartY, dst.StartZ, 0)
		data.CopyPart(DU.Buffer(), du, 0, du.Size()[X], 0, du.Size()[Y], 0, du.Size()[Z], 0, 1, dst.StartX, dst.StartY, dst.StartZ, 0)
		defer func() {
			data.Copy(M.Buffer(), backupM)
			cuda.Recycle(backupM)
			data.Copy(U.Buffer(), backupU)
			cuda.Recycle(backupU)
			data.Copy(DU.Buffer(), backupDU)
			cuda.Recycle(backupDU)
		}()
	}
	RightSideRegion(dst, m, u, f, g, Eta, Rho, Bf)
}

func RightSide(dst, f, g *data.Slice, Eta, Rho *RegionwiseScalar, Bf *Excitation) {
	//No elastodynamics is calculated if density is zero
	if Rho.nonZero() {
		rho, _ := Rho.Slice("rho", true)
		defer cuda.Recycle(rho)

		eta, _ := Eta.Slice("eta", true)
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

func RightSideRegion(dst, m, u, f, g *data.Slice, Eta, Rho *RegionwiseScalar, Bf *Excitation) {
	//No elastodynamics is calculated if density is zero
	if Rho.nonZero() {
		rho, _ := Rho.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(rho)

		eta, _ := Eta.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(eta)

		bf, _ := Bf.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(bf)

		//Elastic part of wave equation
		calcSecondDerivDispRegion(f, u)

		size := f.Size()
		melForce := cuda.Buffer(3, size)
		defer cuda.Recycle(melForce)
		// cuda.Zero(melForce)
		GetMagnetoelasticForceDensityRegion(melForce, m, useFullSample)
		thermalElasticNoise := cuda.Buffer(melForce.NComp(), melForce.Size())
		defer cuda.Recycle(thermalElasticNoise)
		cuda.Zero(thermalElasticNoise)
		F_therm.AddToRegion(thermalElasticNoise)
		cuda.RightSide(dst, f, g, eta, rho, bf, melForce, thermalElasticNoise)
		/*UDebug := dst.HostCopy()
		val, ok := autonum["debug"]
		if !ok {
			autonum["debug"] = 0
			val = 0
		}
		meta := *new(data.Meta)
		meta.Time = Time
		meta.CellSize = Mesh().CellSize()
		meta.Name = "debug"
		queOutput(func() { saveAs_sync(fmt.Sprintf(OD()+"/debug_%d.ovf", val), UDebug, meta, outputFormat) })
		autonum["debug"]++*/
		//Sufficient to only set right to zero because udot2 = udot+right
		//If initial udot!=0, then do also FreezeDisp(udot2)
		FreezeDispRegion(dst, u)

	}
}
