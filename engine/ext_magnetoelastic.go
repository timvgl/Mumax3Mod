package engine

// Mangeto-elastic coupling.

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	//"fmt"
)

var (
	B1             = NewScalarParam("B1", "J/m3", "First magneto-elastic coupling constant")
	B2             = NewScalarParam("B2", "J/m3", "Second magneto-elastic coupling constant")
	exx            = NewScalarExcitation("exx", "", "exx component of the strain tensor")
	eyy            = NewScalarExcitation("eyy", "", "eyy component of the strain tensor")
	ezz            = NewScalarExcitation("ezz", "", "ezz component of the strain tensor")
	exy            = NewScalarExcitation("exy", "", "exy component of the strain tensor")
	exz            = NewScalarExcitation("exz", "", "exz component of the strain tensor")
	eyz            = NewScalarExcitation("eyz", "", "eyz component of the strain tensor")
	B_mel          = NewVectorField("B_mel", "T", "Magneto-elastic filed", AddMagnetoelasticField)
	F_mel          = NewVectorField("F_mel", "N/m3", "Magneto-elastic force density", GetMagnetoelasticForceDensity)
	F_melM         = NewVectorField("F_melM", "N/m3", "Magneto-elastic force density", GetMagnetoelasticForceDensityM)
	F_el           = NewVectorField("F_el", "N/m3", "Elastic force density", GetElasticForceDensity)
	F_elsys        = NewVectorField("F_elsys", "N/m3", "Elastic force density", GetSumElasticForces)
	rhod2udt2      = NewVectorField("rhod2udt2", "N/m3", "Force of displacement", GetDisplacementForceDensity)
	etadudt        = NewVectorField("etadudt", "N/m3", "Force of displacement due to speed", GetDisplacementSpeedForceDensity)
	Edens_mel      = NewScalarField("Edens_mel", "J/m3", "Magneto-elastic energy density", AddMagnetoelasticEnergyDensity)
	E_mel          = NewScalarValue("E_mel", "J", "Magneto-elastic energy", GetMagnetoelasticEnergy)
	DDU            = NewVectorField("ddu", "", "", GetDisplacementAcceleration)
	fixM      bool = false
	zeroMel        = NewScalarParam("_zeroMel", "", "utility zero parameter")
)

func init() {
	registerEnergy(GetMagnetoelasticEnergy, AddMagnetoelasticEnergyDensity)
	registerEnergyElastic(GetMagnetoelasticEnergy, AddMagnetoelasticEnergyDensity)
	DeclVar("fixM", &fixM, "only calculate elastic time evolution and keep m constant")

}

func AddMagnetoelasticField(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	enorm := ValueOf(norm_strain.Quantity)
	defer cuda.Recycle(enorm)

	eshear := ValueOf(shear_strain.Quantity)
	defer cuda.Recycle(eshear)

	b1 := B1.MSlice()
	defer b1.Recycle()

	b2 := B2.MSlice()
	defer b2.Recycle()

	ms := Msat.MSlice()
	defer ms.Recycle()

	cuda.AddMagnetoelasticField(dst, M.Buffer(),
		enorm, eshear,
		b1, b2, ms)
}

func AddMagnetoelasticFieldRegion(dst, m, u *data.Slice, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !useFullSample {
		haveMel := B1.nonZero() || B2.nonZero()
		if !haveMel {
			return
		}
		enorm := cuda.Buffer(dst.NComp(), dst.RegionSize())
		setNormStrainRegion(enorm, u, pbcX, pbcY, pbcZ, useFullSample)
		defer cuda.Recycle(enorm)

		eshear := cuda.Buffer(dst.NComp(), dst.RegionSize())
		setShearStrainRegion(eshear, u, pbcX, pbcY, pbcZ, useFullSample)
		defer cuda.Recycle(eshear)

		b1 := B1.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer b1.Recycle()

		b2 := B2.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer b2.Recycle()

		ms := Msat.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer ms.Recycle()

		cuda.AddMagnetoelasticField(dst, m,
			enorm, eshear,
			b1, b2, ms)
		/*
			buf := cuda.Buffer(dst.NComp(), dst.Size())
			defer cuda.Recycle(buf)
			size2 := dst.Size()
			size2[0] /= 2
			buf2 := cuda.Buffer(dst.NComp(), size2)
			defer cuda.Recycle(buf2)
			cuda.AddMagnetoelasticField(buf, m,
				enorm, eshear,
				b1, b2, ms)
			cuda.Crop(buf2, buf, size2[0]/2, 0, 0)
			UDebug := buf2.HostCopy()
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
	} else {
		B := cuda.Buffer(M.NComp(), M.Buffer().Size())
		cuda.Zero(B)
		AddMagnetoelasticField(B)
		BCropped := cuda.Buffer(dst.NComp(), dst.Size())
		cuda.Crop(BCropped, B, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, BCropped)
		cuda.Recycle(BCropped)
		cuda.Recycle(B)
	}
}

func GetMagnetoelasticForceDensity(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	if !BoolAllowInhomogeniousMECoupling {
		util.AssertMsg(B1.IsUniform() && B2.IsUniform(), "Magnetoelastic: B1, B2 must be uniform")
	}

	b1 := B1.MSlice()
	defer b1.Recycle()

	b2 := B2.MSlice()
	defer b2.Recycle()
	cuda.GetMagnetoelasticForceDensity(dst, M.Buffer(),
		b1, b2, M.Mesh())
}

func GetMagnetoelasticForceDensityRegion(dst, m *data.Slice, useFullSample bool) {
	if !useFullSample {
		haveMel := B1.nonZero() || B2.nonZero()
		if !haveMel {
			return
		}

		if !BoolAllowInhomogeniousMECoupling {
			util.AssertMsg(B1.IsUniform() && B2.IsUniform(), "Magnetoelastic: B1, B2 must be uniform")
		}

		b1 := B1.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer b1.Recycle()

		b2 := B2.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer b2.Recycle()

		cuda.GetMagnetoelasticForceDensity(dst, m,
			b1, b2, Crop(&M, dst.StartX, dst.EndX, dst.StartY, dst.EndY, dst.StartZ, dst.EndZ).Mesh())
	} else {
		F := cuda.Buffer(M.NComp(), M.Buffer().Size())
		cuda.Zero(F)
		GetMagnetoelasticForceDensity(F)
		FCropped := cuda.Buffer(dst.NComp(), dst.Size())
		cuda.Crop(FCropped, F, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, FCropped)
		cuda.Recycle(FCropped)
		cuda.Recycle(F)
	}

}

func GetElasticForceDensity(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	if !BoolAllowInhomogeniousMECoupling {
		util.AssertMsg(B1.IsUniform() && B2.IsUniform(), "Magnetoelastic: B1, B2 must be uniform")
	}

	stress_norm := ValueOf(norm_stress.Quantity)
	defer cuda.Recycle(stress_norm)

	stress_shear := ValueOf(shear_stress.Quantity)
	defer cuda.Recycle(stress_shear)
	cuda.GetElasticForceDensity(dst,
		stress_norm, stress_shear, U.Mesh())

}

func GetDisplacementSpeedForceDensity(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	if !BoolAllowInhomogeniousMECoupling {
		util.AssertMsg(B1.IsUniform() && B2.IsUniform(), "Magnetoelastic: B1, B2 must be uniform")
	}

	eta := Eta.MSlice()
	defer eta.Recycle()

	cuda.MulMSlice(dst, cuda.ToMSlice(DU.Buffer()), eta)
}

func GetDisplacementForceDensity(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	if !BoolAllowInhomogeniousMECoupling {
		util.AssertMsg(B1.IsUniform() && B2.IsUniform(), "Magnetoelastic: B1, B2 must be uniform")
	}

	F_elRef := ValueOf(F_el.Quantity)
	//F_elRef := cuda.Buffer(3, Mesh().Size())
	//GetElasticForceDensity(F_elRef)
	defer cuda.Recycle(F_elRef)

	F_melRef := ValueOf(F_mel.Quantity)
	defer cuda.Recycle(F_melRef)

	etadudtRef := ValueOf(etadudt.Quantity)
	defer cuda.Recycle(etadudtRef)

	bf, _ := Bf.Slice()
	defer cuda.Recycle(bf)
	cuda.Madd4(dst, F_elRef, F_melRef, bf, etadudtRef, 1, 1, 1, -1)
}

func GetMaxDisplacementForceDensity() float64 {
	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	GetDisplacementForceDensity(buf)
	return cuda.MaxVecNorm(buf)
}

func GetDisplacementAcceleration(dst *data.Slice) {
	GetDisplacementForceDensity(dst)
	rho := Rho.MSlice()
	defer rho.Recycle()
	cuda.DivMSlice(dst, cuda.ToMSlice(dst), rho)
}

func GetMaxDisplacementAcceleration() float64 {
	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	GetDisplacementAcceleration(buf)
	return cuda.MaxVecNorm(buf)
}

func GetAverageDisplacementAcceleration() float64 {
	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	GetDisplacementAcceleration(buf)
	avergeDDU := sAverageUniverse(buf)
	return float64(math.Sqrt(math.Pow(avergeDDU[0], 2) + math.Pow(avergeDDU[1], 2) + math.Pow(avergeDDU[2], 2)))
}

func GetAverageDisplacementAccelerationIgnoreNaN() float64 {
	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	GetDisplacementAcceleration(buf)
	avergeDDU := sAverageUniverseIgnoreNaN(buf)
	return float64(math.Sqrt(math.Pow(avergeDDU[0], 2) + math.Pow(avergeDDU[1], 2) + math.Pow(avergeDDU[2], 2)))
}

func AddMagnetoelasticEnergyDensity(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	buf := cuda.Buffer(B_mel.NComp(), B_mel.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf := ValueOf(M_full)
	defer cuda.Recycle(Mf)

	enorm := ValueOf(norm_strain.Quantity)
	defer cuda.Recycle(enorm)

	eshear := ValueOf(shear_strain.Quantity)
	defer cuda.Recycle(eshear)

	b1 := B1.MSlice()
	defer b1.Recycle()

	b2 := B2.MSlice()
	defer b2.Recycle()

	ms := Msat.MSlice()
	defer ms.Recycle()

	zeromel := zeroMel.MSlice()
	defer zeromel.Recycle()

	// 1st
	cuda.Zero(buf)
	cuda.AddMagnetoelasticField(buf, M.Buffer(),
		enorm, eshear,
		b1, zeromel, ms)
	cuda.AddDotProduct(dst, -1./2., buf, Mf)

	// 1nd
	cuda.Zero(buf)
	cuda.AddMagnetoelasticField(buf, M.Buffer(),
		enorm, eshear,
		zeromel, b2, ms)
	cuda.AddDotProduct(dst, -1./1., buf, Mf)
}

// Returns magneto-ell energy in joules.
func GetMagnetoelasticEnergy() float64 {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return float64(0.0)
	}

	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddMagnetoelasticEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}

func GetSumElasticForces(dst *data.Slice) {
	buf := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	GetDisplacementForceDensity(dst)
	GetDisplacementSpeedForceDensity(buf)
	cuda.Madd2(dst, dst, buf, 1, 1)
	GetElasticForceDensity(buf)
	cuda.Madd2(dst, dst, buf, 1, -1)
	GetMagnetoelasticForceDensity(buf)
	cuda.Madd2(dst, dst, buf, 1, -1)
	bf, _ := Bf.Slice()
	defer cuda.Recycle(bf)
	cuda.Madd2(dst, dst, bf, 1, -1)
}

func GetMagnetoelasticForceDensityM(dst *data.Slice) {
	Edens_melV := ValueOf(Edens_mel)
	defer cuda.Recycle(Edens_melV)

	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.MOne(buf)
	cuda.Mul(dst, dst, buf)
	cuda.Grad(dst, Edens_melV, B_mel.Mesh())
}
