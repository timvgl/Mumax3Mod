package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	loadNormStress bool = false
	loadNormStressPath string
	loadShearStress bool = false
	loadShearStressPath string
	loadNormStressConfig bool = false
	normStressConfig Config
	loadShearStressConfig bool = false
	shearStressConfig Config
	norm_stress  = NewVectorField("normStress", "", "Normal stress components", setNormStress)
	shear_stress = NewVectorField("shearStress", "", "Shear stress components", setShearStress)
)

//###################
//Strain
func setNormStress(dst *data.Slice) {
	if loadNormStress == false && loadNormStressConfig == false {
		NormStress(dst, norm_strain.Quantity, C11, C12)
	} else if loadNormStress == true && loadNormStressConfig == false{
		var d *data.Slice
		d = LoadFileDSlice(loadNormStressPath)
		SetArray(dst, d)
		loadNormStress = false
		loadNormStressPath = ""
	} else if loadNormStress == false && loadNormStressConfig == true {
		SetInShape(dst, nil, normStressConfig)
		loadNormStressConfig = false
	} else {
		panic("Cannot load file for normStress and set configuration for normStress in parallel.")
	}
}

func setShearStress(dst *data.Slice) {
	if loadShearStress == false && loadShearStressConfig == false {
		ShearStress(dst, shear_strain.Quantity, C44)
	} else if loadShearStress == true && loadShearStressConfig == false{
		var d *data.Slice
		d = LoadFileDSlice(loadShearStressPath)
		SetArray(dst, d)
		loadShearStress = false
		loadShearStressPath = ""
	} else if loadShearStress == false && loadShearStressConfig == true {
		SetInShape(dst, nil, shearStressConfig)
		loadShearStressConfig = false
	} else {
		panic("Cannot load file for shearStress and set configuration for shearStress in parallel.")
	}
}

func NormStress(dst *data.Slice, eNorm Quantity, C11, C12 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()

	c2 := C12.MSlice()
	defer c2.Recycle()

	enorm := ValueOf(eNorm)
	defer cuda.Recycle(enorm)
	cuda.NormStress(dst, enorm, U.Mesh(), c1, c2)
}

func ShearStress(dst *data.Slice, eShear Quantity, C44 *RegionwiseScalar) {
	c3 := C44.MSlice()
	defer c3.Recycle()

	eshear := ValueOf(eShear)
	defer cuda.Recycle(eshear)
	cuda.ShearStress(dst, eshear, U.Mesh(), c3)
}
