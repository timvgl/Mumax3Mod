package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	loadNormStrain        bool = false
	loadNormStrainPath    string
	loadShearStrain       bool = false
	loadShearStrainPath   string
	loadNormStrainConfig  bool = false
	normStrainConfig      Config
	loadShearStrainConfig bool = false
	shearStrainConfig     Config
	norm_strain           = NewVectorField("normStrain", "", "Normal strain components", setNormStrain)
	shear_strain          = NewVectorField("shearStrain", "", "Shear strain components", setShearStrain)
)

// ###################
// Strain
func setNormStrain(dst *data.Slice) {
	if Solvertype < 7 {
		Exx := exx.MSlice()
		defer Exx.Recycle()
	
		Eyy := eyy.MSlice()
		defer Eyy.Recycle()
	
		Ezz := ezz.MSlice()
		defer Ezz.Recycle()
		cuda.ScalarToVec(dst, Exx, Eyy, Ezz, Mesh())
	} else if loadNormStrain == false && loadNormStrainConfig == false {
		NormStrain(dst, U, C11)
	} else if loadNormStrain == true && loadNormStrainConfig == false {
		var d *data.Slice
		d = LoadFileDSlice(loadNormStrainPath)
		SetArray(dst, d)
		loadNormStrain = false
		loadNormStrainPath = ""
	} else if loadNormStrain == false && loadNormStrainConfig == true {
		SetInShape(dst, nil, normStrainConfig)
		loadNormStrainConfig = false
	} else {
		panic("Cannot load file for normStrain and set configuration for normStrain in parallel.")
	}
}

func setShearStrain(dst *data.Slice) {
	if Solvertype < 7 {
		Exy := exy.MSlice()
		defer Exy.Recycle()
	
		Exz := exz.MSlice()
		defer Exz.Recycle()
	
		Eyz := eyz.MSlice()
		defer Eyz.Recycle()

		cuda.ScalarToVec(dst, Exy, Eyz, Exz, Mesh())
	} else if loadShearStrain == false && loadShearStrainConfig == false {
		ShearStrain(dst, U, C11)
	} else if loadShearStrain == true && loadShearStrainConfig == false {
		var d *data.Slice
		d = LoadFileDSlice(loadShearStrainPath)
		SetArray(dst, d)
		loadShearStrain = false
		loadShearStrainPath = ""
	} else if loadShearStrain == false && loadShearStrainConfig == true {
		SetInShape(dst, nil, shearStrainConfig)
		loadShearStrainConfig = false
	} else {
		panic("Cannot load file for shearStrain and set configuration for shearStrain in parallel.")
	}
}

func NormStrain(dst *data.Slice, u displacement, C11 *RegionwiseScalar) {
	//C11 is necessary to check if we are at edges of free boundary regions
	c1 := C11.MSlice()
	defer c1.Recycle()
	cuda.NormStrain(dst, u.Buffer(), U.Mesh(), c1)
}

func ShearStrain(dst *data.Slice, u displacement, C11 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()
	cuda.ShearStrain(dst, u.Buffer(), U.Mesh(), c1)
}

func SetTime(fname string) {
	var meta data.Meta
	_, meta = LoadFileMeta(fname)
	Time = meta.Time
}

func SetArray(dst, src *data.Slice) {
	if src.Size() != dst.Size() {
		src = data.Resample(src, dst.Size())
	}
	data.Copy(dst, src)
	//b.normalize()
}
