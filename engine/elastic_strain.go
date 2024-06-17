package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	useLoadedStrain				bool = false
	normStrainFile				string
	shearStrainFile				string
	useTimeFromLoadedStrain		bool = false
	norm_strain  = NewVectorField("normStrain", "", "Normal strain components", setNormStrain)
	shear_strain = NewVectorField("shearStrain", "", "Shear strain components", setShearStrain)
)

func init() {
	DeclVar("useLoadedStrain", &useLoadedStrain, "Dont calc strain from u, but take loaded strain")
	DeclVar("useTimeFromLoadedStrain", &useTimeFromLoadedStrain, "")
	DeclVar("normStrainFile", &normStrainFile, "")
	DeclVar("shearStrainFile", &shearStrainFile, "")
}

//###################
//Strain
func setNormStrain(dst *data.Slice) {
	if useLoadedStrain == false {
		NormStrain(dst, U, C11)
	} else {
		if useTimeFromLoadedStrain == false {
			var d *data.Slice
			d = LoadFile(normStrainFile)
			SetArray(dst, d)
		} else {
			var meta data.Meta
			var d *data.Slice
			d, meta = LoadFileMeta(normStrainFile)
			SetArray(dst, d)
			Time = meta.Time
		}
	}
}

func setShearStrain(dst *data.Slice) {
	if useLoadedStrain == false {
		ShearStrain(dst, U, C11)
	} else {
		if useTimeFromLoadedStrain == false {
			var d *data.Slice
			d = LoadFile(shearStrainFile)
			SetArray(dst, d)
		} else {
			var meta data.Meta
			var d *data.Slice
			d, meta = LoadFileMeta(shearStrainFile)
			SetArray(dst, d)
			Time = meta.Time
		}
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

func SetArray(dst, src *data.Slice,) {
	if src.Size() != dst.Size() {
		src = data.Resample(src, dst.Size())
	}
	data.Copy(dst, src)
	//b.normalize()
}