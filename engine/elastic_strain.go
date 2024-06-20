package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	useLoadedStrain				bool = false
	useLoadedStrainNorm			bool = false
	useLoadedStrainShear		bool = false
	setShearStrainZero			bool = false
	setNormStrainZero			bool = false
	normStrainFile				string
	shearStrainFile				string
	useTimeFromLoadedStrain		bool = false
	norm_strain  = NewVectorField("normStrain", "", "Normal strain components", setNormStrain)
	shear_strain = NewVectorField("shearStrain", "", "Shear strain components", setShearStrain)
)

func init() {
	DeclVar("useLoadedStrain", &useLoadedStrain, "Dont calc strain from u, but take loaded strain")
	DeclVar("useLoadedStrainNorm", &useLoadedStrainNorm, "")
	DeclVar("useLoadedStrainShear", &useLoadedStrainShear, "")
	DeclVar("useTimeFromLoadedStrain", &useTimeFromLoadedStrain, "")
	DeclVar("normStrainFile", &normStrainFile, "")
	DeclVar("shearStrainFile", &shearStrainFile, "")
	DeclVar("setShearStrainZero", &setShearStrainZero, "")
	DeclVar("setNormStrainZero", &setNormStrainZero, "")
}

//###################
//Strain
func setNormStrain(dst *data.Slice) {
	if useLoadedStrain == false && useLoadedStrainNorm == false{
		NormStrain(dst, U, C11)
	} else if setNormStrainZero == false {
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
	} else {
		SetInShape(dst, nil, Uniform(0,0,0))
	}
}

func setShearStrain(dst *data.Slice) {
	if useLoadedStrain == false && useLoadedStrainShear == false {
		ShearStrain(dst, U, C11)
	} else if setShearStrainZero == false {
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
	} else {
		SetInShape(dst, nil, Uniform(0,0,0))
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

func SetInShape(dst *data.Slice, region Shape, conf Config) {
	checkMesh()

	if region == nil {
		region = universe
	}
	host := dst.HostCopy()
	h := host.Vectors()
	n := dst.Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					u := conf(x, y, z)
					h[X][iz][iy][ix] = float32(u[X])
					h[Y][iz][iy][ix] = float32(u[Y])
					h[Z][iz][iy][ix] = float32(u[Z])
				}
			}
		}
	}
	SetArray(dst, host)
}