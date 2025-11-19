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
	if loadNormStrain && !loadNormStrainConfig {
		d := LoadFileDSlice(loadNormStrainPath)
		SetArray(dst, d)
		loadNormStrain = false
		loadNormStrainPath = ""
	} else if !loadNormStrain && loadNormStrainConfig {
		SetInShape(dst, nil, normStrainConfig)
		loadNormStrainConfig = false
	} else if UseExcitation {
		Exx := exx.MSlice()
		defer Exx.Recycle()

		Eyy := eyy.MSlice()
		defer Eyy.Recycle()

		Ezz := ezz.MSlice()
		defer Ezz.Recycle()
		cuda.ScalarToVec(dst, Exx, Eyy, Ezz, Mesh())
	} else if !loadNormStrain && !loadNormStrainConfig {
		if !mtxElasto {
			NormStrain(dst, U, C11)
		} else {
			NormStrainMtx(dst, U, M, C11, C12, C44, C13, C33)
		}
	} else {
		panic("Cannot load file for normStrain and set configuration for normStrain in parallel.")
	}
}

func setShearStrain(dst *data.Slice) {
	if loadShearStrain && !loadShearStrainConfig {
		d := LoadFileDSlice(loadShearStrainPath)
		SetArray(dst, d)
		loadShearStrain = false
		loadShearStrainPath = ""
	} else if !loadShearStrain && loadShearStrainConfig {
		SetInShape(dst, nil, shearStrainConfig)
		loadShearStrainConfig = false
	} else if UseExcitation {
		Exy := exy.MSlice()
		defer Exy.Recycle()

		Exz := exz.MSlice()
		defer Exz.Recycle()

		Eyz := eyz.MSlice()
		defer Eyz.Recycle()

		cuda.ScalarToVec(dst, Exy, Eyz, Exz, Mesh())
	} else if !loadShearStrain && !loadShearStrainConfig {
		if !mtxElasto {
			ShearStrain(dst, U, C11)
		} else {
			ShearStrainMtx(dst, U, M, C11, C12, C44, C13, C33)
		}
	} else {
		panic("Cannot load file for shearStrain and set configuration for shearStrain in parallel.")
	}
}

func NormStrain(dst *data.Slice, u displacement, C11 *RegionwiseScalar) {
	// C11 is used for checking edges of free-boundary regions.
	c1 := C11.MSlice()
	defer c1.Recycle()
	cuda.NormStrain(dst, u.Buffer(), U.Mesh(), c1)
}

func NormStrainMtx(dst *data.Slice, u displacement, m magnetization, C11, C12, C44, C13, C33 *RegionwiseScalar) {
	c11 := C11.MSlice()
	defer c11.Recycle()

	c12 := C12.MSlice()
	defer c12.Recycle()

	c13 := C13.MSlice()
	defer c13.Recycle()

	c33 := C33.MSlice()
	defer c33.Recycle()

	c44 := C44.MSlice()
	defer c44.Recycle()

	b1 := B1.MSlice()
	defer b1.Recycle()

	b2 := B2.MSlice()
	defer b2.Recycle()
	normStressTmp := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(normStressTmp)
	cuda.Zero(normStressTmp)
	shearStressTmp := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(shearStressTmp)
	cuda.Zero(shearStressTmp)
	shearStrainTmp := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(shearStrainTmp)
	cuda.Zero(shearStrainTmp)
	cuda.StressWurtzitMtx(normStressTmp, shearStressTmp, dst, shearStrainTmp, u.Buffer(), m.Buffer(), c11, c12, c13, c33, c44, b1, b2, U.Mesh(), cubicElasto)
}

func ShearStrain(dst *data.Slice, u displacement, C11 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()
	cuda.ShearStrain(dst, u.Buffer(), U.Mesh(), c1)
}

func ShearStrainMtx(dst *data.Slice, u displacement, m magnetization, C11, C12, C44, C13, C33 *RegionwiseScalar) {
	c11 := C11.MSlice()
	defer c11.Recycle()

	c12 := C12.MSlice()
	defer c12.Recycle()

	c13 := C13.MSlice()
	defer c13.Recycle()

	c33 := C33.MSlice()
	defer c33.Recycle()

	c44 := C44.MSlice()
	defer c44.Recycle()

	b1 := B1.MSlice()
	defer b1.Recycle()

	b2 := B2.MSlice()
	defer b2.Recycle()
	normStressTmp := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(normStressTmp)
	cuda.Zero(normStressTmp)
	shearStressTmp := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(shearStressTmp)
	cuda.Zero(shearStressTmp)
	normStrainTmp := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(normStrainTmp)
	cuda.Zero(normStrainTmp)
	cuda.StressWurtzitMtx(normStressTmp, shearStressTmp, normStrainTmp, dst, u.Buffer(), m.Buffer(), c11, c12, c13, c33, c44, b1, b2, U.Mesh(), cubicElasto)
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

// --- Additional functions for region-solver support for strain ---

// setNormStrainRegion is analogous to setNormStressRegion.
// It dispatches to the region version of the normal strain calculation.
func setNormStrainRegion(dst, u *data.Slice, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !loadNormStrain && !loadNormStrainConfig {
		NormStrainRegion(dst, u, C11, pbcX, pbcY, pbcZ, useFullSample)
	} else if loadNormStrain && !loadNormStrainConfig {
		panic("Loading ovf files for NormStrain not supported for RegionSolver yet")
		// (Optional: include code to load and crop the file if support is added later.)
	} else if !loadNormStrain && loadNormStrainConfig {
		panic("Loading config for NormStrain not supported for RegionSolver yet")
		// (Optional: include code to apply a configuration if support is added later.)
	} else {
		panic("Cannot load file for normStrain and set configuration for normStrain in parallel.")
	}
}

// setShearStrainRegion is analogous to setShearStressRegion.
func setShearStrainRegion(dst, u *data.Slice, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !loadShearStrain && !loadShearStrainConfig {
		ShearStrainRegion(dst, u, C11, pbcX, pbcY, pbcZ, useFullSample)
	} else if loadShearStrain && !loadShearStrainConfig {
		panic("Loading ovf files for ShearStrain not supported for RegionSolver yet")
	} else if !loadShearStrain && loadShearStrainConfig {
		panic("Loading config for ShearStrain not supported for RegionSolver yet")
	} else {
		panic("Cannot load file for shearStrain and set configuration for shearStrain in parallel.")
	}
}

// NormStrainRegion computes the normal strain on a region.
// If useFullSample is false, it crops the displacement field and computes strain on the region mesh;
// otherwise it computes strain on the full mesh and crops the result.
func NormStrainRegion(dst, u *data.Slice, C11 *RegionwiseScalar, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !useFullSample {
		// Obtain the region version of the regionwise scalar.
		c1 := C11.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c1.Recycle()

		// Crop the displacement buffer to the region.

		// Create a region mesh from the region size and cell sizes.
		regionMesh := data.NewMesh(
			dst.RegionSize()[X], dst.RegionSize()[Y], dst.RegionSize()[Z],
			MeshOf(&U).CellSize()[X], MeshOf(&U).CellSize()[Y], MeshOf(&U).CellSize()[Z],
			pbcX, pbcY, pbcZ)
		cuda.NormStrain(dst, u, regionMesh, c1)
	} else {
		// Compute the strain on the full sample, then crop.
		full := cuda.Buffer(U.Buffer().NComp(), U.Buffer().Size())
		cuda.Zero(full)
		NormStrain(full, U, C11)
		cuda.Crop(dst, full, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(full)
	}
}

// ShearStrainRegion computes the shear strain on a region.
// It follows the same pattern as NormStrainRegion.
func ShearStrainRegion(dst, u *data.Slice, C11 *RegionwiseScalar, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !useFullSample {
		c1 := C11.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c1.Recycle()

		regionMesh := data.NewMesh(
			dst.RegionSize()[X], dst.RegionSize()[Y], dst.RegionSize()[Z],
			MeshOf(&U).CellSize()[X], MeshOf(&U).CellSize()[Y], MeshOf(&U).CellSize()[Z],
			pbcX, pbcY, pbcZ)
		cuda.ShearStrain(dst, u, regionMesh, c1)
	} else {
		full := cuda.Buffer(U.Buffer().NComp(), U.Buffer().Size())
		cuda.Zero(full)
		ShearStrain(full, U, C11)
		cuda.Crop(dst, full, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(full)
	}
}
