package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	loadNormStress        bool = false
	loadNormStressPath    string
	loadShearStress       bool = false
	loadShearStressPath   string
	loadNormStressConfig  bool = false
	normStressConfig      Config
	loadShearStressConfig bool = false
	shearStressConfig     Config
	norm_stress           = NewVectorField("normStress", "", "Normal stress components", setNormStress)
	shear_stress          = NewVectorField("shearStress", "", "Shear stress components", setShearStress)
)

// ###################
// Strain
func setNormStress(dst *data.Slice) {
	if !loadNormStress && !loadNormStressConfig {
		NormStress(dst, norm_strain.Quantity, C11, C12)
	} else if loadNormStress && !loadNormStressConfig {
		var d = LoadFileDSlice(loadNormStressPath)
		SetArray(dst, d)
		loadNormStress = false
		loadNormStressPath = ""
	} else if !loadNormStress && loadNormStressConfig {
		SetInShape(dst, nil, normStressConfig)
		loadNormStressConfig = false
	} else {
		panic("Cannot load file for normStress and set configuration for normStress in parallel.")
	}
}

func setShearStress(dst *data.Slice) {
	if !loadShearStress && !loadShearStressConfig {
		ShearStress(dst, shear_strain.Quantity, C44)
	} else if loadShearStress && !loadShearStressConfig {
		var d = LoadFileDSlice(loadShearStressPath)
		SetArray(dst, d)
		loadShearStress = false
		loadShearStressPath = ""
	} else if !loadShearStress && loadShearStressConfig {
		SetInShape(dst, nil, shearStressConfig)
		loadShearStressConfig = false
	} else {
		panic("Cannot load file for shearStress and set configuration for shearStress in parallel.")
	}
}

func setNormStressRegion(dst *data.Slice, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !loadNormStress && !loadNormStressConfig {
		NormStressRegion(dst, norm_strain.Quantity, C11, C12, pbcX, pbcY, pbcZ, useFullSample)
	} else if loadNormStress && !loadNormStressConfig {
		panic("Loading ovf files for NormStress not supported for RegionSolver yet")
		var d = LoadFileDSlice(loadNormStressPath)
		dRed := cuda.Buffer(dst.NComp(), dst.RegionSize())
		util.AssertMsg(d.Size() == Mesh().Size(), "File for NormStress needs to have the same mesh like the simulation for RegionSolver")
		cuda.Crop(dRed, d, dst.StartX, dst.StartY, dst.StartZ)
		SetArray(dst, dRed)
		loadNormStress = false
		loadNormStressPath = ""
	} else if !loadNormStress && loadNormStressConfig {
		panic("Loading config for NormStress not supported for RegionSolver yet")
		d := cuda.Buffer(dst.NComp(), Mesh().Size())
		SetInShape(d, nil, normStressConfig)
		cuda.Crop(dst, d, dst.StartX, dst.StartY, dst.StartZ)
		loadNormStressConfig = false
	} else {
		panic("Cannot load file for normStress and set configuration for normStress in parallel.")
	}
}

func setShearStressRegion(dst *data.Slice, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !loadShearStress && !loadShearStressConfig {
		ShearStressRegion(dst, shear_strain.Quantity, C44, pbcX, pbcY, pbcZ, useFullSample)
	} else if loadShearStress && !loadShearStressConfig {
		panic("Loading ovf files for ShearStress not supported for RegionSolver yet")
		var d = LoadFileDSlice(loadShearStressPath)
		dRed := cuda.Buffer(dst.NComp(), dst.RegionSize())
		util.AssertMsg(d.Size() == Mesh().Size(), "File for ShearStress needs to have the same mesh like the simulation for RegionSolver")
		cuda.Crop(dRed, d, dst.StartX, dst.StartY, dst.StartZ)
		SetArray(dst, dRed)
		loadShearStress = false
		loadShearStressPath = ""
	} else if !loadShearStress && loadShearStressConfig {
		panic("Loading config for ShearStress not supported for RegionSolver yet")
		d := cuda.Buffer(dst.NComp(), Mesh().Size())
		SetInShape(d, nil, shearStressConfig)
		cuda.Crop(dst, d, dst.StartX, dst.StartY, dst.StartZ)
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

func NormStressRegion(dst *data.Slice, eNorm Quantity, C11, C12 *RegionwiseScalar, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !useFullSample {
		c1 := C11.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c1.Recycle()

		c2 := C12.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c2.Recycle()

		enorm := cuda.Buffer(dst.NComp(), dst.RegionSize())
		ValueOfRegion(eNorm, enorm, dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(enorm)
		cuda.NormStress(dst, enorm, data.NewMesh(dst.RegionSize()[X], dst.RegionSize()[Y], dst.RegionSize()[Z], MeshOf(&U).CellSize()[X], MeshOf(&U).CellSize()[Y], MeshOf(&U).CellSize()[Z], pbcX, pbcY, pbcZ), c1, c2)
	} else {
		norm := cuda.Buffer(M.NComp(), M.Buffer().Size())
		cuda.Zero(norm)
		NormStress(norm, eNorm, C11, C12)
		cuda.Crop(dst, norm, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(norm)
	}
}

func ShearStress(dst *data.Slice, eShear Quantity, C44 *RegionwiseScalar) {
	c3 := C44.MSlice()
	defer c3.Recycle()

	eshear := ValueOf(eShear)
	defer cuda.Recycle(eshear)
	cuda.ShearStress(dst, eshear, U.Mesh(), c3)
}

func ShearStressRegion(dst *data.Slice, eShear Quantity, C44 *RegionwiseScalar, pbcX, pbcY, pbcZ int, useFullSample bool) {
	if !useFullSample {
		c3 := C44.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c3.Recycle()

		eshear := cuda.Buffer(dst.NComp(), dst.RegionSize())
		ValueOfRegion(eShear, eshear, dst.StartX, dst.StartY, dst.StartZ)
		defer cuda.Recycle(eshear)
		cuda.ShearStress(dst, eshear, data.NewMesh(dst.RegionSize()[X], dst.RegionSize()[Y], dst.RegionSize()[Z], MeshOf(&U).CellSize()[X], MeshOf(&U).CellSize()[Y], MeshOf(&U).CellSize()[Z], pbcX, pbcY, pbcZ), c3)
	} else {
		shear := cuda.Buffer(M.NComp(), M.Buffer().Size())
		cuda.Zero(shear)
		ShearStress(shear, eShear, C44)
		cuda.Crop(dst, shear, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(shear)
	}
}
