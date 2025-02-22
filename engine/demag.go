package engine

// Calculation of magnetostatic field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// Demag variables
var (
	Msat        = NewScalarParam("Msat", "A/m", "Saturation magnetization")
	M_full      = NewVectorField("m_full", "A/m", "Unnormalized magnetization", SetMFull)
	B_demag     = NewVectorField("B_demag", "T", "Magnetostatic field", SetDemagField)
	Edens_demag = NewScalarField("Edens_demag", "J/m3", "Magnetostatic energy density", AddEdens_demag)
	E_demag     = NewScalarValue("E_demag", "J", "Magnetostatic energy", GetDemagEnergy)

	EnableDemag  = true // enable/disable global demag field
	NoDemagSpins = NewScalarParam("NoDemagSpins", "", "Disable magnetostatic interaction per region (default=0, set to 1 to disable). "+
		"E.g.: NoDemagSpins.SetRegion(5, 1) disables the magnetostatic interaction in region 5.")
	conv_         *cuda.DemagConvolution // does the heavy lifting
	DemagAccuracy = 6.0                  // Demag accuracy (divide cubes in at most N^3 points)
	regionConv_   = make(map[data.Mesh]*cuda.DemagConvolution)
)

var AddEdens_demag = makeEdensAdder(&B_demag, -0.5)

func init() {

	DeclVar("EnableDemag", &EnableDemag, "Enables/disables demag (default=true)")
	DeclVar("DemagAccuracy", &DemagAccuracy, "Controls accuracy of demag kernel")
	registerEnergy(GetDemagEnergy, AddEdens_demag)
}

// Sets dst to the current demag field
func SetDemagField(dst *data.Slice) {
	if EnableDemag {
		msat := Msat.MSlice()
		defer msat.Recycle()
		if NoDemagSpins.isZero() {
			// Normal demag, everywhere
			demagConv().Exec(dst, M.Buffer(), Geometry.Gpu(), msat)
		} else {
			setMaskedDemagField(dst, msat)
		}
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

func SetDemagFieldRegion(dst, m *data.Slice, useFullSample bool, pbcX, pbcY, pbcZ int) {
	if !useFullSample {
		if EnableDemag {
			msat := Msat.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer msat.Recycle()
			if NoDemagSpins.isZero() {
				// Normal demag, everywhere
				geom := cuda.Buffer(1, dst.RegionSize())
				GeomBig, _ := Geometry.Slice()
				cuda.Crop(geom, GeomBig, dst.StartX, dst.StartY, dst.StartZ)
				defer cuda.Recycle(geom)
				cuda.Recycle(GeomBig)
				demagConvRegion(data.NewMesh(dst.RegionSize()[X], dst.RegionSize()[Y], dst.RegionSize()[Z], MeshOf(&M).CellSize()[X], MeshOf(&M).CellSize()[Y], MeshOf(&M).CellSize()[Z], pbcX, pbcY, pbcZ)).Exec(dst, m, geom, msat)
			} else {
				setMaskedDemagFieldRegion(dst, m, msat, pbcX, pbcY, pbcZ)
			}
		} else {
			cuda.Zero(dst) // will ADD other terms to it
		}
	} else {
		B := cuda.Buffer(M.NComp(), M.Buffer().Size())
		SetDemagField(B)
		cuda.Crop(dst, B, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(B)
	}

}

// Sets dst to the demag field, but cells where NoDemagSpins != 0 do not generate nor recieve field.
func setMaskedDemagField(dst *data.Slice, msat cuda.MSlice) {
	// No-demag spins: mask-out Geometry with zeros where NoDemagSpins is set,
	// so these spins do not generate a field

	buf := cuda.Buffer(SCALAR, Geometry.Gpu().Size()) // masked-out Geometry
	defer cuda.Recycle(buf)

	// obtain a copy of the Geometry mask, which we can overwrite
	geom, r := Geometry.Slice()
	if r {
		defer cuda.Recycle(geom)
	}
	data.Copy(buf, geom)

	// mask-out
	cuda.ZeroMask(buf, NoDemagSpins.gpuLUT1(), regions.Gpu())

	// convolution with masked-out cells.
	demagConv().Exec(dst, M.Buffer(), buf, msat)

	// After convolution, mask-out the field in the NoDemagSpins cells
	// so they don't feel the field generated by others.
	cuda.ZeroMask(dst, NoDemagSpins.gpuLUT1(), regions.Gpu())
}

func setMaskedDemagFieldRegion(dst, m *data.Slice, msat cuda.MSlice, pbcX, pbcY, pbcZ int) {
	// No-demag spins: mask-out Geometry with zeros where NoDemagSpins is set,
	// so these spins do not generate a field

	buf := cuda.Buffer(SCALAR, dst.RegionSize()) // masked-out Geometry
	defer cuda.Recycle(buf)

	// obtain a copy of the Geometry mask, which we can overwrite
	geom := cuda.Buffer(1, dst.RegionSize())
	GeomBig, r := Geometry.Slice()
	cuda.Crop(geom, GeomBig, dst.StartX, dst.StartY, dst.StartZ)
	defer cuda.Recycle(geom)
	cuda.Recycle(GeomBig)
	if r {
		defer cuda.Recycle(geom)
	}
	data.Copy(buf, geom)

	// mask-out
	cuda.ZeroMask(buf, NoDemagSpins.gpuLUT1(), regions.Gpu())

	// convolution with masked-out cells.
	demagConvRegion(data.NewMesh(dst.RegionSize()[X], dst.RegionSize()[Y], dst.RegionSize()[Z], MeshOf(&M).CellSize()[X], MeshOf(&M).CellSize()[Y], MeshOf(&M).CellSize()[Z], pbcX, pbcY, pbcZ)).Exec(dst, m, buf, msat)

	// After convolution, mask-out the field in the NoDemagSpins cells
	// so they don't feel the field generated by others.
	cuda.ZeroMask(dst, NoDemagSpins.gpuLUT1(), regions.Gpu())
}

// Sets dst to the full (unnormalized) magnetization in A/m
func SetMFull(dst *data.Slice) {
	// scale m by Msat...
	msat := Msat.MSlice()
	defer msat.Recycle()
	for c := 0; c < 3; c++ {
		cuda.MulMSlice(dst.Comp(c), cuda.ToMSlice(M.Buffer().Comp(c)), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := Geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

// returns demag convolution, making sure it's initialized
func demagConv() *cuda.DemagConvolution {
	if conv_ == nil || conv_.Size() != Mesh().Size() {
		SetBusy(true)
		defer SetBusy(false)
		kernel := mag.DemagKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, *Flag_cachedir)
		conv_ = cuda.NewDemag(Mesh().Size(), Mesh().PBC(), kernel, *Flag_selftest)
	}
	return conv_
}

func demagConvRegion(mesh *data.Mesh) *cuda.DemagConvolution {
	var ok bool
	conv_, ok = regionConv_[*mesh]
	if !ok {
		SetBusy(true)
		defer SetBusy(false)
		kernel := mag.DemagKernel(mesh.Size(), mesh.PBC(), mesh.CellSize(), DemagAccuracy, *Flag_cachedir)
		conv_ = cuda.NewDemag(mesh.Size(), mesh.PBC(), kernel, *Flag_selftest)
		regionConv_[*mesh] = conv_
	}
	return conv_
}

// Returns the current demag energy in Joules.
func GetDemagEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_demag)
}
