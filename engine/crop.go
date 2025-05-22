package engine

// Cropped quantity refers to a cut-out piece of a large quantity

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var ignoreCropName bool = false

func init() {
	DeclFunc("Crop", Crop, "Crops a quantity to cell ranges [x1,x2[, [y1,y2[, [z1,z2[")
	DeclFunc("CropX", CropX, "Crops a quantity to cell ranges [x1,x2[")
	DeclFunc("CropY", CropY, "Crops a quantity to cell ranges [y1,y2[")
	DeclFunc("CropZ", CropZ, "Crops a quantity to cell ranges [z1,z2[")
	DeclFunc("CropLayer", CropLayer, "Crops a quantity to a single layer")
	DeclFunc("CropRegion", CropRegion, "Crops a quantity to a region")
	DeclVar("ignoreCropName", &ignoreCropName, "")
	DeclFunc("CropXOperator", CropXOperator, "")
	DeclFunc("CropYOperator", CropYOperator, "")
	DeclFunc("CropZOperator", CropZOperator, "")
	DeclFunc("CropOperator", CropOperator, "")
	DeclFunc("CropLayerOperator", CropLayerOperator, "")
	DeclFunc("CropK", CropK, "Crops a quantity to k-space ranges [kx1,kx2[, [ky1,ky2[, [kz1,kz2[")
	DeclFunc("CropKx", CropKx, "Crops a quantity to k-space ranges [kx1,kx2[")
	DeclFunc("CropKy", CropKy, "Crops a quantity to k-space ranges [ky1,ky2[")
	DeclFunc("CropKz", CropKz, "Crops a quantity to k-space ranges [kz1,kz2[")
	DeclFunc("CropKxy", CropKxy, "Crops a quantity to k-space ranges [kx1,kx2[, [ky1,ky2[")
	DeclFunc("CropKOperator", CropKOperator, "")
	DeclFunc("CropKxOperator", CropKxOperator, "")
	DeclFunc("CropKyOperator", CropKyOperator, "")
	DeclFunc("CropKzOperator", CropKzOperator, "")
	DeclFunc("CropKxyOperator", CropKxyOperator, "")
}

type cropped struct {
	parent                 Quantity
	name                   string
	x1, x2, y1, y2, z1, z2 int
}

func CropK(parent Quantity, kx_start, kx_end, ky_start, ky_end, kz_start, kz_end float64, symmetricX ...bool) *cropped {
	mesh := MeshOf(parent)
	size := mesh.Size()
	c := mesh.CellSize()
	startIndexX := 0.0
	if len(symmetricX) == 0 {
		if NegativeKX {
			startIndexX = float64(size[X]) / 2
		}
	} else {
		if symmetricX[0] {
			startIndexX = float64(size[X]) / 2
		}
	}
	x1 := int(math.Ceil(startIndexX + kx_start/c[X]))
	var x2 int
	if kx_end == kx_start {
		x2 = x1 + 1
	} else {
		x2 = int(math.Floor(startIndexX + kx_end/c[X]))
	}
	midY := float64(size[Y]) / 2
	y1 := int(math.Ceil(midY + ky_start/c[Y]))
	var y2 int
	if ky_end == ky_start {
		y2 = y1 + 1
	} else {
		y2 = int(math.Floor(midY + ky_end/c[Y]))
	}
	midZ := float64(size[Z]) / 2
	z1 := int(math.Ceil(midZ + kz_start/c[Z]))
	var z2 int
	if kz_end == kz_start {
		z2 = z1 + 1
	} else {
		z2 = int(math.Floor(midZ + kz_end/c[Z]))
	}
	return Crop(parent,
		x1, x2,
		y1, y2,
		z1, z2)
}

func CropKx(parent Quantity, kx_start, kx_end float64, symmetricX ...bool) *cropped {
	mesh := MeshOf(parent)
	size := mesh.Size()
	c := mesh.CellSize()
	startIndexX := 0.0
	if len(symmetricX) == 0 {
		if NegativeKX {
			startIndexX = float64(size[X]) / 2
		}
	} else {
		if symmetricX[0] {
			startIndexX = float64(size[X]) / 2
		}
	}
	x1 := int(math.Ceil(startIndexX + kx_start/c[X]))
	var x2 int
	if kx_end == kx_start {
		x2 = x1 + 1
	} else {
		x2 = int(math.Floor(startIndexX + kx_end/c[X]))
	}
	return Crop(parent,
		x1, x2,
		0, size[Y],
		0, size[Z])
}

func CropKy(parent Quantity, ky_start, ky_end float64) *cropped {
	mesh := MeshOf(parent)
	size := mesh.Size()
	c := mesh.CellSize()
	midY := float64(size[Y]) / 2
	y1 := int(math.Ceil(midY + ky_start/c[Y]))
	var y2 int
	if ky_end == ky_start {
		y2 = y1 + 1
	} else {
		y2 = int(math.Floor(midY + ky_end/c[Y]))
	}
	return Crop(parent,
		0, size[X],
		y1, y2,
		0, size[Z])
}

func CropKz(parent Quantity, kz_start, kz_end float64) *cropped {
	mesh := MeshOf(parent)
	size := mesh.Size()
	c := mesh.CellSize()
	midZ := float64(size[Z]) / 2
	z1 := int(math.Ceil(midZ + kz_start/c[Z]))
	var z2 int
	if kz_end == kz_start {
		z2 = z1 + 1
	} else {
		z2 = int(math.Floor(midZ + kz_end/c[Z]))
	}
	return Crop(parent,
		0, size[X],
		0, size[Y],
		z1, z2)
}

func CropKxy(parent Quantity, kx_start, kx_end, ky_start, ky_end float64, symmetricX ...bool) *cropped {
	mesh := MeshOf(parent)
	size := mesh.Size()
	c := mesh.CellSize()
	startIndexX := 0.0
	if len(symmetricX) == 0 {
		if NegativeKX {
			startIndexX = float64(size[X]) / 2
		}
	} else {
		if symmetricX[0] {
			startIndexX = float64(size[X]) / 2
		}
	}
	x1 := int(math.Ceil(startIndexX + kx_start/c[X]))
	var x2 int
	if kx_end == kx_start {
		x2 = x1 + 1
	} else {
		x2 = int(math.Floor(startIndexX + kx_end/c[X]))
	}
	midY := float64(size[Y]) / 2
	y1 := int(math.Ceil(midY + ky_start/c[Y]))
	var y2 int
	if ky_end == ky_start {
		y2 = y1 + 1
	} else {
		y2 = int(math.Floor(midY + ky_end/c[Y]))
	}
	return Crop(parent,
		x1, x2,
		y1, y2,
		0, size[Z])
}

func CropKOperator(kx_start, kx_end, ky_start, ky_end, kz_start, kz_end float64) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity {
		return CropK(parent, kx_start, kx_end, ky_start, ky_end, kz_start, kz_end, NegativeKX)
	}
}

func CropKxOperator(kx_start, kx_end float64) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity {
		return CropKx(parent, kx_start, kx_end, NegativeKX)
	}
}

func CropKyOperator(ky_start, ky_end float64) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity {
		return CropKy(parent, ky_start, ky_end)
	}
}

func CropKzOperator(kz_start, kz_end float64) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity {
		return CropKz(parent, kz_start, kz_end)
	}
}

func CropKxyOperator(kx_start, kx_end, ky_start, ky_end float64) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity {
		return CropKxy(parent, kx_start, kx_end, ky_start, ky_end, NegativeKX)
	}
}

// Crop quantity to a box enclosing the given region.
// Used to output a region of interest, even if the region is non-rectangular.
func CropRegion(parent Quantity, region int) *cropped {
	n := MeshOf(parent).Size()
	// use -1 for unset values
	x1, y1, z1 := -1, -1, -1
	x2, y2, z2 := -1, -1, -1
	r := regions.HostArray()
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				if r[iz][iy][ix] == byte(region) {
					// initialize all indices if unset
					if x1 == -1 {
						x1, y1, z1 = ix, iy, iz
						x2, y2, z2 = ix, iy, iz
					}
					if ix < x1 {
						x1 = ix
					}
					if iy < y1 {
						y1 = iy
					}
					if iz < z1 {
						z1 = iz
					}
					if ix > x2 {
						x2 = ix
					}
					if iy > y2 {
						y2 = iy
					}
					if iz > z2 {
						z2 = iz
					}
				}
			}
		}
	}
	return Crop(parent, x1, x2+1, y1, y2+1, z1, z2+1)
}

func CropLayerOperator(layer int) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity { return CropLayer(parent, layer) }
}

func CropXOperator(x1, x2 int) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity { return CropX(parent, x1, x2) }
}

func CropYOperator(y1, y2 int) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity { return CropY(parent, y1, y2) }
}

func CropZOperator(z1, z2 int) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity { return CropZ(parent, z1, z2) }
}

func CropOperator(x1, x2, y1, y2, z1, z2 int) func(parent Quantity) Quantity {
	return func(parent Quantity) Quantity { return Crop(parent, x1, x2, y1, y2, z1, z2) }
}

func CropLayer(parent Quantity, layer int) *cropped {
	n := MeshOf(parent).Size()
	return Crop(parent, 0, n[X], 0, n[Y], layer, layer+1)
}

func CropX(parent Quantity, x1, x2 int) *cropped {
	n := MeshOf(parent).Size()
	return Crop(parent, x1, x2, 0, n[Y], 0, n[Z])
}

func CropY(parent Quantity, y1, y2 int) *cropped {
	n := MeshOf(parent).Size()
	return Crop(parent, 0, n[X], y1, y2, 0, n[Z])
}

func CropZ(parent Quantity, z1, z2 int) *cropped {
	n := MeshOf(parent).Size()
	return Crop(parent, 0, n[X], 0, n[Y], z1, z2)
}

func Crop(parent Quantity, x1, x2, y1, y2, z1, z2 int) *cropped {
	n := MeshOf(parent).Size()
	util.AssertMsg(x1 < x2, "lower end of x >= upper end of x")
	util.AssertMsg(y1 < y2, "lower end of y >= upper end of y")
	util.AssertMsg(z1 < z2, "lower end of z >= upper end of z")
	util.AssertMsg(x1 >= 0, "lower end of x out of bounds")
	util.AssertMsg(y1 >= 0, "lower end of y out of bounds")
	util.AssertMsg(z1 >= 0, "lower end of z out of bounds")
	util.AssertMsg(x2 <= n[X], "upper end of x out of bounds")
	util.AssertMsg(y2 <= n[Y], "upper end of y out of bounds")
	util.AssertMsg(z2 <= n[Z], "upper end of z out of bounds")
	name := NameOf(parent)
	if !ignoreCropName {
		name += "_"
		if x1 != 0 || x2 != n[X] {
			name += "xrange" + rangeStr(x1, x2)
		}
		if y1 != 0 || y2 != n[Y] {
			name += "yrange" + rangeStr(y1, y2)
		}
		if z1 != 0 || z2 != n[Z] {
			name += "zrange" + rangeStr(z1, z2)
		}
	}

	return &cropped{parent, name, x1, x2, y1, y2, z1, z2}
}

func rangeStr(a, b int) string {
	if a+1 == b {
		return fmt.Sprint(a)
	} else {
		return fmt.Sprint(a, "-", b)
	}
	// (trailing underscore to separate from subsequent autosave number)
}

func (q *cropped) NComp() int             { return q.parent.NComp() }
func (q *cropped) Name() string           { return q.name }
func (q *cropped) Unit() string           { return UnitOf(q.parent) }
func (q *cropped) EvalTo(dst *data.Slice) { EvalTo(q, dst) }

func (q *cropped) Mesh() *data.Mesh {
	c := MeshOf(q.parent).CellSize()
	return data.NewMesh(q.x2-q.x1, q.y2-q.y1, q.z2-q.z1, c[X], c[Y], c[Z])
}

func (q *cropped) average() []float64 { return qAverageUniverse(q) } // needed for table
func (q *cropped) Average() []float64 { return q.average() }         // handy for script

func (q *cropped) Slice() (*data.Slice, bool) {
	src := ValueOf(q.parent)
	defer cuda.Recycle(src)
	var dst *data.Slice
	if IsFFT3D(q.parent) {
		size := q.Mesh().Size()
		size[0] *= 2
		dst = cuda.Buffer(q.NComp(), size)
		cuda.Crop(dst, src, 2*q.x1, q.y1, q.z1)
	} else {
		dst = cuda.Buffer(q.NComp(), q.Mesh().Size())
		cuda.Crop(dst, src, q.x1, q.y1, q.z1)
	}
	return dst, true
}

func (q *cropped) AxisFFT() (newSize [3]int, newStartK, newEndK [3]float64, newTransformedAxis []string) {
	parent := q.parent
	if s, ok := parent.(interface {
		AxisFFT() ([3]int, [3]float64, [3]float64, []string)
	}); ok {
		parentSize, parentStartK, parentEndK, parentTransformedAxis := s.AxisFFT()
		newSize = q.Mesh().Size()
		newStartK[X] = parentStartK[X] + float64(q.x1)*MeshOf(parent).CellSize()[X]
		newStartK[Y] = parentStartK[Y] + float64(q.y1)*MeshOf(parent).CellSize()[Y]
		newStartK[Z] = parentStartK[Z] + float64(q.z1)*MeshOf(parent).CellSize()[Z]
		newEndK[X] = parentEndK[X] - float64(parentSize[X]-q.x2)*MeshOf(parent).CellSize()[X]
		newEndK[Y] = parentEndK[Y] - float64(parentSize[Y]-q.y2)*MeshOf(parent).CellSize()[Y]
		newEndK[Z] = parentEndK[Z] - float64(parentSize[Z]-q.z2)*MeshOf(parent).CellSize()[Z]
		newTransformedAxis = parentTransformedAxis
		return newSize, newStartK, newEndK, newTransformedAxis
	} else if s, ok := parent.(interface {
		Axis() ([3]int, [3]float64, [3]float64, []string)
	}); ok {
		if l, ok2 := s.(*fftOperation3D); ok2 {
			parentSize, parentStartK, parentEndK, parentTransformedAxis := l.Axis()
			newSize = q.Mesh().Size()
			newStartK[X] = parentStartK[X] + float64(q.x1)*MeshOf(parent).CellSize()[X]
			newStartK[Y] = parentStartK[Y] + float64(q.y1)*MeshOf(parent).CellSize()[Y]
			newStartK[Z] = parentStartK[Z] + float64(q.z1)*MeshOf(parent).CellSize()[Z]
			newEndK[X] = parentEndK[X] - float64(parentSize[X]-q.x2)*MeshOf(parent).CellSize()[X]
			newEndK[Y] = parentEndK[Y] - float64(parentSize[Y]-q.y2)*MeshOf(parent).CellSize()[Y]
			newEndK[Z] = parentEndK[Z] - float64(parentSize[Z]-q.z2)*MeshOf(parent).CellSize()[Z]
			newTransformedAxis = parentTransformedAxis
			return newSize, newStartK, newEndK, newTransformedAxis
		} else {
			panic("Axis functions not found for " + NameOf(parent))
		}
	} else {
		panic("Axis functions not found for " + NameOf(parent))
	}
}
