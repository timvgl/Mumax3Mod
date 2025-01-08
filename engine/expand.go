package engine

// expanded quantity refers to a cut-out piece of a large quantity

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var ignoreExpandName bool = false

func init() {
	DeclFunc("Expand", Expand, "Expands a quantity to cell ranges [x1,x2[, [y1,y2[, [z1,z2[")
	DeclFunc("ExpandX", ExpandX, "Expands a quantity to cell ranges [x1,x2[")
	DeclFunc("ExpandY", ExpandY, "Expands a quantity to cell ranges [y1,y2[")
	DeclFunc("ExpandZ", ExpandZ, "Expands a quantity to cell ranges [z1,z2[")
	//DeclFunc("ExpandLayer", ExpandLayer, "Expands a quantity to a single layer")
	//DeclFunc("ExpandRegion", ExpandRegion, "Expands a quantity to a region")
	DeclVar("ignoreExpandName", &ignoreExpandName, "")
}

type expanded struct {
	parent                 Quantity
	name                   string
	x1, x2, y1, y2, z1, z2 int
	values                 []float64
}

// Expand quantity to a box enclosing the given region.
// Used to output a region of interest, even if the region is non-rectangular.

func ExpandX(parent Quantity, x1, x2 int, args ...float64) *expanded {
	n := MeshOf(parent).Size()
	return Expand(parent, x1, x2, 0, n[Y], 0, n[Z], args...)
}

func ExpandY(parent Quantity, y1, y2 int, args ...float64) *expanded {
	n := MeshOf(parent).Size()
	return Expand(parent, 0, n[X], y1, y2, 0, n[Z], args...)
}

func ExpandZ(parent Quantity, z1, z2 int, args ...float64) *expanded {
	n := MeshOf(parent).Size()
	return Expand(parent, 0, n[X], 0, n[Y], z1, z2, args...)
}

func createSliceFloat64(n int, l float64) []float64 {
	slice := make([]float64, n)
	for i := range slice {
		slice[i] = l
	}
	return slice
}

func Expand(parent Quantity, x1, x2, y1, y2, z1, z2 int, args ...float64) *expanded {
	n := MeshOf(parent).Size()
	util.Argument(x1 < x2 && y1 < y2 && z1 < z2)
	util.Argument(x1 >= 0 && y1 >= 0 && z1 >= 0)
	util.Argument(x2 >= n[X] && y2 >= n[Y] && z2 >= n[Z])
	name := NameOf(parent)
	if ignoreExpandName == false {
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

	if len(args) != parent.NComp() && len(args) != 0 && len(args) != 1 {
		panic("Either 0, 1 or as many as components available for the quantity allowed for Expand.")
	}
	if len(args) == parent.NComp() {
		return &expanded{parent, name, x1, x2, y1, y2, z1, z2, args}
	} else if len(args) == 1 {
		return &expanded{parent, name, x1, x2, y1, y2, z1, z2, createSliceFloat64(parent.NComp(), args[0])}
	} else {
		return &expanded{parent, name, x1, x2, y1, y2, z1, z2, createSliceFloat64(parent.NComp(), 0)}
	}
}

func (q *expanded) NComp() int             { return q.parent.NComp() }
func (q *expanded) Name() string           { return q.name }
func (q *expanded) Unit() string           { return UnitOf(q.parent) }
func (q *expanded) EvalTo(dst *data.Slice) { EvalTo(q, dst) }

func (q *expanded) Mesh() *data.Mesh {
	c := MeshOf(q.parent).CellSize()
	return data.NewMesh(q.x2-q.x1, q.y2-q.y1, q.z2-q.z1, c[X], c[Y], c[Z])
}

func (q *expanded) average() []float64 { return qAverageUniverse(q) } // needed for table
func (q *expanded) Average() []float64 { return q.average() }         // handy for script

func (q *expanded) Slice() (*data.Slice, bool) {
	src := ValueOf(q.parent)
	defer cuda.Recycle(src)
	dst := cuda.Buffer(q.NComp(), q.Mesh().Size())
	srcNxNyNz := src.Size()
	cuda.Expand(dst, src, (q.x2-q.x1-srcNxNyNz[0])/2, (q.y2-q.y1-srcNxNyNz[1])/2, (q.z2-q.z1-srcNxNyNz[2])/2, q.values)
	return dst, true
}
