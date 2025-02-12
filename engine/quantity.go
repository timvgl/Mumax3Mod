package engine

import (
	"reflect"

	"github.com/mumax/3/cuda"

	//"fmt"
	"github.com/mumax/3/data"
)

var Quantities = make(map[string]Quantity)

func wrapInt(vs1, vs2, vs3 int) [3]int {
	return [3]int{vs1, vs2, vs3}
}

// Arbitrary physical quantity.
type Quantity interface {
	NComp() int
	EvalTo(dst *data.Slice)
}

func MeshSize() [3]int {
	return Mesh().Size()
}

func addQuantity(name string, value interface{}, doc string) {
	_ = doc
	if v, ok := value.(Quantity); ok {
		Quantities[name] = v
	}
}

func SizeOf(q Quantity) [3]int {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Mesh() *data.Mesh
	}); ok {
		return s.Mesh().Size()
	}
	// otherwise: default mesh
	return MeshSize()
}

func AverageOf(q Quantity) []float64 {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		average() []float64
	}); ok {
		return s.average()
	}
	// otherwise: default mesh
	buf := ValueOf(q)
	defer cuda.Recycle(buf)
	return sAverageMagnet(buf)
}

func NameOf(q Quantity) string {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Name() string
	}); ok {
		return s.Name()
	}
	return "unnamed." + reflect.TypeOf(q).String()
}

func UnitOf(q Quantity) string {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Unit() string
	}); ok {
		return s.Unit()
	}
	return "?"
}

func MeshOf(q Quantity) *data.Mesh {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Mesh() *data.Mesh
	}); ok {
		return s.Mesh()
	}
	return Mesh()
}

func ValueOf(q Quantity) *data.Slice {
	// TODO: check for Buffered() implementation
	if s, ok := q.(interface {
		FFTOutputSize() [3]int
	}); ok {
		//fmt.Println(s.FFTOutputSize())
		buf := cuda.BufferComplex(q.NComp(), s.FFTOutputSize())
		q.EvalTo(buf)
		return buf
	} else {
		buf := cuda.Buffer(q.NComp(), SizeOf(q))
		q.EvalTo(buf)
		return buf
	}
}

func AxisOf(q Quantity) ([3]int, [3]float64, [3]float64, []string) {
	if s, ok := q.(interface {
		Axis() ([3]int, [3]float64, [3]float64, []string)
	}); ok {
		//fmt.Println(s.FFTOutputSize())
		return s.Axis()
	} else {
		c := MeshOf(q).CellSize()
		s := MeshOf(q).Size()
		return s, [3]float64{0., 0., 0.}, [3]float64{c[X] * float64(s[X]), c[Y] * float64(s[Y]), c[Y] * float64(s[Y])}, []string{}
	}
}

func SymmetricXOf(q Quantity) bool {
	if s, ok := q.(interface {
		SymmetricX() bool
	}); ok {
		return s.SymmetricX()
	} else {
		return false
	}
}

func SymmetricYOf(q Quantity) bool {
	if s, ok := q.(interface {
		SymmetricY() bool
	}); ok {
		return s.SymmetricY()
	} else {
		return false
	}
}

// Temporary shim to fit Slice into EvalTo
func EvalTo(q interface {
	Slice() (*data.Slice, bool)
}, dst *data.Slice) {
	v, r := q.Slice()
	if r {
		defer cuda.Recycle(v)
	}
	data.Copy(dst, v)
}
