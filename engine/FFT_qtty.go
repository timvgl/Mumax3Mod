package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"math"
)

func init() {
	DeclFunc("FFT3D", FFT3D, "performs FFT in x, y and z")
}

type fftOperation struct {
	fieldOp
	name string
}

func FFT3D(q Quantity) *fftOperation {
	if !cuda.FFT3DR2CPlanInitialized {
		s := Mesh().Size()
		//fmt.Println(fmt.Sprintf("Initializing with %d, %d and %d", s[X], s[Y], s[Z]))
		cuda.InitializeR2CFFT(s[X], s[Y], s[Z])
	}
	return &fftOperation{fieldOp{q, q, q.NComp()}, "k_x_y_z_" + NameOf(q)}

}

func (d *fftOperation) EvalTo(dst *data.Slice) {
	data := ValueOf(d.a)
	defer cuda.Recycle(data)
	cuda.Zero(dst)
	for i := range(d.nComp) {
		cuda.PerformR2CFFT(data.Comp(i), dst.Comp(i))
	}
}

func (d *fftOperation) Mesh() *data.Mesh{
	s := d.FFTOutputSize()
	c := Mesh().CellSize()
	return data.NewMesh(s[X] / 2, s[Y], s[Z], 2*math.Pi / c[X], 2*math.Pi / c[Y], 2*math.Pi / c[Z])
}

func (d *fftOperation) Name() string {
	return d.name
}

func (d *fftOperation) Unit() string { return "a.u." }

func (d *fftOperation) FFTOutputSize() [3]int {
	var NxOP, NyOP, NzOP = cuda.FFT3DR2CPlan.OutputSizeFloats()
	return [3]int{NxOP, NyOP, NzOP}
}