package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("FFT3D", FFT3D, "performs FFT in x, y and z")
	//DeclFunc("FFT2D", FFT2D, "performs FFT in x, y and z")
}

type fftOperation3D struct {
	fieldOp
	name string
}

type fftOperation2D struct {
	fieldOp
	name string
	axis [2]string
}

func FFT3D(q Quantity) *fftOperation3D {
	if !cuda.FFT3DR2CPlanInitialized {
		s := Mesh().Size()
		//fmt.Println(fmt.Sprintf("Initializing with %d, %d and %d", s[X], s[Y], s[Z]))
		cuda.Initialize3DR2CFFT(s[X], s[Y], s[Z])
	}
	return &fftOperation3D{fieldOp{q, q, q.NComp()}, "k_x_y_z_" + NameOf(q)}

}

func (d *fftOperation3D) EvalTo(dst *data.Slice) {
	data := ValueOf(d.a)
	defer cuda.Recycle(data)
	cuda.Zero(dst)
	for i := range d.nComp {
		cuda.Perform3DR2CFFT(data.Comp(i), dst.Comp(i))
	}
}

func (d *fftOperation3D) Mesh() *data.Mesh {
	s := d.FFTOutputSize()
	c := Mesh().CellSize()
	return data.NewMesh(s[X]/2, s[Y], s[Z], 1/(2*c[X]*float64(s[X])), 1/(2*c[Y]*float64(s[Y])), 1/(2*c[Z]*float64(s[Z])))
}

func (d *fftOperation3D) Name() string {
	return d.name
}

func (d *fftOperation3D) Unit() string { return "a.u." }

func (d *fftOperation3D) FFTOutputSize() [3]int {
	var NxOP, NyOP, NzOP = cuda.FFT3DR2CPlan.OutputSizeFloats()
	return [3]int{NxOP, NyOP, NzOP}
}

func (d *fftOperation3D) Axis() ([3]int, [3]float64, [3]float64, []string) {
	c := Mesh().CellSize()
	s := d.FFTOutputSize()
	s[0] /= 2
	return s, [3]float64{0., -1 / (2 * c[Y]), -1 / (2 * c[Z])}, [3]float64{1 / (2 * c[Y]), 1 / (2 * c[Y]), 1 / (2 * c[Z])}, []string{"x", "y", "z"}
}

/*
func FFT2D(q Quantity, axis1, axis2 string) *fftOperation2D {
	if !cuda.FFT3DR2CPlanInitialized {
		s := Mesh().Size()
		//fmt.Println(fmt.Sprintf("Initializing with %d, %d and %d", s[X], s[Y], s[Z]))
		cuda.InitializeR2CFFT(s[X], s[Y], s[Z])
	}
	return &fftOperation2D{fieldOp{q, q, q.NComp()}, "k_" + axis1 + "_" + axis2 + "_" + NameOf(q), [2]string{axis1, axis2}}

}

func (d *fftOperation2D) EvalTo(dst *data.Slice) {
	data := ValueOf(d.a)
	defer cuda.Recycle(data)
	cuda.Zero(dst)
	for i := range d.nComp {
		cuda.PerformR2CFFT(data.Comp(i), dst.Comp(i))
	}
}

func (d *fftOperation2D) Mesh() *data.Mesh {
	s := d.FFTOutputSize()
	c := Mesh().CellSize()
	return data.NewMesh(s[X]/2, s[Y], s[Z], 2*math.Pi/c[X], 2*math.Pi/c[Y], 2*math.Pi/c[Z])
}

func (d *fftOperation2D) Name() string {
	return d.name
}

func (d *fftOperation2D) Unit() string { return "a.u." }

func (d *fftOperation2D) FFTOutputSize() [3]int {
	var NxOP, NyOP, NzOP = cuda.FFT2DR2CPlan.OutputSizeFloats()
	return [3]int{NxOP, NyOP, NzOP}
}
*/
