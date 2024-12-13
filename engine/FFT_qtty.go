package engine

import (
	"fmt"
	"slices"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DeclVarFFTDyn      = []Quantity{}
	DeclVarFFTDynAlias = []string{}
	FFT3DData          = make(map[Quantity]*data.Slice)
	FFT3DR2CPlans      = make(map[Quantity]interface{})
	FFTEvaluated       = make(map[Quantity]bool)
	FFTEvaluatedReal   = make(map[Quantity]bool)
	FFTEvaluatedImag   = make(map[Quantity]bool)
)

func init() {
	DeclFunc("FFT3D", FFT3D, "performs FFT in x, y and z")
	//DeclFunc("FFT2D", FFT2D, "performs FFT in x, y and z")
}

type fftOperation3DReal struct {
	fieldOp
	name string
	q    Quantity
	op   fftOperation3D
}

type fftOperation3DImag struct {
	fieldOp
	name string
	q    Quantity
	op   fftOperation3D
}

type fftOperation3D struct {
	fieldOp
	name string
	q    Quantity
}

type fftOperation2D struct {
	fieldOp
	name string
	axis [2]string
}

func FFT3D(q Quantity) *fftOperation3D {

	s := MeshOf(q).Size()
	//fmt.Println(fmt.Sprintf("Initializing with %d, %d and %d", s[X], s[Y], s[Z]))
	FFT3DR2CPlans[q] = cuda.Initialize3DR2CFFT(s[X], s[Y], s[Z])
	fftOP3D := &fftOperation3D{fieldOp{q, q, q.NComp()}, "k_x_y_z_" + NameOf(q), q}
	FFTEvaluated[q] = false
	FFTEvaluatedReal[q] = false
	FFTEvaluatedImag[q] = false
	if !slices.Contains(DeclVarFFTDyn, q) {
		if q.NComp() == 3 {
			if !slices.Contains(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_real") {
				NewVectorFieldFFT("FFT_"+NameOf(q)+"_real", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Real().EvalTo, fftOP3D.Real().Mesh(), fftOP3D.Real().Axis, fftOP3D.Real().SymmetricX(), fftOP3D.Real().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Real())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_real")
			}
			if !slices.Contains(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_imag") {
				NewVectorFieldFFT("FFT_"+NameOf(q)+"_imag", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Imag().EvalTo, fftOP3D.Imag().Mesh(), fftOP3D.Imag().Axis, fftOP3D.Imag().SymmetricX(), fftOP3D.Imag().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Imag())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_imag")
			}
			FFT3DData[q] = cuda.Buffer(3, fftOP3D.FFTOutputSize())
			cuda.Zero(FFT3DData[q])
			//fmt.Println(FFT3DData[q].HostCopy().Tensors())
		} else if q.NComp() == 1 {
			if !slices.Contains(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_real") {
				NewScalarFieldFFT("FFT_"+NameOf(q)+"_real", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Real().EvalTo, fftOP3D.Real().Mesh(), fftOP3D.Real().Axis, fftOP3D.Real().SymmetricX(), fftOP3D.Real().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Real())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_real")
			}
			if !slices.Contains(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_imag") {
				NewScalarFieldFFT("FFT_"+NameOf(q)+"_imag", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Imag().EvalTo, fftOP3D.Imag().Mesh(), fftOP3D.Imag().Axis, fftOP3D.Imag().SymmetricX(), fftOP3D.Imag().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Imag())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, "FFT_"+NameOf(q)+"_imag")
			}
			FFT3DData[q] = cuda.Buffer(1, fftOP3D.FFTOutputSize())
			cuda.Zero(FFT3DData[q])
		}
	}
	return fftOP3D

}

// ///not working!!!!
func FFT3DAs(q Quantity, name string) *fftOperation3D {

	s := MeshOf(q).Size()
	fmt.Println(fmt.Sprintf("Initializing with %d, %d and %d", s[X], s[Y], s[Z]))
	FFT3DR2CPlans[q] = cuda.Initialize3DR2CFFT(s[X], s[Y], s[Z])
	fftOP3D := &fftOperation3D{fieldOp{q, q, q.NComp()}, "k_x_y_z_" + NameOf(q), q}
	FFTEvaluated[q] = false
	FFTEvaluatedReal[q] = false
	FFTEvaluatedImag[q] = false
	if !slices.Contains(DeclVarFFTDyn, q) {
		if q.NComp() == 3 {
			if !slices.Contains(DeclVarFFTDynAlias, name+"_real") {
				NewVectorFieldFFT(name+"_real", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Real().EvalTo, fftOP3D.Real().Mesh(), fftOP3D.Real().Axis, fftOP3D.Real().SymmetricX(), fftOP3D.Real().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Real())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, name+"_real")
			}
			if !slices.Contains(DeclVarFFTDynAlias, name+"_imag") {
				NewVectorFieldFFT(name+"_imag", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Imag().EvalTo, fftOP3D.Imag().Mesh(), fftOP3D.Imag().Axis, fftOP3D.Imag().SymmetricX(), fftOP3D.Imag().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Imag())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, name+"_imag")
			}
			FFT3DData[q] = cuda.Buffer(3, fftOP3D.FFTOutputSize())
			cuda.Zero(FFT3DData[q])
			//fmt.Println(FFT3DData[q].HostCopy().Tensors())
		} else if q.NComp() == 1 {
			if !slices.Contains(DeclVarFFTDynAlias, name+"_real") {
				NewScalarFieldFFT(name+"_real", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Real().EvalTo, fftOP3D.Real().Mesh(), fftOP3D.Real().Axis, fftOP3D.Real().SymmetricX(), fftOP3D.Real().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Real())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, name+"_real")
			}
			if !slices.Contains(DeclVarFFTDynAlias, name+"_imag") {
				NewScalarFieldFFT(name+"_imag", "a.u.", "get FFT of last save run - to costly otherwise", fftOP3D.Imag().EvalTo, fftOP3D.Imag().Mesh(), fftOP3D.Imag().Axis, fftOP3D.Imag().SymmetricX(), fftOP3D.Imag().SymmetricY())
				DeclVarFFTDyn = append(DeclVarFFTDyn, fftOP3D.Imag())
				DeclVarFFTDynAlias = append(DeclVarFFTDynAlias, name+"_imag")
			}
			FFT3DData[q] = cuda.Buffer(1, fftOP3D.FFTOutputSize())
			cuda.Zero(FFT3DData[q])
		}
	}
	return fftOP3D

}

func (d *fftOperation3D) EvalTo(dst *data.Slice) {
	FFTEvaluated[d.q] = true
	d.evalIntern()
	data.Copy(dst, FFT3DData[d.q])

}

func (d *fftOperation3D) evalIntern() {
	input := ValueOf(d.a)
	defer cuda.Recycle(input)
	buf := cuda.Buffer(d.nComp, d.FFTOutputSize())
	cuda.Zero(buf)
	defer cuda.Recycle(buf)
	for i := range d.nComp {
		cuda.Perform3DR2CFFT(input.Comp(i), buf.Comp(i), FFT3DR2CPlans[d.q])
	}
	cuda.ReorderCufftData(FFT3DData[d.q], buf)
}

func (d *fftOperation3D) Mesh() *data.Mesh {
	s := d.FFTOutputSize()
	c := Mesh().CellSize()
	return data.NewMesh(s[X]/2, s[Y], s[Z], 1/(c[X]*float64(s[X])), 1/(c[Y]*float64(s[Y])), 1/(c[Z]*float64(s[Z])))
}

func (d *fftOperation3D) Name() string {
	return d.name
}

func (d *fftOperation3D) Unit() string { return "a.u." }

func (d *fftOperation3D) FFTOutputSize() [3]int {
	var NxOP, NyOP, NzOP = cuda.OutputSizeFloatsFFT3D(FFT3DR2CPlans[d.q])
	return [3]int{NxOP, NyOP, NzOP}
}

func (d *fftOperation3D) Axis() ([3]int, [3]float64, [3]float64, []string) {
	c := Mesh().CellSize()
	s := d.FFTOutputSize()
	s[0] /= 2
	return s, [3]float64{0., -1 / (2 * c[Y]), -1 / (2 * c[Z])}, [3]float64{1 / (2 * c[Y]), 1 / (2 * c[Y]), 1 / (2 * c[Z])}, []string{"x", "y", "z"}
}

func (d *fftOperation3D) Real() *fftOperation3DReal {
	return &fftOperation3DReal{fieldOp{d.q, d.q, d.q.NComp()}, "k_x_y_z_" + NameOf(d.q) + "_real", d.q, *d}
}

func (d *fftOperation3D) Imag() *fftOperation3DImag {
	return &fftOperation3DImag{fieldOp{d.q, d.q, d.q.NComp()}, "k_x_y_z_" + NameOf(d.q) + "_imag", d.q, *d}
}

func (d *fftOperation3D) SymmetricX() bool {
	return false
}

func (d *fftOperation3D) SymmetricY() bool {
	return true
}

func (d *fftOperation3DReal) EvalTo(dst *data.Slice) {
	if !FFTEvaluated[d.q] && !FFTEvaluatedImag[d.q] {
		d.op.evalIntern()
		FFTEvaluatedReal[d.q] = true
	}
	cuda.Real(dst, FFT3DData[d.q])
}

func (d *fftOperation3DReal) Mesh() *data.Mesh {
	s := d.fftOutputSize()
	c := Mesh().CellSize()
	return data.NewMesh(s[X], s[Y], s[Z], 1/(c[X]*float64(s[X])), 1/(c[Y]*float64(s[Y])), 1/(c[Z]*float64(s[Z])))
}

func (d *fftOperation3DReal) Name() string {
	return d.name
}

func (d *fftOperation3DReal) Unit() string { return "a.u." }

func (d *fftOperation3DReal) fftOutputSize() [3]int {
	var NxOP, NyOP, NzOP = cuda.OutputSizeFloatsFFT3D(FFT3DR2CPlans[d.q])
	return [3]int{int(NxOP / 2), NyOP, NzOP}
}

func (d *fftOperation3DReal) Axis() ([3]int, [3]float64, [3]float64, []string) {
	c := Mesh().CellSize()
	s := d.fftOutputSize()
	return s, [3]float64{0., -1 / (2 * c[Y]), -1 / (2 * c[Z])}, [3]float64{1 / (2 * c[Y]), 1 / (2 * c[Y]), 1 / (2 * c[Z])}, []string{"x", "y", "z"}
}

func (d *fftOperation3DReal) SymmetricX() bool {
	return false
}

func (d *fftOperation3DReal) SymmetricY() bool {
	return true
}

func (d *fftOperation3DImag) EvalTo(dst *data.Slice) {
	if !FFTEvaluated[d.q] && !FFTEvaluatedReal[d.q] {
		d.op.evalIntern()
		FFTEvaluatedImag[d.q] = true
	}
	cuda.Imag(dst, FFT3DData[d.q])
}

func (d *fftOperation3DImag) Mesh() *data.Mesh {
	s := d.fftOutputSize()
	c := Mesh().CellSize()
	return data.NewMesh(s[X], s[Y], s[Z], 1/(c[X]*float64(s[X])), 1/(c[Y]*float64(s[Y])), 1/(c[Z]*float64(s[Z])))
}

func (d *fftOperation3DImag) Name() string {
	return d.name
}

func (d *fftOperation3DImag) Unit() string { return "a.u." }

func (d *fftOperation3DImag) fftOutputSize() [3]int {
	var NxOP, NyOP, NzOP = cuda.OutputSizeFloatsFFT3D(FFT3DR2CPlans[d.q])
	return [3]int{int(NxOP / 2), NyOP, NzOP}
}

func (d *fftOperation3DImag) Axis() ([3]int, [3]float64, [3]float64, []string) {
	c := Mesh().CellSize()
	s := d.fftOutputSize()
	return s, [3]float64{0., -1 / (2 * c[Y]), -1 / (2 * c[Z])}, [3]float64{1 / (2 * c[Y]), 1 / (2 * c[Y]), 1 / (2 * c[Z])}, []string{"x", "y", "z"}
}

func (d *fftOperation3DImag) SymmetricX() bool {
	return false
}

func (d *fftOperation3DImag) SymmetricY() bool {
	return true
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
