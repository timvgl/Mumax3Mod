package engine

import (
	"fmt"
	"math"
	"reflect"
	"sync"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
)

var mapSetExcitation sync.Map

func SetExcitation(name string, s ExcitationSlice) {
	tmp, ok := mapSetExcitation.Load(name)
	if ok {
		slces := tmp.(ExcitationSlices)
		var slcInterfaces = make([]SetSlice, 0)
		for _, slc := range slces.slc {
			slcInterfaces = append(slcInterfaces, slc)
		}
		overlapIndices := GetOverlapIndex(s, slcInterfaces)
		FreeMemoryIndices(slcInterfaces, overlapIndices)
		slces.slc = RemoveIndices(slces.slc, overlapIndices)
		slces.slc = append(slces.slc, s)
		mapSetExcitation.Store(name, slces)
	} else {
		slc := ExcitationSlices{name: name, slc: []ExcitationSlice{s}}
		mapSetExcitation.Store(name, slc)
	}
}

func EraseSetExcitation(name string) {
	tmp, ok := mapSetExcitation.Load(name)
	if ok {
		s := tmp.(ExcitationSlices)
		for _, slc := range s.slc {
			if slc.d.GPUAccess() {
				cuda.Recycle(slc.d)
			} else {
				slc.d.Free()
			}
		}
		mapSetExcitation.Delete(name)
	} else {
		panic(fmt.Sprintf("EraseSetExcitation: %s not found", name))
	}
}

type ExcitationSlices struct {
	name string
	slc  []ExcitationSlice
}

type ExcitationSlice struct {
	start         [3]int
	end           [3]int
	d             *data.Slice
	ncomp         int
	timedependent bool
	stringFct     StringFunction
}

func (e ExcitationSlice) StartAt() [3]int {
	return e.start
}

func (e ExcitationSlice) EndAt() [3]int {
	return e.end
}

func (e ExcitationSlice) Buffer() *data.Slice {
	return e.d
}

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type Excitation struct {
	name       string
	perRegion  RegionwiseVector // Region-based excitation
	extraTerms []mulmask3D      // add extra mask*multiplier terms
}

// space-dependent mask plus time dependent multiplier
type mulmask struct {
	mul  func() float64
	mask *data.Slice
}

type mulmask3D struct {
	mul  []func() float64
	mask *data.Slice
}

func NewExcitation(name, unit, desc string) *Excitation {
	e := new(Excitation)
	e.name = name
	e.perRegion.init(3, "_"+name+"_perRegion", unit, nil) // name starts with underscore: unexported
	DeclLValue(name, e, cat(desc, unit))
	return e
}

func (p *Excitation) MSlice() cuda.MSlice {
	buf, r := p.Slice()
	util.Assert(r)
	return cuda.ToMSlice(buf)
}

func (p *Excitation) RenderFunction(equation StringFunction) {
	util.AssertMsg(!equation.IsScalar(), "RenderFunction: Need vector function.")
	d, timeDep := GenerateSliceFromFunctionStringTimeDep(equation, p.Mesh())
	SetExcitation(p.name, ExcitationSlice{start: [3]int{0, 0, 0}, end: p.Mesh().Size(), d: d, timedependent: timeDep, stringFct: equation})
}

func (p *Excitation) RenderFunctionLimit(equation StringFunction, xStart, xEnd, yStart, yEnd, zStart, zEnd int) {
	n := MeshOf(p).Size()
	util.Argument(xStart < xEnd && yStart < yEnd && zStart < zEnd)
	util.Argument(xStart >= 0 && yStart >= 0 && zStart >= 0)
	util.Argument(xEnd <= n[X] && yEnd <= n[Y] && zEnd <= n[Z])
	util.AssertMsg(!equation.IsScalar(), "RenderFunction: Need vector function.")
	d, timeDep := GenerateSliceFromFunctionStringTimeDep(equation, p.Mesh())
	SetExcitation(p.name, ExcitationSlice{start: [3]int{xStart, yStart, zStart}, end: [3]int{xEnd, yEnd, zEnd}, d: d, timedependent: timeDep, stringFct: equation})
}

func (p *Excitation) RenderFunctionLimitX(equation StringFunction, xStart, xEnd int) {
	n := MeshOf(p).Size()
	p.RenderFunctionLimit(equation, xStart, xEnd, 0, n[Y], 0, n[Z])
}

func (p *Excitation) RenderFunctionLimitY(equation StringFunction, yStart, yEnd int) {
	n := MeshOf(p).Size()
	p.RenderFunctionLimit(equation, 0, n[X], yStart, yEnd, 0, n[Z])
}

func (p *Excitation) RenderFunctionLimitZ(equation StringFunction, zStart, zEnd int) {
	n := MeshOf(p).Size()
	p.RenderFunctionLimit(equation, 0, n[X], 0, n[Y], zStart, zEnd)
}

func (p *Excitation) RemoveRenderedFunction(equation StringFunction) {
	EraseSetExcitation(p.name)
}

func (p *Excitation) MSliceRegion(size [3]int, offsetX, offsetY, offsetZ int) cuda.MSlice {
	buf, r := p.Slice()
	bufRed := cuda.Buffer(buf.NComp(), size)
	cuda.Crop(bufRed, buf, offsetX, offsetY, offsetZ)
	cuda.Recycle(buf)
	util.Assert(r)
	return cuda.ToMSlice(bufRed)
}

func (e *Excitation) AddTo(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddV(dst, e.perRegion.gpuLUT(), regions.Gpu())
	}

	for _, t := range e.extraTerms {
		MulSlice := make([]float32, len(t.mul))
		for i := range MulSlice {
			if t.mul[i] != nil {
				MulSlice[i] = float32(t.mul[i]())
			} else {
				MulSlice[i] = 0
			}
		}

		OneSlice := make([]float32, dst.NComp())
		for i := range OneSlice {
			OneSlice[i] = 1.
		}
		cuda.Madd2Comp(dst, dst, t.mask, OneSlice, MulSlice)
	}
}

func (e *Excitation) AddToRegion(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddV(dst, e.perRegion.gpuLUT(), regions.Gpu())
	}

	for _, t := range e.extraTerms {
		MulSlice := make([]float32, len(t.mul))
		for i := range MulSlice {
			if t.mul[i] != nil {
				MulSlice[i] = float32(t.mul[i]())
			} else {
				MulSlice[i] = 0
			}
		}

		OneSlice := make([]float32, dst.NComp())
		for i := range OneSlice {
			OneSlice[i] = 1.
		}
		maskRed := cuda.Buffer(t.mask.NComp(), dst.RegionSize())
		cuda.Crop(maskRed, t.mask, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Madd2Comp(dst, dst, maskRed, OneSlice, MulSlice)
		cuda.Recycle(maskRed)
	}
}

func (e *Excitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *Excitation) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(e.NComp(), e.Mesh().Size())
	cuda.Zero(buf)
	e.AddTo(buf)
	tmp, ok := mapSetExcitation.Load(e.name)
	if ok {
		setExcitations := tmp.(ExcitationSlices)
		for _, setExcitation := range setExcitations.slc {
			if setExcitation.timedependent {
				d := GenerateSliceFromFunctionString(setExcitation.stringFct, e.Mesh())
				setExcitation.d = d
			}
			newData := setExcitation.d
			regionStart := setExcitation.start
			regionEnd := setExcitation.end
			data.CopyPart(buf, newData, 0, regionEnd[X]-regionStart[X], 0, regionEnd[Y]-regionStart[Y], 0, regionEnd[Z]-regionStart[Z], 0, 1, regionStart[X], regionStart[Y], regionStart[Z], 0)
		}
	}
	return buf, true
}

func (e *Excitation) SliceRegion(size [3]int, offsetX, offsetY, offsetZ int) (*data.Slice, bool) {
	buf, ok := e.Slice()
	bufRed := cuda.Buffer(buf.NComp(), size)
	cuda.Crop(bufRed, buf, offsetX, offsetY, offsetZ)
	cuda.Recycle(buf)
	return bufRed, ok
}

// After resizing the mesh, the extra terms don't fit the grid anymore
// and there is no reasonable way to resize them. So remove them and have
// the user re-add them.
func (e *Excitation) RemoveExtraTerms() {
	if len(e.extraTerms) == 0 {
		return
	}

	LogOut("REMOVING EXTRA TERMS FROM", e.Name())
	for _, m := range e.extraTerms {
		m.mask.Free()
	}
	e.extraTerms = nil
}

// Add an extra mask*multiplier term to the excitation.
func (e *Excitation) AddComp(mask *data.Slice, f0, f1, f2 script.ScalarFunction) {
	var mul0, mul1, mul2 func() float64
	if f0 != nil {
		if IsConst(f0) {
			val := f0.Float()
			mul0 = func() float64 {
				return val
			}
		} else {
			mul0 = func() float64 {
				return f0.Float()
			}
		}
	}
	if f1 != nil {
		if IsConst(f1) {
			val := f1.Float()
			mul1 = func() float64 {
				return val
			}
		} else {
			mul1 = func() float64 {
				return f1.Float()
			}
		}
	}
	if f2 != nil {
		if IsConst(f2) {
			val := f2.Float()
			mul2 = func() float64 {
				return val
			}
		} else {
			mul2 = func() float64 {
				return f2.Float()
			}
		}
	}
	e.AddGo(mask, mul0, mul1, mul2)
}

func (e *Excitation) Add(mask *data.Slice, f script.ScalarFunction) {
	var mul func() float64
	if f != nil {
		if IsConst(f) {
			val := f.Float()
			mul = func() float64 {
				return val
			}
		} else {
			mul = func() float64 {
				return f.Float()
			}
		}
	}
	e.AddGo(mask, mul, mul, mul)
}

// An Add(mask, f) equivalent for Go use
func (e *Excitation) AddGo(mask *data.Slice, mul0, mul1, mul2 func() float64) {
	if mask != nil {
		checkNaN(mask, e.Name()+".add()") // TODO: in more places
		mask = data.Resample(mask, e.Mesh().Size())
		mask = assureGPU(mask)
	}
	funcSlice := make([]func() float64, 3)
	funcSlice[0] = mul0
	funcSlice[1] = mul1
	funcSlice[2] = mul2

	e.extraTerms = append(e.extraTerms, mulmask3D{funcSlice, mask})
}

func (e *Excitation) SetRegion(region int, f script.VectorFunction) { e.perRegion.SetRegion(region, f) }
func (e *Excitation) SetValue(v interface{})                        { e.perRegion.SetValue(v) }
func (e *Excitation) Set(v data.Vector)                             { e.perRegion.setRegions(0, NREGION, slice(v)) }
func (e *Excitation) getRegion(region int) []float64                { return e.perRegion.getRegion(region) } // for gui

func (e *Excitation) SetRegionFn(region int, f func() [3]float64) {
	e.perRegion.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (e *Excitation) average() []float64      { return qAverageUniverse(e) }
func (e *Excitation) Average() data.Vector    { return unslice(qAverageUniverse(e)) }
func (e *Excitation) IsUniform() bool         { return e.perRegion.IsUniform() }
func (e *Excitation) Name() string            { return e.name }
func (e *Excitation) Unit() string            { return e.perRegion.Unit() }
func (e *Excitation) NComp() int              { return e.perRegion.NComp() }
func (e *Excitation) Mesh() *data.Mesh        { return Mesh() }
func (e *Excitation) Region(r int) *vOneReg   { return vOneRegion(e, r) }
func (e *Excitation) Comp(c int) ScalarField  { return Comp(e, c) }
func (e *Excitation) Eval() interface{}       { return e }
func (e *Excitation) Type() reflect.Type      { return reflect.TypeOf(new(Excitation)) }
func (e *Excitation) InputType() reflect.Type { return script.VectorFunction_t }
func (e *Excitation) EvalTo(dst *data.Slice)  { EvalTo(e, dst) }
func (e *Excitation) EvalRegionTo(dst *data.Slice) {
	buf := cuda.Buffer(e.NComp(), e.Mesh().Size())
	e.EvalTo(buf)
	cuda.Crop(dst, buf, dst.StartX, dst.StartY, dst.StartZ)
}

func checkNaN(s *data.Slice, name string) {
	h := s.Host()
	for _, h := range h {
		for _, v := range h {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				util.Fatal("NaN or Inf in", name)
			}
		}
	}
}

func (e *Excitation) GetRegionToString(region int) string {
	v := e.perRegion.GetRegion(region)
	return fmt.Sprintf("(%g,%g,%g)", v[0], v[1], v[2])
}
