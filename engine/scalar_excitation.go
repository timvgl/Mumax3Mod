package engine

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
)

var mapSetScalarExcitation sync.Map

func SetScalarExcitation(name string, s ScalarExcitationSlice) {
	tmp, ok := mapSetScalarExcitation.Load(name)
	if ok {
		slces := tmp.(ScalarExcitationSlices)
		var slcInterfaces = make([]SetSlice, 0)
		for _, slc := range slces.slc {
			slcInterfaces = append(slcInterfaces, slc)
		}
		overlapIndices := GetOverlapIndex(s, slcInterfaces)
		FreeMemoryIndices(slcInterfaces, overlapIndices)
		slces.slc = RemoveIndices(slces.slc, overlapIndices)
		slces.slc = append(slces.slc, s)
		mapSetScalarExcitation.Store(name, slces)
	} else {
		slc := ScalarExcitationSlices{name: name, slc: []ScalarExcitationSlice{s}}
		mapSetScalarExcitation.Store(name, slc)
	}
}

func EraseSetScalarExcitation(name string) {
	tmp, ok := mapSetScalarExcitation.Load(name)
	if ok {
		s := tmp.(ScalarExcitationSlices)
		for _, s := range s.slc {
			if s.d.GPUAccess() {
				cuda.Recycle(s.d)
			} else {
				s.d.Free()
			}
		}
		mapSetScalarExcitation.Delete(name)
	} else {
		panic(fmt.Sprintf("EraseSetScalarExcitation: %s not found", name))
	}
}

type ScalarExcitationSlices struct {
	name string
	slc  []ScalarExcitationSlice
}

type ScalarExcitationSlice struct {
	start         [3]int
	end           [3]int
	d             *data.Slice
	timedependent bool
	stringFct     StringFunction
}

func (s ScalarExcitationSlice) StartAt() [3]int {
	return s.start
}

func (s ScalarExcitationSlice) EndAt() [3]int {
	return s.end
}

func (s ScalarExcitationSlice) Buffer() *data.Slice {
	return s.d
}

// An excitation, typically field or current,
// can be defined region-wise plus extra mask*multiplier terms.
type ScalarExcitation struct {
	name       string
	perRegion  RegionwiseScalar // Region-based excitation
	extraTerms []mulmask        // add extra mask*multiplier terms
}

func NewScalarExcitation(name, unit, desc string) *ScalarExcitation {
	e := new(ScalarExcitation)
	e.name = name
	e.perRegion.init("_"+name+"_perRegion", unit, desc, nil) // name starts with underscore: unexported
	DeclLValue(name, e, cat(desc, unit))
	return e
}

func (p *ScalarExcitation) LoadFile(path string, xOffset, yOffset, zOffset int) {
	d := LoadFileDSlice(path)
	SetScalarExcitation(p.name, ScalarExcitationSlice{[3]int{xOffset, yOffset, zOffset}, [3]int{xOffset + d.Size()[X], yOffset + d.Size()[Y], zOffset + d.Size()[Z]}, d, false, StringFunction{[3]string{"", "", ""}, false}})

}

func (p *ScalarExcitation) MSlice() cuda.MSlice {
	buf, r := p.Slice()
	util.Assert(r)
	return cuda.ToMSlice(buf)
}

func (p *ScalarExcitation) RenderFunction(equation StringFunction) {
	util.AssertMsg(equation.IsScalar(), "RenderFunction: Need scalar function.")
	d, timeDep := GenerateSliceFromFunctionStringTimeDep(equation, p.Mesh())
	SetScalarExcitation(p.name, ScalarExcitationSlice{[3]int{0, 0, 0}, p.Mesh().Size(), d, timeDep, equation})
}

func (p *ScalarExcitation) RenderFunctionLimit(equation StringFunction, xStart, xEnd, yStart, yEnd, zStart, zEnd int) {
	util.AssertMsg(equation.IsScalar(), "RenderFunction: Need scalar function.")
	n := MeshOf(p).Size()
	util.Argument(xStart < xEnd && yStart < yEnd && zStart < zEnd)
	util.Argument(xStart >= 0 && yStart >= 0 && zStart >= 0)
	util.Argument(xEnd <= n[X] && yEnd <= n[Y] && zEnd <= n[Z])
	d, timeDep := GenerateSliceFromFunctionStringTimeDep(equation, p.Mesh())
	SetScalarExcitation(p.name, ScalarExcitationSlice{[3]int{xStart, yStart, zStart}, [3]int{xEnd, yEnd, zEnd}, d, timeDep, equation})
}

func (p *ScalarExcitation) RenderFunctionLimitX(equation StringFunction, xStart, xEnd int) {
	n := MeshOf(p).Size()
	p.RenderFunctionLimit(equation, xStart, xEnd, 0, n[Y], 0, n[Z])
}

func (p *ScalarExcitation) RenderFunctionLimitY(equation StringFunction, yStart, yEnd int) {
	n := MeshOf(p).Size()
	p.RenderFunctionLimit(equation, 0, n[X], yStart, yEnd, 0, n[Z])
}

func (p *ScalarExcitation) RenderFunctionLimitZ(equation StringFunction, zStart, zEnd int) {
	n := MeshOf(p).Size()
	p.RenderFunctionLimit(equation, 0, n[X], 0, n[Y], zStart, zEnd)
}

func (p *ScalarExcitation) RemoveRenderedFunction() {
	EraseSetScalarExcitation(p.name)
}

func (e *ScalarExcitation) AddTo(dst *data.Slice) {
	if !e.perRegion.isZero() {
		cuda.RegionAddS(dst, e.perRegion.gpuLUT1(), regions.Gpu())
	}

	for _, t := range e.extraTerms {
		var mul float32 = 1
		if t.mul != nil {
			mul = float32(t.mul())
		}
		cuda.Madd2(dst, dst, t.mask, 1, mul)
	}
}

func (e *ScalarExcitation) isZero() bool {
	return e.perRegion.isZero() && len(e.extraTerms) == 0
}

func (e *ScalarExcitation) Slice() (*data.Slice, bool) {
	size := e.Mesh().Size()
	buf := cuda.Buffer(e.NComp(), size)
	cuda.Zero(buf)
	e.AddTo(buf)
	tmp, ok := mapSetScalarExcitation.Load(e.name)
	if ok {
		setExcitations := tmp.(ScalarExcitationSlices)
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

// After resizing the mesh, the extra terms don't fit the grid anymore
// and there is no reasonable way to resize them. So remove them and have
// the user re-add them.
func (e *ScalarExcitation) RemoveExtraTerms() {
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
func (e *ScalarExcitation) Add(mask *data.Slice, f script.ScalarFunction) {
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
	e.AddGo(mask, mul)
}

// An Add(mask, f) equivalent for Go use
func (e *ScalarExcitation) AddGo(mask *data.Slice, mul func() float64) {
	if mask != nil {
		checkNaN(mask, e.Name()+".add()") // TODO: in more places
		mask = data.Resample(mask, e.Mesh().Size())
		mask = assureGPU(mask)
	}
	e.extraTerms = append(e.extraTerms, mulmask{mul, mask})
}

func (e *ScalarExcitation) SetRegion(region int, f script.ScalarFunction) {
	e.perRegion.SetRegion(region, f)
}
func (e *ScalarExcitation) SetValue(v interface{})         { e.perRegion.SetValue(v) }
func (e *ScalarExcitation) Set(v float64)                  { e.perRegion.setRegions(0, NREGION, []float64{v}) }
func (e *ScalarExcitation) getRegion(region int) []float64 { return e.perRegion.getRegion(region) } // for gui

func (e *ScalarExcitation) SetRegionFn(region int, f func() [3]float64) {
	e.perRegion.setFunc(region, region+1, func() []float64 {
		return slice(f())
	})
}

func (e *ScalarExcitation) average() float64        { return qAverageUniverse(e)[0] }
func (e *ScalarExcitation) Average() float64        { return e.average() }
func (e *ScalarExcitation) IsUniform() bool         { return e.perRegion.IsUniform() }
func (e *ScalarExcitation) Name() string            { return e.name }
func (e *ScalarExcitation) Unit() string            { return e.perRegion.Unit() }
func (e *ScalarExcitation) NComp() int              { return e.perRegion.NComp() }
func (e *ScalarExcitation) Mesh() *data.Mesh        { return Mesh() }
func (e *ScalarExcitation) Region(r int) *vOneReg   { return vOneRegion(e, r) }
func (e *ScalarExcitation) Comp(c int) ScalarField  { return Comp(e, c) }
func (e *ScalarExcitation) Eval() interface{}       { return e }
func (e *ScalarExcitation) Type() reflect.Type      { return reflect.TypeOf(new(ScalarExcitation)) }
func (e *ScalarExcitation) InputType() reflect.Type { return script.ScalarFunction_t }
func (e *ScalarExcitation) EvalTo(dst *data.Slice)  { EvalTo(e, dst) }
func (e *ScalarExcitation) EvalRegionTo(dst *data.Slice) {
	buf := cuda.Buffer(e.NComp(), e.Mesh().Size())
	e.EvalTo(buf)
	cuda.Crop(dst, buf, dst.StartX, dst.StartY, dst.StartZ)
}

func (e *ScalarExcitation) GetRegionToString(region int) string {
	return fmt.Sprintf("%g", e.perRegion.GetRegion(region))
}
