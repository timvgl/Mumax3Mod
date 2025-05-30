package engine

import (
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// look-up table for region based parameters
type lut struct {
	gpu_buf cuda.LUTPtrs       // gpu copy of cpu buffer, only transferred when needed
	gpu_ok  bool               // gpu cache up-to date with cpu source?
	cpu_buf [][NREGION]float32 // table data on cpu
	source  updater            // updates cpu data
	name    string
}

type updater interface {
	update() // updates cpu lookup table
}

func (p *lut) init(nComp int, source updater) {
	p.gpu_buf = make(cuda.LUTPtrs, nComp)
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.source = source
}

// get an up-to-date version of the lookup-table on CPU
func (p *lut) cpuLUT() [][NREGION]float32 {
	p.source.update()
	return p.cpu_buf
}

// get an up-to-date version of the lookup-table on GPU
func (p *lut) gpuLUT() cuda.LUTPtrs {
	p.source.update()
	if !p.gpu_ok {
		// upload to GPU
		p.assureAlloc()
		cuda.Sync() // sync previous kernels, may still be using gpu lut
		for c := range p.gpu_buf {
			cuda.MemCpyHtoD(p.gpu_buf[c], unsafe.Pointer(&p.cpu_buf[c][0]), cu.SIZEOF_FLOAT32*NREGION)
		}
		p.gpu_ok = true
		cuda.Sync() //sync upload
	}
	return p.gpu_buf
}

// utility for LUT of single-component data
func (p *lut) gpuLUT1() cuda.LUTPtr {
	util.Assert(len(p.gpu_buf) == 1)
	return cuda.LUTPtr(p.gpuLUT()[0])
}

// all data is 0?
func (p *lut) isZero() bool {
	_, ok := mapSetParam.Load(p.name)
	if ok {
		return false
	}
	v := p.cpuLUT()
	for c := range v {
		for i := 0; i < NREGION; i++ {
			if v[c][i] != 0 {
				return false
			}
		}
	}
	return true
}

func (p *lut) nonZero() bool { return !p.isZero() }

func (p *lut) assureAlloc() {
	if p.gpu_buf[0] == nil {
		for i := range p.gpu_buf {
			p.gpu_buf[i] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
	}
}

func (b *lut) NComp() int { return len(b.cpu_buf) }

// uncompress the table to a full array with parameter values per cell.
func (p *lut) Slice() (*data.Slice, bool) {
	b := cuda.Buffer(p.NComp(), Mesh().Size())
	p.EvalTo(b)
	return b, true
}

func (p *lut) SliceRegion(paramName string, param bool, size [3]int, offsetX, offsetY, offsetZ int) (*data.Slice, bool) {
	b, ok := p.Slice()
	c := cuda.Buffer(p.NComp(), size)
	cuda.Crop(c, b, offsetX, offsetY, offsetZ)
	if ok {
		cuda.Recycle(b)
	}
	return c, true
}

// uncompress the table to a full array in the dst Slice with parameter values per cell.
func (p *lut) EvalTo(dst *data.Slice) {
	gpu := p.gpuLUT()
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(dst.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
	}
	tmp, ok := mapSetParam.Load(p.name)
	if ok {
		setParams := tmp.(ParameterSlices)
		for _, setParam := range setParams.slc {
			if setParam.timedependent {
				for i := range setParam.renderers {
					setParam.renderers[i].Vars["t"] = Time
				}
				xEnd := setParam.end[X]
				yEnd := setParam.end[Y]
				zEnd := setParam.end[Z]
				xStart := setParam.start[X]
				yStart := setParam.start[Y]
				zStart := setParam.start[Z]
				mesh := data.NewMesh(xEnd-xStart, yEnd-yStart, zEnd-zStart, MeshOf(p).CellSize()[X], MeshOf(p).CellSize()[Y], MeshOf(p).CellSize()[Z])
				d, _ := GenerateSliceFromReadyToRenderFct(setParam.renderers, mesh)
				data.Copy(setParam.d, d)
				cuda.Recycle(d)
			}
			if setParam.renderedShape == nil && setParam.inversedRenderedShape == nil {
				newData := setParam.d
				regionStart := setParam.start
				regionEnd := setParam.end
				data.CopyPart(dst, newData, 0, regionEnd[X]-regionStart[X], 0, regionEnd[Y]-regionStart[Y], 0, regionEnd[Z]-regionStart[Z], 0, 1, regionStart[X], regionStart[Y], regionStart[Z], 0)
			} else if setParam.renderedShape != nil && setParam.inversedRenderedShape != nil {
				cuda.Mul(dst, dst, setParam.inversedRenderedShape)
				cuda.Mul(setParam.d, setParam.d, setParam.renderedShape)
				cuda.Add(dst, dst, setParam.d)
			} else {
				panic("Invalid option for rendering function in parameters encountered.")
			}
		}
	}
}
