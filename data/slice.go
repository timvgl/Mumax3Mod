package data

// Slice stores N-component GPU or host data.

import (
	"bytes"
	"fmt"
	"log"
	"reflect"
	"unsafe"

	"github.com/mumax/3/util"
)

var DataSliceSlice = make([]*Slice, 0)

// Slice is like a [][]float32, but may be stored in GPU or host memory.
type Slice struct {
	ptrs    []unsafe.Pointer
	size    [3]int
	memType int8
	LengthF int
	StartX  int
	EndX    int
	StartY  int
	EndY    int
	StartZ  int
	EndZ    int
}

// this package must not depend on CUDA. If CUDA is
// loaded, these functions are set to cu.MemFree, ...
// NOTE: cpyDtoH and cpuHtoD are only needed to support 32-bit builds,
// otherwise, it could be removed in favor of memCpy only.
var (
	memFree, memFreeHost           func(unsafe.Pointer)
	memCpyDtoH, memCpyHtoD         func(dst, src unsafe.Pointer, bytes int64)
	memCpy                         func(dst, src unsafe.Pointer, bytes int64, args ...string)
	memCpyDtoHPart, memCpyHtoDPart func(dst, src unsafe.Pointer, offset_dst, offset_src, bytes int64)
	memCpyPart                     func(dst, src unsafe.Pointer, offset_dst, offset_src, bytes int64, args ...string)
	createStream                   func(key string)
	destroyStream                  func(key string)
	setCurrentCtx                  func()
	MAX_STREAMS                    = 5
	cudaCopyPart                   func(dst, src *Slice,
		xStart_src, xEnd_src,
		yStart_src, yEnd_src,
		zStart_src, zEnd_src,
		fStart_src, fEnd_src,
		xStart_dst, yStart_dst,
		zStart_dst, fStart_dst int)
)

// Internal: enables slices on GPU. Called upon cuda init.
func EnableGPU(free, freeHost func(unsafe.Pointer),
	cpyDtoH, cpyHtoD func(dst, src unsafe.Pointer, bytes int64), cpy func(dst, src unsafe.Pointer, bytes int64, args ...string), cpyDtoHPart, cpyHtoDPart func(dst, src unsafe.Pointer, offset_dst, offset_src, bytes int64), cpyPart func(dst, src unsafe.Pointer, offset_dst, offset_src, bytes int64, args ...string), CreateStream, DestroyStream func(key string), SetCurrentCtx func(), CudaCopyPart func(dst, src *Slice, xStart_src, xEnd_src, yStart_src, yEnd_src, zStart_src, zEnd_src, fStart_src, fEnd_src, xStart_dst, yStart_dst, zStart_dst, fStart_dst int)) {
	memFree = free
	memFreeHost = freeHost
	memCpy = cpy
	memCpyDtoH = cpyDtoH
	memCpyHtoD = cpyHtoD
	memCpyDtoHPart = cpyDtoHPart
	memCpyHtoDPart = cpyHtoDPart
	memCpyPart = cpyPart
	createStream = CreateStream
	destroyStream = DestroyStream
	setCurrentCtx = SetCurrentCtx
	cudaCopyPart = CudaCopyPart
}

func Zero(data *Slice) {
	array := data.Tensors()
	size := data.size
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < data.NComp(); c++ {
					array[c][iz][iy][ix] = 0
				}
			}
		}
	}
}

// Make a CPU Slice with nComp components of size length.
func NewSlice(nComp int, size [3]int) *Slice {
	length := prod(size)
	ptrs := make([]unsafe.Pointer, nComp)
	for i := range ptrs {
		ptrs[i] = unsafe.Pointer(&(make([]float32, length)[0]))
	}
	slc := SliceFromPtrs(size, CPUMemory, ptrs)
	DataSliceSlice = append(DataSliceSlice, slc)
	return slc
}

func SliceFromArray(data [][]float32, size [3]int) *Slice {
	nComp := len(data)
	length := prod(size)
	ptrs := make([]unsafe.Pointer, nComp)
	for i := range ptrs {
		if len(data[i]) != length {
			panic("size mismatch")
		}
		ptrs[i] = unsafe.Pointer(&data[i][0])
	}
	return SliceFromPtrs(size, CPUMemory, ptrs)
}

func SliceFromSlices(data []*Slice, size [3]int) *Slice {
	nComp := len(data)
	length := prod(size)
	lengthF := data[0].LengthF
	memType := data[0].memType

	ptrs := make([]unsafe.Pointer, nComp)
	for i := range ptrs {
		if data[i].Len() != length {
			panic("size mismatch")
		}
		if data[i].LengthF != lengthF {
			panic("lengthF mismatch")
		}
		if data[i].memType != memType {
			panic("memType mismatch")
		}
		if data[i].NComp() != 1 {
			panic("slice merge only with one comp slices possible")
		}
		ptrs[i] = data[i].DevPtr(0)
	}
	slc := SliceFromPtrs(size, GPUMemory, ptrs)
	for _, s := range data {
		idx := 0
		gotValue := false
		for i := range DataSliceSlice {
			if DataSliceSlice[i] == s {
				idx = i
				gotValue = true
				break
			}
		}
		if !gotValue {
			panic("Could not find buffer in buffer list.")
		}
		DataSliceSlice = append(DataSliceSlice[:idx], DataSliceSlice[idx+1:]...)
	}
	slc.LengthF = data[0].LengthF
	slc.memType = data[0].memType
	DataSliceSlice = append(DataSliceSlice, slc)
	return slc
}

// Return a slice without underlying storage. Used to represent a mask containing all 1's.
func NilSlice(nComp int, size [3]int) *Slice {
	slc := SliceFromPtrs(size, GPUMemory, make([]unsafe.Pointer, nComp))
	DataSliceSlice = append(DataSliceSlice, slc)
	return slc
}

// Internal: construct a Slice using bare memory pointers.
func SliceFromPtrs(size [3]int, memType int8, ptrs []unsafe.Pointer) *Slice {
	length := prod(size)
	nComp := len(ptrs)
	util.Argument(nComp > 0 && length > 0)
	s := new(Slice)
	s.ptrs = make([]unsafe.Pointer, nComp)
	s.size = size
	for c := range ptrs {
		s.ptrs[c] = ptrs[c]
	}
	s.memType = memType
	s.LengthF = 1
	s.StartX = 0
	s.EndX = size[X]
	s.StartY = 0
	s.EndY = size[Y]
	s.StartZ = 0
	s.EndZ = size[Z]
	return s
}

func SliceFromPtrsF(size [3]int, fLength int, memType int8, ptrs []unsafe.Pointer) *Slice {
	length := prod(size) * fLength
	nComp := len(ptrs)
	util.Argument(nComp > 0 && length > 0)
	s := new(Slice)
	s.ptrs = make([]unsafe.Pointer, nComp)
	s.size = size
	for c := range ptrs {
		s.ptrs[c] = ptrs[c]
	}
	s.memType = memType
	s.LengthF = fLength
	s.StartX = 0
	s.EndX = size[X]
	s.StartY = 0
	s.EndY = size[Y]
	s.StartZ = 0
	s.EndZ = size[Z]
	return s
}

func (s *Slice) SetSolverRegion(startX, endX, startY, endY, startZ, endZ int) {
	s.StartX = startX
	s.StartY = startY
	s.StartZ = startZ
	s.EndX = endX
	s.EndY = endY
	s.EndZ = endZ
}

func (s *Slice) Get_lengthF() int {
	if s == nil {
		panic("Slice is nil")
	}
	return s.LengthF
}

// Frees the underlying storage and zeros the Slice header to avoid accidental use.
// Slices sharing storage will be invalid after Free. Double free is OK.
func (s *Slice) Free() {
	if s == nil {
		return
	}
	idx := 0
	gotValue := false
	for i := range DataSliceSlice {
		if DataSliceSlice[i] == s {
			idx = i
			gotValue = true
			break
		}
	}
	if !gotValue {
		panic("Could not find buffer in buffer list.")
	}
	DataSliceSlice = append(DataSliceSlice[:idx], DataSliceSlice[idx+1:]...)
	// free storage
	switch s.memType {
	case 0:
		return // already freed
	case GPUMemory:
		for _, ptr := range s.ptrs {
			memFree(ptr)
		}
	//case UnifiedMemory:
	//	for _, ptr := range s.ptrs {
	//		memFreeHost(ptr)
	//	}
	case CPUMemory:
		// nothing to do
	default:
		panic("invalid memory type")
	}
	s.Disable()
}

// INTERNAL. Overwrite struct fields with zeros to avoid
// accidental use after Free.
func (s *Slice) Disable() {
	s.ptrs = s.ptrs[:0]
	s.size = [3]int{0, 0, 0}
	s.memType = 0
}

// value for Slice.memType
const (
	CPUMemory = 1 << 0
	GPUMemory = 1 << 1
	//UnifiedMemory = CPUMemory | GPUMemory
)

// MemType returns the memory type of the underlying storage:
// CPUMemory, GPUMemory or UnifiedMemory
func (s *Slice) MemType() int {
	return int(s.memType)
}

// GPUAccess returns whether the Slice is accessible by the GPU.
// true means it is either stored on GPU or in unified host memory.
func (s *Slice) GPUAccess() bool {
	return s.memType&GPUMemory != 0
}

// CPUAccess returns whether the Slice is accessible by the CPU.
// true means it is stored in host memory.
func (s *Slice) CPUAccess() bool {
	return s.memType&CPUMemory != 0
}

// NComp returns the number of components.
func (s *Slice) NComp() int {
	return len(s.ptrs)
}

// Len returns the number of elements per component.
func (s *Slice) Len() int {
	return prod(s.size)
}

func (s *Slice) Size() [3]int {
	if s == nil {
		return [3]int{0, 0, 0}
	}
	return s.size
}

// Comp returns a single component of the Slice.
func (s *Slice) Comp(i int) *Slice {
	sl := new(Slice)
	sl.ptrs = make([]unsafe.Pointer, 1)
	sl.ptrs[0] = s.ptrs[i]
	sl.size = s.size
	sl.memType = s.memType
	size := s.Size()
	s.StartX = 0
	s.EndX = size[X]
	s.StartY = 0
	s.EndY = size[Y]
	s.StartZ = 0
	s.EndZ = size[Z]
	return sl
}

// Ptrs returns a copy of the ptrs slice
func (s *Slice) Ptrs() []unsafe.Pointer {
	ptrs := make([]unsafe.Pointer, len(s.ptrs))
	copy(ptrs, s.ptrs)
	return ptrs
}

func (s *Slice) RegionSize() [3]int {
	return [3]int{s.EndX - s.StartX, s.EndY - s.StartY, s.EndZ - s.StartZ}
}

// Subslice returns a subslice with components ranging from
// minComp to maxComp (exclusive)
func (s *Slice) SubSlice(minComp, maxComp int) *Slice {
	sl := new(Slice)
	sl.ptrs = s.ptrs[minComp:maxComp]
	sl.size = s.size
	sl.memType = s.memType
	size := s.size
	s.StartX = 0
	s.EndX = size[X]
	s.StartY = 0
	s.EndY = size[Y]
	s.StartZ = 0
	s.EndZ = size[Z]
	return sl
}

// DevPtr returns a CUDA device pointer to a component.
// Slice must have GPUAccess.
// It is safe to call on a nil slice, returns NULL.
func (s *Slice) DevPtr(component int) unsafe.Pointer {
	if s == nil {
		return nil
	}
	if !s.GPUAccess() {
		panic("slice not accessible by GPU")
	}
	return s.ptrs[component]
}

const SIZEOF_FLOAT32 = 4

// Host returns the Slice as a [][]float32 indexed by component, cell number.
// It should have CPUAccess() == true.
func (s *Slice) Host() [][]float32 {
	if !s.CPUAccess() {
		log.Panic("slice not accessible by CPU")
	}
	list := make([][]float32, s.NComp())
	for c := range list {
		hdr := (*reflect.SliceHeader)(unsafe.Pointer(&list[c]))
		hdr.Data = uintptr(s.ptrs[c])
		hdr.Len = s.Len()
		hdr.Cap = hdr.Len
	}
	return list
}

// Returns a copy of the Slice, allocated on CPU.
func (s *Slice) HostCopy() *Slice {
	cpy := NewSlice(s.NComp(), s.Size())
	Copy(cpy, s)
	return cpy
}

func (s *Slice) HostCopyPart(xStart, xEnd, yStart, yEnd, zStart, zEnd, fStart, fEnd int) *Slice {
	cpy := NewSlice(s.NComp(), [3]int{xEnd - xStart, yEnd - yStart, zEnd - zStart})
	CopyPart(cpy, s, xStart, xEnd, yStart, yEnd, zStart, zEnd, fStart, fEnd, 0, 0, 0, 0)
	return cpy
}

func Copy(dst, src *Slice, args ...string) {
	if dst.NComp() != src.NComp() || dst.Len() != src.Len() {
		panic(fmt.Sprintf("slice copy: illegal sizes: dst: %vx%v, src: %vx%v", dst.NComp(), dst.Len(), src.NComp(), src.Len()))
	}
	d, s := dst.GPUAccess(), src.GPUAccess()
	bytes := SIZEOF_FLOAT32 * int64(dst.Len())
	if len(args) > 1 {
		panic("Only one string arg allowed in Copy.")
	}
	switch {
	default:
		panic("bug")
	case d && s:
		for c := 0; c < dst.NComp(); c++ {
			memCpy(dst.DevPtr(c), src.DevPtr(c), bytes, args...)
		}
	case s && !d:
		for c := 0; c < dst.NComp(); c++ {
			memCpyDtoH(dst.ptrs[c], src.DevPtr(c), bytes)
		}
	case !s && d:
		for c := 0; c < dst.NComp(); c++ {
			memCpyHtoD(dst.DevPtr(c), src.ptrs[c], bytes)
		}
	case !d && !s:
		dst, src := dst.Host(), src.Host()
		for c := range dst {
			copy(dst[c], src[c])
		}
	}
}

func CopyComp(dst, src *Slice, comp int, args ...string) {
	if dst.NComp() != 1 || dst.Len() != src.Len() && comp > 2 || comp < 0 {
		panic(fmt.Sprintf("slice copy: illegal sizes: dst: %vx%v, src: %vx%v with %v being selected", dst.NComp(), dst.Len(), src.NComp(), src.Len(), comp))
	}
	d, s := dst.GPUAccess(), src.GPUAccess()
	bytes := SIZEOF_FLOAT32 * int64(dst.Len())
	if len(args) > 1 {
		panic("Only one string arg allowed in Copy.")
	}
	switch {
	default:
		panic("bug")
	case d && s:
		memCpy(dst.DevPtr(0), src.DevPtr(comp), bytes, args...)
	case s && !d:
		memCpyDtoH(dst.ptrs[0], src.DevPtr(comp), bytes)
	case !s && d:
		memCpyHtoD(dst.DevPtr(0), src.ptrs[comp], bytes)
	case !d && !s:
		dst, src := dst.Host(), src.Host()
		copy(dst[0], src[comp])
	}
}

func CopyPart(dst, src *Slice,
	xStart_src, xEnd_src,
	yStart_src, yEnd_src,
	zStart_src, zEnd_src,
	fStart_src, fEnd_src,
	xStart_dst, yStart_dst,
	zStart_dst, fStart_dst int,
	args ...string) {

	if dst.NComp() != src.NComp() {
		panic(fmt.Sprintf("slice copy: illegal sizes: dst: %vx%v, src: %vx%v",
			dst.NComp(), dst.Len(), src.NComp(), src.Len()))
	}

	d, s := dst.GPUAccess(), src.GPUAccess()
	sizeSrc := src.Size()
	sizeDst := dst.Size()

	// Strides for each dimension in source and destination
	strideXSrc := 1
	strideYSrc := sizeSrc[X]
	strideZSrc := sizeSrc[X] * sizeSrc[Y]
	strideFSrc := sizeSrc[X] * sizeSrc[Y] * sizeSrc[Z]

	strideXDst := 1
	strideYDst := sizeDst[X]
	strideZDst := sizeDst[X] * sizeDst[Y]
	strideFDst := sizeDst[X] * sizeDst[Y] * sizeDst[Z]

	// Calculate number of elements to copy along each dimension
	xCount := xEnd_src - xStart_src
	yCount := yEnd_src - yStart_src
	zCount := zEnd_src - zStart_src
	fCount := fEnd_src - fStart_src
	if d && !s || !d && s || !d && !s {
		for f := 0; f < fCount; f++ {
			for z := 0; z < zCount; z++ {
				for y := 0; y < yCount; y++ {
					// Calculate the source and destination offsets for this slice
					offsetSrc := int64(
						xStart_src*strideXSrc+
							(yStart_src+y)*strideYSrc+
							(zStart_src+z)*strideZSrc+
							(fStart_src+f)*strideFSrc,
					) * int64(SIZEOF_FLOAT32)

					offsetDst := int64(
						xStart_dst*strideXDst+
							(yStart_dst+y)*strideYDst+
							(zStart_dst+z)*strideZDst+
							(fStart_dst+f)*strideFDst,
					) * int64(SIZEOF_FLOAT32)

					// Calculate the number of bytes to copy for the x dimension
					bytes := int64(xCount) * int64(SIZEOF_FLOAT32)

					// Perform the copy based on the access types
					if s && !d {
						for c := 0; c < dst.NComp(); c++ {
							memCpyDtoHPart(
								dst.ptrs[c],
								src.DevPtr(c),
								offsetDst,
								offsetSrc,
								bytes,
							)
						}
					} else if !s && d {
						for c := 0; c < dst.NComp(); c++ {
							memCpyHtoDPart(
								dst.DevPtr(c),
								src.ptrs[c],
								offsetDst,
								offsetSrc,
								bytes,
							)
						}
					} else { // !d && !s
						for c := 0; c < dst.NComp(); c++ {
							hostDst := dst.Host()[c]
							hostSrc := src.Host()[c]
							copy(
								hostDst[offsetDst/int64(SIZEOF_FLOAT32):(offsetDst+bytes)/int64(SIZEOF_FLOAT32)],
								hostSrc[offsetSrc/int64(SIZEOF_FLOAT32):(offsetSrc+bytes)/int64(SIZEOF_FLOAT32)],
							)
						}
					}
				}
			}
		}
	} else {
		cudaCopyPart(dst, src, xStart_src, xEnd_src, yStart_src, yEnd_src, zStart_src, zEnd_src, fStart_src, fEnd_src, xStart_dst, yStart_dst, zStart_dst, fStart_dst)
	}
}

/*
func CopyPart(dst, src *Slice, xStart_src, xEnd_src, yStart_src, yEnd_src, zStart_src, zEnd_src, fStart_src, fEnd_src, xStart_dst, yStart_dst, zStart_dst, fStart_dst int, args ...string) {
	if dst.NComp() != src.NComp() {
		panic(fmt.Sprintf("slice copy: illegal sizes: dst: %vx%v, src: %vx%v", dst.NComp(), dst.Len(), src.NComp(), src.Len()))
	}
	//fmt.Println(dst.Len())
	d, s := dst.GPUAccess(), src.GPUAccess()
	sizeSrc := src.Size()
	sizeDst := dst.Size()
	offset_src := int64(xStart_src + sizeSrc[X]*(yStart_src+sizeSrc[Y]*(zStart_src+sizeSrc[Z]*fStart_src)))
	offset_dst := int64(xStart_dst + sizeDst[X]*(yStart_dst+sizeDst[Y]*(zStart_dst+sizeDst[Z]*fStart_dst)))
	xEnd_src -= 1
	yEnd_src -= 1
	zEnd_src -= 1
	fEnd_src -= 1
	bytes := int64(xEnd_src+sizeSrc[X]*(yEnd_src+sizeSrc[Y]*(zEnd_src+sizeSrc[Z]*fEnd_src))) - offset_src
	//fmt.Println("bytes", bytes, "offset", offset)
	//bytes := int64(dst.Len())
	//fmt.Println("bytes", bytes)
	bytes *= int64(SIZEOF_FLOAT32)
	offset_src *= int64(SIZEOF_FLOAT32)
	offset_dst *= int64(SIZEOF_FLOAT32)

	if len(args) > 1 {
		panic("Only one string arg allowed in Copy.")
	}
	switch {
	default:
		panic("bug")
	case d && s:
		for c := 0; c < dst.NComp(); c++ {
			memCpyPart(dst.DevPtr(c), src.DevPtr(c), offset_dst, offset_src, bytes, args...)
		}
	case s && !d:
		for c := 0; c < dst.NComp(); c++ {
			memCpyDtoHPart(dst.ptrs[c], src.DevPtr(c), offset_dst, offset_src, bytes)
		}
	case !s && d:
		for c := 0; c < dst.NComp(); c++ {
			memCpyHtoDPart(dst.DevPtr(c), src.ptrs[c], offset_dst, offset_src, bytes)
		}
	case !d && !s:
		dst, src := dst.Host(), src.Host()
		for c := range dst {
			copy(dst[c][offset_dst/SIZEOF_FLOAT32:(offset_dst+bytes)/SIZEOF_FLOAT32], src[c][offset_src/SIZEOF_FLOAT32:(offset_src+bytes)/SIZEOF_FLOAT32])
		}
	}
}
*/

// Floats returns the data as 3D array,
// indexed by cell position. Data should be
// scalar (1 component) and have CPUAccess() == true.
func (f *Slice) Scalars() [][][]float32 {
	x := f.Tensors()
	if len(x) != 1 {
		panic(fmt.Sprintf("expecting 1 component, got %v", f.NComp()))
	}
	return x[0]
}

// Vectors returns the data as 4D array,
// indexed by component, cell position. Data should have
// 3 components and have CPUAccess() == true.
func (f *Slice) Vectors() [3][][][]float32 {
	x := f.Tensors()
	if len(x) != 3 {
		panic(fmt.Sprintf("expecting 3 components, got %v", f.NComp()))
	}
	return [3][][][]float32{x[0], x[1], x[2]}
}

// Tensors returns the data as 4D array,
// indexed by component, cell position.
// Requires CPUAccess() == true.
func (f *Slice) Tensors() [][][][]float32 {
	tensors := make([][][][]float32, f.NComp())
	host := f.Host()
	for i := range tensors {
		tensors[i] = reshape(host[i], f.Size())
	}
	return tensors
}

// IsNil returns true if either s is nil or s.pointer[0] == nil
func (s *Slice) IsNil() bool {
	if s == nil {
		return true
	}
	return s.ptrs[0] == nil
}

func (s *Slice) String() string {
	if s == nil {
		return "nil"
	}
	var buf bytes.Buffer
	util.Fprint(&buf, s.Tensors())
	return buf.String()
}

func (s *Slice) Set(comp, ix, iy, iz int, value float64) {
	s.checkComp(comp)
	s.Host()[comp][s.Index(ix, iy, iz)] = float32(value)
}

func (s *Slice) SetVector(ix, iy, iz int, v Vector) {
	i := s.Index(ix, iy, iz)
	for c := range v {
		s.Host()[c][i] = float32(v[c])
	}
}

func (s *Slice) SetScalar(ix, iy, iz int, v float64) {
	s.Host()[0][s.Index(ix, iy, iz)] = float32(v)
}

func (s *Slice) Get(comp, ix, iy, iz int) float64 {
	s.checkComp(comp)
	return float64(s.Host()[comp][s.Index(ix, iy, iz)])
}

func (s *Slice) checkComp(comp int) {
	if comp < 0 || comp >= s.NComp() {
		panic(fmt.Sprintf("slice: invalid component index: %v (number of components=%v)\n", comp, s.NComp()))
	}
}

func (s *Slice) Index(ix, iy, iz int) int {
	return Index(s.Size(), ix, iy, iz)
}

func Index(size [3]int, ix, iy, iz int) int {
	if ix < 0 || ix >= size[X] || iy < 0 || iy >= size[Y] || iz < 0 || iz >= size[Z] {
		panic(fmt.Sprintf("Slice index out of bounds: %v,%v,%v (bounds=%v)\n", ix, iy, iz, size))
	}
	return (iz*size[Y]+iy)*size[X] + ix
}
