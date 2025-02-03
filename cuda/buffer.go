package cuda

// Pool of re-usable GPU buffers.
// Synchronization subtlety:
// async kernel launches mean a buffer may already be recycled when still in use.
// That should be fine since the next launch run in the same stream (0), and will
// effectively wait for the previous operation on the buffer.

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
)

var (
	buf_pool  = make(map[int][]unsafe.Pointer)    // pool of GPU buffers indexed by size
	buf_check = make(map[unsafe.Pointer]struct{}) // checks if pointer originates here to avoid unintended recycle
)

var buf_max = 100 // maximum number of buffers to allocate (detect memory leak early)

func IncreaseBufMax(val int) {
	buf_max += val
}
func DecreaseBufMax(val int) {
	buf_max -= val
}

// Returns a GPU slice for temporary use. To be returned to the pool with Recycle
func Buffer(nComp int, size [3]int) *data.Slice {
	if Synchronous {
		Sync()
	}

	ptrs := make([]unsafe.Pointer, nComp)

	// re-use as many buffers as possible form our stack
	N := prod(size)
	pool := buf_pool[N]
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	buf_pool[N] = pool[:len(pool)-nFromPool]

	// allocate as much new memory as needed
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic("too many buffers in use, possible memory leak")
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as mine
	}
	slc := data.SliceFromPtrs(size, data.GPUMemory, ptrs)
	data.DataSliceSlice = append(data.DataSliceSlice, slc)
	return slc
}

func BufferFFT_T(nComp int, size [3]int, key string) *data.Slice {
	if Synchronous {
		SyncFFT_T(key)
	}

	ptrs := make([]unsafe.Pointer, nComp)

	// re-use as many buffers as possible form our stack
	N := prod(size)
	pool := buf_pool[N]
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	buf_pool[N] = pool[:len(pool)-nFromPool]

	// allocate as much new memory as needed
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic(fmt.Sprintf("too many buffers in use, possible memory leak: %v", buf_max))
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as mine
	}

	return data.SliceFromPtrs(size, data.GPUMemory, ptrs)
}

func BufferFFT_T_F(nComp int, size [3]int, fLength int, key string) *data.Slice {
	if Synchronous {
		SyncFFT_T(key)
	}

	ptrs := make([]unsafe.Pointer, nComp)

	// re-use as many buffers as possible form our stack
	N := prod(size) * fLength
	pool := buf_pool[N]
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	buf_pool[N] = pool[:len(pool)-nFromPool]

	// allocate as much new memory as needed
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic(fmt.Sprintf("too many buffers in use, possible memory leak: %v", buf_max))
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as mine
	}

	return data.SliceFromPtrsF(size, fLength, data.GPUMemory, ptrs)
}

// Returns a buffer obtained from GetBuffer to the pool.
func Recycle(s *data.Slice) {
	if Synchronous {
		Sync()
	}
	idx := 0
	gotValue := false
	for i, _ := range data.DataSliceSlice {
		if data.DataSliceSlice[i] == s {
			idx = i
			gotValue = true
			break
		}
	}
	if !gotValue {
		panic("Could not find buffer in buffer list.")
	}
	data.DataSliceSlice = append(data.DataSliceSlice[:idx], data.DataSliceSlice[idx+1:]...)

	N := s.Len()
	pool := buf_pool[N]
	// put each component buffer back on the stack
	for i := 0; i < s.NComp(); i++ {
		ptr := s.DevPtr(i)
		if ptr == unsafe.Pointer(uintptr(0)) {
			continue
		}
		if _, ok := buf_check[ptr]; !ok {
			log.Panic("recyle: was not obtained with getbuffer")
		}
		pool = append(pool, ptr)
	}
	s.Disable() // make it unusable, protect against accidental use after recycle
	buf_pool[N] = pool
}

func Recycle_FFT_T(s *data.Slice, key string) {
	if Synchronous {
		SyncFFT_T(key)
	}

	N := s.Len()
	pool := buf_pool[N]
	// put each component buffer back on the stack
	for i := 0; i < s.NComp(); i++ {
		ptr := s.DevPtr(i)
		if ptr == unsafe.Pointer(uintptr(0)) {
			continue
		}
		if _, ok := buf_check[ptr]; !ok {
			log.Panic("recyle: was not obtained with getbuffer")
		}
		pool = append(pool, ptr)
	}
	s.Disable() // make it unusable, protect against accidental use after recycle
	buf_pool[N] = pool
}

// Frees all buffers. Called after mesh resize.
func FreeBuffers() {
	Sync()
	for _, size := range buf_pool {
		for i := range size {
			cu.DevicePtr(uintptr(size[i])).Free()
			size[i] = nil
		}
	}
	buf_pool = make(map[int][]unsafe.Pointer)
	buf_check = make(map[unsafe.Pointer]struct{})
}
