package cuda

// Pool of re-usable GPU buffers.
// Synchronization subtlety:
// async kernel launches mean a buffer may already be recycled when still in use.
// That should be fine since the next launch run in the same stream (0), and will
// effectively wait for the previous operation on the buffer.

import (
	"fmt"
	"log"
	"sync"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
)

var (
	// buf_pool is now a sync.Map mapping buffer sizes (int) to a slice of unsafe.Pointer.
	buf_pool  sync.Map
	buf_check = make(map[unsafe.Pointer]struct{}) // checks if pointer originates here to avoid unintended recycle
)

var buf_max = 250 // maximum number of buffers to allocate (detect memory leak early)

func IncreaseBufMax(val int) {
	buf_max += val
}
func DecreaseBufMax(val int) {
	buf_max -= val
}

// Returns a GPU slice for temporary use. To be returned to the pool with Recycle.
func Buffer(nComp int, size [3]int) *data.Slice {
	if Synchronous {
		Sync()
	}

	ptrs := make([]unsafe.Pointer, nComp)
	N := prod(size)

	// Try to re-use as many buffers as possible from our pool.
	var pool []unsafe.Pointer
	if rawPool, ok := buf_pool.Load(N); ok {
		pool = rawPool.([]unsafe.Pointer)
	}
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	// Remove the used pointers and store back the remaining pool.
	pool = pool[:len(pool)-nFromPool]
	buf_pool.Store(N, pool)

	// Allocate new memory for any remaining components.
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic("too many buffers in use, possible memory leak")
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as ours
	}
	slc := data.SliceFromPtrs(size, data.GPUMemory, ptrs)
	return slc
}

func PrintBufLength() {
	fmt.Println("Buffer pool lengths:", len(buf_check))
}

func BufferFFT_T(nComp int, size [3]int, key string) *data.Slice {
	if Synchronous {
		SyncFFT_T(key)
	}

	ptrs := make([]unsafe.Pointer, nComp)
	N := prod(size)

	// Re-use as many buffers as possible from the pool.
	var pool []unsafe.Pointer
	if rawPool, ok := buf_pool.Load(N); ok {
		pool = rawPool.([]unsafe.Pointer)
	}
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	pool = pool[:len(pool)-nFromPool]
	buf_pool.Store(N, pool)

	// Allocate new memory as needed.
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic(fmt.Sprintf("too many buffers in use, possible memory leak: %v", buf_max))
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as ours
	}

	return data.SliceFromPtrs(size, data.GPUMemory, ptrs)
}

func BufferFFT_T_F(nComp int, size [3]int, fLength int, key string) *data.Slice {
	if Synchronous {
		SyncFFT_T(key)
	}

	ptrs := make([]unsafe.Pointer, nComp)
	N := prod(size) * fLength

	// Re-use as many buffers as possible from the pool.
	var pool []unsafe.Pointer
	if rawPool, ok := buf_pool.Load(N); ok {
		pool = rawPool.([]unsafe.Pointer)
	}
	nFromPool := iMin(nComp, len(pool))
	for i := 0; i < nFromPool; i++ {
		ptrs[i] = pool[len(pool)-i-1]
	}
	pool = pool[:len(pool)-nFromPool]
	buf_pool.Store(N, pool)

	// Allocate new memory as needed.
	for i := nFromPool; i < nComp; i++ {
		if len(buf_check) >= buf_max {
			log.Panic(fmt.Sprintf("too many buffers in use, possible memory leak: %v", buf_max))
		}
		ptrs[i] = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N))
		buf_check[ptrs[i]] = struct{}{} // mark this pointer as ours
	}

	return data.SliceFromPtrsF(size, fLength, data.GPUMemory, ptrs)
}

// Returns a buffer obtained from Buffer back to the pool.
func Recycle(s *data.Slice) {
	if Synchronous {
		Sync()
	}

	N := s.Len()
	var pool []unsafe.Pointer
	if rawPool, ok := buf_pool.Load(N); ok {
		pool = rawPool.([]unsafe.Pointer)
	}
	// Put each component buffer back on the pool.
	for i := 0; i < s.NComp(); i++ {
		ptr := s.DevPtr(i)
		if ptr == unsafe.Pointer(uintptr(0)) {
			continue
		}
		if _, ok := buf_check[ptr]; !ok {
			log.Panic("recycle: was not obtained with getbuffer")
		}
		pool = append(pool, ptr)
	}
	s.Disable() // Make the slice unusable after recycle.
	buf_pool.Store(N, pool)
}

func Recycle_FFT_T(s *data.Slice, key string) {
	if Synchronous {
		SyncFFT_T(key)
	}

	N := s.Len()
	var pool []unsafe.Pointer
	if rawPool, ok := buf_pool.Load(N); ok {
		pool = rawPool.([]unsafe.Pointer)
	}
	// Return each component buffer to the pool.
	for i := 0; i < s.NComp(); i++ {
		ptr := s.DevPtr(i)
		if ptr == unsafe.Pointer(uintptr(0)) {
			continue
		}
		if _, ok := buf_check[ptr]; !ok {
			log.Panic("recycle: was not obtained with getbuffer")
		}
		pool = append(pool, ptr)
	}
	s.Disable() // Disable the slice.
	buf_pool.Store(N, pool)
}

// Frees all buffers. Called after mesh resize.
func FreeBuffers() {
	Sync()
	// Iterate over all entries in the pool.
	buf_pool.Range(func(key, value interface{}) bool {
		pool := value.([]unsafe.Pointer)
		for i := range pool {
			cu.DevicePtr(uintptr(pool[i])).Free()
			pool[i] = nil
		}
		buf_pool.Delete(key)
		return true
	})
	buf_check = make(map[unsafe.Pointer]struct{})
}
