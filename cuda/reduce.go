package cuda

import (
	"math"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

//#include "reduce.h"
import "C"

// Block size for reduce kernels.
const REDUCE_BLOCKSIZE = C.REDUCE_BLOCKSIZE

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	k_reducesum_async(in.DevPtr(0), out, 0, in.Len(), reducecfg)
	return copyback(out)
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	out := reduceBuf(0)
	// not async over components
	for c := 0; c < nComp; c++ {
		k_reducedot_async(a.DevPtr(c), b.DevPtr(c), out, 0, a.Len(), reducecfg) // all components add to out
	}
	return copyback(out)
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0)
	k_reducemaxabs_async(in.DevPtr(0), out, 0, in.Len(), reducecfg)
	return copyback(out)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
//
//	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	out := reduceBuf(0)
	k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

// // Minimum of the norms of all vectors (x[i], y[i], z[i]).
// // 	min_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MinVecNorm(v *data.Slice) float64 {
	out := reduceBuf(0)
	k_reduceminvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

func MaxvecCellZComp(data, vals, idx, blk_vals, blk_idxs *data.Slice, mesh *data.Mesh) ([]int, float32, float32, float32, float32, float32) {
	outIdxX := reduceBufInt(0)
	outIdxY := reduceBufInt(0)
	outIdxZ := reduceBufInt(0)
	outVal := reduceBuf(0.0)
	outVal_n1X := reduceBuf(0)
	outVal_p1X := reduceBuf(0)
	outVal_n1Y := reduceBuf(0)
	outVal_p1Y := reduceBuf(0)
	blk_num := reduceBufInt(0)
	N := mesh.Size()

	k_max_idx_kernel_async(data.DevPtr(2), vals.DevPtr(0), idx.DevPtr(0), blk_vals.DevPtr(0), blk_idxs.DevPtr(0), blk_num, N[X], N[Y], N[Z], outIdxX, outIdxY, outIdxZ, outVal, outVal_n1X, outVal_p1X, outVal_n1Y, outVal_p1Y, reducecfg)
	outIdx := make([]int, 3)
	outIdx[0] = int(copybackInt(outIdxX))
	outIdx[1] = int(copybackInt(outIdxY))
	outIdx[2] = int(copybackInt(outIdxZ))
	return outIdx, float32(copyback(outVal)), float32(copyback(outVal_n1X)), float32(copyback(outVal_p1X)), float32(copyback(outVal_n1Y)), float32(copyback(outVal_p1Y))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0)
	k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

// Relative Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = ((x1, y1, z1) - (x2, y2, z2))/(x1, y1, z1)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func RelMaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0)
	k_reducerelmaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducecfg)
	return math.Sqrt(float64(copyback(out)))
}

var reduceBuffers chan unsafe.Pointer // pool of 1-float CUDA buffers for reduce

// return a 1-float CUDA reduction buffer from a pool
// initialized to initVal
func reduceBuf(initVal float32) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), math.Float32bits(initVal), 1, stream0)
	return buf
}

func reduceBufFFT_T(initVal float32, key string) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), math.Float32bits(initVal), 1, Get_Stream(key))
	return buf
}

func reduceBufInt(initVal int) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), uint32(initVal), 1, stream0)
	return buf
}

func reduceBufIntFFT_T(initVal int, key string) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), uint32(initVal), 1, Get_Stream(key))
	return buf
}

// copy back single float result from GPU and recycle buffer
func copyback(buf unsafe.Pointer) float32 {
	var result float32
	MemCpyDtoH(unsafe.Pointer(&result), buf, cu.SIZEOF_FLOAT32)
	reduceBuffers <- buf
	return result
}

func copybackInt(buf unsafe.Pointer) int {
	var result int
	MemCpyDtoH(unsafe.Pointer(&result), buf, cu.SIZEOF_INT)
	reduceBuffers <- buf
	return result
}

// initialize pool of 1-float CUDA reduction buffers
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan unsafe.Pointer, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(1 * cu.SIZEOF_FLOAT32)
	}
}

// launch configuration for reduce kernels
// 8 is typ. number of multiprocessors.
// could be improved but takes hardly ~1% of execution time
var reducecfg = &config{Grid: cu.Dim3{X: 8, Y: 1, Z: 1}, Block: cu.Dim3{X: REDUCE_BLOCKSIZE, Y: 1, Z: 1}}
