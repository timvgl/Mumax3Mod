package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

import "C"

func FFT_T_Step_Partial_Compressed(dst, src *data.SliceBinary, phase float32, compressedSize int, key string) {
	cfg := make1DConf(compressedSize)
	k_transformCompressedComplexKernel_async(dst.DevPtr(), src.DevPtr(), phase, (C.size_t)(compressedSize), key, cfg)
}

func MAdd2_Compressed(dst, src1, src2 *data.SliceBinary, fac1, fac2 float32, compressedSize1, compressedSize2 int, key string) int {
	var compressedSize int
	if compressedSize1 > compressedSize {
		compressedSize = compressedSize1
	} else {
		compressedSize = compressedSize2
	}
	cfg := make1DConf(compressedSize)
	var output_size C.size_t

	k_Madd2CompressedArraysKernel_async(dst.DevPtr(), (C.size_t)(dst.Length()), src1.DevPtr(), (C.size_t)(src1.Length()), src2.DevPtr(), (C.size_t)(src2.Length()), fac1, fac2, unsafe.Pointer(&output_size), key, cfg)
	return (int)(output_size)
}

func FFT_T_Step(dst, src1, src2 *data.SliceBinary, phase float32, n int, key string) int {
	util.Assert(dst.Len() == src1.Len() && src1.Len() == src2.Len() && dst.NComp() == src1.NComp() && src1.NComp() == src2.NComp())
	FFT_T_Step_Partial_Compressed(src2, src2, phase, src2.Length(), key)
	return MAdd2_Compressed(dst, src1, src2, 1, 1/float32(n), src1.Length(), src2.Length(), key)
	//size := dst.Size()
	//cfg := make3DConf(size)
	//k_FFT_Step_async(dst.DevPtr(), src1.DevPtr(), src2.DevPtr(), size[X], size[Y], size[Z], angleReal, angleImag, n, cfg)
}
