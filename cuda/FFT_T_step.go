package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func FFT_T_Step_MEM_Complex(dst, src1, src2 *data.Slice, minF, dF, fftT float32, key string) {
	size := dst.Size()
	util.Assert(src1.NComp() == src2.NComp() && src1.NComp() == dst.NComp())
	util.Assert(src1.Size() == src2.Size() && src1.Size() == dst.Size())
	util.Assert(dst.LengthF == src1.LengthF)
	//fmt.Println("size:", size, "LengthF:", src1.LengthF)
	cfg := make1DConf(prod(size) / 2)
	for c := range dst.NComp() {
		k_FFT_Step_MEM_Complex_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), size[X], size[Y], size[Z], dst.LengthF, minF, dF, fftT, float32(dst.LengthF), key, cfg)
	}
	//SyncFFT_T(key)
}

func FFT_T_Step_MEM_Real(dst, src1, src2 *data.Slice, minF, dF, fftT float32, key string) {
	size := dst.Size()
	size[X] /= 2
	util.Assert(src1.NComp() == src2.NComp() && src1.NComp() == dst.NComp())
	util.Assert(src2.Size() == size && src1.Size() == dst.Size())
	util.Assert(dst.LengthF == src1.LengthF)
	//fmt.Println("size:", size, "LengthF:", src1.LengthF)
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_FFT_Step_MEM_Real_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), 2*size[X], size[Y], size[Z], dst.LengthF, minF, dF, fftT, float32(dst.LengthF), key, cfg)
	}
	//SyncFFT_T(key)
}

func FFT_T_Step_Complex(dst, src1, src2 *data.Slice, phase float32, n int, key string) {
	size := dst.Size()
	size[X] /= 2
	util.Assert(src1.NComp() == src2.NComp() && src1.NComp() == dst.NComp())
	util.Assert(src1.Size() == src2.Size() && src1.Size() == dst.Size())
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_FFT_Step_Complex_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), size[X], size[Y], size[Z], phase, float32(n), key, cfg)
	}
}

func FFT_T_Step_Real(dst, src1, src2 *data.Slice, phase float32, n int, key string) {
	size := dst.Size()
	size[X] /= 2
	util.Assert(src1.NComp() == src2.NComp() && src1.NComp() == dst.NComp())
	util.Assert(src2.Size() == size && src1.Size() == dst.Size())
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_FFT_Step_Real_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), size[X], size[Y], size[Z], phase, float32(n), key, cfg)
	}
}
