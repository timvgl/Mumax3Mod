package cuda

import "github.com/mumax/3/data"

func FFT_T_Step(dst, src1, src2 *data.Slice, phase float32, n int, key string) {
	size := dst.Size()
	cfg := make3DConf(size)
	k_FFT_Step_async(dst.DevPtr(0), src1.DevPtr(0), src2.DevPtr(0), size[X], size[Y], size[Z], phase, float32(n), key, cfg)
}
