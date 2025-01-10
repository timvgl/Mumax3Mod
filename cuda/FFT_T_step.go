package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func FFT_T_Step(dst, src1, src2 *data.Slice, phase float32, n int, key string) {
	size := dst.Size()
	util.Assert(src1.NComp() == src2.NComp() && src1.NComp() == dst.NComp())
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_FFT_Step_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), size[X], size[Y], size[Z], phase, float32(n), key, cfg)
	}
}
