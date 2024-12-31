package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func FFT_T_Step(dst, src1, src2 *data.SliceBinary, angleReal, angleImag float32, n int) {
	util.Assert(dst.Len() == src1.Len() && src1.Len() == src2.Len() && dst.NComp() == src1.NComp() && src1.NComp() == src2.NComp())
	size := dst.Size()
	cfg := make3DConf(size)
	k_FFT_Step_async(dst.DevPtr(), src1.DevPtr(), src2.DevPtr(), size[X], size[Y], size[Z], angleReal, angleImag, n, cfg)
}
