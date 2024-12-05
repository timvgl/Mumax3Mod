package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func Real(dst, src *data.Slice) {
	dstNxNyNz := dst.Size()
	srcNxNyNz := src.Size()
	util.Argument(dstNxNyNz[0] == int(srcNxNyNz[0]/2) && dstNxNyNz[1] == srcNxNyNz[1] && dstNxNyNz[2] == srcNxNyNz[2])

	N := dst.Len()
	cfg := make1DConf(N)

	k_real_async(dst.DevPtr(X), src.DevPtr(X), N, cfg)
	k_real_async(dst.DevPtr(Y), src.DevPtr(Y), N, cfg)
	k_real_async(dst.DevPtr(Z), src.DevPtr(Z), N, cfg)
}

func Imag(dst, src *data.Slice) {
	dstNxNyNz := dst.Size()
	srcNxNyNz := src.Size()
	util.Argument(dstNxNyNz[0] == int(srcNxNyNz[0]/2) && dstNxNyNz[1] == srcNxNyNz[1] && dstNxNyNz[2] == srcNxNyNz[2])

	N := dst.Len()
	cfg := make1DConf(N)

	k_imag_async(dst.DevPtr(X), src.DevPtr(X), N, cfg)
	k_imag_async(dst.DevPtr(Y), src.DevPtr(Y), N, cfg)
	k_imag_async(dst.DevPtr(Z), src.DevPtr(Z), N, cfg)
}
