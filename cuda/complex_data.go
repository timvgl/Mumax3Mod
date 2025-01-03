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

	for c := range dst.NComp() {
		k_real_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func Imag(dst, src *data.Slice) {
	dstNxNyNz := dst.Size()
	srcNxNyNz := src.Size()
	util.Argument(dstNxNyNz[0] == int(srcNxNyNz[0]/2) && dstNxNyNz[1] == srcNxNyNz[1] && dstNxNyNz[2] == srcNxNyNz[2])

	N := dst.Len()
	cfg := make1DConf(N)
	for c := range dst.NComp() {
		k_imag_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}
