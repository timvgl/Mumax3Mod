package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func Real(dst, src *data.Slice) {
	dstNxNyNz := dst.Size()
	srcNxNyNz := src.Size()
	util.Argument(dstNxNyNz[0] == int(srcNxNyNz[0]/2) && dstNxNyNz[1] == srcNxNyNz[1] && dstNxNyNz[2] == srcNxNyNz[2])
	util.Argument(dst.NComp() == src.NComp())

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
	util.Argument(dst.NComp() == src.NComp())

	N := dst.Len()
	cfg := make1DConf(N)
	for c := range dst.NComp() {
		k_imag_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func ComplexToPolar(dst, src *data.Slice) {
	dstNxNyNz := dst.Size()
	srcNxNyNz := src.Size()
	util.Argument(dstNxNyNz == srcNxNyNz)
	util.Argument(dst.NComp() == src.NComp())

	N := dst.Len()
	cfg := make1DConf(dst.LengthF * N / 2)
	for c := range dst.NComp() {
		k_complexToPolar_async(dst.DevPtr(c), src.DevPtr(c), dst.LengthF*N/2, cfg)
	}
}

func ComplexConjugate(dst, src *data.Slice) {
	dstNxNyNz := dst.Size()
	srcNxNyNz := src.Size()
	util.Argument(dstNxNyNz == srcNxNyNz)
	util.Argument(dst.NComp() == src.NComp())

	N := dst.Len()
	cfg := make1DConf(dst.LengthF * N / 2)
	for c := range dst.NComp() {
		k_complexConjugate_async(dst.DevPtr(c), src.DevPtr(c), dst.LengthF*N/2, cfg)
	}
}

func ReverseX(dst, src *data.Slice) {
	size := dst.Size()
	size[0] /= 4
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_reverseX_async(dst.DevPtr(c), src.DevPtr(c), size[X], size[Y], size[Z], cfg)
	}
}
