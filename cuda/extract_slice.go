package cuda

import (
	"math"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func ExtractSlice(dst, src *data.Slice, iX, iY, iZ float64, key string) {
	S := src.Size()
	util.Argument(dst.NComp() == src.NComp())
	if math.IsNaN(iX) {
		yInt := int(iY)
		zInt := int(iZ)
		util.Argument(yInt < S[Y] && zInt < S[Z])
		cfg := make1DConf(dst.Len())
		for c := 0; c < dst.NComp(); c++ {
			k_extractXSlice_async(dst.DevPtr(c),
				src.DevPtr(c), S[X], S[Y], S[Z],
				yInt, zInt, key, cfg)
		}
	} else if math.IsNaN(iY) {
		xInt := int(iX)
		zInt := int(iZ)
		util.Argument(xInt < S[X] && zInt < S[Z])
		cfg := make1DConf(dst.Len())
		for c := 0; c < dst.NComp(); c++ {
			k_extractYSlice_async(dst.DevPtr(c),
				src.DevPtr(c), S[X], S[Y], S[Z],
				xInt, zInt, key, cfg)
		}
	} else if math.IsNaN(iZ) {
		xInt := int(iX)
		yInt := int(iY)
		util.Argument(xInt < S[X] && yInt < S[Y])
		cfg := make1DConf(dst.Len())
		for c := 0; c < dst.NComp(); c++ {
			k_extractZSlice_async(dst.DevPtr(c),
				src.DevPtr(c), S[X], S[Y], S[Z],
				xInt, yInt, key, cfg)
		}
	} else {
		util.Argument(false)
	}
}
