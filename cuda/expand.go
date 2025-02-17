package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Crop stores in dst a rectangle cropped from src at given offset position.
// dst size may be smaller than src.
func Expand(dst, src *data.Slice, offX, offY, offZ int, value []float64) {
	D := dst.Size()
	S := src.Size()
	util.Argument(dst.NComp() == src.NComp())
	util.Argument((D[X]-2*offX == S[X] || D[X]-2*offX-1 == S[X]) && (D[Y]-2*offY == S[Y] || D[Y]-2*offY-1 == S[Y]) && (D[Z]-2*offZ == S[Z] || D[Z]-2*offZ-1 == S[Z]))
	util.Argument(len(value) == dst.NComp())

	cfg := make3DConf(D)

	for c := 0; c < dst.NComp(); c++ {
		k_expand_async(dst.DevPtr(c), D[X], D[Y], D[Z],
			src.DevPtr(c), S[X], S[Y], S[Z],
			offX, offY, offZ, float32(value[c]), cfg)
	}
}
