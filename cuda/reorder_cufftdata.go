package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func ReorderCufftData(output, input *data.Slice) {
	util.Assert(output.Size() == input.Size() && output.NComp() == input.NComp())
	size := output.Size()
	cfg := make3DConf(size)
	if input.NComp() == 1 {
		k_fftshift3D_partial_async(output.DevPtr(0), input.DevPtr(0), size[X], size[Y], size[Z], cfg)
	} else if input.NComp() == 3 {
		for c := range 3 {
			k_fftshift3D_partial_async(output.DevPtr(c), input.DevPtr(c), size[X], size[Y], size[Z], cfg)
		}
	} else {
		panic("Data has to have one or three component(s) if data is supposed to be reordered.")
	}
}
