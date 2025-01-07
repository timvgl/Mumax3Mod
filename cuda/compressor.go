package cuda

// #include <stdint.h>

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)
import "C"

func Compress(output *data.SliceBinary, input *data.Slice, key string) int {
	util.Assert(output.Len() >= input.Len() && output.NComp() == input.NComp())
	var output_size C.size_t
	cfg := make1DConf(input.Len())
	for i := range input.NComp() {
		k_compressRLESingleKernel_async(output.DevPtr(), input.DevPtr(i), (C.size_t)(input.Len()), unsafe.Pointer(&output_size), key, cfg)
	}
	return (int)(output_size)
}
