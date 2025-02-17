package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
var (
	FrozenDispLoc = NewScalarParam("frozenDispLoc", "", "Defines displacment region that should be fixed")
	FrozenDispVal = NewVectorParam("frozenDispVal", "", "Defines fixed displacement value")
)

func FreezeDisp(dst *data.Slice) {
	if !FrozenDispLoc.isZero() {
		//Us, _ := U.Slice()
		//defer cuda.Recycle(Us)

		//Set rhs to zero
		cuda.ZeroMask(dst, FrozenDispLoc.gpuLUT1(), regions.Gpu())
		//Set displacment to the given value
		cuda.CopyMask(U.Buffer(), FrozenDispLoc.gpuLUT1(), FrozenDispVal.gpuLUT(), regions.Gpu())
		//Put du also to zero?
	}
}

func FreezeDispRegion(dst *data.Slice) {
	if !FrozenDispLoc.isZero() {
		//Us, _ := U.Slice()
		//defer cuda.Recycle(Us)

		//Set rhs to zero
		cuda.ZeroMask(dst, FrozenDispLoc.gpuLUT1(), regions.Gpu())
		//Set displacment to the given value
		u := cuda.Buffer(dst.NComp(), dst.RegionSize())
		cuda.Crop(u, U.Buffer(), dst.StartX, dst.StartY, dst.StartZ)
		cuda.CopyMask(u, FrozenDispLoc.gpuLUT1(), FrozenDispVal.gpuLUT(), regions.Gpu())
		//Put du also to zero?
	}
}

func SetFreezeDisp() {
	if !FrozenDispLoc.isZero() {
		//Set displacment to the given value
		cuda.CopyMask(U.Buffer(), FrozenDispLoc.gpuLUT1(), FrozenDispVal.gpuLUT(), regions.Gpu())
		//Put du also to zero?
	}
}
