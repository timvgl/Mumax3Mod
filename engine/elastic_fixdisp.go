package engine

import (
	"fmt"

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
		fmt.Println("heres")
		//Us, _ := U.Slice()
		//defer cuda.Recycle(Us)

		//Set rhs to zero
		cuda.ZeroMask(dst, FrozenDispLoc.gpuLUT1(), regions.Gpu())
		//Set displacment to the given value
		cuda.CopyMask(U.Buffer(), FrozenDispLoc.gpuLUT1(), FrozenDispVal.gpuLUT(), regions.Gpu())
		//Put du also to zero?
	}
}

func FreezeDispRegion(dst, u *data.Slice) {
	if !FrozenDispLoc.isZero() {
		//Us, _ := U.Slice()
		//defer cuda.Recycle(Us)

		//Set rhs to zero
		dstExp := cuda.Buffer(U.NComp(), U.Buffer().Size())
		cuda.Expand(dstExp, dst, (Mesh().Size()[X]-u.Size()[X])/2, (Mesh().Size()[Y]-u.Size()[Y])/2, (Mesh().Size()[Z]-u.Size()[Z])/2, (dst.StartX - (dst.Size()[X] - dst.EndX)), (dst.StartY - (dst.Size()[Y] - dst.EndY)), (dst.StartZ - (dst.Size()[Z] - dst.EndZ)), []float64{0})
		cuda.ZeroMask(dstExp, FrozenDispLoc.gpuLUT1(), regions.Gpu())
		cuda.Crop(dst, dstExp, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(dstExp)
		//Set displacment to the given value
		UExp := cuda.Buffer(U.NComp(), U.Buffer().Size())
		cuda.Expand(UExp, u, (Mesh().Size()[X]-u.Size()[X])/2, (Mesh().Size()[Y]-u.Size()[Y])/2, (Mesh().Size()[Z]-u.Size()[Z])/2, (dst.StartX - (dst.Size()[X] - dst.EndX)), (dst.StartY - (dst.Size()[Y] - dst.EndY)), (dst.StartZ - (dst.Size()[Z] - dst.EndZ)), []float64{0})
		/*UDebug := UExp.HostCopy()
		val, ok := autonum["debug"]
		if !ok {
			autonum["debug"] = 0
			val = 0
		}
		meta := *new(data.Meta)
		meta.Time = Time
		meta.CellSize = Mesh().CellSize()
		meta.Name = "debug"
		queOutput(func() { saveAs_sync(fmt.Sprintf(OD()+"/debug_%d", val), UDebug, meta, outputFormat) })*/
		cuda.CopyMask(UExp, FrozenDispLoc.gpuLUT1(), FrozenDispVal.gpuLUT(), regions.Gpu())
		cuda.Crop(u, UExp, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Recycle(UExp)
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
