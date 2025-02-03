package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	SecondDerivDisp = NewVectorField("force", "", "Force/volume", calcSecondDerivDisp)
	C11             = NewScalarParam("C11", "N/m2", "Stiffness constant C11")
	C12             = NewScalarParam("C12", "N/m2", "Stiffness constant C12")
	C44             = NewScalarParam("C44", "N/m2", "Stiffness constant C44")
)

func calcSecondDerivDisp(dst *data.Slice) {
	SecondDerivative(dst, U, C11, C12, C44)
}

func calcSecondDerivDispRegion(dst *data.Slice) {
	SecondDerivativeRegion(dst, U, C11, C12, C44)
}

func SecondDerivative(dst *data.Slice, u displacement, C1, C2, C3 *RegionwiseScalar) {
	c1 := C1.MSlice()
	defer c1.Recycle()

	c2 := C2.MSlice()
	defer c2.Recycle()

	c3 := C3.MSlice()
	defer c3.Recycle()
	cuda.SecondDerivative(dst, u.Buffer(), U.Mesh(), c1, c2, c3)
}

func SecondDerivativeRegion(dst *data.Slice, u displacement, C1, C2, C3 *RegionwiseScalar) {
	c1 := C1.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
	defer c1.Recycle()

	c2 := C2.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
	defer c2.Recycle()

	c3 := C3.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
	defer c3.Recycle()
	uRed := cuda.Buffer(dst.NComp(), dst.RegionSize())
	defer cuda.Recycle(uRed)
	cuda.Crop(uRed, u.Buffer(), dst.StartX, dst.StartY, dst.StartZ)
	cuda.SecondDerivative(dst, uRed, Crop(&U, dst.StartX, dst.EndX, dst.StartY, dst.EndY, dst.StartZ, dst.EndZ).Mesh(), c1, c2, c3)
}
