package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	SecondDerivDisp = NewVectorField("force", "", "Force/volume", calcSecondDerivDisp)
	C11             = NewScalarParam("C11", "N/m2", "Stiffness constant C11")
	C12             = NewScalarParam("C12", "N/m2", "Stiffness constant C12")
	C13             = NewScalarParam("C13", "N/m2", "Stiffness constant C13")
	C33             = NewScalarParam("C33", "N/m2", "Stiffness constant C33")
	C44             = NewScalarParam("C44", "N/m2", "Stiffness constant C44")
	C66             = NewScalarParam("C66", "N/m2", "Stiffness constant C66")
	mtxElasto       = false
	cubicElasto     = true
)

func init() {
	DeclVar("mtxElasto", &mtxElasto, "calc elastics in matrix")
}

func calcSecondDerivDisp(dst *data.Slice) {
	SecondDerivative(dst, U, M, C11, C12, C44, C13, C33, B1, B2)
}

func calcSecondDerivDispRegion(dst, u *data.Slice) {
	if useFullSample {
		backupU := cuda.Buffer(U.Buffer().NComp(), U.Buffer().Size())
		data.Copy(backupU, U.Buffer())
		data.CopyPart(U.Buffer(), u, 0, u.Size()[X], 0, u.Size()[Y], 0, u.Size()[Z], 0, 1, dst.StartX, dst.StartY, dst.StartZ, 0)
		defer func() {
			data.Copy(U.Buffer(), backupU)
			cuda.Recycle(backupU)
		}()
	}
	SecondDerivativeRegion(dst, u, C11, C12, C44, C13, C33)
}

func SecondDerivative(dst *data.Slice, u displacement, m magnetization, C11, C12, C44, C13, C33, B1, B2 *RegionwiseScalar) {
	c11 := C11.MSlice()
	defer c11.Recycle()

	c12 := C12.MSlice()
	defer c12.Recycle()

	c13 := C13.MSlice()
	defer c13.Recycle()

	c33 := C33.MSlice()
	defer c33.Recycle()

	c44 := C44.MSlice()
	defer c44.Recycle()

	b1 := B1.MSlice()
	defer b1.Recycle()

	b2 := B2.MSlice()
	defer b2.Recycle()
	if !mtxElasto {
		cuda.SecondDerivative(dst, u.Buffer(), U.Mesh(), c11, c12, c44)
	} else {
		normStressTmp := cuda.Buffer(3, dst.Size())
		defer cuda.Recycle(normStressTmp)
		cuda.Zero(normStressTmp)
		shearStressTmp := cuda.Buffer(3, dst.Size())
		defer cuda.Recycle(shearStressTmp)
		cuda.Zero(shearStressTmp)
		cuda.StressWurtzitMtx(normStressTmp, shearStressTmp, u.Buffer(), m.Buffer(), c11, c12, c13, c33, c44, b1, b2, U.Mesh(), cubicElasto)
		cuda.ForceWurtzitMtx(dst, normStressTmp, shearStressTmp, U.Mesh())
	}
}

func SecondDerivativeRegion(dst, u *data.Slice, C11, C12, C44, C13, C33 *RegionwiseScalar) {
	if !useFullSample {
		c11 := C11.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c11.Recycle()

		c12 := C12.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c12.Recycle()

		c44 := C44.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		defer c44.Recycle()
		cuda.SecondDerivative(dst, u, Crop(&U, dst.StartX, dst.EndX, dst.StartY, dst.EndY, dst.StartZ, dst.EndZ).Mesh(), c11, c12, c44)
	} else {
		ddU := cuda.Buffer(M.NComp(), M.Buffer().Size())
		cuda.Zero(ddU)
		SecondDerivative(ddU, U, M, C11, C12, C44, C13, C33, B1, B2)
		ddUCropped := cuda.Buffer(dst.NComp(), dst.Size())
		cuda.Crop(ddUCropped, ddU, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, ddUCropped)
		cuda.Recycle(ddUCropped)
		cuda.Recycle(ddU)
	}
}
