package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func calcBndry() {
	Bndry(U, C11, C12)
}

func calcBndryRegion(u *data.Slice, pbcX, pbcY, pbcZ int) {
	BndryRegion(u, C11, C12, pbcX, pbcY, pbcZ)
}

func Bndry(u displacement, C1, C2 *RegionwiseScalar) {
	// c1 := C1.MSlice()
	// defer c1.Recycle()

	// c2 := C2.MSlice()
	// defer c2.Recycle()
	c1 := float32(C1.getRegion(2)[0])
	c2 := float32(C2.getRegion(2)[0])
	cuda.Bndry(u.Buffer(), U.Mesh(), c1, c2)
}

func BndryRegion(u *data.Slice, C1, C2 *RegionwiseScalar, pbcX, pbcY, pbcZ int) {
	// c1 := C1.MSlice()
	// defer c1.Recycle()

	// c2 := C2.MSlice()
	// defer c2.Recycle()
	c1 := float32(C1.getRegion(2)[0])
	c2 := float32(C2.getRegion(2)[0])
	cuda.Bndry(u, data.NewMesh(u.RegionSize()[X], u.RegionSize()[Y], u.RegionSize()[Z], MeshOf(&M).CellSize()[X], MeshOf(&M).CellSize()[Y], MeshOf(&M).CellSize()[Z], pbcX, pbcY, pbcZ), c1, c2)
}
