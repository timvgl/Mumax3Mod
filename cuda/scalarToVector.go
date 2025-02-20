package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func ScalarToVec(out *data.Slice, a, b, c MSlice, mesh *data.Mesh) {
	util.Argument(out.Size() == mesh.Size())
    util.Assert(out.NComp() == 3)
    util.Assert(out.Size() == a.Size())
    util.Assert(out.Size() == b.Size())
    util.Assert(out.Size() == c.Size())

	N := mesh.Size()
	cfg := make3DConf(N)
	k_scalarToVector_async(out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z), a.DevPtr(0), a.Mul(0), b.DevPtr(0), b.Mul(0), c.DevPtr(X), c.Mul(0), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}