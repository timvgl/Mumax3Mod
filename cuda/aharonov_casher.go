package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// AddAharonovCasherField adds the Aharonov-Casher effective field (Tesla) to B.
//
// The AC effect is the SO(3) gauge connection A_alpha = xi*(gamma/c^2)*(e_alpha x E)
// that a magnetic moment picks up per unit length when transported through an electric
// field. Minimal coupling d_alpha m -> d_alpha m + A_alpha x m in the exchange energy
// yields the additive field computed here,
//
//	B_AC = (2 Aex / Ms) * ( D^2 - laplacian ) m
//
// which shifts the exchange magnon parabola to k = k_AC = xi*(gamma/c^2)*(q x E),
// adds an interfacial-DMI-like nonreciprocity with D_AC = -2*Aex*xi*(gamma/c^2)*|E|,
// and a hard axis along E at order E^2.
//
// Arguments:
//
//	B       : effective field, added to
//	m       : normalized magnetization
//	E       : electric field, V/m
//	xi      : dimensionless AC coupling (screening/enhancement), region-wise
//	q       : magnon quantization axis; only read when projectAxis is true
//	Aex_red : inter-region exchange stiffness LUT (the same table as plain exchange)
//	acConst : gamma/c^2, in m^-1 per V/m
//	projectAxis: project the connection onto q (U(1) linear-spin-wave mode)
//	openBC  : drop boundary bonds instead of applying the covariant Neumann condition
//
// see aharonov_casher.cu
func AddAharonovCasherField(B, m, E *data.Slice, xi, q, Msat MSlice, Aex_red SymmLUT,
	regions *Bytes, mesh *data.Mesh, acConst float32, projectAxis, openBC bool) {

	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	k_addaharonovcasher_async(
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		E.DevPtr(X), E.DevPtr(Y), E.DevPtr(Z),
		xi.DevPtr(0), xi.Mul(0),
		q.DevPtr(X), q.Mul(X), q.DevPtr(Y), q.Mul(Y), q.DevPtr(Z), q.Mul(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		unsafe.Pointer(Aex_red), regions.Ptr,
		wx, wy, wz,
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		N[X], N[Y], N[Z],
		acConst, boolToByte(projectAxis), boolToByte(openBC), pbc, cfg)
}

func boolToByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}
