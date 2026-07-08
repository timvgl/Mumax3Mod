package cuda

// Launchers for the magnum.np-port (NP) fully coupled magnetoelastic kernels.
// See the corresponding .cu files and PORT_NOTES_MAGNUMNP.md for the numerics.

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// helper: device pointer of component c, or nil if the slice is nil
func devPtrOrNil(s *data.Slice, c int) unsafe.Pointer {
	if s == nil {
		return nil
	}
	return s.DevPtr(c)
}

// MelasStiffness bundles the 9 independent stiffness components of the block-diagonal
// Voigt stiffness matrix (paper Eq. 24) as MSlices.
type MelasStiffness struct {
	C11, C12, C13, C22, C23, C33, C44, C55, C66 MSlice
}

// Jump-aware first derivative of component fcomp of f along x, y, z into g (3 components).
// Cx/Cy/Cz: jump coefficients per direction. Bl*/Br*: B jump data per direction (component
// bcomp of each 3-component slice, may be nil). w: window bounds [x0,x1,y0,y1,z0,z1).
func MelasGradJumpNP(g, f *data.Slice, fcomp int, Cx, Cy, Cz MSlice,
	Blx, Brx, Bly, Bry, Blz, Brz *data.Slice, bcomp int,
	w [6]int, mesh *data.Mesh, so bool) {
	N := mesh.Size()
	c := mesh.CellSize()
	cfg := make3DConf(N)
	soInt := 0
	if so {
		soInt = 1
	}
	k_MelasGradJumpNP_async(g.DevPtr(X), g.DevPtr(Y), g.DevPtr(Z),
		f.DevPtr(fcomp),
		Cx.DevPtr(0), Cx.Mul(0), Cy.DevPtr(0), Cy.Mul(0), Cz.DevPtr(0), Cz.Mul(0),
		devPtrOrNil(Blx, bcomp), devPtrOrNil(Brx, bcomp),
		devPtrOrNil(Bly, bcomp), devPtrOrNil(Bry, bcomp),
		devPtrOrNil(Blz, bcomp), devPtrOrNil(Brz, bcomp),
		w[0], w[1], w[2], w[3], w[4], w[5],
		N[X], N[Y], N[Z],
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		soInt, mesh.PBC_code(), cfg)
}

// Magnetic-stress part of the B jump parameters from m and its exchange-jump gradients.
// Outputs Bl/Br per direction, components indexed by displacement component.
func MelasBSigMNP(Blx, Brx, Bly, Bry, Blz, Brz *data.Slice,
	m, gmx, gmy, gmz *data.Slice,
	l100, l111 MSlice, C MelasStiffness, mesh *data.Mesh) {
	N := mesh.Size()
	c := mesh.CellSize()
	cfg := make3DConf(N)
	k_MelasBSigMNP_async(
		Blx.DevPtr(X), Blx.DevPtr(Y), Blx.DevPtr(Z),
		Brx.DevPtr(X), Brx.DevPtr(Y), Brx.DevPtr(Z),
		Bly.DevPtr(X), Bly.DevPtr(Y), Bly.DevPtr(Z),
		Bry.DevPtr(X), Bry.DevPtr(Y), Bry.DevPtr(Z),
		Blz.DevPtr(X), Blz.DevPtr(Y), Blz.DevPtr(Z),
		Brz.DevPtr(X), Brz.DevPtr(Y), Brz.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		gmx.DevPtr(X), gmx.DevPtr(Y), gmx.DevPtr(Z),
		gmy.DevPtr(X), gmy.DevPtr(Y), gmy.DevPtr(Z),
		gmz.DevPtr(X), gmz.DevPtr(Y), gmz.DevPtr(Z),
		l100.DevPtr(0), l100.Mul(0), l111.DevPtr(0), l111.Mul(0),
		C.C11.DevPtr(0), C.C11.Mul(0), C.C12.DevPtr(0), C.C12.Mul(0), C.C13.DevPtr(0), C.C13.Mul(0),
		C.C22.DevPtr(0), C.C22.Mul(0), C.C23.DevPtr(0), C.C23.Mul(0), C.C33.DevPtr(0), C.C33.Mul(0),
		C.C44.DevPtr(0), C.C44.Mul(0), C.C55.DevPtr(0), C.C55.Mul(0), C.C66.DevPtr(0), C.C66.Mul(0),
		N[X], N[Y], N[Z],
		float32(c[X]), float32(c[Y]), float32(c[Z]), cfg)
}

// Iterative B update: B = B_sigM + harmonic-mean displacement-gradient part.
func MelasBEpsNP(Blx, Brx, Bly, Bry, Blz, Brz *data.Slice,
	SlBlx, SlBrx, SlBly, SlBry, SlBlz, SlBrz *data.Slice,
	gux, guy, guz *data.Slice,
	C MelasStiffness, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasBEpsNP_async(
		Blx.DevPtr(X), Blx.DevPtr(Y), Blx.DevPtr(Z),
		Brx.DevPtr(X), Brx.DevPtr(Y), Brx.DevPtr(Z),
		Bly.DevPtr(X), Bly.DevPtr(Y), Bly.DevPtr(Z),
		Bry.DevPtr(X), Bry.DevPtr(Y), Bry.DevPtr(Z),
		Blz.DevPtr(X), Blz.DevPtr(Y), Blz.DevPtr(Z),
		Brz.DevPtr(X), Brz.DevPtr(Y), Brz.DevPtr(Z),
		SlBlx.DevPtr(X), SlBlx.DevPtr(Y), SlBlx.DevPtr(Z),
		SlBrx.DevPtr(X), SlBrx.DevPtr(Y), SlBrx.DevPtr(Z),
		SlBly.DevPtr(X), SlBly.DevPtr(Y), SlBly.DevPtr(Z),
		SlBry.DevPtr(X), SlBry.DevPtr(Y), SlBry.DevPtr(Z),
		SlBlz.DevPtr(X), SlBlz.DevPtr(Y), SlBlz.DevPtr(Z),
		SlBrz.DevPtr(X), SlBrz.DevPtr(Y), SlBrz.DevPtr(Z),
		gux.DevPtr(X), gux.DevPtr(Y), gux.DevPtr(Z),
		guy.DevPtr(X), guy.DevPtr(Y), guy.DevPtr(Z),
		guz.DevPtr(X), guz.DevPtr(Y), guz.DevPtr(Z),
		C.C12.DevPtr(0), C.C12.Mul(0), C.C13.DevPtr(0), C.C13.Mul(0), C.C23.DevPtr(0), C.C23.Mul(0),
		C.C44.DevPtr(0), C.C44.Mul(0), C.C55.DevPtr(0), C.C55.Mul(0), C.C66.DevPtr(0), C.C66.Mul(0),
		N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

// Elastic stress sigma_el = C:(eps - eps_m): sd = (xx,yy,zz), so = (yz,xz,xy).
func MelasStressNP(sd, so *data.Slice, gux, guy, guz, m *data.Slice,
	l100, l111 MSlice, C MelasStiffness, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasStressNP_async(
		sd.DevPtr(X), sd.DevPtr(Y), sd.DevPtr(Z),
		so.DevPtr(X), so.DevPtr(Y), so.DevPtr(Z),
		gux.DevPtr(X), gux.DevPtr(Y), gux.DevPtr(Z),
		guy.DevPtr(X), guy.DevPtr(Y), guy.DevPtr(Z),
		guz.DevPtr(X), guz.DevPtr(Y), guz.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		l100.DevPtr(0), l100.Mul(0), l111.DevPtr(0), l111.Mul(0),
		C.C11.DevPtr(0), C.C11.Mul(0), C.C12.DevPtr(0), C.C12.Mul(0), C.C13.DevPtr(0), C.C13.Mul(0),
		C.C22.DevPtr(0), C.C22.Mul(0), C.C23.DevPtr(0), C.C23.Mul(0), C.C33.DevPtr(0), C.C33.Mul(0),
		C.C44.DevPtr(0), C.C44.Mul(0), C.C55.DevPtr(0), C.C55.Mul(0), C.C66.DevPtr(0), C.C66.Mul(0),
		N[X], N[Y], N[Z], cfg)
}

// Bulk force columns fcx/fcy/fcz (component = force row).
func MelasForceNP(fcx, fcy, fcz *data.Slice,
	u, gux, guy, guz, gmx, gmy, gmz, m *data.Slice,
	Blx, Brx, Bly, Bry, Blz, Brz *data.Slice,
	l100, l111 MSlice, C MelasStiffness, mesh *data.Mesh, so bool) {
	N := mesh.Size()
	c := mesh.CellSize()
	cfg := make3DConf(N)
	soInt := 0
	if so {
		soInt = 1
	}
	k_MelasForceNP_async(
		fcx.DevPtr(X), fcx.DevPtr(Y), fcx.DevPtr(Z),
		fcy.DevPtr(X), fcy.DevPtr(Y), fcy.DevPtr(Z),
		fcz.DevPtr(X), fcz.DevPtr(Y), fcz.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		gux.DevPtr(X), gux.DevPtr(Y), gux.DevPtr(Z),
		guy.DevPtr(X), guy.DevPtr(Y), guy.DevPtr(Z),
		guz.DevPtr(X), guz.DevPtr(Y), guz.DevPtr(Z),
		gmx.DevPtr(X), gmx.DevPtr(Y), gmx.DevPtr(Z),
		gmy.DevPtr(X), gmy.DevPtr(Y), gmy.DevPtr(Z),
		gmz.DevPtr(X), gmz.DevPtr(Y), gmz.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Blx.DevPtr(X), Blx.DevPtr(Y), Blx.DevPtr(Z),
		Brx.DevPtr(X), Brx.DevPtr(Y), Brx.DevPtr(Z),
		Bly.DevPtr(X), Bly.DevPtr(Y), Bly.DevPtr(Z),
		Bry.DevPtr(X), Bry.DevPtr(Y), Bry.DevPtr(Z),
		Blz.DevPtr(X), Blz.DevPtr(Y), Blz.DevPtr(Z),
		Brz.DevPtr(X), Brz.DevPtr(Y), Brz.DevPtr(Z),
		l100.DevPtr(0), l100.Mul(0), l111.DevPtr(0), l111.Mul(0),
		C.C11.DevPtr(0), C.C11.Mul(0), C.C12.DevPtr(0), C.C12.Mul(0), C.C13.DevPtr(0), C.C13.Mul(0),
		C.C22.DevPtr(0), C.C22.Mul(0), C.C23.DevPtr(0), C.C23.Mul(0), C.C33.DevPtr(0), C.C33.Mul(0),
		C.C44.DevPtr(0), C.C44.Mul(0), C.C55.DevPtr(0), C.C55.Mul(0), C.C66.DevPtr(0), C.C66.Mul(0),
		N[X], N[Y], N[Z],
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		soInt, mesh.PBC_code(), cfg)
}

// Neumann boundary overrides + column summation + damping/mass/mask handling.
// outMode 0: dst = d(du)/dt; outMode 1: dst = f_el (force diagnostic).
func MelasAssembleNP(dst, fcx, fcy, fcz, sd, so, traction, du *data.Slice,
	eta, rho, mask, frozen MSlice, bn, outMode int, mesh *data.Mesh) {
	N := mesh.Size()
	c := mesh.CellSize()
	cfg := make3DConf(N)
	k_MelasAssembleNP_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		fcx.DevPtr(X), fcx.DevPtr(Y), fcx.DevPtr(Z),
		fcy.DevPtr(X), fcy.DevPtr(Y), fcy.DevPtr(Z),
		fcz.DevPtr(X), fcz.DevPtr(Y), fcz.DevPtr(Z),
		sd.DevPtr(X), sd.DevPtr(Y), sd.DevPtr(Z),
		so.DevPtr(X), so.DevPtr(Y), so.DevPtr(Z),
		devPtrOrNil(traction, X), devPtrOrNil(traction, Y), devPtrOrNil(traction, Z),
		du.DevPtr(X), du.DevPtr(Y), du.DevPtr(Z),
		eta.DevPtr(0), eta.Mul(0), rho.DevPtr(0), rho.Mul(0),
		mask.DevPtr(0), mask.Mul(0), frozen.DevPtr(0), frozen.Mul(0),
		bn, outMode,
		N[X], N[Y], N[Z],
		float32(c[X]), float32(c[Y]), float32(c[Z]), mesh.PBC_code(), cfg)
}

// Adds the nonlinear magnetoelastic field (B, in Tesla) to dst.
func MelasHFieldNP(dst, m, sd, so *data.Slice, l100, l111, Ms MSlice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasHFieldNP_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		sd.DevPtr(X), sd.DevPtr(Y), sd.DevPtr(Z),
		so.DevPtr(X), so.DevPtr(Y), so.DevPtr(Z),
		l100.DevPtr(0), l100.Mul(0), l111.DevPtr(0), l111.Mul(0),
		Ms.DevPtr(0), Ms.Mul(0),
		N[X], N[Y], N[Z], cfg)
}

// Energy density (J/m3): mode 0 = U_el, mode 1 = magnetoelastic (LLG budget), mode 2 = U.
func MelasEnergyNP(dst *data.Slice, gux, guy, guz, m *data.Slice,
	l100, l111 MSlice, C MelasStiffness, mode int, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasEnergyNP_async(dst.DevPtr(0),
		gux.DevPtr(X), gux.DevPtr(Y), gux.DevPtr(Z),
		guy.DevPtr(X), guy.DevPtr(Y), guy.DevPtr(Z),
		guz.DevPtr(X), guz.DevPtr(Y), guz.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		l100.DevPtr(0), l100.Mul(0), l111.DevPtr(0), l111.Mul(0),
		C.C11.DevPtr(0), C.C11.Mul(0), C.C12.DevPtr(0), C.C12.Mul(0), C.C13.DevPtr(0), C.C13.Mul(0),
		C.C22.DevPtr(0), C.C22.Mul(0), C.C23.DevPtr(0), C.C23.Mul(0), C.C33.DevPtr(0), C.C33.Mul(0),
		C.C44.DevPtr(0), C.C44.Mul(0), C.C55.DevPtr(0), C.C55.Mul(0), C.C66.DevPtr(0), C.C66.Mul(0),
		mode,
		N[X], N[Y], N[Z], cfg)
}

// Strain diagnostic: mode 0 = eps, mode 1 = eps - eps_m, mode 2 = eps_m.
func MelasStrainNP(nrm, shr *data.Slice, gux, guy, guz, m *data.Slice,
	l100, l111 MSlice, mode int, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasStrainNP_async(
		nrm.DevPtr(X), nrm.DevPtr(Y), nrm.DevPtr(Z),
		shr.DevPtr(X), shr.DevPtr(Y), shr.DevPtr(Z),
		gux.DevPtr(X), gux.DevPtr(Y), gux.DevPtr(Z),
		guy.DevPtr(X), guy.DevPtr(Y), guy.DevPtr(Z),
		guz.DevPtr(X), guz.DevPtr(Y), guz.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		l100.DevPtr(0), l100.Mul(0), l111.DevPtr(0), l111.Mul(0),
		mode,
		N[X], N[Y], N[Z], cfg)
}

// du/dt = du, masked by the elastic and Dirichlet masks.
func MelasDudNP(dst, du *data.Slice, mask, frozen MSlice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasDudNP_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		du.DevPtr(X), du.DevPtr(Y), du.DevPtr(Z),
		mask.DevPtr(0), mask.Mul(0), frozen.DevPtr(0), frozen.Mul(0),
		N[X], N[Y], N[Z], cfg)
}

// Applies u = val where the Dirichlet mask is nonzero (space/time-dependent Dirichlet BC).
func MelasDirichletNP(u, val *data.Slice, frozen MSlice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasDirichletNP_async(
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		val.DevPtr(X), val.DevPtr(Y), val.DevPtr(Z),
		frozen.DevPtr(0), frozen.Mul(0),
		N[X], N[Y], N[Z], cfg)
}

// Gilbert dissipation density factor Ms*alpha/(1+alpha^2)*|m x B|^2.
func MelasGilbertNP(dst, m, b *data.Slice, alpha, Ms MSlice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_MelasGilbertNP_async(dst.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0), Ms.DevPtr(0), Ms.Mul(0),
		N[X], N[Y], N[Z], cfg)
}
