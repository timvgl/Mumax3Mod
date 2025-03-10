package engine

import (
	"math/rand"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var limitElasticToGeometry = false

func init() {
	DeclFunc("SetGeom", SetGeom, "Sets the Geometry to a given shape")
	DeclVar("EdgeSmooth", &edgeSmooth, "Geometry edge smoothing with edgeSmooth^3 samples per cell, 0=staircase, ~8=very smooth")
	//DeclVar("limitElasticToGeometry", &limitElasticToGeometry, "")
	Geometry.init()
}

var (
	Geometry   geom
	edgeSmooth int = 0 // disabled by default
)

type geom struct {
	info
	Buffer *data.Slice
	shape  Shape
}

func (g *geom) init() {
	g.Buffer = nil
	g.info = info{1, "geom", ""}
	DeclROnly("geom", g, "Cell fill fraction (0..1)")
}

func spaceFill() float64 {
	if Geometry.Gpu().IsNil() {
		return 1
	} else {
		return float64(cuda.Sum(Geometry.Buffer)) / float64(Geometry.Mesh().NCell())
	}
}

func (g *geom) Gpu() *data.Slice {
	if g.Buffer == nil {
		g.Buffer = data.NilSlice(1, g.Mesh().Size())
	}
	return g.Buffer
}

func (g *geom) Slice() (*data.Slice, bool) {
	s := g.Gpu()
	if s.IsNil() {
		s := cuda.Buffer(g.NComp(), g.Mesh().Size())
		cuda.Memset(s, 1)
		return s, true
	} else {
		return s, false
	}
}

func (q *geom) EvalTo(dst *data.Slice) { EvalTo(q, dst) }

var _ Quantity = &Geometry

func (g *geom) average() []float64 {
	s, r := g.Slice()
	if r {
		defer cuda.Recycle(s)
	}
	return sAverageUniverse(s)
}

func (g *geom) Average() float64 { return g.average()[0] }

func SetGeom(s Shape) {
	Geometry.setGeom(s)
}

func (Geometry *geom) setGeom(s Shape) {
	SetBusy(true)
	defer SetBusy(false)

	if s == nil {
		// TODO: would be nice not to save volume if entirely filled
		s = universe
	}

	Geometry.shape = s
	if Geometry.Gpu().IsNil() {
		Geometry.Buffer = cuda.NewSlice(1, Geometry.Mesh().Size())
	}

	host := data.NewSlice(1, Geometry.Gpu().Size())
	array := host.Scalars()
	V := host
	v := array
	n := Geometry.Mesh().Size()
	c := Geometry.Mesh().CellSize()
	cx, cy, cz := c[X], c[Y], c[Z]

	progress, progmax := 0, n[Y]*n[Z]

	var ok bool
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {

			progress++
			util.Progress(progress, progmax, "Initializing Geometry")

			for ix := 0; ix < n[X]; ix++ {

				r := Index2Coord(ix, iy, iz)
				x0, y0, z0 := r[X], r[Y], r[Z]

				// check if center and all vertices lie inside or all outside
				allIn, allOut := true, true
				if s(x0, y0, z0) {
					allOut = false
				} else {
					allIn = false
				}

				if edgeSmooth != 0 { // center is sufficient if we're not really smoothing
					for _, Δx := range []float64{-cx / 2, cx / 2} {
						for _, Δy := range []float64{-cy / 2, cy / 2} {
							for _, Δz := range []float64{-cz / 2, cz / 2} {
								if s(x0+Δx, y0+Δy, z0+Δz) { // inside
									allOut = false
								} else {
									allIn = false
								}
							}
						}
					}
				}

				switch {
				case allIn:
					v[iz][iy][ix] = 1
					ok = true
				case allOut:
					v[iz][iy][ix] = 0
				default:
					v[iz][iy][ix] = Geometry.cellVolume(ix, iy, iz)
					ok = ok || (v[iz][iy][ix] != 0)
				}
			}
		}
	}

	if !ok {
		util.Fatal("SetGeom: Geometry completely empty")
	}

	data.Copy(Geometry.Buffer, V)

	// M inside geom but previously outside needs to be re-inited
	needupload := false
	geomlist := host.Host()[0]
	mhost := M.Buffer().HostCopy()
	m := mhost.Host()
	rng := rand.New(rand.NewSource(0))
	for i := range m[0] {
		if geomlist[i] != 0 {
			mx, my, mz := m[X][i], m[Y][i], m[Z][i]
			if mx == 0 && my == 0 && mz == 0 {
				needupload = true
				rnd := randomDir(rng)
				m[X][i], m[Y][i], m[Z][i] = float32(rnd[X]), float32(rnd[Y]), float32(rnd[Z])
			}
		}
	}
	if needupload {
		data.Copy(M.Buffer(), mhost)
	}

	M.normalize() // removes m outside vol

	// U inside geom but previously outside needs to be re-inited
	needupload2 := false
	geomlist2 := host.Host()[0]
	uhost := U.Buffer().HostCopy()
	u := uhost.Host()
	rng2 := rand.New(rand.NewSource(0))
	for i := range u[0] {
		if geomlist2[i] != 0 {
			ux, uy, uz := u[X][i], u[Y][i], u[Z][i]
			if ux == 0 && uy == 0 && uz == 0 {
				needupload2 = true
				rnd := randomDir(rng2)
				u[X][i], u[Y][i], u[Z][i] = float32(rnd[X]), float32(rnd[Y]), float32(rnd[Z])
			}
		}
	}
	if limitElasticToGeometry {
		buf := cuda.Buffer(uhost.NComp(), uhost.Size())
		data.Copy(buf, uhost)
		cuda.LimitToGeometry(buf, Geometry.Buffer)
		data.Copy(U.Buffer(), buf)
		cuda.Recycle(buf)
	}
	if needupload2 && !limitElasticToGeometry {
		data.Copy(U.Buffer(), uhost)
	}

	//U.normalize() // removes m outside vol

	// du inside geom but previously outside needs to be re-inited
	needupload3 := false
	geomlist3 := host.Host()[0]
	duhost := DU.Buffer().HostCopy()
	du := duhost.Host()
	rng3 := rand.New(rand.NewSource(0))
	for i := range du[0] {
		if geomlist3[i] != 0 {
			dux, duy, duz := du[X][i], du[Y][i], du[Z][i]
			if dux == 0 && duy == 0 && duz == 0 {
				needupload3 = true
				rnd := randomDir(rng3)
				du[X][i], du[Y][i], du[Z][i] = float32(rnd[X]), float32(rnd[Y]), float32(rnd[Z])
			}
		}
	}
	if limitElasticToGeometry {
		buf := cuda.Buffer(duhost.NComp(), duhost.Size())
		data.Copy(buf, duhost)
		cuda.LimitToGeometry(buf, Geometry.Buffer)
		data.Copy(DU.Buffer(), buf)
		cuda.Recycle(buf)
	}
	if needupload3 && !limitElasticToGeometry {
		data.Copy(DU.Buffer(), duhost)
	}

	//U.normalize() // removes m outside vol
}

// Sample edgeSmooth^3 points inside the cell to estimate its volume.
func (g *geom) cellVolume(ix, iy, iz int) float32 {
	r := Index2Coord(ix, iy, iz)
	x0, y0, z0 := r[X], r[Y], r[Z]

	c := Geometry.Mesh().CellSize()
	cx, cy, cz := c[X], c[Y], c[Z]
	s := Geometry.shape
	var vol float32

	N := edgeSmooth
	S := float64(edgeSmooth)

	for dx := 0; dx < N; dx++ {
		Δx := -cx/2 + (cx / (2 * S)) + (cx/S)*float64(dx)
		for dy := 0; dy < N; dy++ {
			Δy := -cy/2 + (cy / (2 * S)) + (cy/S)*float64(dy)
			for dz := 0; dz < N; dz++ {
				Δz := -cz/2 + (cz / (2 * S)) + (cz/S)*float64(dz)

				if s(x0+Δx, y0+Δy, z0+Δz) { // inside
					vol++
				}
			}
		}
	}
	return vol / float32(N*N*N)
}

func (g *geom) shift(dx int) {
	// empty mask, nothing to do
	if g == nil || g.Buffer.IsNil() {
		return
	}

	// allocated mask: shift
	s := g.Buffer
	s2 := cuda.Buffer(1, g.Mesh().Size())
	defer cuda.Recycle(s2)
	newv := float32(1) // initially fill edges with 1's
	cuda.ShiftX(s2, s, dx, newv, newv)
	data.Copy(s, s2)

	n := Mesh().Size()
	x1, x2 := shiftDirtyRange(dx)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				if !g.shape(r[X], r[Y], r[Z]) {
					cuda.SetCell(g.Buffer, 0, ix, iy, iz, 0) // a bit slowish, but hardly reached
				}
			}
		}
	}

}

func (g *geom) shiftY(dy int) {
	// empty mask, nothing to do
	if g == nil || g.Buffer.IsNil() {
		return
	}

	// allocated mask: shift
	s := g.Buffer
	s2 := cuda.Buffer(1, g.Mesh().Size())
	defer cuda.Recycle(s2)
	newv := float32(1) // initially fill edges with 1's
	cuda.ShiftY(s2, s, dy, newv, newv)
	data.Copy(s, s2)

	n := Mesh().Size()
	y1, y2 := shiftDirtyRange(dy)

	for iz := 0; iz < n[Z]; iz++ {
		for ix := 0; ix < n[X]; ix++ {
			for iy := y1; iy < y2; iy++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				if !g.shape(r[X], r[Y], r[Z]) {
					cuda.SetCell(g.Buffer, 0, ix, iy, iz, 0) // a bit slowish, but hardly reached
				}
			}
		}
	}

}

// x range that needs to be refreshed after shift over dx
func shiftDirtyRange(dx int) (x1, x2 int) {
	nx := Mesh().Size()[X]
	util.Argument(dx != 0)
	if dx < 0 {
		x1 = nx + dx
		x2 = nx
	} else {
		x1 = 0
		x2 = dx
	}
	return
}

func (g *geom) Mesh() *data.Mesh { return Mesh() }
