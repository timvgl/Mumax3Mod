package engine

import (
	"reflect"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var U displacement // displacement [m]
//var UOVERLAY displacement

func init() {
	DeclLValue("u", &U, `displacement [m]`)
	//DeclLValue("uOverlay", &UOVERLAY, `displacement [m] added after each step`)
}

// Special buffered quantity to store displacement
// makes sure it's normalized etc.
type displacement struct {
	buffer_ *data.Slice
}

func (u *displacement) Mesh() *data.Mesh    { return Mesh() }
func (u *displacement) NComp() int          { return 3 }
func (u *displacement) Name() string        { return "u" }
func (u *displacement) Unit() string        { return "m" }
func (u *displacement) Buffer() *data.Slice { return u.buffer_ } // todo: rename Gpu()?

func (u *displacement) Comp(c int) ScalarField  { return Comp(u, c) }
func (u *displacement) SetValue(v interface{})  { u.SetInShape(nil, v.(Config)) }
func (u *displacement) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (u *displacement) Type() reflect.Type      { return reflect.TypeOf(new(displacement)) }
func (u *displacement) Eval() interface{}       { return u }
func (u *displacement) average() []float64      { return sAverageMagnet(u.Buffer()) }
func (u *displacement) Average() data.Vector    { return unslice(u.average()) }
func (u *displacement) EvalRegionTo(dst *data.Slice) {
	cuda.Crop(dst, u.buffer_, dst.StartX, dst.StartY, dst.StartZ)
}

//func (u *displacement) normalize()              { cuda.Normalize(u.Buffer(), Geometry.Gpu()) }
//func (u *displacement) Strain()              { return Strain(u) }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (u *displacement) alloc() {
	u.buffer_ = cuda.NewSlice(3, u.Mesh().Size())
	u.Set(Uniform(0, 0, 0)) // sane starting config
}

func (b *displacement) SetArray(src *data.Slice) {
	if src.Size() != b.Mesh().Size() {
		src = data.Resample(src, b.Mesh().Size())
	}
	data.Copy(b.Buffer(), src)
	//b.normalize()
}

func (u *displacement) Set(c Config) {
	checkMesh()
	u.SetInShape(nil, c)
}

func (u *displacement) LoadFile(fname string) {
	u.SetArray(LoadFileDSlice(fname))
}

func (u *displacement) LoadFileMyDir(fname string) {
	u.SetArray(LoadFileDSliceMyDir(fname))
}

func (u *displacement) RenderFunction(equation StringFunction) {
	util.AssertMsg(!equation.IsScalar(), "RenderFunction: Need vector function.")
	renderers := make([]*ReadyToRenderFunction, 3)
	for i := range 3 {
		renderers[i] = RenderStringToReadyToRenderFct(equation.functions[i], u.Mesh())
	}
	d, _ := GenerateSliceFromReadyToRenderFct(renderers, u.Mesh())
	u.SetArray(d)
	cuda.Recycle(d)
}

func (u *displacement) SetTime(fname string) {
	var meta data.Meta
	_, meta = LoadFileMeta(fname)
	Time = meta.Time
}

func (u *displacement) LoadFileSetTime(fname string) {
	var meta data.Meta
	var d *data.Slice
	d, meta = LoadFileMeta(fname)
	u.SetArray(d)
	Time = meta.Time
}

func (u *displacement) Slice() (s *data.Slice, recycle bool) {
	return u.Buffer(), false
}

func (u *displacement) EvalTo(dst *data.Slice) {
	data.Copy(dst, u.buffer_)
}

func (u *displacement) Region(r int) *vOneReg { return vOneRegion(u, r) }

func (u *displacement) String() string { return util.Sprint(u.Buffer().HostCopy()) }

// Set the value of one cell.
func (u *displacement) SetCell(ix, iy, iz int, v data.Vector) {
	for c := 0; c < 3; c++ {
		cuda.SetCell(u.Buffer(), c, ix, iy, iz, float32(v[c]))
	}
}

// Get the value of one cell.
func (u *displacement) GetCell(ix, iy, iz int) data.Vector {
	ux := float64(cuda.GetCell(u.Buffer(), X, ix, iy, iz))
	uy := float64(cuda.GetCell(u.Buffer(), Y, ix, iy, iz))
	uz := float64(cuda.GetCell(u.Buffer(), Z, ix, iy, iz))
	return Vector(ux, uy, uz)
}

func (u *displacement) Quantity() []float64 { return slice(u.Average()) }

// Sets the displacement inside the shape
func (u *displacement) SetInShape(region Shape, conf Config) {
	checkMesh()

	if region == nil {
		region = universe
	}
	host := u.Buffer().HostCopy()
	h := host.Vectors()
	n := u.Mesh().Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					u := conf(x, y, z)
					h[X][iz][iy][ix] = float32(u[X])
					h[Y][iz][iy][ix] = float32(u[Y])
					h[Z][iz][iy][ix] = float32(u[Z])
				}
			}
		}
	}
	u.SetArray(host)
}

// set u to config in region
func (u *displacement) SetRegion(region int, conf Config) {
	host := u.Buffer().HostCopy()
	h := host.Vectors()
	n := u.Mesh().Size()
	r := byte(region)

	regionsArr := regions.HostArray()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				pos := Index2Coord(ix, iy, iz)
				x, y, z := pos[X], pos[Y], pos[Z]
				if regionsArr[iz][iy][ix] == r {
					u := conf(x, y, z)
					h[X][iz][iy][ix] = float32(u[X])
					h[Y][iz][iy][ix] = float32(u[Y])
					h[Z][iz][iy][ix] = float32(u[Z])
				}
			}
		}
	}
	u.SetArray(host)
}

func (u *displacement) resize() {
	backup := u.Buffer().HostCopy()
	s2 := Mesh().Size()
	resized := data.Resample(backup, s2)
	u.buffer_.Free()
	u.buffer_ = cuda.NewSlice(VECTOR, s2)
	data.Copy(u.buffer_, resized)
}
