package engine

// Utilities for setting magnetic configurations.

import (
	"math"
	"math/rand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("Uniform", Uniform, "Uniform magnetization in given direction")
	DeclFunc("SinShape", SinShape, "Sin in space")
	DeclFunc("Vortex", Vortex, "Vortex magnetization with given circulation and core polarization")
	DeclFunc("VortexAsym", VortexAsym, "Vortex magnetization with given circulation and core polarization")
	DeclFunc("DisplacedVortex", DisplacedVortex, "")
	DeclFunc("DisplacedVortexAsym", DisplacedVortexAsym, "")
	DeclFunc("Antivortex", AntiVortex, "Antivortex magnetization with given circulation and core polarization")
	DeclFunc("NeelSkyrmion", NeelSkyrmion, "Néél skyrmion magnetization with given charge and core polarization")
	DeclFunc("BlochSkyrmion", BlochSkyrmion, "Bloch skyrmion magnetization with given chirality and core polarization")
	DeclFunc("TwoDomain", TwoDomain, "Twodomain magnetization with with given magnetization in left domain, wall, and right domain")
	DeclFunc("VortexWall", VortexWall, "Vortex wall magnetization with given mx in left and right domain and core circulation and polarization")
	DeclFunc("RandomMag", RandomMag, "Random magnetization")
	DeclFunc("RandomMagSeed", RandomMagSeed, "Random magnetization with given seed")
	DeclFunc("Conical", Conical, "Conical state for given wave vector, cone direction, and cone angle")
	DeclFunc("Helical", Helical, "Helical state for given wave vector")
	DeclFunc("GaussianSpherical", GaussianSpherical, "Gaussian pulse with spherical symmetry")
	DeclFunc("GaussianSpherical_outplane", GaussianSpherical_outplane, "Gaussian pulse with spherical distribution out of plane")
	DeclFunc("GaussianUniform", GaussianUniform, "Gaussian pulse in one direction")
	DeclFunc("RotationField", RotationField, "Vectorfield that rotates homogeniously around the position defined by the radius and the angle.")
	DeclFunc("ArtificialVortex", ArtificialVortex, "Vectorfield mimics a vortex in the elastic part by making use of a gaussian.")
}

// Magnetic configuration returns m vector for position (x,y,z)
type Config func(x, y, z float64) data.Vector

// Random initial magnetization.
func RandomMag() Config {
	return RandomMagSeed(0)
}

// Random initial magnetization,
// generated from random seed.
func RandomMagSeed(seed int) Config {
	rng := rand.New(rand.NewSource(int64(seed)))
	return func(x, y, z float64) data.Vector {
		return randomDir(rng)
	}
}

// generate anisotropic random unit vector
func randomDir(rng *rand.Rand) data.Vector {
	theta := 2 * rng.Float64() * math.Pi
	z := 2 * (rng.Float64() - 0.5)
	b := math.Sqrt(1 - z*z)
	x := b * math.Cos(theta)
	y := b * math.Sin(theta)
	return data.Vector{x, y, z}
}

// Returns a uniform magnetization state. E.g.:
// 	M = Uniform(1, 0, 0) // saturated along X
func Uniform(mx, my, mz float64) Config {
	return func(x, y, z float64) data.Vector {
		return data.Vector{mx, my, mz}
	}
}

func SinShape(ax, ay, az, kx, ky, kz, phx, phy, phz float64, typex, typey, typez string) Config {
	return func(x, y, z float64) data.Vector {
		var mx func(fx float64) float64
		var my func(fy float64) float64
		var mz func(fz float64) float64
		if typex == "sin" {
			mx = math.Sin
		} else if typex == "cos" {
			mx = math.Cos
		} else if typex == "" {
			mx = func(fx float64) float64 {return 1}
		} else {
			util.AssertMsg(false, "String has to be sin, cos or empty")
		}
		if typey == "sin" {
			my = math.Sin
		} else if typey == "cos" {
			my = math.Cos
		} else if typey == "" {
			my = func(fy float64) float64 {return 1}
		} else {
			util.AssertMsg(false, "String has to be sin, cos or empty")
		}
		if typez == "sin" {
			mz = math.Sin
		} else if typez == "cos" {
			mz = math.Cos
		} else if typez == "" {
			mz = func(fz float64) float64 {return 1}
		} else {
			util.AssertMsg(false, "String has to be sin, cos or empty")
		}
		return data.Vector{ax*mx(x*kx + phx), ay*my(y*ky + phy), az*mz(z*kz + phz)}
	}
}

// Make a vortex magnetization with given circulation and core polarization (+1 or -1).
// The core is smoothed over a few exchange lengths and should easily relax to its ground state.
func Vortex(circ, pol int) Config {
	diam2 := 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mx := -y * float64(circ) / r
		my := x * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func VortexAsym(circ, pol int, diam2Prop, angle, openAngle, diam2BasicFac float64) Config {
	diam2 := diam2BasicFac * 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mz := 0.0
		posAngle := float64(int(math.Atan2(y, x) * 180 / math.Pi + 360) % 360) * math.Pi / 180
		if posAngle < angle + math.Pi / 4 && posAngle > angle - math.Pi / 4 {
			if posAngle < angle + math.Pi / 4 && posAngle > angle {
				mz = 1.5 * float64(pol) * math.Exp(-r2/(diam2 * (- (diam2Prop -1) / openAngle * posAngle + diam2Prop + (diam2Prop -1) / openAngle * angle)))

			} else {
				mz = 1.5 * float64(pol) * math.Exp(-r2/(diam2 * ((diam2Prop -1) / openAngle * posAngle + diam2Prop - (diam2Prop -1) / openAngle * angle)))
			}
		} else {
			mz = 1.5 * float64(pol) * math.Exp(-r2/diam2)
		}
		mx := -y * float64(circ) / r
		my := x * float64(circ) / r
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func DisplacedVortex(circ, pol int, deltaX, deltaY float64) Config {
	diam2 := 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := math.Pow(x - deltaX, 2) + math.Pow(y - deltaY, 2)
		r := math.Sqrt(r2)
		mx := -(y - deltaY) * float64(circ) / r
		my := (x - deltaX) * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func DisplacedVortexAsym(circ, pol int, deltaX, deltaY, diam2Prop, angle, openAngle, diam2BasicFac float64) Config {
	diam2 := diam2BasicFac * 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := math.Pow(x - deltaX, 2) + math.Pow(y - deltaY, 2)
		r := math.Sqrt(r2)
		mz := 0.0
		posAngle := float64(int(math.Atan2(y, x) * 180 / math.Pi + 360) % 360) * math.Pi / 180
		if posAngle < angle + math.Pi / 4 && posAngle > angle - math.Pi / 4 {
			if posAngle < angle + math.Pi / 4 && posAngle > angle {
				mz = 1.5 * float64(pol) * math.Exp(-r2/(diam2 * (- (diam2Prop -1) / openAngle * posAngle + diam2Prop + (diam2Prop -1) / openAngle * angle)))

			} else {
				mz = 1.5 * float64(pol) * math.Exp(-r2/(diam2 * ((diam2Prop -1) / openAngle * posAngle + diam2Prop - (diam2Prop -1) / openAngle * angle)))
			}
		} else {
			mz = 1.5 * float64(pol) * math.Exp(-r2/diam2)
		}
		mx := -(y - deltaY) * float64(circ) / r
		my := (x - deltaX) * float64(circ) / r
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func GaussianSpherical_outplane(A, pos_x, pos_y, sig_x, sig_y float64) Config {
	return func(x, y, z float64) data.Vector {
		norm := A * math.Exp(-((x-pos_x)*(x-pos_x)/(2*sig_x*sig_x) + (y-pos_y)*(y-pos_y)/(2*sig_y*sig_y)))
		ux := 0.0
		uy := 0.0
		uz := norm
		return data.Vector{ux, uy, uz}
	}
}

func GaussianSpherical(A, pos_x, pos_y, sig_x, sig_y, angle float64) Config {
	return func(x, y, z float64) data.Vector {
		norm := A * math.Exp(-((x-pos_x)*(x-pos_x)/(2*sig_x*sig_x) + (y-pos_y)*(y-pos_y)/(2*sig_y*sig_y)))
		angle := angle * 3.1415 / 180
		ux := norm * math.Cos(angle)
		uy := norm * math.Sin(angle)
		uz := 0.0
		return data.Vector{ux, uy, uz}
	}
}

func GaussianUniform(A, pos, sig, angle1, angle2 float64) Config {
	return func(x, y, z float64) data.Vector {
		angle1 := angle1 * 3.1415 / 180
		angle2 := angle2 * 3.1415 / 180
		rot := math.Cos(angle1)*x + math.Sin(angle1)*y
		norm := A * math.Exp(-((rot - pos) * (rot - pos) / (2 * sig * sig)))
		ux := math.Cos(angle2) * norm
		uy := math.Sin(angle2) * norm
		uz := 0.0
		return data.Vector{ux, uy, uz}
	}
}

func RotationField(A, pos_center_r, pos_center_phi float64) Config {
	return func(x, y, z float64) data.Vector {
		pos_center_phi_rad := pos_center_phi * 3.1415 / 180
		ux := (-y - pos_center_r * math.Sin(pos_center_phi_rad))
		uy := (x - pos_center_r * math.Cos(pos_center_phi_rad))
		uz := 0.0
		norm := math.Sqrt(math.Pow(ux, 2) + math.Pow(uy, 2) + math.Pow(uz, 2))
		return data.Vector{A*ux/norm, A*uy/norm, A*uz/norm}
	}
}

func ArtificialVortex(A, pos_center_r, pos_center_phi, sig_x, sig_y float64) Config {
	return func(x, y, z float64) data.Vector {
		pos_center_phi_rad := pos_center_phi * 3.1415 / 180
		pos_x := pos_center_r * math.Cos(pos_center_phi_rad)
		pos_y := pos_center_r * math.Sin(pos_center_phi_rad)
		ux := (-y - pos_y)
		uy := (x - pos_x)
		uz := math.Exp(-((x-pos_x)*(x-pos_x)/(2*sig_x*sig_x) + (y-pos_y)*(y-pos_y)/(2*sig_y*sig_y)))
		norm := math.Sqrt(math.Pow(ux, 2) + math.Pow(uy, 2) + math.Pow(uz, 2))
		return data.Vector{A*ux/norm, A*uy/norm, A*uz/norm}
	}
}

func NeelSkyrmion(charge, pol int) Config {
	w := 8 * Mesh().CellSize()[X]
	w2 := w * w
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mz := 2 * float64(pol) * (math.Exp(-r2/w2) - 0.5)
		mx := (x * float64(charge) / r) * (1 - math.Abs(mz))
		my := (y * float64(charge) / r) * (1 - math.Abs(mz))
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func BlochSkyrmion(charge, pol int) Config {
	w := 8 * Mesh().CellSize()[X]
	w2 := w * w
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mz := 2 * float64(pol) * (math.Exp(-r2/w2) - 0.5)
		mx := (-y * float64(charge) / r) * (1 - math.Abs(mz))
		my := (x * float64(charge) / r) * (1 - math.Abs(mz))
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

func AntiVortex(circ, pol int) Config {
	diam2 := 2 * sqr64(Mesh().CellSize()[X])
	return func(x, y, z float64) data.Vector {
		r2 := x*x + y*y
		r := math.Sqrt(r2)
		mx := -x * float64(circ) / r
		my := y * float64(circ) / r
		mz := 1.5 * float64(pol) * math.Exp(-r2/diam2)
		return noNaN(data.Vector{mx, my, mz}, pol)
	}
}

// Make a vortex wall configuration.
func VortexWall(mleft, mright float64, circ, pol int) Config {
	h := Mesh().WorldSize()[Y]
	v := Vortex(circ, pol)
	return func(x, y, z float64) data.Vector {
		if x < -h/2 {
			return data.Vector{mleft, 0, 0}
		}
		if x > h/2 {
			return data.Vector{mright, 0, 0}
		}
		return v(x, y, z)
	}
}

func noNaN(v data.Vector, pol int) data.Vector {
	if math.IsNaN(v[X]) || math.IsNaN(v[Y]) || math.IsNaN(v[Z]) {
		return data.Vector{0, 0, float64(pol)}
	} else {
		return v
	}
}

// Make a 2-domain configuration with domain wall.
// (mx1, my1, mz1) and (mx2, my2, mz2) are the magnetizations in the left and right domain, respectively.
// (mxwall, mywall, mzwall) is the magnetization in the wall. The wall is smoothed over a few cells so it will
// easily relax to its ground state.
// E.g.:
// 	TwoDomain(1,0,0,  0,1,0,  -1,0,0) // head-to-head domains with transverse (Néel) wall
// 	TwoDomain(1,0,0,  0,0,1,  -1,0,0) // head-to-head domains with perpendicular (Bloch) wall
// 	TwoDomain(0,0,1,  1,0,0,   0,0,-1)// up-down domains with Bloch wall
func TwoDomain(mx1, my1, mz1, mxwall, mywall, mzwall, mx2, my2, mz2 float64) Config {
	ww := 2 * Mesh().CellSize()[X] // wall width in cells
	return func(x, y, z float64) data.Vector {
		var m data.Vector
		if x < 0 {
			m = data.Vector{mx1, my1, mz1}
		} else {
			m = data.Vector{mx2, my2, mz2}
		}
		gauss := math.Exp(-sqr64(x / ww))
		m[X] = (1-gauss)*m[X] + gauss*mxwall
		m[Y] = (1-gauss)*m[Y] + gauss*mywall
		m[Z] = (1-gauss)*m[Z] + gauss*mzwall
		return m
	}
}

// Conical magnetization configuration.
// The magnetization rotates on a cone defined by coneAngle and coneDirection.
// q is the wave vector of the conical magnetization configuration.
// The magnetization is
//
//	m = u*cos(coneAngle) + sin(coneAngle)*( ua*cos(q*r) + ub*sin(q*r) )
//
// with ua and ub unit vectors perpendicular to u (normalized coneDirection)
func Conical(q, coneDirection data.Vector, coneAngle float64) Config {
	u := coneDirection.Div(coneDirection.Len())
	// two unit vectors perpendicular to each other and to the cone direction u
	p := math.Sqrt(1 - u[Z]*u[Z])
	ua := data.Vector{u[X] * u[Z], u[Y] * u[Z], u[Z]*u[Z] - 1}.Div(p)
	ub := data.Vector{-u[Y], u[X], 0}.Div(p)
	// cone direction along z direction? -> oops devided by zero, let's fix this
	if u[Z]*u[Z] == 1 {
		ua = data.Vector{1, 0, 0}
		ub = data.Vector{0, 1, 0}
	}
	sina, cosa := math.Sincos(coneAngle)
	return func(x, y, z float64) data.Vector {
		sinqr, cosqr := math.Sincos(q[X]*x + q[Y]*y + q[Z]*z)
		return u.Mul(cosa).MAdd(sina*cosqr, ua).MAdd(sina*sinqr, ub)
	}
}

func Helical(q data.Vector) Config {
	return Conical(q, q, math.Pi/2)
}

// Transl returns a translated copy of configuration c. E.g.:
// 	M = Vortex(1, 1).Transl(100e-9, 0, 0)  // vortex with center at x=100nm
func (c Config) Transl(dx, dy, dz float64) Config {
	return func(x, y, z float64) data.Vector {
		return c(x-dx, y-dy, z-dz)
	}
}

// Scale returns a scaled copy of configuration c.
func (c Config) Scale(sx, sy, sz float64) Config {
	return func(x, y, z float64) data.Vector {
		return c(x/sx, y/sy, z/sz)
	}
}

// Rotates the configuration around the Z-axis, over θ radians.
func (c Config) RotZ(θ float64) Config {
	cos := math.Cos(θ)
	sin := math.Sin(θ)
	return func(x, y, z float64) data.Vector {
		x_ := x*cos + y*sin
		y_ := -x*sin + y*cos
		m := c(x_, y_, z)
		mx_ := m[X]*cos - m[Y]*sin
		my_ := m[X]*sin + m[Y]*cos
		return data.Vector{mx_, my_, m[Z]}
	}
}

// Returns a new magnetization equal to c + weight * other.
// E.g.:
// 	Uniform(1, 0, 0).Add(0.2, RandomMag())
// for a uniform state with 20% random distortion.
func (c Config) Add(weight float64, other Config) Config {
	return func(x, y, z float64) data.Vector {
		return c(x, y, z).MAdd(weight, other(x, y, z))
	}
}
