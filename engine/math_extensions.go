package engine

import (
	"math"

	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("RotVector", RotVector, "Rotate the vector x around an arbitrary vector n about alpha in rad.")
}

func RotVector(x data.Vector, n data.Vector, alpha float64) data.Vector {
	return (n.Mul(n.Dot(x)).Add(((n.Cross(x)).Cross(n)).Mul(math.Cos(alpha)))).Add((n.Cross(x)).Mul(math.Sin(alpha)))
}