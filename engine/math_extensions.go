package engine

import (
	"math"

	"github.com/mumax/3/data"
)
const Mu0 = 4 * math.Pi * 1e-7

func init() {
	DeclFunc("RotVector", RotVector, "Rotate the vector x around an arbitrarz vector n about alpha in rad.")
	DeclFunc("Stripline_Hx", Stripline_Hx, "x component of field around stripline")
	DeclFunc("Stripline_Hz", Stripline_Hz, "x component of field around stripline")
}

func RotVector(x data.Vector, n data.Vector, alpha float64) data.Vector {
	return (n.Mul(n.Dot(x)).Add(((n.Cross(x)).Cross(n)).Mul(math.Cos(alpha)))).Add((n.Cross(x)).Mul(math.Sin(alpha)))
}

func Stripline_Hx(x, z, a, b, I float64) float64{
	term1 := (b - z) * (b - z) + (a - x) * (a - x)
    term2 := (- b - z) * ( - b - z) + (a - x) * (a - x)
	term3 := (b - z) * (b - z) + (- a - x) * (- a - x)
    term4 := (- a - x) * ( - a - x) + (- b - z) * (- b - z)
    ln1 := 0.5 * math.Log(term1 / term2)
	ln2 := 0.5 * math.Log(term3 / term4)
    atan1 := math.Atan((a - x) / (b - z))
    atan2 := math.Atan((a - x) / (- b - z))
    atan3 := math.Atan((-a - x) / (b - z))
    atan4 := math.Atan((-a - x) / (-b - z))
    
    result := -Mu0 * I / (8 * math.Pi * a * b) * (
		(a - x) * (ln1 +
			(b - z) / (a - x) * atan1 -
			(- b - z) / (a - x) * atan2) -
		(- a - x) * (ln2 +
			(b - z) / (-a - x) * atan3 -
			(-b - z) / (-a - x) * atan4))
	return result
}

// Function to calculate hz(x, z)
func Stripline_Hz(x, z, a, b, I float64) float64{
	term1 := (b - z) * (b - z) + (a - x) * (a - x)
    term2 := (- a - x) * ( - a - x) + (b - z) * (b - z)
	term3 := (a - x) * (a - x) + (- b - z) * (- b - z)
    term4 := (- a - x) * ( - a - x) + (- b - z) * (- b - z)
    ln1 := 0.5 * math.Log(term1 / term2)
	ln2 := 0.5 * math.Log(term3 / term4)
    atan1 := math.Atan((b - z) / (a - x))
    atan2 := math.Atan((b - z) / (- a - x))
    atan3 := math.Atan((-b - z) / (a - x))
    atan4 := math.Atan((-b - z) / (-a - x))
    
    result := - Mu0 * I / (8 * math.Pi * a * b) * (
		(b - z) * (ln1 + 
			(a - x) / (b - z) * atan1 - 
			(- a - x) / (b - z) * atan2) -
		(- b - z) * (ln2 + 
			(a - x) / (-b - z) * atan3 - 
			(-a - x) / (-b - z) * atan4))
    return result
}
