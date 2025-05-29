package engine

import (
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/govaluate"
)

type ExprEvaluator struct {
	expression *govaluate.EvaluableExpression
	rng        *rand.Rand
	mu         sync.Mutex // Ensures thread-safe access to rng
}

func init() {
	DeclFunc("CreateFloatSlice", CreateFloatSlice, "")
	DeclFunc("CreateFloatSliceZero", CreateFloatSliceZero, "")
	DeclFunc("CreateFloatSliceOne", CreateFloatSliceOne, "")
	DeclFunc("CreateFloatSliceArb", CreateFloatSliceArb, "")
	DeclFunc("CreateString", CreateString, "")
}
func CreateString(name string) string {
	return name
}

// NewExprEvaluator compiles the expression string and prepares the evaluator
func NewExprEvaluator(expressionStr string) (*ExprEvaluator, error) {
	evaluator := &ExprEvaluator{
		// Initialize the random number generator with a seed
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// Define custom functions
	functions := map[string]govaluate.ExpressionFunction{
		// One-argument functions (using float or DataSlice)
		"abs": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("abs() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Abs(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AbsGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for abs(): %v", args[0])
		},
		"acos": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("acos() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Acos(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AcosGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for acos(): %v", args[0])
		},
		"acosh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("acosh() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Acosh(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AcoshGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for acosh(): %v", args[0])
		},
		"asin": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("asin() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Asin(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AsinGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for asin(): %v", args[0])
		},
		"asinh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("asinh() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Asinh(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AsinhGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for asinh(): %v", args[0])
		},
		"atan": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("atan() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Atan(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AtanGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for atan(): %v", args[0])
		},
		"atanh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("atanh() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Atanh(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.AtanhGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for atanh(): %v", args[0])
		},
		"cbrt": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("cbrt() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Cbrt(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.CbrtGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for cbrt(): %v", args[0])
		},
		"ceil": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("ceil() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Ceil(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.CeilGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for ceil(): %v", args[0])
		},
		"cos": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("cos() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Cos(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.CosGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for cos(): %v", args[0])
		},
		"cosh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("cosh() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Cosh(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.CoshGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for cosh(): %v", args[0])
		},
		"erf": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("erf() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Erf(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.ErfGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for erf(): %v", args[0])
		},
		"erfc": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("erfc() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Erfc(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.ErfcGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for erfc(): %v", args[0])
		},
		"exp": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("exp() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Exp(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.ExpGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for exp(): %v", args[0])
		},
		"exp2": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("exp2() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Exp2(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Exp2Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for exp2(): %v", args[0])
		},
		"expm1": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("expm1() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Expm1(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Expm1Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for expm1(): %v", args[0])
		},
		"floor": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("floor() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Floor(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.FloorGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for floor(): %v", args[0])
		},
		"gamma": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("gamma() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Gamma(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.GammaGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for gamma(): %v", args[0])
		},
		"j0": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("j0() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.J0(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.J0Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for j0(): %v", args[0])
		},
		"j1": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("j1() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.J1(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.J1Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for j1(): %v", args[0])
		},
		"log": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Log(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.LogGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for log(): %v", args[0])
		},
		"log10": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log10() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Log10(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Log10Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for log10(): %v", args[0])
		},
		"log1p": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log1p() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Log1p(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Log1pGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for log1p(): %v", args[0])
		},
		"log2": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log2() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Log2(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Log2Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for log2(): %v", args[0])
		},
		"logb": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("logb() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Logb(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.LogbGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for logb(): %v", args[0])
		},
		"sin": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sin() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Sin(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.SinGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for sin(): %v", args[0])
		},
		"sinh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sinh() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Sinh(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.SinhGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for sinh(): %v", args[0])
		},
		"sqrt": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sqrt() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Sqrt(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.SqrtGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for sqrt(): %v", args[0])
		},
		"tan": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("tan() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Tan(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.TanGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for tan(): %v", args[0])
		},
		"tanh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("tanh() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Tanh(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.TanhGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for tanh(): %v", args[0])
		},
		"trunc": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("trunc() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Trunc(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.TruncGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for trunc(): %v", args[0])
		},
		"y0": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("y0() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Y0(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Y0Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for y0(): %v", args[0])
		},
		"y1": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("y1() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Y1(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Y1Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for y1(): %v", args[0])
		},
		"ilogb": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("ilogb() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return math.Ilogb(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.IlogbGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for ilogb(): %v", args[0])
		},

		// Custom one-argument functions
		"heaviside": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("heaviside() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return heaviside(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.HeavisideGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for heaviside(): %v", args[0])
		},
		"norm": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("norm() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return norm(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.NormGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for norm(): %v", args[0])
		},
		"sinc": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sinc() expects exactly one argument")
			}
			if x, ok := toFloat64(args[0]); ok {
				return sinc(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.SincGovaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for sinc(): %v", args[0])
		},

		// Functions with two arguments (left unchanged) -> changed manually
		"yn": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("yn() expects exactly two arguments")
			}
			n, ok1 := toInt(args[0])
			x, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				d1, ok3 := toDataSlice(args[0])
				d2, ok4 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for yn(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, float64(n), x, ok3, ok4, cuda.YnGovaluate3X3, cuda.YnGovaluate3X1, cuda.YnGovaluate1X3), nil
			}
			return math.Yn(n, x), nil
		},
		"atan2": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("atan2() expects exactly two arguments")
			}
			y, ok1 := toFloat64(args[0])
			x, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				d2, ok4 := toDataSlice(args[0])
				d1, ok3 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for atan2(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, x, y, ok3, ok4, cuda.Atan2Govaluate3X3, cuda.Atan2Govaluate3X1, cuda.Atan2Govaluate1X3), nil
			}
			return math.Atan2(y, x), nil
		},
		"hypot": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("hypot() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				d1, ok3 := toDataSlice(args[0])
				d2, ok4 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for hypot(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, x, y, ok3, ok4, cuda.HypotGovaluate3X3, cuda.HypotGovaluate3X1, cuda.HypotGovaluate1X3), nil
			}
			return math.Hypot(x, y), nil
		},
		"remainder": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("remainder() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				d1, ok3 := toDataSlice(args[0])
				d2, ok4 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for remainder(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, x, y, ok3, ok4, cuda.RemainderGovaluate3X3, cuda.RemainderGovaluate3X1, cuda.RemainderGovaluate1X3), nil
			}
			return math.Remainder(x, y), nil
		},
		"max": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 && len(args) != 1 {
				return nil, fmt.Errorf("max() expects two or one argument(s)")
			}
			if len(args) == 2 {
				x, ok1 := toFloat64(args[0])
				y, ok2 := toFloat64(args[1])
				if !ok1 || !ok2 {
					return nil, fmt.Errorf("invalid arguments for max(): %v, %v", args[0], args[1])
				}
				return math.Max(x, y), nil
			} else {
				d, ok3 := toDataSlice(args[0])
				if ok3 {
					dCPU := d.HostCopy()
					s := dCPU.Size()
					tensor := dCPU.Tensors()
					Nx, Ny, Nz := s[X], s[Y], s[Z]

					max := float32(math.Inf(-1))
					for c := 0; c < dCPU.NComp(); c++ {
						for z := 0; z < Nz; z++ {
							// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
							for y := 0; y < Ny; y++ {
								for x := 0; x < Nx; x++ {
									vee := tensor[c][z][y][x]
									if vee > max {
										max = vee
									}
								}
							}
						}
					}
					fmt.Println("max:", max)
					return float64(max), nil
				} else {
					return nil, fmt.Errorf("invalid argument for max(): %v", args[0])
				}
			}
		},
		"min": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 && len(args) != 1 {
				return nil, fmt.Errorf("min() expects two or one argument(s)")
			}
			if len(args) == 2 {
				x, ok1 := toFloat64(args[0])
				y, ok2 := toFloat64(args[1])
				if !ok1 || !ok2 {
					return nil, fmt.Errorf("invalid arguments for min(): %v, %v", args[0], args[1])
				}
				return math.Min(x, y), nil
			} else {
				d, ok3 := toDataSlice(args[0])
				if ok3 {
					dCPU := d.HostCopy()
					s := dCPU.Size()
					tensor := dCPU.Tensors()
					Nx, Ny, Nz := s[X], s[Y], s[Z]

					min := float32(math.Inf(1))
					for c := 0; c < dCPU.NComp(); c++ {
						for z := 0; z < Nz; z++ {
							// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
							for y := 0; y < Ny; y++ {
								for x := 0; x < Nx; x++ {
									vee := tensor[c][z][y][x]
									if vee < min {
										min = vee
									}
								}
							}
						}
					}
					return float64(min), nil
				} else {
					return nil, fmt.Errorf("invalid argument for min(): %v", args[0])
				}
			}
		},
		"mod": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("mod() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				d1, ok3 := toDataSlice(args[0])
				d2, ok4 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for mod(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, x, y, ok3, ok4, cuda.ModGovaluate3X3, cuda.ModGovaluate3X1, cuda.ModGovaluate1X3), nil
			}
			return math.Mod(x, y), nil
		},
		"pow": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("pow() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				d1, ok3 := toDataSlice(args[0])
				d2, ok4 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for pow(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, x, y, ok3, ok4, cuda.PowGovaluate3X3, cuda.PowGovaluate3X1, cuda.PowGovaluate1X3), nil
			}
			return math.Pow(x, y), nil
		},
		"ldexp": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("ldexp() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			exp, ok2 := toInt(args[1])
			if !ok1 || !ok2 {
				d1, ok3 := toDataSlice(args[0])
				d2, ok4 := toDataSlice(args[1])
				if !ok1 && !ok2 && !ok3 && !ok4 {
					return nil, fmt.Errorf("invalid arguments for ldexp(): %v, %v", args[0], args[1])
				}
				return govaluate.GPUCalc(d1, d2, x, float64(exp), ok3, ok4, cuda.LdexpGovaluate3X3, cuda.LdexpGovaluate3X1, cuda.LdexpGovaluate1X3), nil
			}
			return math.Ldexp(x, exp), nil
		},
		"pow10": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("pow10() expects exactly one argument")
			}
			if x, ok := toInt(args[0]); ok {
				return math.Pow10(x), nil
			} else if d, ok := toDataSlice(args[0]); ok {
				cuda.Pow10Govaluate(d)
				return d, nil
			}
			return nil, fmt.Errorf("invalid argument for pow10(): %v", args[0])
		},

		// Random functions (unchanged)
		"randSeed": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("randSeed() expects exactly one argument")
			}
			seed, ok := toInt(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for randSeed(): %v", args[0])
			}
			evaluator.mu.Lock()
			evaluator.rng.Seed(int64(seed))
			evaluator.mu.Unlock()
			return nil, nil
		},
		"rand": func(args ...interface{}) (interface{}, error) {
			if len(args) != 0 {
				return nil, fmt.Errorf("rand() expects no arguments")
			}
			evaluator.mu.Lock()
			val := evaluator.rng.Float64()
			evaluator.mu.Unlock()
			return val, nil
		},
		"randExp": func(args ...interface{}) (interface{}, error) {
			if len(args) != 0 {
				return nil, fmt.Errorf("randExp() expects no arguments")
			}
			evaluator.mu.Lock()
			val := evaluator.rng.ExpFloat64()
			evaluator.mu.Unlock()
			return val, nil
		},
		"randNorm": func(args ...interface{}) (interface{}, error) {
			if len(args) != 0 {
				return nil, fmt.Errorf("randNorm() expects no arguments")
			}
			evaluator.mu.Lock()
			val := evaluator.rng.NormFloat64()
			evaluator.mu.Unlock()
			return val, nil
		},
		"randInt": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("randInt() expects exactly one argument")
			}
			upper, ok := toInt(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for randInt(): %v", args[0])
			}
			evaluator.mu.Lock()
			val := evaluator.rng.Intn(upper)
			evaluator.mu.Unlock()
			return float64(val), nil // cast to float64
		},
	}

	// Compile the expression with custom functions
	expr, err := govaluate.NewEvaluableExpressionWithFunctions(expressionStr, functions)
	if err != nil {
		return nil, fmt.Errorf("error parsing expression: %w", err)
	}

	evaluator.expression = expr

	return evaluator, nil
}

// Evaluate computes the result of the expression given a map of variables
func (e *ExprEvaluator) Evaluate(vars map[string]interface{}) (interface{}, error) {
	result, err := e.expression.Evaluate(vars)
	if err != nil {
		return 0, fmt.Errorf("error evaluating expression: %w", err)
	}

	return result, nil
}

// toFloat64 safely converts various numeric types to float64
func toFloat64(arg interface{}) (float64, bool) {
	switch v := arg.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	default:
		return 0, false
	}
}

func toDataSlice(arg interface{}) (*data.Slice, bool) {
	v, ok := arg.(*data.Slice)
	return v, ok

}

// toInt safely converts various numeric types to int
func toInt(arg interface{}) (int, bool) {
	switch v := arg.(type) {
	case int:
		return v, true
	case int64:
		return int(v), true
	case int32:
		return int(v), true
	case float64:
		return int(v), true
	case float32:
		return int(v), true
	default:
		return 0, false
	}
}

// Custom function implementations

// heaviside returns 1 if x > 0, 0 if x < 0, and 0.5 if x == 0
func heaviside(x float64) float64 {
	switch {
	default:
		return 1
	case x == 0:
		return 0.5
	case x < 0:
		return 0
	}
}

// norm computes the standard normal distribution function
func norm(x float64) float64 {
	return (1 / math.Sqrt(2*math.Pi)) * math.Exp(-0.5*x*x)
}

// sinc returns sin(x)/x. If x == 0, returns 1
func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(x) / x
}

// Function wraps ExprEvaluator to provide a callable interface
type Function struct {
	evaluator *ExprEvaluator
	required  []string
}

// NewFunction creates a new Function instance with the given expression
func NewFunction(expressionStr string) (*Function, error) {
	evaluator, err := NewExprEvaluator(expressionStr)
	if err != nil {
		return nil, err
	}

	requiredVars := evaluator.expression.Vars()

	return &Function{
		evaluator: evaluator,
		required:  requiredVars,
	}, nil
}

// Call evaluates the function with the provided variables
func (f *Function) Call(vars map[string]interface{}) (interface{}, error) {
	// Validate variables
	if err := validateVariables(f.required, vars); err != nil {
		return 0, err
	}

	return f.evaluator.Evaluate(vars)
}

// validateVariables ensures that all required variables are present
func validateVariables(required []string, provided map[string]interface{}) error {
	for _, varName := range required {
		if _, exists := provided[varName]; !exists {
			return fmt.Errorf("missing variable: %s", varName)
		}
	}
	return nil
}

func InitializeVars(varNames []string) (map[string]interface{}, error) {
	varsMap := make(map[string]interface{}, len(varNames))
	for _, name := range varNames {
		varsMap[name] = math.NaN()
	}
	return varsMap, nil
}

func GenerateSliceFromExpr(render *ReadyToRenderFunction, cellsize [3]float64, areaStart, areaEnd [3]int) *data.Slice {
	size := [3]int{areaEnd[X] - areaStart[X], areaEnd[Y] - areaStart[Y], areaEnd[Z] - areaStart[Z]}

	vars := render.Vars
	xDep := render.xDep
	yDep := render.yDep
	zDep := render.zDep
	function := render.f
	if zDep {
		vars["z_length"] = size[Z]
		vars["z_factor"] = cellsize[Z]
	}
	if yDep {
		vars["y_length"] = size[Y]
		vars["y_factor"] = cellsize[Y]
	}
	if xDep {
		vars["x_length"] = size[X]
		vars["x_factor"] = cellsize[X]
	}
	result, err := function.Call(vars)
	if err != nil {
		panic(fmt.Sprintf("Failed to call function: %v", err))
	}
	resultDataSlice, ok1 := result.(*data.Slice)
	resultFloat, ok2 := result.(float64)
	if ok1 {
		if resultDataSlice.Size() != size {
			buf := cuda.Buffer(resultDataSlice.NComp(), size)
			cuda.Zero(buf)
			cuda.AddGovaluate3X3(buf, buf, resultDataSlice)
			cuda.Recycle(resultDataSlice)
			return buf
		} else {
			return resultDataSlice
		}
	}
	if ok2 {
		buf := cuda.Buffer(1, size)
		cuda.Zero(buf)
		cuda.AddGovaluate3X1(buf, buf, float32(resultFloat))
		return buf
	} else {
		panic("Function does not return a data.Slice or float64")
	}

}

func GenerateExprFromFunctionString(functionStr string) (*Function, map[string]interface{}, error) {
	constants := map[string]interface{}{
		"pi":  math.Pi,
		"inf": math.Inf(1),
	}
	function, err := NewFunction(preprocessExpressionScientific(functionStr))
	if err != nil {
		return nil, nil, err
	}

	vars, err := InitializeVars(function.required)
	if err != nil {
		return nil, nil, err
	}

	for key, value := range constants {
		_, ok := vars[key]
		if ok {
			vars[key] = value
		}
	}

	return function, vars, nil
}

func preprocessExpressionScientific(expr string) string {
	if expr == "" {
		return "0"
	}
	// Regular expression to match scientific notation (e.g., 1e7, 3.14e-2)
	re := regexp.MustCompile(`(\d+(\.\d+)?)[eE]([+-]?\d+)`)

	// Replace each scientific notation occurrence with its evaluated value
	return re.ReplaceAllStringFunc(expr, func(match string) string {
		// Parse the matched string into a float64
		f, err := strconv.ParseFloat(match, 64)
		if err != nil {
			// If parsing fails, return the original match
			return match
		}
		// Format the float without wissenschaftlicher Notation
		return strconv.FormatFloat(f, 'f', -1, 64)
	})
}

// preprocessExpressionExpandSum expands sum expressions in the input string using
// different max values provided in the 'max' slice. It expects expressions of the form:
//
//	sum_<index>(<innerExpr>)
//
// For each found sum, it expands the inner expression by replacing occurrences of the
// index variable (both as part of an identifier like slc_i and as a standalone token)
// with numbers from 1 up to the provided max value.
func preprocessExpressionExpandSum(expr string, max []int) string {
	var output strings.Builder
	sumIndex := 0
	i := 0

	for i < len(expr) {
		// Find the next occurrence of "sum_"
		idx := strings.Index(expr[i:], "sum_")
		if idx == -1 {
			// Append the rest of the string if no more sums are found.
			output.WriteString(expr[i:])
			break
		}
		// Write everything up to the found "sum_"
		output.WriteString(expr[i : i+idx])
		i += idx

		// Check that we have at least "sum_X" (where X is the index variable)
		if i+4 >= len(expr) {
			output.WriteString(expr[i:])
			break
		}
		// The character after "sum_" is our index variable.
		indexVar := expr[i+4]
		j := i + 5
		// Skip any whitespace after the index variable.
		for j < len(expr) && (expr[j] == ' ' || expr[j] == '\t') {
			j++
		}
		// The next character must be '(' for a valid sum expression.
		if j >= len(expr) || expr[j] != '(' {
			output.WriteString(expr[i : i+4])
			i += 4
			continue
		}
		// Find the matching closing parenthesis.
		endParen, err := findMatchingParen(expr, j)
		if err != nil {
			panic(err)
		}
		// Extract the inner expression (without the surrounding parentheses).
		innerExpr := expr[j+1 : endParen]

		// If not enough max values are provided, leave this sum expression unchanged.
		if sumIndex >= len(max) {
			output.WriteString(expr[i : endParen+1])
			i = endParen + 1
			continue
		}
		currentMax := max[sumIndex]
		sumIndex++

		var terms []string
		// Expand the inner expression for each index value.
		for k := 0; k < currentMax; k++ {
			replaced := innerExpr

			// Replace occurrences of an underscore followed by the index variable.
			// The regex captures an underscore+index variable and ensures the next character is not a word character.
			underscorePattern := fmt.Sprintf(`(_%c)([^0-9A-Za-z]|$)`, indexVar)
			reUnderscore := regexp.MustCompile(underscorePattern)
			replaced = reUnderscore.ReplaceAllStringFunc(replaced, func(match string) string {
				groups := reUnderscore.FindStringSubmatch(match)
				// groups[1] is the matched "_<indexVar>" and groups[2] is the following character (if any)
				trailing := ""
				if len(groups) >= 3 {
					trailing = groups[2]
				}
				return fmt.Sprintf("_%d%s", k, trailing)
			})

			// Replace standalone occurrences of the index variable as a whole word.
			standalonePattern := fmt.Sprintf(`\b%c\b`, indexVar)
			reStandalone := regexp.MustCompile(standalonePattern)
			replaced = reStandalone.ReplaceAllString(replaced, fmt.Sprintf("%d", k))

			terms = append(terms, replaced)
		}
		// Join all expanded terms with " + " and append to the output.
		output.WriteString(strings.Join(terms, " + "))
		i = endParen + 1
	}
	return output.String()
}

// findMatchingParen returns the index of the matching closing parenthesis for the opening
// parenthesis at position 'start' in the string s. It handles nested parentheses.
func findMatchingParen(s string, start int) (int, error) {
	if s[start] != '(' {
		return -1, fmt.Errorf("expected '(' at position %d", start)
	}
	depth := 0
	for i := start; i < len(s); i++ {
		switch s[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i, nil
			}
		}
	}
	return -1, fmt.Errorf("no matching closing parenthesis found")
}

// containsIndex checks if the given indexPart contains the index variable (as a single character).
func containsIndex(indexPart, indexVar string) bool {
	for _, ch := range indexPart {
		if string(ch) == indexVar {
			return true
		}
	}
	return false
}

// extractIndexedVars extracts indexed variables from each sum expression in the input expression.
// It handles nested parentheses so that functions like sin, cos, etc. can appear.
// Only variables with an underscore index (e.g. a_i, b_j, etc.) that actually include the current
// sum's index variable are returned.
func extractIndexedVars(expr string) [][]string {
	var result [][]string
	i := 0

	// Loop through the entire expression to find sum expressions.
	for i < len(expr) {
		// Find the next occurrence of "sum_"
		idx := strings.Index(expr[i:], "sum_")
		if idx == -1 {
			break
		}
		i += idx

		// Check for a valid "sum_<letter>" pattern.
		if i+4 >= len(expr) {
			break
		}
		// The character after "sum_" is our index variable (e.g., 'i' in "sum_i")
		indexVar := expr[i+4]
		// Skip whitespace until the '(' is reached.
		j := i + 5
		for j < len(expr) && (expr[j] == ' ' || expr[j] == '\t') {
			j++
		}
		if j >= len(expr) || expr[j] != '(' {
			i = j
			continue
		}

		// Find the matching closing parenthesis for this sum expression.
		endParen, err := findMatchingParen(expr, j)
		if err != nil {
			panic(err)
		}
		// Extract the inner expression (without the surrounding parentheses).
		innerExpr := expr[j+1 : endParen]

		// Regex to find variables that have an underscore index.
		// It matches an identifier that contains an underscore followed by at least one letter/digit.
		reVar := regexp.MustCompile(`\b([a-zA-Z]\w*_[a-zA-Z][a-zA-Z0-9]*)\b`)
		matches := reVar.FindAllStringSubmatch(innerExpr, -1)
		unique := map[string]bool{}

		// For each matched variable, verify that the part after '_' contains the current index variable.
		reIndexPart := regexp.MustCompile(`_([a-zA-Z0-9]+)`)
		for _, m := range matches {
			fullVar := m[1] // e.g., "a_i" or "b_j"
			parts := reIndexPart.FindStringSubmatch(fullVar)
			if len(parts) == 2 {
				if containsIndex(parts[1], string(indexVar)) {
					unique[fullVar] = true
				}
			}
		}
		var group []string
		for key := range unique {
			group = append(group, key)
		}
		result = append(result, group)
		i = endParen + 1
	}
	return result
}

func removeIndexStructure(s string, doPanic bool) string {
	// Compile a regular expression that matches underscore followed by one or more letters/digits.
	re := regexp.MustCompile(`_[0-9A-Za-z]+`)

	// Check if the string contains the index structure.
	if !re.MatchString(s) && doPanic {
		panic("Index structure not found in string")
	}

	// Remove all occurrences of the pattern from the string.
	return re.ReplaceAllString(s, "")
}

func extractIndexNumber(s string) int {
	// Regular expression to match an underscore followed by one or more digits.
	re := regexp.MustCompile(`_[0-9]+`)

	// Find the first occurrence that matches the pattern.
	match := re.FindString(s)
	if match == "" {
		panic("no index number found in " + s)
	}

	// Remove the underscore to get only the digits.
	numStr := match[1:]

	// Convert the digit string to an integer.
	num, err := strconv.Atoi(numStr)
	if err != nil {
		panic(fmt.Sprintf("failed to convert %s to integer: %v", numStr, err))
	}

	return num
}

type ReadyToRenderFunction struct {
	f       *Function
	Vars    map[string]interface{}
	TimeDep bool
	xDep    bool
	yDep    bool
	zDep    bool
}

func RenderStringToReadyToRenderFct(functionStr string, mesh *data.Mesh) *ReadyToRenderFunction {
	indexedVars := extractIndexedVars(functionStr)
	worldVars := World.GetRuntimeVariables()
	lengthIndexedVars := make([]int, 0)
	for _, sum := range indexedVars {
		referenceLen := 0
		for _, vari := range sum {
			if value, ok := worldVars[strings.ToLower(vari)]; ok {
				if slice, ok := value.([]float64); ok {
					if referenceLen == 0 {
						referenceLen = len(slice)
						lengthIndexedVars = append(lengthIndexedVars, referenceLen)
					} else if referenceLen != len(slice) {
						panic("Indexed variables need to have the same length per sum")
					}
				} else {
					panic("Unsuppported type in indexed variables.")
				}
			} else {
				panic("Could not find variable: " + vari)
			}
		}
	}
	function, vars, err := GenerateExprFromFunctionString(preprocessExpressionExpandSum(preprocessExpressionScientific(functionStr), lengthIndexedVars))
	if err != nil {
		panic(fmt.Sprintf("Failed to generate function: %v\n%s", err, preprocessExpressionExpandSum(preprocessExpressionScientific(functionStr), lengthIndexedVars)))
	}
	var (
		xDep    = false
		yDep    = false
		zDep    = false
		TimeDep = false
	)
	for key := range vars {
		if key != "x_length" && key != "y_length" && key != "z_length" && key != "x_factor" && key != "y_factor" && key != "z_factor" && key != "t" && math.IsNaN(vars[key].(float64)) {
			//fmt.Println(key, s.variablesStart[j][key].vector[comp], s.variablesEnd[j][key].vector[comp])
			foundVal := false
			for _, sum := range indexedVars {
				breakLoop := false
				for _, vari := range sum {
					if removeIndexStructure(vari, true) == removeIndexStructure(key, false) {
						if value, ok := worldVars[strings.ToLower(vari)]; ok {
							if slice, ok := value.([]float64); ok {
								vars[key] = slice[extractIndexNumber(key)]
								breakLoop = true
								foundVal = true
								break
							} else {
								panic("Unsuppported type in indexed variables.")
							}
						} else {
							panic(fmt.Sprintf("Variable %s not defined.", key))
						}
					}
				}
				if breakLoop {
					break
				}
			}
			if !foundVal {
				if value, ok := worldVars[strings.ToLower(key)]; ok {
					vars[key] = value
				} else {
					panic(fmt.Sprintf("Variable %s not defined.", key))
				}
			}

		} else if strings.ToLower(key) == "t" {
			vars[key] = Time
			TimeDep = true
		} else if key == "x_factor" {
			xDep = true
		} else if key == "y_factor" {
			yDep = true
		} else if key == "z_factor" {
			zDep = true
		}
	}
	renderFct := new(ReadyToRenderFunction)
	renderFct.f = function
	renderFct.Vars = vars
	renderFct.xDep = xDep
	renderFct.yDep = yDep
	renderFct.zDep = zDep
	renderFct.TimeDep = TimeDep
	return renderFct
}

func GenerateSliceFromReadyToRenderFct(renderers []*ReadyToRenderFunction, mesh *data.Mesh) (*data.Slice, bool) {
	var dataFused *data.Slice
	TimeDep := renderers[0].TimeDep
	data0 := GenerateSliceFromExpr(renderers[0], mesh.CellSize(), [3]int{0, 0, 0}, mesh.Size())
	if len(renderers) == 3 {
		data123 := make([]*data.Slice, 3)
		for c := 1; c <= 2; c++ {
			if renderers[c].TimeDep {
				TimeDep = true
			}
			data123[c] = GenerateSliceFromExpr(renderers[c], mesh.CellSize(), [3]int{0, 0, 0}, mesh.Size())
		}
		data123[0] = data0
		dataFused = data.SliceFromSlices(data123, mesh.Size())

	} else {
		dataFused = data.SliceFromSlices([]*data.Slice{data0}, mesh.Size())
	}
	return dataFused, TimeDep
}

func CreateFloatSlice(args ...float64) []float64 {
	return args
}

func CreateFloatSliceZero(size int) []float64 {
	ar := make([]float64, size)
	for i := range ar {
		ar[i] = 0
	}
	return ar
}

func CreateFloatSliceOne(size int) []float64 {
	ar := make([]float64, size)
	for i := range ar {
		ar[i] = 1
	}
	return ar
}

func CreateFloatSliceArb(size int, val float64) []float64 {
	ar := make([]float64, size)
	for i := range ar {
		ar[i] = val
	}
	return ar
}
