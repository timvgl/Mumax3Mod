package engine

import (
	"fmt"
	"math"
	"math/rand"
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
					return cuda.MaxGovaluate(d), nil
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
					return cuda.MinGovaluate(d), nil
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

func GenerateSliceFromExpr(xDep, yDep, zDep bool, vars map[string]interface{}, cellsize [3]float64, function *Function, areaStart, areaEnd [3]int) *data.Slice {
	size := [3]int{areaEnd[X] - areaStart[X], areaEnd[Y] - areaStart[Y], areaEnd[Z] - areaStart[Z]}
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
	function, err := NewFunction(functionStr)
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
