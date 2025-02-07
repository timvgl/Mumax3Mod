package engine

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/Knetic/govaluate"
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
		// Basic math functions from the math package
		"abs": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("abs() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for abs(): %v", args[0])
			}
			return math.Abs(x), nil
		},
		"acos": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("acos() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for acos(): %v", args[0])
			}
			return math.Acos(x), nil
		},
		"acosh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("acosh() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for acosh(): %v", args[0])
			}
			return math.Acosh(x), nil
		},
		"asin": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("asin() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for asin(): %v", args[0])
			}
			return math.Asin(x), nil
		},
		"asinh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("asinh() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for asinh(): %v", args[0])
			}
			return math.Asinh(x), nil
		},
		"atan": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("atan() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for atan(): %v", args[0])
			}
			return math.Atan(x), nil
		},
		"atanh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("atanh() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for atanh(): %v", args[0])
			}
			return math.Atanh(x), nil
		},
		"cbrt": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("cbrt() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for cbrt(): %v", args[0])
			}
			return math.Cbrt(x), nil
		},
		"ceil": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("ceil() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for ceil(): %v", args[0])
			}
			return math.Ceil(x), nil
		},
		"cos": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("cos() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for cos(): %v", args[0])
			}
			return math.Cos(x), nil
		},
		"cosh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("cosh() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for cosh(): %v", args[0])
			}
			return math.Cosh(x), nil
		},
		"erf": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("erf() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for erf(): %v", args[0])
			}
			return math.Erf(x), nil
		},
		"erfc": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("erfc() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for erfc(): %v", args[0])
			}
			return math.Erfc(x), nil
		},
		"exp": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("exp() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for exp(): %v", args[0])
			}
			return math.Exp(x), nil
		},
		"exp2": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("exp2() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for exp2(): %v", args[0])
			}
			return math.Exp2(x), nil
		},
		"expm1": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("expm1() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for expm1(): %v", args[0])
			}
			return math.Expm1(x), nil
		},
		"floor": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("floor() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for floor(): %v", args[0])
			}
			return math.Floor(x), nil
		},
		"gamma": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("gamma() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for gamma(): %v", args[0])
			}
			return math.Gamma(x), nil
		},
		"j0": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("j0() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for j0(): %v", args[0])
			}
			return math.J0(x), nil
		},
		"j1": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("j1() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for j1(): %v", args[0])
			}
			return math.J1(x), nil
		},
		"log": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for log(): %v", args[0])
			}
			return math.Log(x), nil
		},
		"log10": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log10() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for log10(): %v", args[0])
			}
			return math.Log10(x), nil
		},
		"log1p": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log1p() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for log1p(): %v", args[0])
			}
			return math.Log1p(x), nil
		},
		"log2": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("log2() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for log2(): %v", args[0])
			}
			return math.Log2(x), nil
		},
		"logb": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("logb() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for logb(): %v", args[0])
			}
			return math.Logb(x), nil
		},
		"sin": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sin() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for sin(): %v", args[0])
			}
			return math.Sin(x), nil
		},
		"sinh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sinh() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for sinh(): %v", args[0])
			}
			return math.Sinh(x), nil
		},
		"sqrt": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sqrt() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for sqrt(): %v", args[0])
			}
			return math.Sqrt(x), nil
		},
		"tan": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("tan() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for tan(): %v", args[0])
			}
			return math.Tan(x), nil
		},
		"tanh": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("tanh() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for tanh(): %v", args[0])
			}
			return math.Tanh(x), nil
		},
		"trunc": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("trunc() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for trunc(): %v", args[0])
			}
			return math.Trunc(x), nil
		},
		"y0": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("y0() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for y0(): %v", args[0])
			}
			return math.Y0(x), nil
		},
		"y1": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("y1() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for y1(): %v", args[0])
			}
			return math.Y1(x), nil
		},
		"yn": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("yn() expects exactly two arguments")
			}
			n, ok1 := toInt(args[0])
			x, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("invalid arguments for yn(): %v, %v", args[0], args[1])
			}
			return math.Yn(n, x), nil
		},
		"ilogb": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("ilogb() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for ilogb(): %v", args[0])
			}
			return math.Ilogb(x), nil
		},
		"pow10": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("pow10() expects exactly one argument")
			}
			x, ok := toInt(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for pow10(): %v", args[0])
			}
			return math.Pow10(x), nil
		},
		"atan2": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("atan2() expects exactly two arguments")
			}
			y, ok1 := toFloat64(args[0])
			x, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("invalid arguments for atan2(): %v, %v", args[0], args[1])
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
				return nil, fmt.Errorf("invalid arguments for hypot(): %v, %v", args[0], args[1])
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
				return nil, fmt.Errorf("invalid arguments for remainder(): %v, %v", args[0], args[1])
			}
			return math.Remainder(x, y), nil
		},
		"max": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("max() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("invalid arguments for max(): %v, %v", args[0], args[1])
			}
			return math.Max(x, y), nil
		},
		"min": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("min() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("invalid arguments for min(): %v, %v", args[0], args[1])
			}
			return math.Min(x, y), nil
		},
		"mod": func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("mod() expects exactly two arguments")
			}
			x, ok1 := toFloat64(args[0])
			y, ok2 := toFloat64(args[1])
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("invalid arguments for mod(): %v, %v", args[0], args[1])
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
				return nil, fmt.Errorf("invalid arguments for pow(): %v, %v", args[0], args[1])
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
				return nil, fmt.Errorf("invalid arguments for ldexp(): %v, %v", args[0], args[1])
			}
			return math.Ldexp(x, exp), nil
		},

		// Custom functions
		"heaviside": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("heaviside() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for heaviside(): %v", args[0])
			}
			return heaviside(x), nil
		},
		"norm": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("norm() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for norm(): %v", args[0])
			}
			return norm(x), nil
		},
		"sinc": func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("sinc() expects exactly one argument")
			}
			x, ok := toFloat64(args[0])
			if !ok {
				return nil, fmt.Errorf("invalid argument for sinc(): %v", args[0])
			}
			return sinc(x), nil
		},
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
			return float64(val), nil // Cast to float64
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
func (e *ExprEvaluator) Evaluate(vars map[string]interface{}) (float64, error) {
	result, err := e.expression.Evaluate(vars)
	if err != nil {
		return 0, fmt.Errorf("error evaluating expression: %w", err)
	}

	floatResult, ok := toFloat64(result)
	if !ok {
		return 0, fmt.Errorf("result is not a float64: %v", result)
	}

	return floatResult, nil
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
func (f *Function) Call(vars map[string]interface{}) (float64, error) {
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
