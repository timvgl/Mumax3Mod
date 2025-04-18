package govaluate

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

const (
	logicalErrorFormat    string = "Value '%v' cannot be used with the logical operator '%v', it is not a bool"
	modifierErrorFormat   string = "Value '%v' cannot be used with the modifier '%v', it is not a number"
	comparatorErrorFormat string = "Value '%v' cannot be used with the comparator '%v', it is not a number"
	ternaryErrorFormat    string = "Value '%v' cannot be used with the ternary operator '%v', it is not a bool"
	prefixErrorFormat     string = "Value '%v' cannot be used with the prefix '%v'"
)

type evaluationOperator func(left interface{}, right interface{}, parameters Parameters) (interface{}, error)
type stageTypeCheck func(value interface{}) bool
type stageCombinedTypeCheck func(left interface{}, right interface{}) bool

type evaluationStage struct {
	symbol OperatorSymbol

	leftStage, rightStage *evaluationStage

	// the operation that will be used to evaluate this stage (such as adding [left] to [right] and return the result)
	operator evaluationOperator

	// ensures that both left and right values are appropriate for this stage. Returns an error if they aren't operable.
	leftTypeCheck  stageTypeCheck
	rightTypeCheck stageTypeCheck

	// if specified, will override whatever is used in "leftTypeCheck" and "rightTypeCheck".
	// primarily used for specific operators that don't care which side a given type is on, but still requires one side to be of a given type
	// (like string concat)
	typeCheck stageCombinedTypeCheck

	// regardless of which type check is used, this string format will be used as the error message for type errors
	typeErrorFormat string

	GPULeft  bool
	GPURight bool
}

var (
	_true  = interface{}(true)
	_false = interface{}(false)
)

func (this *evaluationStage) swapWith(other *evaluationStage) {

	temp := *other
	other.setToNonStage(*this)
	this.setToNonStage(temp)
}

func (this *evaluationStage) setToNonStage(other evaluationStage) {
	this.symbol = other.symbol
	this.operator = other.operator
	this.leftTypeCheck = other.leftTypeCheck
	this.rightTypeCheck = other.rightTypeCheck
	this.typeCheck = other.typeCheck
	this.typeErrorFormat = other.typeErrorFormat
}

func (this *evaluationStage) isShortCircuitable() bool {

	switch this.symbol {
	case AND:
		fallthrough
	case OR:
		fallthrough
	case TERNARY_TRUE:
		fallthrough
	case TERNARY_FALSE:
		fallthrough
	case COALESCE:
		return true
	}

	return false
}

func noopStageRight(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return right, nil
}

func GPUCalc(gpuLeft, gpuRight *data.Slice, leftFloat, rightFloat float64, gpuLeftOk, gpuRightOk bool, operator3X3 func(dst, left, right *data.Slice), operator3X1 func(dst, left *data.Slice, right interface{}), operator1X3 func(dst *data.Slice, left interface{}, right *data.Slice)) *data.Slice {
	var buf *data.Slice
	if gpuLeftOk && gpuRightOk {
		var size = [3]int{}
		sizeLeft := gpuLeft.Size()
		sizeRight := gpuRight.Size()
		if sizeLeft[0] == 1 && sizeRight[0] > 1 {
			size[0] = sizeRight[0]
		} else if sizeLeft[0] > 1 && sizeRight[0] == 1 {
			size[0] = sizeLeft[0]
		} else if sizeLeft[0] == sizeRight[0] {
			size[0] = sizeLeft[0]
		} else {
			panic("Nx has either to be the same or one of Nx has to be one")
		}
		if sizeLeft[1] == 1 && sizeRight[1] > 1 {
			size[1] = sizeRight[1]
		} else if sizeLeft[1] > 1 && sizeRight[1] == 1 || sizeLeft[1] == sizeRight[1] {
			size[1] = sizeLeft[1]
		} else {
			panic("Ny has either to be the same or one of Ny has to be one")
		}
		if sizeLeft[2] == 1 && sizeRight[2] > 1 {
			size[2] = sizeRight[2]
		} else if sizeLeft[2] > 1 && sizeRight[2] == 1 || sizeLeft[2] == sizeRight[2] {
			size[2] = sizeLeft[2]
		} else {
			panic("Ny has either to be the same or one of Ny has to be one")
		}
		buf = cuda.Buffer(1, size)
		operator3X3(buf, gpuLeft, gpuRight)
		cuda.Recycle(gpuLeft)
		cuda.Recycle(gpuRight)
	} else if gpuLeftOk && !gpuRightOk {
		buf = cuda.Buffer(1, gpuLeft.Size())
		operator3X1(buf, gpuLeft, float32(rightFloat))
		cuda.Recycle(gpuLeft)
	} else if !gpuLeftOk && gpuRightOk {
		buf = cuda.Buffer(1, gpuRight.Size())
		operator1X3(buf, float32(leftFloat), gpuRight)
		cuda.Recycle(gpuRight)
	}
	return buf
}

func addStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {

	// string concat if either are strings
	if isString(left) || isString(right) {
		return fmt.Sprintf("%v%v", left, right), nil
	}
	leftFloat, okLeft := left.(float64)
	rightFloat, okRight := right.(float64)
	gpuLeft, gpuLeftOk := left.(*data.Slice)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okLeft && okRight {
		return leftFloat + rightFloat, nil
	} else {
		return GPUCalc(gpuLeft, gpuRight, leftFloat, rightFloat, gpuLeftOk, gpuRightOk, cuda.AddGovaluate3X3, cuda.AddGovaluate3X1, cuda.AddGovaluate1X3), nil
	}
}
func subtractStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	leftFloat, okLeft := left.(float64)
	rightFloat, okRight := right.(float64)
	gpuLeft, gpuLeftOk := left.(*data.Slice)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okLeft && okRight {
		return leftFloat - rightFloat, nil
	} else {
		return GPUCalc(gpuLeft, gpuRight, leftFloat, rightFloat, gpuLeftOk, gpuRightOk, cuda.SubGovaluate3X3, cuda.SubGovaluate3X1, cuda.SubGovaluate1X3), nil
	}
}
func multiplyStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	leftFloat, okLeft := left.(float64)
	rightFloat, okRight := right.(float64)
	gpuLeft, gpuLeftOk := left.(*data.Slice)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okLeft && okRight {
		return leftFloat * rightFloat, nil
	} else {
		return GPUCalc(gpuLeft, gpuRight, leftFloat, rightFloat, gpuLeftOk, gpuRightOk, cuda.MulGovaluate3X3, cuda.MulGovaluate3X1, cuda.MulGovaluate1X3), nil
	}
}
func divideStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	leftFloat, okLeft := left.(float64)
	rightFloat, okRight := right.(float64)
	gpuLeft, gpuLeftOk := left.(*data.Slice)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okLeft && okRight {
		return leftFloat / rightFloat, nil
	} else {
		return GPUCalc(gpuLeft, gpuRight, leftFloat, rightFloat, gpuLeftOk, gpuRightOk, cuda.DivGovaluate3X3, cuda.DivGovaluate3X1, cuda.DivGovaluate1X3), nil
	}
}
func exponentStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	leftFloat, okLeft := left.(float64)
	rightFloat, okRight := right.(float64)
	gpuLeft, gpuLeftOk := left.(*data.Slice)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okLeft && okRight {
		return math.Pow(leftFloat, rightFloat), nil
	} else {
		return GPUCalc(gpuLeft, gpuRight, leftFloat, rightFloat, gpuLeftOk, gpuRightOk, cuda.PowGovaluate3X3, cuda.PowGovaluate3X1, cuda.PowGovaluate1X3), nil
	}
}
func modulusStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	leftFloat, okLeft := left.(float64)
	rightFloat, okRight := right.(float64)
	gpuLeft, gpuLeftOk := left.(*data.Slice)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okLeft && okRight {
		return math.Mod(leftFloat, rightFloat), nil
	} else {
		return GPUCalc(gpuLeft, gpuRight, leftFloat, rightFloat, gpuLeftOk, gpuRightOk, cuda.ModGovaluate3X3, cuda.ModGovaluate3X1, cuda.ModGovaluate1X3), nil
	}
}
func gteStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	if isString(left) && isString(right) {
		return boolIface(left.(string) >= right.(string)), nil
	}
	return boolIface(left.(float64) >= right.(float64)), nil
}
func gtStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	if isString(left) && isString(right) {
		return boolIface(left.(string) > right.(string)), nil
	}
	return boolIface(left.(float64) > right.(float64)), nil
}
func lteStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	if isString(left) && isString(right) {
		return boolIface(left.(string) <= right.(string)), nil
	}
	return boolIface(left.(float64) <= right.(float64)), nil
}
func ltStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	if isString(left) && isString(right) {
		return boolIface(left.(string) < right.(string)), nil
	}
	return boolIface(left.(float64) < right.(float64)), nil
}
func equalStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return boolIface(reflect.DeepEqual(left, right)), nil
}
func notEqualStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return boolIface(!reflect.DeepEqual(left, right)), nil
}
func andStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return boolIface(left.(bool) && right.(bool)), nil
}
func orStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return boolIface(left.(bool) || right.(bool)), nil
}
func negateStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	rightFloat, okRight := right.(float64)
	gpuRight, gpuRightOk := right.(*data.Slice)
	if okRight {
		return -rightFloat, nil
	} else {
		return GPUCalc(nil, gpuRight, 0, rightFloat, false, gpuRightOk, nil, nil, cuda.NegateGovaluate), nil
	}
}
func invertStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return boolIface(!right.(bool)), nil
}
func bitwiseNotStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return float64(^int64(right.(float64))), nil
}
func ternaryIfStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	if left.(bool) {
		return right, nil
	}
	return nil, nil
}
func ternaryElseStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	if left != nil {
		return left, nil
	}
	return right, nil
}

func regexStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {

	var pattern *regexp.Regexp
	var err error

	switch right.(type) {
	case string:
		pattern, err = regexp.Compile(right.(string))
		if err != nil {
			return nil, errors.New(fmt.Sprintf("Unable to compile regexp pattern '%v': %v", right, err))
		}
	case *regexp.Regexp:
		pattern = right.(*regexp.Regexp)
	}

	return pattern.Match([]byte(left.(string))), nil
}

func notRegexStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {

	ret, err := regexStage(left, right, parameters)
	if err != nil {
		return nil, err
	}

	return !(ret.(bool)), nil
}

func bitwiseOrStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return float64(int64(left.(float64)) | int64(right.(float64))), nil
}
func bitwiseAndStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return float64(int64(left.(float64)) & int64(right.(float64))), nil
}
func bitwiseXORStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return float64(int64(left.(float64)) ^ int64(right.(float64))), nil
}
func leftShiftStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return float64(uint64(left.(float64)) << uint64(right.(float64))), nil
}
func rightShiftStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
	return float64(uint64(left.(float64)) >> uint64(right.(float64))), nil
}

func makeParameterStage(parameterName string) evaluationOperator {

	return func(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
		value, err := parameters.Get(parameterName)
		if err != nil {
			return nil, err
		}

		return value, nil
	}
}

func makeParameterStageGPU(parameterName string) evaluationOperator {
	return func(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
		value_length, err := parameters.Get(parameterName + "_length")
		if err != nil {
			return nil, err
		}
		value_factor, err := parameters.Get(parameterName + "_factor")
		if err != nil {
			return nil, err
		}
		var buf *data.Slice
		if parameterName == "x" {
			buf = cuda.Buffer(1, [3]int{int(value_length.(float64)), 1, 1})

		} else if parameterName == "y" {
			buf = cuda.Buffer(1, [3]int{1, int(value_length.(float64)), 1})

		} else if parameterName == "z" {
			buf = cuda.Buffer(1, [3]int{1, 1, int(value_length.(float64))})

		}
		cuda.Fill1DWithCoords(buf, float32(value_factor.(float64)))

		return buf, nil
	}
}

func makeLiteralStage(literal interface{}) evaluationOperator {
	return func(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {
		return literal, nil
	}
}

func makeFunctionStage(function ExpressionFunction) evaluationOperator {

	return func(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {

		if right == nil {
			return function()
		}

		switch right.(type) {
		case []interface{}:
			return function(right.([]interface{})...)
		default:
			return function(right)
		}

	}
}

func separatorStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {

	var ret []interface{}

	switch left.(type) {
	case []interface{}:
		ret = append(left.([]interface{}), right)
	default:
		ret = []interface{}{left, right}
	}

	return ret, nil
}

func inStage(left interface{}, right interface{}, parameters Parameters) (interface{}, error) {

	for _, value := range right.([]interface{}) {
		if left == value {
			return true, nil
		}
	}
	return false, nil
}

//

func isString(value interface{}) bool {

	switch value.(type) {
	case string:
		return true
	}
	return false
}

func isRegexOrString(value interface{}) bool {

	switch value.(type) {
	case string:
		return true
	case *regexp.Regexp:
		return true
	}
	return false
}

func isBool(value interface{}) bool {
	switch value.(type) {
	case bool:
		return true
	}
	return false
}

func isFloat64(value interface{}) bool {
	switch value.(type) {
	case float64:
		return true
	}
	return false
}

func isDataSlice(value interface{}) bool {
	switch value.(type) {
	case *data.Slice:
		return true
	}
	return false
}

/*
Addition usually means between numbers, but can also mean string concat.
String concat needs one (or both) of the sides to be a string.
*/
func additionTypeCheck(left interface{}, right interface{}) bool {

	if isFloat64(left) && isFloat64(right) {
		return true
	}
	if isDataSlice(left) && isDataSlice(right) || isFloat64(left) && isDataSlice(right) || isDataSlice(left) && isFloat64(right) {
		return true
	}
	if !isString(left) && !isString(right) {
		return false
	}
	return true
}

/*
Comparison can either be between numbers, or lexicographic between two strings,
but never between the two.
*/
func comparatorTypeCheck(left interface{}, right interface{}) bool {

	if isFloat64(left) && isFloat64(right) {
		return true
	}
	if isString(left) && isString(right) {
		return true
	}
	return false
}

func isArray(value interface{}) bool {
	switch value.(type) {
	case []interface{}:
		return true
	}
	return false
}

/*
Converting a boolean to an interface{} requires an allocation.
We can use interned bools to avoid this cost.
*/
func boolIface(b bool) interface{} {
	if b {
		return _true
	}
	return _false
}
