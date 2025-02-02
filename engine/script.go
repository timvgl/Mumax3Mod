package engine

// declare functionality for interpreted input scripts

import (
	"reflect"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/logUI"
	"github.com/mumax/3/script"
)

var QuantityChanged = make(map[string]bool)

func CompileFile(fname string) (*script.BlockStmt, error) {
	bytes, err := httpfs.Read(fname)
	if err != nil {
		return nil, err
	}
	return World.Compile(string(bytes))
}

func Eval(code string) {
	tree, err := World.Compile(code)
	if err != nil {
		LogIn(code)
		LogErr(err.Error())
		logUI.Log.Command(code)
		logUI.Log.Err("%v", err.Error())
		return
	}
	LogIn(rmln(tree.Format()))
	tree.Eval()
}

func Eval1Line(code string) interface{} {
	tree, err := World.Compile(code)
	if err != nil {
		LogErr(err.Error())
		return nil
	}
	if len(tree.Children) != 1 {
		LogErr("expected single statement:" + code)
		return nil
	}
	return tree.Children[0].Eval()
}

// holds the script state (variables etc)
var World = script.NewWorld()

// Add a function to the script world
func DeclFunc(name string, f interface{}, doc string) {
	World.Func(name, f, doc)
}

// Add a constant to the script world
func DeclConst(name string, value float64, doc string) {
	World.Const(name, value, doc)
}

// Add a read-only variable to the script world.
// It can be changed, but not by the user.
func DeclROnly(name string, value interface{}, doc string) {
	World.ROnly(name, value, doc)
	addQuantity(name, value, doc)
}

func Export(q interface {
	Name() string
	Unit() string
}, doc string) {
	DeclROnly(q.Name(), q, cat(doc, q.Unit()))
}

// Add a (pointer to) variable to the script world
func DeclVar(name string, value interface{}, doc string) {
	World.Var(name, value, doc)
	addQuantity(name, value, doc)
}

// Hack for fixing the closure caveat:
// Defines "t", the time variable, handled specially by Fix()
func DeclTVar(name string, value interface{}, doc string) {
	World.TVar(name, value, doc)
	addQuantity(name, value, doc)
}

// Add an LValue to the script world.
// Assign to LValue invokes SetValue()
func DeclLValue(name string, value LValue, doc string) {
	addParameter(name, value, doc)
	World.LValue(name, newLValueWrapper(name, value), doc)
	addQuantity(name, value, doc)
}

// LValue is settable
type LValue interface {
	SetValue(interface{}) // assigns a new value
	Eval() interface{}    // evaluate and return result (nil for void)
	Type() reflect.Type   // type that can be assigned and will be returned by Eval
}

// evaluate code, exit on error (behavior for input files)
func EvalFile(code *script.BlockStmt) {
	for i := range code.Children {
		formatted := rmln(script.Format(code.Node[i]))
		LogIn(formatted)
		logUI.Log.Command(formatted)
		code.Children[i].Eval()
	}
}

// wraps LValue and provides empty Child()
type lValueWrapper struct {
	LValue
	name string
}

func newLValueWrapper(name string, lv LValue) script.LValue {
	return &lValueWrapper{name: name, LValue: lv}
}

func (w *lValueWrapper) Child() []script.Expr { return nil }
func (w *lValueWrapper) Fix() script.Expr     { return script.NewConst(w) }

func (w *lValueWrapper) InputType() reflect.Type {
	if i, ok := w.LValue.(interface {
		InputType() reflect.Type
	}); ok {
		return i.InputType()
	} else {
		return w.Type()
	}
}

func (w *lValueWrapper) SetValue(val interface{}) {
	w.LValue.SetValue(val)
	QuantityChanged[w.name] = true
}

func EvalTryRecover(code string) {
	defer func() {
		if err := recover(); err != nil {
			if userErr, ok := err.(UserErr); ok {
				logUI.Log.Err("%v", userErr)
			} else {
				panic(err)
			}
		}
	}()
	Eval(code)
}
