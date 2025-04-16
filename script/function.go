package script

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"reflect"

	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

type function struct {
	reflect.Value
}

func newFunction(fn interface{}) *function {
	val := reflect.ValueOf(fn)
	if val.Type().Kind() != reflect.Func {
		panic(fmt.Errorf("not a function: %v", val.Type()))
	}
	if val.Type().NumOut() > 1 {
		panic(fmt.Errorf("multiple return values not allowed: %v", val.Type()))
	}
	return &function{val}
}

// type of the function itself (when not called)
func (f *function) Type() reflect.Type    { return f.Value.Type() }
func (f *function) NumIn() int            { return f.Type().NumIn() }
func (f *function) In(i int) reflect.Type { return f.Type().In(i) }
func (f *function) Eval() interface{}     { return f.Value.Interface() }
func (f *function) Child() []Expr         { return nil }
func (f *function) Fix() Expr             { return f }

func reflectTypeFromFuncType(ft *ast.FuncType) reflect.Type {
	// Erzeuge einen neuen token.FileSet.
	fset := token.NewFileSet()

	// Formatiere den *ast.FuncType in einen String, z.B. "func(x int) bool".
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, ft); err != nil {
		panic(fmt.Errorf("Fehler beim Formatieren des FuncType: %w", err))
	}
	typeStr := buf.String()

	// Konstruiere einen vollständigen Funktionsliteral-String,
	// indem ein Dummy-Body angehängt wird.
	completeFuncStr := fmt.Sprintf("%s { panic(0) }", typeStr)

	// Erzeuge einen neuen yaegi-Interpreter und binde die Standardbibliothek ein.
	i := interp.New(interp.Options{})
	i.Use(stdlib.Symbols)

	// Evaluiere den Funktionsliteral-String.
	v, err := i.Eval(completeFuncStr)
	if err != nil {
		panic(fmt.Errorf("Fehler beim Evaluieren des Funktionsliterals: %w", err))
	}

	// Gib den Laufzeittyp (reflect.Type) des ausgewerteten Funktionswerts zurück.
	return reflect.TypeOf(v.Interface())
}

func (w *World) compileFuncDecl(d *ast.FuncDecl) *Function {
	// 1. Funktionsnamen extrahieren.
	w.EnterScope()
	defer w.ExitScope()
	name := d.Name.Name

	// 2. Parameter extrahieren.
	var params []string
	if d.Type.Params != nil {
		for _, field := range d.Type.Params.List {
			for _, ident := range field.Names {
				params = append(params, ident.Name)
				w.safeDeclare(ident.Name, &nop{})
			}
		}
	}
	_, ok := w.Lookup(params[0])
	if ok {
		fmt.Println("yeeeeeeeeeeee")
		fmt.Println(params[0])
	} else {
		fmt.Println("npooo")
	}
	var body Expr
	if d.Body != nil {
		body = w.compileStmt(d.Body)
	} else {
		body = &nop{}
	}

	fn := &Function{
		Name:                name,
		Params:              params,
		Body:                &body,
		FunctionTypeReflect: reflectTypeFromFuncType(d.Type),
		FunctionType:        d.Type,
	}
	w.Identifiers[name] = fn
	return fn
}

func (w *World) compileFuncLit(d *ast.FuncLit) *Function {
	// 1. Funktionsnamen extrahieren.
	w.EnterScope()
	defer w.ExitScope()

	// 2. Parameter extrahieren.
	var params []string
	if d.Type.Params != nil {
		for _, field := range d.Type.Params.List {
			for _, ident := range field.Names {
				params = append(params, ident.Name)
			}
		}
	}
	var body Expr
	if d.Body != nil {
		body = w.compileBlockStmt_noScope(d.Body)
	} else {
		body = &nop{}
	}

	fn := &Function{
		Name:                "",
		Params:              params,
		Body:                &body,
		FunctionTypeReflect: reflectTypeFromFuncType(d.Type),
		FunctionType:        d.Type,
	}
	return fn
}
