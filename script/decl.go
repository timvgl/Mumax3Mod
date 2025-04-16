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

// an expression can be evaluated
type Function struct {
	Name                string
	Params              []string
	Body                *Expr
	FunctionTypeReflect reflect.Type
	FunctionType        *ast.FuncType
	void
}

func (f *Function) Eval(args ...interface{}) interface{} {
	// 1. Den AST des Funktions-Typs in einen String formatieren.
	fset := token.NewFileSet()
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, f.FunctionType); err != nil {
		panic(fmt.Errorf("Fehler beim Formatieren des FuncType: %w", err))
	}
	typeStr := buf.String() // z.B. "func(x int) bool"

	// 2. Den Body als String abrufen. (Hier muss f.Body.String() implementiert sein.)
	bodyStr := f.Body

	// 3. Einen vollständigen Funktionsliteral-String konstruieren.
	// Dabei wird der Typ mit dem Body kombiniert.
	completeFuncStr := fmt.Sprintf("%s { %s }", typeStr, bodyStr)
	fmt.Println(completeFuncStr)

	// 4. Einen neuen yaegi-Interpreter erstellen und die Standardbibliothek einbinden.
	i := interp.New(interp.Options{})
	i.Use(stdlib.Symbols)

	// 5. Den kompletten Funktionsliteral-String evaluieren.
	v, err := i.Eval(completeFuncStr)
	if err != nil {
		panic(fmt.Errorf("Fehler beim Evaluieren des Funktionsliterals: %w", err))
	}

	// 6. Den Laufzeittyp des erzeugten Funktionswerts speichern.
	f.FunctionTypeReflect = reflect.TypeOf(v.Interface())

	// 7. Den Funktionswert zurückgeben.
	return v.Interface()
}

// Child gibt den Funktionskörper als einziges Kind zurück (zur Traversierung des AST).
func (f *Function) Child() []Expr {
	return []Expr{*f.Body}
}

func (e *Function) Fix() Expr { return e }

func (e *Function) Type() reflect.Type { return e.FunctionTypeReflect }

// compiles an expression
func (w *World) compileDecl(e ast.Decl) Expr {
	switch e := e.(type) {
	default:
		panic(err(e.Pos(), "not allowed:", typ(e)))
	case *ast.FuncDecl:
		return w.compileFuncDecl(e)
	}

}
