// mumax3 main command
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"strconv"
	"sync"
	"time"

	"github.com/mumax/3/api"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/logUI"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
)

var (
	flag_failfast    = flag.Bool("failfast", false, "If one simulation fails, stop entire batch immediately")
	flag_test        = flag.Bool("test", false, "Cuda test (internal)")
	flag_version     = flag.Bool("v", true, "Print version")
	flag_vet         = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	flag_template    = flag.Bool("template", false, "use template method from amumax")
	flag_flat        = flag.Bool("flat", false, "flat structure for template")
	flag_pipeline    = flag.Int("pipelineLength", 1, "")
	flag_encapsle    = flag.Bool("encapsle", false, "")
	flag_example     = flag.Bool("example", false, "")
	flag_stringCode  = flag.String("c", "", "")
	flag_GPUTreshold = flag.Int("gputreshold", 100, "")
	flag_maxPerGPU   = flag.Int("maxPerGPU", 1, "Maximum number of concurrent simulations per GPU")
	// more flags in engine/gofiles.go
)

func main() {
	flag.Parse()
	log.SetPrefix("")
	log.SetFlags(0)

	if *flag_vet {
		vet()
		return
	}
	if *flag_template && *flag_example || !*flag_template && !*flag_example {
		cuda.Init(*engine.Flag_gpu)
	}

	cuda.Synchronous = *engine.Flag_sync
	if *flag_version {
		printVersion()
	}

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *flag_test {
		fmt.Println(cuda.GPUInfo)
		os.Exit(0)
	}

	defer engine.Close() // flushes pending output, if any

	stringMode := false
	if *flag_stringCode != "" && flag.NArg() == 0 {
		runStringAndServe(*flag_stringCode)
		stringMode = true
	}

	if *flag_template && !stringMode {
		args := flag.Args()
		if len(args) > *flag_pipeline {
			panic(fmt.Sprintf("Found unparsed args.\n %v", args))
		}
		var wg sync.WaitGroup
		for i := range args {
			mmx3ScriptsTmp, SweepExec, err := template(args[i], *flag_flat)
			if err != nil {
				panic(err)
			}
			if !*flag_example {
				RunQueue(mmx3ScriptsTmp)
				if len(SweepExec) == 1 {
					wg.Add(1)
					if SweepExec[0].Dir {
						go func(arg, path string) {
							engine.RunExecDirSweep(arg, path)
							wg.Done()
						}(SweepExec[0].Arg, engine.ODSweep())
					} else {
						go func(arg string) {
							engine.RunExec(arg)
							wg.Done()
						}(SweepExec[0].Arg)
					}
				} else {
					for j := range SweepExec {
						if SweepExec[j].Dir {
							engine.RunExecDirSweep(SweepExec[j].Arg, engine.ODSweep())
						} else {
							engine.RunExec(SweepExec[j].Arg)
						}
					}
				}
			} else {
				runFileAndServe(mmx3ScriptsTmp[0])
			}
		}
		wg.Wait()
		os.Exit(int(exitStatus))
	} else if !*flag_template && !stringMode {
		switch flag.NArg() {
		case 0:
			if *engine.Flag_interactive {
				runInteractive()
			}
		case 1:
			runFileAndServe(flag.Arg(0))
		default:
			RunQueue(flag.Args())
		}
	}
}

func runInteractive() {
	fmt.Println("//no input files: starting interactive session")
	//initEngine()

	// setup outut dir
	now := time.Now()
	outdir := fmt.Sprintf("mumax-%v-%02d-%02d_%02dh%02d.out", now.Year(), int(now.Month()), now.Day(), now.Hour(), now.Minute())
	engine.InitIO(outdir, outdir, *engine.Flag_forceclean)

	engine.Timeout = 365 * 24 * time.Hour // basically forever

	// set up some sensible start configuration
	engine.Eval(`SetGridSize(128, 64, 1)
		SetCellSize(4e-9, 4e-9, 4e-9)
		Msat = 1e6
		Aex = 10e-12
		alpha = 1
		m = RandomMag()`)
	addr := goServeGUI()
	castedPort, ok := strconv.Atoi(*engine.Flag_port)
	if ok != nil {
		panic("Provide port as integer.")
	}
	host, port, path, err := parseAddrPath(*engine.Flag_webUIHost, castedPort)
	logUI.Log.PanicIfError(err)
	go api.Start(host, port, path, *engine.Flag_tunnel, *engine.Flag_debug)
	openbrowser("http://127.0.0.1:" + addr)
	engine.RunInteractive()
}

func runFileAndServe(fname string) {
	if path.Ext(fname) == ".go" {
		runGoFile(fname)
	} else {
		runScript(fname)
	}
	engine.WaitFFTs4DDone()
}

func runStringAndServe(script string) {
	runScriptString(script)
	engine.WaitFFTs4DDone()
}

func runScriptString(scriptStr string) {
	fname := "script.mx3"
	err := os.WriteFile(fname, []byte(scriptStr), 0644)
	if err != nil {
		panic(err)
	}
	outDir := util.NoExt(fname) + ".out"
	if *engine.Flag_od != "" {
		outDir = *engine.Flag_od
	}
	engine.InitIO(fname, outDir, *engine.Flag_forceclean)

	fname = engine.InputFile

	var code *script.BlockStmt
	var err2 error
	if fname != "" {
		// first we compile the entire file into an executable tree
		code, err2 = engine.CompileFile(fname)
		util.FatalErr(err2)
	}

	// now the parser is not used anymore so it can handle web requests
	castedPort, ok := strconv.Atoi(*engine.Flag_port)
	if ok != nil {
		panic("Provide port as integer.")
	}
	host, port, path, err := parseAddrPath(*engine.Flag_webUIHost, castedPort)
	logUI.Log.PanicIfError(err)
	go api.Start(host, port, path, *engine.Flag_tunnel, *engine.Flag_debug)

	if *engine.Flag_interactive {
		openbrowser("http://127.0.0.1:" + *engine.Flag_port)
	}

	// start executing the tree, possibly injecting commands from web gui
	engine.EvalFile(code)

	if *engine.Flag_interactive {
		engine.RunInteractive()
	}
}

func runScript(fname string) {
	outDir := util.NoExt(fname) + ".out"
	if *engine.Flag_od != "" {
		outDir = *engine.Flag_od
	}
	engine.InitIO(fname, outDir, *engine.Flag_forceclean)

	fname = engine.InputFile

	var code *script.BlockStmt
	var err2 error
	if fname != "" {
		// first we compile the entire file into an executable tree
		code, err2 = engine.CompileFile(fname)
		util.FatalErr(err2)
	}

	// now the parser is not used anymore so it can handle web requests
	castedPort, ok := strconv.Atoi(*engine.Flag_port)
	if ok != nil {
		panic("Provide port as integer.")
	}
	host, port, path, err := parseAddrPath(*engine.Flag_webUIHost, castedPort)
	logUI.Log.PanicIfError(err)
	go api.Start(host, port, path, *engine.Flag_tunnel, *engine.Flag_debug)

	if *engine.Flag_interactive {
		openbrowser("http://127.0.0.1:" + *engine.Flag_port)
	}

	// start executing the tree, possibly injecting commands from web gui
	engine.EvalFile(code)

	if *engine.Flag_interactive {
		engine.RunInteractive()
	}
}

func runGoFile(fname string) {

	// pass through flags
	flags := []string{"run", fname}
	flag.Visit(func(f *flag.Flag) {
		if f.Name != "o" {
			flags = append(flags, fmt.Sprintf("-%v=%v", f.Name, f.Value))
		}
	})

	if *engine.Flag_od != "" {
		flags = append(flags, fmt.Sprintf("-o=%v", *engine.Flag_od))
	}

	cmd := exec.Command("go", flags...)
	log.Println("go", flags)
	cmd.Stdout = os.Stdout
	cmd.Stdin = os.Stdin
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		os.Exit(1)
	}
}

// start Gui server and return server address
func goServeGUI() string {
	if *engine.Flag_port == "" {
		log.Println(`//not starting GUI (-http="")`)
		return ""
	}
	addr := engine.GoServe(*engine.Flag_port)
	fmt.Print("//starting GUI at http://127.0.0.1", addr, "\n")
	return addr
}

// print version to stdout
func printVersion() {
	engine.LogOut(engine.UNAME)
	engine.LogOut(fmt.Sprintf("GPU info: %s, using cc=%d PTX", cuda.GPUInfo, cuda.UseCC))
	engine.LogOut(fmt.Sprintf("Running on GPU %v", cuda.GPUIndex))
	engine.LogOut("(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium")
	engine.LogOut("This is free software without any warranty. See license.txt")
	engine.LogOut("********************************************************************//")
	engine.LogOut("  If you use mumax in any work or publication,                      //")
	engine.LogOut("  we kindly ask you to cite the references in references.bib        //")
	engine.LogOut("********************************************************************//")
}
