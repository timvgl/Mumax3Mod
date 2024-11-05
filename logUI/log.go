package logUI

// Logging and error reporting utility functions

import (
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/fatih/color"
	"github.com/mumax/3/httpfs"
)

var Log Logs

var logOnlyForUI = true

type Logs struct {
	Hist    string                   // console history for GUI
	logfile httpfs.WriteCloseFlusher // saves history of input commands +  output
	debug   bool
	path    string
}

func (l *Logs) AutoFlushToFile() {
	for {
		l.FlushToFile()
		time.Sleep(5 * time.Second)
	}
}

func (l *Logs) FlushToFile() {
	if l.logfile != nil && !logOnlyForUI {
		l.logfile.Flush()

	}
}

func (l *Logs) SetDebug(debug bool) {
	l.debug = debug
}

func (l *Logs) Init(zarrPath string) {
	if !logOnlyForUI {
		l.path = zarrPath + "/log.txt"
		l.createLogFile()
		l.writeToFile(l.Hist)
	}
}

func (l *Logs) createLogFile() {
	if !logOnlyForUI {
		var err error
		l.logfile, err = httpfs.Create(l.path)
		if err != nil {
			color.Red(fmt.Sprintf("Error creating the log file: %v", err))
		}
	}
}

func (l *Logs) writeToFile(msg string) {
	if !logOnlyForUI {
		if l.logfile == nil {
			return
		}
		_, err := l.logfile.Write([]byte(msg))
		if err != nil {
			if err.Error() == "short write" {
				color.Yellow("Error writing to log file, trying to recreate it...")
				l.createLogFile()
				_, _ = l.logfile.Write([]byte(msg))
			} else {
				color.Red(fmt.Sprintf("Error writing to log file: %v", err))
			}
		}
	}
}

func (l *Logs) addAndWrite(msg string) {
	l.Hist += msg
	l.writeToFile(msg)
}

func (l *Logs) Command(msg ...interface{}) {
	fmt.Println(fmt.Sprint(msg...))
	l.addAndWrite(fmt.Sprint(msg...) + "\n")
}

func (l *Logs) Info(msg string, args ...interface{}) {
	formattedMsg := "// " + fmt.Sprintf(msg, args...) + "\n"
	color.Green(formattedMsg)
	l.addAndWrite(formattedMsg)
}

func (l *Logs) Warn(msg string, args ...interface{}) {
	formattedMsg := "// " + fmt.Sprintf(msg, args...) + "\n"
	color.Yellow(formattedMsg)
	l.addAndWrite(formattedMsg)
}

func (l *Logs) Debug(msg string, args ...interface{}) {
	if l.debug {
		formattedMsg := "// " + fmt.Sprintf(msg, args...) + "\n"
		color.Blue(formattedMsg)
		l.addAndWrite(formattedMsg)
	}
}

func (l *Logs) Err(msg string, args ...interface{}) {
	formattedMsg := "// " + fmt.Sprintf(msg, args...) + "\n"
	color.Red(formattedMsg)
	l.addAndWrite(formattedMsg)
}

func (l *Logs) PanicIfError(err error) {
	if err != nil {
		_, file, line, _ := runtime.Caller(1)
		color.Red(fmt.Sprint("// ", file, ":", line, err) + "\n")
		panic(err)
	}
}

func (l *Logs) ErrAndExit(msg string, args ...interface{}) {
	l.Err(msg, args...)
	os.Exit(1)
}

// Panics with msg if test is false
func AssertMsg(test bool, msg interface{}) {
	if !test {
		Log.ErrAndExit("%v", msg)
	}
}
