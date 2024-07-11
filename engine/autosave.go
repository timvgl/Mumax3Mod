package engine

// Bookkeeping for auto-saving quantities at given intervals.

import "fmt"

var (
	output  = make(map[Quantity]*autosave) // when to save quantities
	autonum = make(map[string]int)         // auto number for out file
	autonumPrefix = make(map[string]int)         // auto number for out file
)

func init() {
	DeclFunc("SetAutoNumPrefixTo", SetAutoNumPrefixTo, "")
	DeclFunc("SetAutoNumTo", SetAutoNumTo, "")
	DeclFunc("AutoSave", AutoSave, "Auto save space-dependent quantity every period (s).")
	DeclFunc("AutoSnapshot", AutoSnapshot, "Auto save image of quantity every period (s).")
}

func SetAutoNumTo(v int) {
	for key, _ := range autonum {
		autonum[key] = v
	}
}

func SetAutoNumPrefixTo(v int) {
	for key, _ := range autonumPrefix {
		autonumPrefix[key] = v
	}
}

// Periodically called by run loop to save everything that's needed at this time.
func DoOutput() {
	for q, a := range output {
		if a.needSave() {
			a.save(q)
			a.count++
		}
	}
	if Table.needSave() {
		Table.Save()
	}
}

func DoOutputPrefix(prefix string) {
	for q, a := range output {
		if a.needSave() {
			a.savePrefix(q, prefix)
			a.count++
		}
	}
	if Table.needSave() {
		Table.SavePrefix(prefix)
	}
}

// Register quant to be auto-saved every period.
// period == 0 stops autosaving.
func AutoSave(q Quantity, period float64) {
	autoSave(q, period, Save)
}

// Register quant to be auto-saved as image, every period.
func AutoSnapshot(q Quantity, period float64) {
	autoSave(q, period, Snapshot)
}

// register save(q) to be called every period
func autoSave(q Quantity, period float64, save func(Quantity)) {
	if period == 0 {
		delete(output, q)
	} else {
		output[q] = &autosave{period, Time, -1, save, savePrefix} // init count to -1 allows save at t=0
	}
}

// generate auto file name based on save count and FilenameFormat. E.g.:
// 	m000001.ovf
func autoFname(name string, format OutputFormat, num int) string {
	return fmt.Sprintf(OD()+FilenameFormat+"."+StringFromOutputFormat[format], name, num)
}

func autoFnamePrefix(prefix, name string, format OutputFormat, num int) string {
	return fmt.Sprintf(OD()+FilenameFormat+"."+StringFromOutputFormat[format], prefix + "_" + name, num)
}

// keeps info needed to decide when a quantity needs to be periodically saved
type autosave struct {
	period float64        // How often to save
	start  float64        // Starting point
	count  int            // Number of times it has been autosaved
	save   func(Quantity) // called to do the actual save
	savePrefix   func(Quantity, string)
}

// returns true when the time is right to save.
func (a *autosave) needSave() bool {
	t := Time - a.start
	return a.period != 0 && t-float64(a.count)*a.period >= a.period
}
