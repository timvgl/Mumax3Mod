package engine

// Bookkeeping for auto-saving quantities at given intervals.

import "fmt"

var (
	output                   = make(map[Quantity]*autosave) // when to save quantities
	autonum                  = make(map[string]int)         // auto number for out file
	autonumPrefix            = make(map[string]int)         // auto number for out file
	autonumSnapshots         = make(map[string]int)         // auto number for out file
	autonumSnapshotsPrefix   = make(map[string]int)         // auto number for out file
	autonumAs                = make(map[string]int)         // auto number for out file
	autonumPrefixAs          = make(map[string]int)         // auto number for out file
	autonumSnapshotsAs       = make(map[string]int)         // auto number for out file
	autonumSnapshotsPrefixAs = make(map[string]int)         // auto number for out file
)

func init() {
	DeclFunc("SetAutoNumPrefixTo", SetAutoNumPrefixTo, "")
	DeclFunc("SetAutoNumTo", SetAutoNumTo, "")
	DeclFunc("SetAutoNumSnapshotPrefixTo", SetAutoNumSnapshotPrefixTo, "")
	DeclFunc("SetAutoNumSnapshotTo", SetAutoNumSnapshotTo, "")
	DeclFunc("SetAutoNumPrefixToAs", SetAutoNumPrefixToAs, "")
	DeclFunc("SetAutoNumToAs", SetAutoNumToAs, "")
	DeclFunc("SetAutoNumSnapshotPrefixToAs", SetAutoNumSnapshotPrefixToAs, "")
	DeclFunc("SetAutoNumSnapshotToAs", SetAutoNumSnapshotToAs, "")
	DeclFunc("AutoSave", AutoSave, "Auto save space-dependent quantity every period (s).")
	DeclFunc("AutoSaveAs", AutoSaveAs, "Auto save space-dependent quantity every period (s).")
	DeclFunc("AutoSnapshot", AutoSnapshot, "Auto save image of quantity every period (s).")
	DeclFunc("AutoSnapshotAs", AutoSnapshotAs, "Auto save image of quantity every period (s).")
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

func SetAutoNumSnapshotTo(v int) {
	for key, _ := range autonumSnapshots {
		autonumSnapshots[key] = v
	}
}

func SetAutoNumSnapshotPrefixTo(v int) {
	for key, _ := range autonumSnapshotsPrefix {
		autonumSnapshotsPrefix[key] = v
	}
}

func SetAutoNumToAs(v int) {
	for key, _ := range autonumAs {
		autonumAs[key] = v
	}
}

func SetAutoNumPrefixToAs(v int) {
	for key, _ := range autonumPrefixAs {
		autonumPrefixAs[key] = v
	}
}

func SetAutoNumSnapshotToAs(v int) {
	for key, _ := range autonumSnapshotsAs {
		autonumSnapshotsAs[key] = v
	}
}

func SetAutoNumSnapshotPrefixToAs(v int) {
	for key, _ := range autonumSnapshotsPrefixAs {
		autonumSnapshotsPrefixAs[key] = v
	}
}

// Periodically called by run loop to save everything that's needed at this time.
func DoOutput() {
	for q, a := range output {
		if a.needSave() {
			if a.qname == "" {
				a.save(q)
				a.count++
			} else {
				a.saveAs(q, a.qname)
				a.count++
			}
		}
	}
	if Table.needSave() {
		Table.Save()
	}
}

func DoOutputPrefix(prefix string) {
	for q, a := range output {
		if a.needSave() {
			if a.qname == "" {
				a.savePrefix(q, prefix)
				a.count++
			} else {
				a.saveAsPrefix(q, prefix, a.qname)
				a.count++
			}
		}
	}
	if Table.needSave() {
		Table.SavePrefix(prefix)
	}
}

// Register quant to be auto-saved every period.
// period == 0 stops autosaving.
func AutoSave(q Quantity, period float64) {
	autonum[NameOf(q)] = 0
	autonumPrefix[NameOf(q)] = 0
	autoSave(q, period, Save, SavePrefix)
}
func AutoSaveAs(q Quantity, period float64, name string) {
	autonumAs[name] = 0
	autonumPrefixAs[name] = 0
	autoSaveAs(q, period, Save, SavePrefix, SaveAsOverwrite, SaveAsOverwritePrefix, name)
}

// Register quant to be auto-saved as image, every period.
func AutoSnapshot(q Quantity, period float64) {
	autonumSnapshots[NameOf(q)] = 0
	autonumSnapshotsPrefix[NameOf(q)] = 0
	autoSave(q, period, Snapshot, SnapshotPrefix)
}

func AutoSnapshotAs(q Quantity, period float64, name string) {
	autonumSnapshotsAs[name] = 0
	autonumSnapshotsPrefixAs[name] = 0
	autoSaveAs(q, period, Snapshot, SnapshotPrefix, SnapshotAsOverwrite, SnapshotAsOverwritePrefix, name)
}

// register save(q) to be called every period
func autoSave(q Quantity, period float64, save func(Quantity), savePrefix func(Quantity, string)) {
	if period == 0 {
		delete(output, q)
	} else {
		output[q] = &autosave{period, Time, -1, save, savePrefix, nil, nil, ""} // init count to -1 allows save at t=0
	}
}

// register save(q) to be called every period
func autoSaveAs(q Quantity, period float64, save func(Quantity), savePrefix, SaveAsOverwrite func(Quantity, string), SaveAsOverwritePrefix func(Quantity, string, string), name string) {
	if period == 0 {
		delete(output, q)
	} else {
		output[q] = &autosave{period, Time, -1, save, savePrefix, SaveAsOverwrite, SaveAsOverwritePrefix, name} // init count to -1 allows save at t=0
	}
}

// generate auto file name based on save count and FilenameFormat. E.g.:
//
//	m000001.ovf
func autoFname(name string, format OutputFormat, num int) string {
	return fmt.Sprintf(OD()+FilenameFormat+"."+StringFromOutputFormat[format], name, num)
}

func autoFnamePrefix(prefix, name string, format OutputFormat, num int) string {
	if usePrefixOutputRelax {
		return fmt.Sprintf(OD()+FilenameFormat+"."+StringFromOutputFormat[format], prefix+"_"+name, num)
	} else {
		return fmt.Sprintf(OD()+FilenameFormat+"."+StringFromOutputFormat[format], name, num)
	}
}

// keeps info needed to decide when a quantity needs to be periodically saved
type autosave struct {
	period       float64                // How often to save
	start        float64                // Starting point
	count        int                    // Number of times it has been autosaved
	save         func(Quantity)         // called to do the actual save
	savePrefix   func(Quantity, string) //prefix if autosave is runned during relaxing
	saveAs       func(Quantity, string)
	saveAsPrefix func(Quantity, string, string)
	qname        string //rename file so that if data is cropped good names are possible
}

// returns true when the time is right to save.
func (a *autosave) needSave() bool {
	t := Time - a.start
	return a.period != 0 && t-float64(a.count)*a.period >= a.period
}
