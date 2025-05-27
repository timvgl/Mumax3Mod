==================================================
# Important Notes

- **Not an official repository:**
  - This repository is **not** the official mumax3 repository. The official repository can be found [here](https://github.com/mumax/3).
  - This repository is **not** the official mumax3-ME repository. The official repository can be found [here](https://github.com/Fredericvdv/Magnetoelasticity_MuMax3).
  - This repository is **not** the official amumax repository. The official repository can be found [here](https://github.com/MathieuMoalic/amumax).

==================================================
# Overview

This repository contains an extended version of Mumax3-ME. The web interface is taken from the [amumax repository](https://github.com/MathieuMoalic/amumax).  
The functions have been mainly tested for researching vortices in magnetic systems. Below are the key extensions and commands.

==================================================

<img src="https://raw.githubusercontent.com/timvgl/Mumax3Mod/main/logo.png" width="35%" height="35%">

==================================================
# Extended Extraction Variables

## Space-resolved Variables (ovf files)
- **F_melM:** Force from the elastic system onto the magnetic system
- **F_el:** Elastic force within the elastic PDE
- **F_elsys:** Reformulation of the elastic PDE such that one side is zero (the side that must be zero)
- **rhod2udt2:** Force of the elastic PDE resulting from the acceleration of displacement
- **etadudt:** Damping force of the elastic PDE resulting from the velocity of displacement
- **GradMx, GradMy, GradMz:** Gradients of the x, y, and z components of the magnetization
- **ddu:** Acceleration of displacement

## Space-independent Variables (Table)
- **CorePosR:** Radius to the core position of the vortex

==================================================
# Commands for Vortex Functions

- **VortexAsym:** Creates a deformed vortex  
  *Args:* int (circulation), int (polarity), float (scaling factor for the FWHM of the out-of-plane Gaussian), float (angle for FWHM change), float (angle defining the width until the FWHM returns to its usual value)

- **DisplacedVortex:** A vortex displaced from the center  
  *Args:* int (circulation), int (polarity), float (deltaX), float (deltaY)

- **DisplacedVortexAsym:** A deformed and displaced vortex  
  *Args:* int (circulation), int (polarity), float (deltaX), float (deltaY), float (FWHM scaling), float (angle for FWHM change), float (angle defining the width until the FWHM returns to its usual value)

- **Activate_corePosScriptAccess:** Uses the vortex's core position for runWhile methods within the mumax script  
  - *Note:* The variable **CorePosRT** corresponds to this function.

- **Activate_corePosScriptAccessR:** Uses the radius to the core position (from the center) for runWhile methods within the mumax script  
  - *Note:* The variable **CorePosRTR** corresponds to this function.

==================================================
# Commands for Loading Data

- **LoadFileWithoutMem:** Loads an ovf file into a variable without memory allocation (needed for strain).  
  *Args:* __Quantity__ and file path.  
  *Note:* Available for normStrain, shearStrain, normStress, and shearStress.  
  Best practice: Set all dependencies to zero (see __SetQuantityWithoutMemToConfig__).

- **SetQuantityWithoutMemToConfig:** Sets a quantity without allocated memory to a configuration (e.g., Uniform(...)).  
  *Args:* __Quantity__ and __config__.

- **LoadFileVector:** Assigns a vector variable with memory using an '='; e.g.,  
  ```
  LoadFileVector("path_to_file")
  ```
  
  *Comment:* Loading strains is experimental – if you want to use the strain as an initial state, activate loading for one step only, otherwise the newly calculated displacement will not be properly considered.

==================================================
# Commands for Saving Data

- **SaveAsAt:** Saves a quantity as an ovf file with a specific name at a provided path.  
  *Args:* __Quantity__, filename, path (string).

- **AutoSaveAs:** Autosaves a quantity and relabels the file.  
  *Args:* __Quantity__, dt (float), and name (string).

- **AutoSnapshotAs:** Similar to AutoSaveAs, but for snapshots.  
  *Args:* __Quantity__, dt (float), and name (string).

- **ignoreCropName:** Set to true to disable automatic range labeling of ovf files when a cropped quantity is saved.

- **File Numbering Commands:**  
  These commands set the numbering of the ovf files (e.g., if a simulation is to be continued):
  - SetAutoNumTo  
  - SetAutoNumPrefixTo  
  - SetAutoNumSnapshotTo  
  - SetAutoNumSnapshotPrefixTo  
  - SetAutoNumToAs  
  - SetAutoNumPrefixToAs  
  - SetAutoNumSnapshotToAs  
  - SetAutoNumSnapshotPrefixToAs  
  *All commands require an integer argument.*

- **createNewTable:** Set to false to continue the existing table.txt instead of overwriting it.

- **rewriteHeaderTable:** Set to true to append the table header to table.txt if it is continued.

==================================================
# Commands for File Checking and Management

- **IsFile:** Checks if a file exists at the given path.  
  *Args:* path_to_file (string)

- **IsFileMyDir:** Checks if a file exists in the current mumax directory.  
  *Args:* filename (string)

- **EraseOD:** Deletes all files in the current mumax directory.

- **GSDIR:** Since mumax3 does not allow strings in the script by default, this variable can be assigned the path to a folder where ground states are stored. In combination with __Suffix__ and __ConcStr__, complex ground state ovf files can be stored using schematic names.

- **Suffix:** See __GSDIR__.

- **ConcStr:** Concatenates as many strings as desired.  
  *Args:* strings (string)  
  *Example for GS:*  
  ```
  GSDir = "/home/test/GS/"
  Suffix = "_GS_3mTX_Vortex_512_512_256_256_eta_300_CoFeB.ovf"
  if IsFile(ConcStr(GSDir, ConcStr("m", Suffix))) && IsFile(ConcStr(GSDir, ConcStr("u", Suffix))) && IsFile(ConcStr(GSDir, ConcStr("du", Suffix))) {
        m.LoadFile(ConcStr(GSDir, ConcStr("m", Suffix)))
        u.LoadFile(ConcStr(GSDir, ConcStr("u", Suffix)))
        du = Uniform(0, 0, 0)
  } else {
        B_ext = Vector(0, 0.003, 0)
        RelaxFullCoupled = true
        Relax()
        SaveAsAt(m, ConcStr("m", Suffix), GSDir)
        SaveAsAt(u, ConcStr("u", Suffix), GSDir)
        SaveAsAt(du, ConcStr("du", Suffix), GSDir)
        B_ext = Vector(0, 0, 0)
        du = Uniform(0, 0, 0)
  }
  ```

==================================================
# Commands for Fully Elastically Coupled Systems

- **useBoundaries:** Activates boundaries for the elastic system.
- **AllowInhomogeniousMECoupling:** Deactivates the blocking of inhomogeneous B1 and B2 values.

==================================================
# Other Useful Commands

- **RotVector:** Rotates a vector around another vector b by an angle c.  
  *Args:* b, c (vector)

- **IDT:** Generates a one-sided IDT.  
  *Args:* IDTWidthFinger, IDTDistanceFinger, IDTFingerLength (float64), AmountFingers (int)

- **WriteNUndoneToLog:** Writes the number of reversed steps to the log.

- **absPath:** Defines the absolute path to the simulation folder without the folder itself – used in __ExecDir__.

- **Exec:** Executes a bash command.  
  *Args:* cmd (string)

- **ExecDir:** Executes a bash command but appends the absolute path to the output directory as an argument, or replaces all occurrences of %v and %s with the path.  
  *Example:*  
  ```
  ExecDir("python3 foo.py %v/fee %s/faa")
  ```
  -> Executes:  
  ```
  Exec("python3 foo.py path_to_outputdir/fee path_to_outputdir/faa")
  ```

- **ExecSweep:** Executes a bash command after finishing a sweep from a template-
  *Args:* cmd (string)

- **ExecSweepDir:** Executes a bash command after sweep from a template but appends the absolute path to the output directory as an argument, or replaces all occurrences of %v and %s with the path.
  *Args:* cmd (string)

- **string:** Casts a float or int to a string.  
  *Args:* interface

- **int:** Casts a value to an integer.  
  *Args:* value (float)

==================================================
# Commands for Relaxation

- **outputRelax:** Set to true to extract ovf files and table data during relaxation (depending on AutoSave etc.).  
  The files will have a prefix (modifiable with __prefix_relax__) to avoid conflicts with time evolution.

- **RelaxFullCoupled:** Set to true to relax the fully coupled system (requires __FixDt__).

- **useHighEta:** Sets eta homogeneously to 1e6 during relaxation (requires __RelaxFullCoupled__ = true).

- **__\__printSlope__\__:** Prints the slope or the value of the variable being minimized (requires __RelaxFullCoupled__ = true).

- **SlopeTresholdEnergyRelax:** Threshold from which it can be assumed that the energy slope (over time) is zero (default is 1e-12; may vary depending on the system, requires __RelaxFullCoupled__ = true).

- **RelaxDDUThreshold:** Similar to __RelaxTorqueThreshold__ but for the second derivative in time of displacement (requires __RelaxFullCoupled__ = true).

- **RelaxDUThreshold:** Similar to __RelaxTorqueThreshold__ but for the first derivative in time of displacement (requires __RelaxFullCoupled__ = true).

- **IterativeHalfing:** Runs the minimization process N times and halves __FixDt__ as well as the thresholds (__SlopeTresholdEnergyRelax__, __RelaxDDUThreshold__, __RelaxDUThreshold__) each time (requires __RelaxFullCoupled__ = true).

==================================================
# Commands for Defining Regions

- **homogeniousRegionZ:** When set to true, defines a region for only one z-layer and copies it to the other layers.

- **LimitRenderRegionX:** Limits the area in x where rendering is performed.  
  *Args:* from, to (int, in cells)

- **LimitRenderRegionY:** Similar to x, but for y.  
  *Args:* from, to (int, in cells)

- **LimitRenderRegionZ:** Similar to x, but for z.  
  *Args:* from, to (int, in cells)

- **ReDefRegion:** Reassigns cells of one region to a different one (0 to remove the region).  
  *Args:* oldRegion, newRegion (int)

- **EraseAllRegions:** Erases all regions – resets all cells to region 0 (more efficient than redefining each region separately).

==================================================
# Commands for expanding Quantities
- **Expand:** Expands the array of a quantity and fills the new created entries with constant (default 0).
  *Args:* q (Quantity), fromX, toX, fromY, toY, fromZ, toZ, shiftX, shiftY, shiftZ (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)
- **ExpandX:** Same as **Expand** but only for x dimension
  *Args:* q (Quantity), fromX, toX, shiftX (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)
- **ExpandY:** Same as **Expand** but only for y dimension
  *Args:* q (Quantity), fromY, toY, shiftY (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)
- **ExpandZ** Same as **Expand** but only for z dimension
  *Args:* q (Quantity), fromZ, toZ, shiftZ (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)

==================================================
# Commands for generating Operators

## Crop
- **CropOperator:** Returns a function with the cropping area being defined already but the quantity missing.
  *Args:* fromX, toX, fromY, toY, fromZ, toZ (int, in cells)
- **CropXOperator:** Same as **CropOperator** but only for x dimension
  *Args:* fromX, toX (int, in cells)
- **CropYOperator:** Same as **CropOperator** but only for y dimension
  *Args:* fromY, toY (int, in cells)
- **CropZOperator:** Same as **CropOperator** but only for z dimension
  *Args:* fromZ, toZ (int, in cells)
## CropK
- **CropK:** Like Crop, but fromX, toX, fromY, toY, fromZ and toZ are the wave numbers and not the cells of the mesh to be cropped to. This is supposed to be used with the FFT3D function or with the __operatorsKSpace__ variable. Choosing the same value for from* and to* will result in cropping to just this single value instead of a region.
- **CropKx:** Like CropK, but only for Kx
- **CropKy:** Like CropK, but only for Ky
- **CropKz:** Like CropK, but only for Kz
- **CropKxy:** Like CropK, but only for Kx and Ky
- **CropKOperator:** Mix between CropK and CropOperator. Can be used to crop FFT in space in FFT4D to certain k-values
- **CropKxOperator:** Like CropKOperator but only for Kx
- **CropKyOperator:** Like CropKOperator but only for ky
- **CropKzOperator:** Like CropKOperator but only for kz
- **CropKxyOperaotr:** Like CropKOperator but only for kx and ky

## Expand
- **ExpandOperator:** Returns a function with the expanding area being defined already but the quantity missing:
  *Args:* fromX, toX, fromY, toY, fromZ, toZ, shiftX, shiftY, shiftZ (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)
- **ExpandXOperator:** Same as **ExpandOperator** but only for x dimension
  *Args:* fromX, toX, shiftX (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)
- **ExpandYOperator:** Same as **ExpandOperator** but only for y dimension
  *Args:* fromY, toY, shiftY (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)
- **ExpandZOperator:** Same as **ExpandOperator** but only for z dimension
  *Args:* fromZ, toZ, shiftZ (int, in cells), value to set for the expanded area (optional, can be set componentwise, but does not have to. Default is 0. float)

*Example*
```
operator := CropXOperator(0, 32)
Save(operator(m))
Save(operator(B_ext))
Save(CropKxy(FFT3D(m), -1e7, 1e7, 0, 0)) //crops to kx from -1e7 to 1e7, ky = 0 and leaves kz unchanged
```


==================================================
# FFT Functions

## FFT3D
- **FFT3D:** Computes a 3D FFT (x, y, z) and, for example, saves it to an ovf file.  
  When this command is run, the real and imaginary parts of the FFT are visible in the Web UI.  
  *Methods on the result:*
  - __.Abs()__ – Length of the complex FFT value
  - __.Phi()__ – Angle of the complex FFT value
  - __.ToPolar()__ – Exports length and angle alternately along x (saves computation time if .Abs() and .Phi() are not called separately)
  - __.Real()__ and __.Imag()__
  
  *Example:*  
  ```
  Save(FFT3D(m).Abs())
  ```

## FFT4D / FFT_T
- **FFT4D / FFT_T:**  
  - **FFT4D:** Computes a 3D FFT (x, y, z) and performs an incremental Fourier transform in time.
    *Args:* q Quantity, period float
  - **FFT_T:** Performs a Fourier transform in time only (using real-space data).
    *Args:* q Quantity, period float
  - __FFT_T_IN_MEM__ – if set to false, required data is stored continuously in ovf files (lower GPU memory consumption but slower)
  - __minFrequency__, __maxFrequency__, __dFrequency__ (floats)
  - __FFT4D_Label__ – a string to use instead of the quantity name
  - __negativeKx__ - set to false to not get negtative kx values
  - __operatorsKSpace__ - fuse as many operators (e.g. ExpandOperator or CropOperator) as wished together with MergeOperators(...) and assign the result to this variable to crop or expand the k-space before performing the fouriertransform in time
    *Example:*
    ```
    operatorsKSpace = MergeOperators(CropXOperator(0, 64), ExpandXOperator(0, 128, -32)) //only consider everything in the area from x = 0 to x = 64, and then expand to the right side
    ```
  - __interpolate__ - set to true - interpolate between time steps for fouriertransform - need when FixDt not set or the calculation period not a multiple from the calculation period (experimental)
  - Methods on the result: __.SaveAbs()__, __.SavePhi()__, __.ToPolar()__
  
  *Examples:*  
  ```
  FFT4D(m, 10e-10).SaveAbs().SavePhi()  //10e-10 is the calculation period - like in AutoSave, but for incremental updating fouriertransform data in time. Saves Length and Angle of complex number in distincted ovf files
  FFT4D(m, 10e-10).ToPolar() //saves complex number in polar form in single ovf file
  FFT4D(m, 10e-10) //saves complex number in single ovf file
  ```
  
*Note:* The FFT4D code creates twice as many values for kx to hold the imaginary and real parts.  
It can be imported into xarray using mumaxXR. Please do not mix non-FFT data with FFT data.

==================================================
# RenderFunctions

- **RenderFunction:** Converts a string into a function (using CreateFunction(...)) and renders it as a scalar excitation or parameter (excluding exchange and DMI).  
  Available variables: All variables defined in the script and standard mathematical functions in mumax3.  
  *x, y, and z* are coordinates in the mesh, and *t* is replaced with the current simulation time.  
  **Important:** Do not use the internally used variables (x_factor, y_factor, z_factor, x_length, y_length, and z_length).  
  *Args:* StringFunction  
  *Examples:*  
  ```
  B_ext.RenderFunction(CreateFunction("", "amplitude*1e7*sin(2*pi*t*freq*1e6)", ""))
  exx.RenderFunction(CreateFunction("amplitude*1e7*sin(2*pi*t*freqElastic*1e6)"))
  ```

- **RenderFunctionLimit:** Same as RenderFunction, but spatially limited (mesh still starts at 0 for x, y, z).  
  *Args:* StringFunction, startX (int), endX (int), startY (int), endY (int), startZ (int), endZ (int)

- **RenderFunctionLimitX / LimitY / LimitZ:**  
  Similar functions that limit only in x, y, or z respectively.  
  *Example:*  
  ```
  alpha.RenderFunctionLimitX(CreateFunction("alphaLow+(alphaHigh-alphaLow)*tanh((128*dx-x)/(128*dx))"), 0, 128)
  ```

- **RenderFunctionShape:** Same es RenderFunction, but the shape, given as an argument is layed over the result, of the individual function and is being merged with prior defintions.
  *Args:* StringFunction, Shape

Render Finite Sums: With this you can render finite sums e.g. fourier synthesis. Order of the sum is increased with passing more values to the slice. Multiple sums and indexed variables can be used in one expression. The index of the sum is also available. Note that the index of the sum and of the variable names that you want to use have to match
```
a_i := CreateFloatSlice(500e6, 1e9) //Create slice containing floats 500e6 and 1e6
exx.RenderFunction(CreateFunction("sum_i(sin(2*pi*a_i*t))")) //expands internally to sin(2*pi*500e6*t)+sin(2*pi*1e9*t)

a_i := CreateFloatSliceOne(50) //Create a slice of floats with value one with lengths of 50
exx.RenderFunction(CreateFunction("1/2 + sum_i(a_i*2/(pi*(2*i+1))*sin(2*pi*(2*i+1)*t))")) // create a rectangle signal up intill the order of 50
```

==================================================
# Queue

When a queue is started in mumax, it cannot be paused by default. An environment variable "MumaxQueue_%d" is created (with %d being the queue index if multiple queues are running).  
- **Queue Control:**  
  - When the queue starts, the environment variable is set to "running".  
  - Setting this environment variable to "pause" prevents new simulations from starting until it is set back to "running".  
  - Removing the environment variable resumes the queue.

==================================================
# Template Strings

*The following section originates from the amumax repository and has been slightly modified.*

## Syntax

Template strings are placeholders within your `.mx3` files that define how parameters should vary across generated files.  
The syntax is:  
```
"{key1=value1;key2=value2;...}"
```
Each template string is enclosed in quotes and curly braces and contains key-value pairs separated by semicolons.

## Available Keys

- **array:** Define a specific set of values  
  *Example:* `array=[1,2,3]`
- **start**, **end**, **step:** Define a range of values (similar to `numpy.arange`)  
  *Example:* `start=0;end=1;step=0.1`
- **start**, **end**, **count:** Define a range of values with a specific count (similar to `numpy.linspace`)  
  *Example:* `start=0;end=1;count=5`
- **prefix:** A string to be added before the value in the generated filenames  
  *Example:* `prefix=param_`
- **suffix:** A string to be added after the value in the generated filenames  
  *Example:* `suffix=_test`
- **format:** Specifies the formatting of the value in the filename (must be a float format like `%f`)  
  *Example:* `format=%.2f`

## Examples

### Example 1: Varying a Single Parameter

**Template File (template.mx3):**  
```
Aex := "{start=0;end=2;step=1}"
```

**Command:**  
```
mumax3 --template template1.mx3 template2.mx3 template3.mx3
```

**Generated Files:**  
- 0.mx3  
- 1.mx3  
- 2.mx3

*Explanation:* The placeholder `{start=0;end=2;step=1}` is replaced with values from 0 to 2 in steps of 1, generating a new `.mx3` file for each value.

### Example 2: Using an Array of Values and Formatting

**Template File (template.mx3):**  
```
x := "{array=[1,2,3];format=%02.0f}"
```

**Command:**  
```
mumax3 --template template.mx3
```

**Generated Files:**  
- 01.mx3  
- 02.mx3  
- 03.mx3

*Explanation:* The placeholder is replaced with the values from the array, and `format=%02.0f` ensures that the values in the filenames are formatted with at least two digits (padded with zeros if necessary).

### Example 3: Combining Multiple Template Strings

**Template File (template.mx3):**  
```
x := "{prefix=alpha; array=[1,2];format=%02.0f}"
y := "{prefix=beta; array=[3,4];format=%.0f}"
```

**Command:**  
```
mumax3 --template template.mx3
```

**Generated Files:**  
- alpha01/beta3.mx3  
- alpha01/beta4.mx3  
- alpha02/beta3.mx3  
- alpha02/beta4.mx3

*Explanation:* All combinations of x and y values are generated. Each combination results in a new `.mx3` file with the corresponding values.

### Example 4: Using Flat Output

**Template File (template.mx3):**  
```
x := "{array=[1,2];format=%02.0f}"
y := "{array=[3,4];format=%.0f}"
```

**Command:**  
```
mumax --template --flat template.mx3
```

**Generated Files:**  
- 013.mx3  
- 014.mx3  
- 023.mx3  
- 024.mx3

*Explanation:* The `--flat` option generates files without creating subdirectories. The filenames are concatenated with the formatted values.

### Example 5: Using Prefix and Suffix

**Template File (template.mx3):**  
```
Temperature := "{prefix=T; array=[300,350]; suffix=K; format=%.0f}"
```

**Command:**  
```
mumax3 --template template.mx3
```

**Generated Files:**  
- T300K.mx3  
- T350K.mx3

*Explanation:* The keys `prefix` and `suffix` add strings before and after the value in the filename, with `format=%.0f` ensuring no decimal places.

## Notes

- **Formatting:** Only `%f` float formats (e.g., `%.2f`, `%03.0f`) are allowed. Formats like `%d` are not permitted.  
- **Error Handling:** The template parser will report errors if the syntax is incorrect or required keys are missing.  
- **Variable Replacement:** Within the `.mx3` files, placeholders are replaced with numerical values.

### Advanced Example

**Template File (template.mx3):**  
```
alpha := "{prefix=alpha_; array=[0.01, 0.02, 0.03]; format=%.2f}"
beta := "{prefix=beta_; start=0.1; end=0.3; step=0.1; format=%.1f}"
```

**Command:**  
```
mumax3 --template template.mx3
```

**Generated Files (excerpt):**  
- alpha_0.01/beta_0.1.mx3  
- alpha_0.01/beta_0.2.mx3  
- alpha_0.01/beta_0.3.mx3  
- alpha_0.02/beta_0.1.mx3  
- ...

*Explanation:* Generates all combinations of alpha and beta values. Each generated file contains the corresponding replaced values.

==================================================
# Final Note

This README provides a comprehensive overview of the extended functions, commands, and template strings in this repository. For questions or issues, please refer to the respective official repositories or the documentation of the individual functions.

Happy simulating!
