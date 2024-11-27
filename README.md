*IMPORTANT NOTE: this repository is not the official mumax3 repository. The official mumax3 repository can be found [here](https://github.com/mumax/3).*

*IMPORTANT NOTE: this repository is not the official mumax3-ME repository. The official mumax3-ME repository can be found [here](https://github.com/Fredericvdv/Magnetoelasticity_MuMax3).*

*IMPORTANT NOTE: this repository is not the official amumax repository. The official amumax repository can be found [here](https://github.com/MathieuMoalic/amumax).*


This repository contains an extended Mumax3-ME version.<br />
The webinterface is used from [here](https://github.com/MathieuMoalic/amumax).
The functions have mainly been tested for reasearching vortices in magnetic systems.<br />
The following changes are implemented:<br />

New extraction variables:<br />
* Space-resolved variables (ovf-files):
    + __F_melM__ - force from the elastic system onto the magnetic system
    + __F_el__ - elastic force within the elastic PDE
    + __F_elsys__ - rewriting the elastic PDE such that one side has to be zero - this is the side that has to be zero
    + __rhod2udt2__ - the force of the elastic PDE resulting from the acceleration of the displacement
    + __etadudt__ - the force of the elastic PDE resulting from the velocity of the displacement - damping force
    + __GradMx__ - Gradient of the x component of the magnetization
    + __GradMy__ - Gradient of the y component of the magnetization
    + __GradMz__ - Gradient of the z component of the magnetization
    + __ddu__ - acceleration of displacement

* Space-independent variables (table):
    + __CorePosR__ - Radius to core position of vortex
    
* Further commands for Vortices:
    + __VortexAsym__ - deformed vortex. Args: int for circulation, int for polarity, float for changing the FWHM of the out of plane gaussian by this factor, float for the angle at which FWHM is supposed to be changed by this factor, float for the angle to define the width until FWHM is supposed to evolve to the usual value
    + __DisplacedVortex__ - vortex that has been displaced from center. Args: int for circulation, int for polarity, float for deltaX, float for deltaY
    + __DisplacedVortexAsym__ - deformed and displaced vortex.  Args: int for circulation, int for polarity, float for deltaX, float for deltaY, float for changing the FWHM of the out of plane gaussian by this factor, float for the angle at which FWHM is supposed to be changed by this factor, float for the angle to define the width until FWHM is supposed to evolve to the usual value
    + __Activate_corePosScriptAccess__ - use the core position of the vortex for runWhile methods within the mumax-script
        - __CorePosRT__ is the corresponding variable
    + __Activate_corePosScriptAccessR__ - use the radius to the core position of the vortex from the center for __runWhile__ methods within the mumax-script
        - __CorePosRTR__ is the corresponding variable

* Further commands for loading data:
    + __LoadFileWithoutMem__ - load ovf file into variable that has no memory allocated - needed for strain. Args: __Quantity__ and path to file. Available for normStrain, shearStrain, normStress and shearStress. The variables that are usually being used for calculating this __Quantity__ remain unchanged. Best practice is to set all dependencies to zero. See __SetQuantityWithoutMemToConfig__
    + __SetQuantityWithoutMemToConfig__ - set a quantity that has no memory being allocated to a config like Uniform(...) etc. Args: __Quantity__ and __config__.
    + __LoadFileVector__ - assign vector variable with memory with an =; m = __LoadFileVector__("path_to_file")
Comment: Loading strains is highly experimental - if you want to use the strain as an inital state, activate the loading only for one step, otherwise the new calculated displacement is never concidered properly

* Further commands for saving data:
    + __SaveAsAt__ - save a quantity as an ovf file with a certain name at a provided path. Args: __Quantity__, filename, path __string__
    + __AutoSaveAs__ - autosave quantity but relabel it for the files. Args: __Quantity__, dt __float__ and name __string__
    + __AutoSnapshotAs__ - autosnapshot quantity but relabel it for the files. Args: __Quantity__, dt __float__ and name __string__
    + __ignoreCropName__ - set to true in order to prohibit range labeling of ovf files, if quantity is cropped and saved to file.
    + For setting the numeration of the ovf files (for example if a simulation is supposed to be continued) to a specific value:
        - SetAutoNumTo - defines numeration of usual ovf files
        - SetAutoNumPrefixTo - defines numeration of ovf files with prefix (created during relaxing with output)
        - SetAutoNumSnapshotTo - s.a. for snapshots
        - SetAutoNumSnapshotPrefixTo - s.a. for snapshots
        - SetAutoNumToAs - s.a. in case that AutoSaveAs instead of AutoSave is used
        - SetAutoNumPrefixToAs - s.a. in case that AutoSaveAs instead of AutoSave is used and files are generated during relaxing process
        - SetAutoNumSnapshotToAs - s.a. for snapshots
        - SetAutoNumSnapshotPrefixToAs - s.a. for snapshots
    
        All need an integer as arg
    + __createNewTable__ set to false the old table.txt file will be continued and not overwritten
    + __rewriteHeaderTable__ set to true this appends the header of the table into the table.txt if table is continued

* Further commands for checking/retrieving files:
    + __IsFile__ - checks if file at give path exists. Args: path_to_file __string__
    + __IsFileMyDir__ - checks if file exists in the current used mumax directory. Args: filename __string__
    + __EraseOD__ - delete all files in current mumax directory
    + __GSDIR__ - mumax3 does not allow strings in the script by default. If groundstates of various systems are stored in a single folder as a library this var can be assigned with the path. In combination with __Suffix__ and __ConcatStr__ complex groundstate ovf files can be stored using schematic names. See example below
    + __Suffix__ - see __GSDir__
    + __ConcStr__ - concatenate as many strings as wished. Args: strings... __string__
    + Example for GS:
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

* Further commands for fully elastic coupled systems:
    + __useBoundaries__ - activate boundaries for elastic system
    + __AllowInhomogeniousMECoupling__ - deactivate blocking of inhomogenious B1 and B2 values

* Further useful commands:
    + __RotVector__ - needs to be applied onto a vector. Rotates this vector around another vector b with the angle c. Args: b, c __vector__
 
* Further shapes:
    + __IDT__ - generate one sided IDT. Args: IDTWidthFinger, IDTDistanceFinger, IDTFingerLength float64, AmountFingers __int__

* Helpful commands:
    + __WriteNUndoneToLog__ - writes amount of reversed steps to log
    + __absPath__ - define the absolute path to the simulation folder without the folder itself -> being used in __ExecDir__
    + __Exec__ - run bash command. Args: cmd __string__
    + __ExecDir__ - run bash command but append absolute path to output directory as arg for executed command, or replace all found %v and %s substrings with the path. Args: cmd __string__
        - ``` ExecDir("python3 foo.py %v/fee %s/faa") -> Exec("python3 foo.py path_to_outputdir/fee path_to_outputdir/faa")```
    + __string__ - cast float or int to string. Args: interface
    + __int__ - cast castable values to integer. Args: value __float__


* Relaxing:
    + __outputRelax__ - set to true in order to extract ovf-files and table data during relaxing (depending on the definitions of AutoSave etc.) - the ovf-files and the time evolution in the table are going to have a prefix (can be modified with __\__\__prefix_relax______) so that the names are not interfering with a time evolution
    + __RelaxFullCoupled__ - set to true in order to relax fully coupled system and not only the magnetic system (needs __FixDt__ to be set)
    + __useHighEta__ - set eta homogeniously to 1e6 during relaxing process (needs __RelaxFullCoupled__ = true)
    + __\__\__printSlope______ - print slope or value itself of variable that is getting minimized (needs __RelaxFullCoupled__ = true)
    + __SlopeTresholdEnergyRelax__ - treshold from which it can be assumed that slope of energy (in time) is zero - the default value is 1e-12 - can be different depending on the system (needs __RelaxFullCoupled__ = true)
    + __RelaxDDUThreshold__ - same as __RelaxTorqueThreshold__ but for second derivative in time of u (needs __RelaxFullCoupled__ = true)
    + __RelaxDUThreshold__ - same as __RelaxTorqueThreshold__ but for first derivative in time of u (needs __RelaxFullCoupled__ = true)
    + __IterativeHalfing__ - run the minizing process N times and half __FixDt__ each time and __SlopeTresholdEnergyRelax__/__RelaxDDUThreshold__/__RelaxDUThreshold__ each time (needs __RelaxFullCoupled__ = true)
    + __\__\__usePrefixOutputRelax______ - suppress prefix for ovf files even though output is generated during relaxing

* Regions:
    + __homogeniousRegionZ__ - set to true renders region only for one z-layer and copies to the other layers. if the region is homogenious in the z direction but the sample has n layers into the z-direction, the region is going to be rendered for one z-layer and being copied to the others
    + __LimitRenderRegionX__ - limit the area in which the rendering process is supposed to be done in the x direction. Reduces computational time for generating a region. Args: from, to __int__ (cells)
    + __LimitRenderRegionY__ - limit the area in which the rendering process is supposed to be done in the y direction. Reduces computational time for generating a region. Args: from, to __int__ (cells)
    + __LimitRenderRegionZ__ - limit the area in which the rendering process is supposed to be done in the z direction. Reduces computational time for generating a region. Args: from, to __int__ (cells)
    + __ReDefRegion__ - if a region has been set it cannot be removed by default - assign the cells of one region to a different one (0 for removing it). Args: oldRegion, newRegion __int__
    + __EraseAllRegions__ - erase all regions - sets all regions back to region 0 - more efficent than redefining all regions seperate 

* FFT:
    + __FFT3D__ - calculate FFT in 3D (x, y, z) and e.q. save to ovf file. Args: __Quantity__

    Created OVF file has twice the amount of values calculated for kx for holding imag and real value.
    
    VERY EXPERIMENTAL - NOT TESTED
* Queue:
   When a queue is started in mumax it cannot be paused by default. Here an env "MumaxQueue_%d" is created
   with %d being the queue index in case that multiple queues are running. When the queue is being started this env is set to "running". By setting this env to pause no new simulation is going to be started until it is set to running again. When the env is removed the queue is also going to be continued.

#### The upcoming part of the README file originates from the amumax repository but has been slightly modified.
### Template Strings

#### Syntax

Template strings are placeholders within your `.mx3` files that define how parameters should vary across generated files. The syntax for a template string is:

```go
"{key1=value1;key2=value2;...}"
```

Each template string is enclosed in `"{...}"` and contains key-value pairs separated by semicolons.

#### Available Keys

- **array**: Define a specific set of values.
  - Example: `array=[1,2,3]`
- **start**, **end**, **step**: Define a range of values using start, end, and step (similar to `numpy.arange`).
  - Example: `start=0;end=1;step=0.1`
- **start**, **end**, **count**: Define a range of values with a specific count (similar to `numpy.linspace`).
  - Example: `start=0;end=1;count=5`
- **prefix**: A string to be added before the value in the generated filenames.
  - Example: `prefix=param_`
- **suffix**: A string to be added after the value in the generated filenames.
  - Example: `suffix=_test`
- **format**: Specifies the formatting of the value in the filename (must be a float format like `%f`).
  - Example: `format=%.2f`

#### Examples

##### Example 1: Varying a Single Parameter

**Template File (`template.mx3`):**

```go
Aex := "{start=0;end=2;step=1}"
```

**Command:**

```bash
mumax3 --template template1.mx3 template2.mx3 template3.mx3
```

The simulations can be encapsled into subfolders with
```bash
mumax3 --template --encapsle template.mx3
```
Each template string creates another deeper subfolder with this.


**Generated Files:**

- `0.mx3`
- `1.mx3`
- `2.mx3`

**Explanation:**

- The placeholder `{start=0;end=2;step=1}` is replaced with values from `0` to `2` in steps of `1`.
- For each value, a new `.mx3` file is generated with the value assigned to `Aex`.

##### Example 2: Using an Array of Values and Formatting

**Template File (`template.mx3`):**

```go
x := "{array=[1,2,3];format=%02.0f}"
```

**Command:**

```bash
mumax3 --template template.mx3
```

**Generated Files:**

- `01.mx3`
- `02.mx3`
- `03.mx3`

**Explanation:**

- The placeholder `{array=[1,2,3];format=%02.0f}` is replaced with each value in the array.
- The `format=%02.0f` ensures that the values in the filenames are formatted with at least two digits, padding with zeros if necessary.

##### Example 3: Combining Multiple Template Strings

**Template File (`template.mx3`):**

```go
x := "{prefix=alpha; array=[1,2];format=%02.0f}"
y := "{prefix=beta; array=[3,4];format=%.0f}"
```

**Command:**

```bash
mumax3 --template template.mx3
```

**Generated Files:**

- `alpha01/beta3.mx3`
- `alpha01/beta4.mx3`
- `alpha02/beta3.mx3`
- `alpha02/beta4.mx3`

**Explanation:**

- All combinations of `x` and `y` values are generated.
- Each combination results in a new `.mx3` file with the corresponding values assigned.

##### Example 4: Using Flat Output

**Template File (`template.mx3`):**

```go
x := "{array=[1,2];format=%02.0f}"
y := "{array=[3,4];format=%.0f}"
```

**Command:**

```bash
mumax --template --flat template.mx3
```

**Generated Files:**

- `013.mx3`
- `014.mx3`
- `023.mx3`
- `024.mx3`

**Explanation:**

- The `--flat` option generates files without creating subdirectories.
- Filenames are concatenated with the formatted values.

##### Example 5: Using Prefix and Suffix

**Template File (`template.mx3`):**

```go
Temperature := "{prefix=T; array=[300,350]; suffix=K; format=%.0f}"
```

**Command:**

```bash
mumax3 --template template.mx3
```

**Generated Files:**

- `T300K.mx3`
- `T350K.mx3`

**Explanation:**

- The `prefix` and `suffix` keys add strings before and after the value in the filename.
- The `format=%.0f` ensures no decimal places are included.

#### Notes

- **Formatting:** Only `%f` float formats are allowed in the `format` key (e.g., `%.2f`, `%03.0f`). Formats like `%d` are not allowed.
- **Error Handling:** The template parser will report errors if the syntax is incorrect or required keys are missing.
- **Variables Replacement:** Inside the `.mx3` files, the placeholders are replaced with the numerical values.

#### Advanced Example

**Template File (`template.mx3`):**

```go
alpha := "{prefix=alpha_; array=[0.01, 0.02, 0.03]; format=%.2f}"
beta := "{prefix=beta_; start=0.1; end=0.3; step=0.1; format=%.1f}"
```

**Command:**

```bash
mumax3 --template template.mx3
```

**Generated Files:**

- `alpha_0.01/beta_0.1.mx3`
- `alpha_0.01/beta_0.2.mx3`
- `alpha_0.01/beta_0.3.mx3`
- `alpha_0.02/beta_0.1.mx3`
- ...

**Explanation:**

- Generates all combinations of `alpha` and `beta`.
- Each generated file contains the `alpha` and `beta` values replaced in the `.mx3` file.

