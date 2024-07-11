*IMPORTANT NOTE: this repository is not the official mumax3 repository. The official mumax3 repository can be found [here](https://github.com/mumax/3).*

*IMPORTANT NOTE: this repository is not the official mumax3-ME repository. The official mumax3-ME repository can be found [here](https://github.com/Fredericvdv/Magnetoelasticity_MuMax3).*


This repository contains an extended Mumax3-ME version.<br />
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
    
* Further commands:
    + __AllowInhomogeniousMECoupling__ - deactivate blocking of inhomogenious B1 and B2 values
    + __Activate_corePosScriptAccess__ - use the core position of the vortex for runWhile methods within the mumax-script
        - __CorePosRT__ is the corresponding variable
    + __Activate_corePosScriptAccessR__ - use the radius to the core position of the vortex from the center for __runWhile__ methods within the mumax-script
        - __CorePosRTR__ is the corresponding variable
    + __useBoundaries__ - activate boundaries for elastic system
    + __LoadFileWithoutMem__ - load ovf file into variable that has no memory allocated - needed for strain. Args: __Quantity__ and path to file. Available for normStrain, shearStrain, normStress and shearStress. The variables that are usually being used for calculating this __Quantity__ remain unchanged. Best practice is to set all dependencies to zero. See __SetQuantityWithoutMemToConfig__
    + __SetQuantityWithoutMemToConfig__ - set a quantity that has no memory being allocated to a config like Uniform(...) etc. Args: __Quantity__ and config.
    + __LoadFileVector__ - assign vector variable with memory with an =; m = __LoadFileVector__("path_to_file")
  
Comment: Loading strains is highly experimental - if you want to use the strain as an inital state, activate the loading only for one step, otherwise the new calculated displacement is never concidered properly


* Relaxing:
    + __outputRelax__ - set to true in order to extract ovf-files and table data during relaxing - the ovf-files and the time evolution in the table are going to have a prefix (can be modified with __\__\__prefix_relax______) so that the names are not interfering with a time evolution
    + __RelaxFullCoupled__ - set to true in order to relax fully coupled system and not only the magnetic system (needs __FixDt__ to be set)
    + __useHighEta__ - set eta homogeniously to 1e6 during relaxing process (needs __RelaxFullCoupled__ = true)
    + __\__\__printSlope______ - print slope or value itself of variable that is getting minimized (needs __RelaxFullCoupled__ = true)
    + __SlopeTresholdEnergyRelax__ - treshold from which it can be assumed that slope of energy (in time) is zero - the default value is 1e-12 - can be different depending on the system (needs __RelaxFullCoupled__ = true)
    + __RelaxDDUThreshold__ - same as __RelaxTorqueThreshold__ but for second derivative in time of u (needs __RelaxFullCoupled__ = true)
    + __RelaxDUThreshold__ - same as __RelaxTorqueThreshold__ but for first derivative in time of u (needs __RelaxFullCoupled__ = true)
    + __IterativeHalfing__ - run the minizing process N times and half __FixDt__ each time and __SlopeTresholdEnergyRelax__/__RelaxDDUThreshold__/__RelaxDUThreshold__ each time (needs __RelaxFullCoupled__ = true)



