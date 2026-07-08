# Implementation report: magnum.np fully coupled magnetoelastic solver port ("NP solver")

Port of magnum.np's `LLGWithLESolver` + `MagnetoElasticField` (P. Flauger et al.,
PRApplied 25, 034050 (2026)) into this mumax fork, as a **separate, explicitly selected**
solver. The old fully coupled implementation (SetSolver 9/12, B1/B2) is fully preserved.

## 1. Files changed / added

**New CUDA kernels** (each: `.cu` → PTX for cc 50–80 → Go wrapper via `cuda2go`,
following the existing PTX-string workflow; build script `cuda/build_melas_np.sh`,
single-file rebuild `cuda/rebuild_one.sh`):

| file | kernel | purpose |
|---|---|---|
| `cuda/melas_np.h` | – | device helpers: safe division (nan_to_num), eps_m, block-C sigma, harmonic mean, D2± pair terms, mixed derivative |
| `cuda/melasgradjumpnp.cu` | `MelasGradJumpNP` | jump-aware first derivative (∂x,∂y,∂z) of one scalar field, window + PBC + 3-pt boundary option |
| `cuda/melasbsigmnp.cu` | `MelasBSigMNP` | magnetic-stress B jump parameters from interface m values |
| `cuda/melasbepsnp.cu` | `MelasBEpsNP` | harmonic-mean displacement-gradient B (iterative step), added to B_sigM |
| `cuda/melasstressnp.cu` | `MelasStressNP` | sigma_el = C:(eps−eps_m), Voigt diag + offdiag |
| `cuda/melasforcenp.cu` | `MelasForceNP` | force columns: jump-aware 2nd + mixed derivatives − magnetic force terms |
| `cuda/melasassemblenp.cu` | `MelasAssembleNP` | Neumann boundary overrides (bn=1/2/3), column sum, −eta·du, /rho, masks |
| `cuda/melashfieldnp.cu` | `MelasHFieldNP` | nonlinear magnetoelastic field B_mel [T] |
| `cuda/melasenergynp.cu` | `MelasEnergyNP` | energy densities U_el / E_mel / U |
| `cuda/melasstrainnp.cu` | `MelasStrainNP` | strain diagnostics eps / eps−eps_m / eps_m |
| `cuda/melasdudnp.cu` | `MelasDudNP` | du/dt with elastic + Dirichlet masks |
| `cuda/melasgilbertnp.cu` | `MelasGilbertNP` | Gilbert dissipation density factor |
| `cuda/melas_np.go` | – | Go launchers for all of the above |

**New engine files:**

| file | contents |
|---|---|
| `engine/magelas_np.go` | parameters (C13/C22/C23/C33/C55/C66, lambda100/111, elasticMaskNP, tractionNP), script vars/functions, quantities, CFL estimate |
| `engine/magelas_np_rhs.go` | pipeline (gradients → B iteration → stress), force assembly, field term, quantity setters, stress cache |
| `engine/magelas_np_solver.go` | `magelasNPRKF45` stepper (9 components, per-variable tolerances) |
| `engine/magelas_np_energy.go` | U_elNP, U_totNP, T_elNP, E_melNP (registered energy), GilbertDissNP |

**Modified existing files (minimal, additive):**

- `engine/run.go`: new solver constant `MAGELAS_RKF45_NP = 13` + `SetSolver` case.
- `engine/effectivefield.go`: one line `AddMagnetoelasticFieldNP(dst)` (no-op unless the NP
  solver is active and lambda parameters are set).
- `engine/relax.go`: `Relax()` with `RelaxFullCoupled=true` keeps the NP solver selected if
  it was active (previously hard-switched to the old solver 9/12); old behavior unchanged
  otherwise.

Nothing else was touched; all existing commands, solvers, quantities and scripts behave as
before.

## 2. New public API / script commands

- `UseMagnumNPSolver(true|false)`, `SetSolver(13)`
- Parameters: `C13 C22 C23 C33 C55 C66` (N/m²), `lambda100`, `lambda111`, `elasticMaskNP`,
  excitation `tractionNP` (N/m²); existing `C11 C12 C44`, `rho`, `eta`, `frozenDispLoc`,
  `frozenDispVal`, `Msat`, `Aex`, `alpha` are reused.
- Variables: `BoundaryNodesNP` (default 3), `IterationDepthNP` (default 1), `RtolNP`,
  `AtolMNP`, `AtolUNP`, `AtolDUNP`, `EnforceCFLNP`, `CFLFactorNP`.
- Functions: `SetMagneticWindowNP(x0,x1,y0,y1,z0,z1)`,
  `SetStiffnessCubicNP(...)`, `SetStiffnessCubicRegionNP(...)`,
  `SetStiffnessIsotropicNP(...)`, `SetStiffnessIsotropicRegionNP(...)`.
- Quantities: `F_elNP`, `B_melNP`, `dduNP`, `pNP`, `normStrainNP`, `shearStrainNP`,
  `normStrainElNP`, `shearStrainElNP`, `normStressNP`, `shearStressNP`,
  `U_elNP`, `U_totNP`, `T_elNP`, `E_melNP`, `GilbertDissNP`.

## 3. Mathematical definitions implemented

State `v = (m, u, du)` per cell (du = velocity; magnum.np's momentum p = rho·du):

```
dm/dt  = -gamma/(1+alpha^2) [ m x B + alpha m x (m x B) ]        (existing torque path)
du/dt  = du                        (identity; masked)             ~ magnum.np dud = p/rho
d(du)/dt = ( div(sigma - sigma_m) - eta*du ) / rho  (masked)      ~ magnum.np dpd/rho
```

with `sigma = C:eps`, `sigma_m = C:eps_m`, `eps = sym(grad u)` from jump-aware gradients,
`eps_m` per paper Eq. (8). The magnetoelastic field is the **nonlinear** field
`H_i = -1/(mu0 Ms) (sigma - sigma_m) : d eps_m/d m_i` (magnum.np `MagnetoElasticField`, not
`LinearMagnetoElasticField`).

## 4. Voigt convention

`[xx, yy, zz, yz, xz, xy]` with **engineering shear** (`eps[3] = dyuz + dzuy = 2 eps_yz`,
`eps[4] = dxuz + dzux`, `eps[5] = dxuy + dyux`), identical to magnum.np. Off-diagonal
outputs (`shearStressNP` etc.) are ordered (yz, xz, xy) — magnum.np's "inverse" ordering.

## 5. Sign/factor audit

- `eps_m`: diag `3/2·lambda100·(m_i²−1/3)`; shear `3·lambda111·m_i·m_j` (Voigt, includes
  the factor 2) — `melas_np.h::melas_eps_m`, matches `strain.py::epsilon_m` exactly.
- `sigma_m = C:eps_m` with the block C (row 2 uses C12/C22/C23, row 3 C13/C23/C33) —
  matches `stress.py::sigma` restricted to the block form.
- `H_magEl`: `B_x = +3/Ms (lambda100·sig_el[xx]·m_x + lambda111(sig_el[xy]·m_y + sig_el[xz]·m_z))`
  etc. Sign verified two ways: term-by-term against `MagnetoElasticField.h`
  (`h *= -1/(mu0 Ms)` on `sig_el·(−3lambda·m)` products), and by the functional derivative
  `H = −1/(mu0 Ms) dE/dm` of `E = ∫(1/2 eps_m − eps):C:eps_m` giving
  `+1/(mu0 Ms)(sigma−sigma_m):d eps_m/dm`. mumax stores B = mu0·H, hence the mu0 cancels.
  Torque sign: the standard mumax LLTorque acts on this B; energy consistency confirmed by
  the derivative check above.
- `p_dot`: `div(sigma−sigma_m)`: the force kernel adds the sigma-derivative terms and
  **subtracts** the analytic sigma_m derivative terms (`f_ij -= fm` in magnum.np);
  `−eta·du` damping; division by rho maps momentum to velocity form.
- Elastic energy: `U_el = 1/2 Σ_v (eps−eps_m)_v (C:(eps−eps_m))_v` — the Voigt dot with
  engineering shear equals the tensor Frobenius product; matches `LLGWithLESolver.U_el`.
- Kinetic energy `T_el = ∫ 1/2 rho du² = ∫ p²/(2rho)` — identical by p = rho·du.

## 6. Boundary conditions

- **Dirichlet**: `frozenDispLoc`/`frozenDispVal` cells: u forced to the (possibly
  time-dependent) value at every solver stage *after* the stage time is set; du/dt and
  d(du)/dt are zeroed there (magnum.np `state.bcs["ud"]` + `_get_mask_elastic`).
- **Neumann**: on every non-periodic outer face the entire force column ∂_d(σ−σm)_(i,d) at
  the boundary cells is replaced by the boundary estimate built from the traction and the
  total stress (paper Eqs. 49–51):
  - `BoundaryNodesNP=3`: midpoint/3-pt formula with hl = dx/2, hr = dx (uniform grid),
    `g = −s(s2·hl² − t·hr² + (hr²−hl²)s1)/(hr·hl² + hl·hr²)`;
  - `=2`: `g = s(t−s2)/(1.5dx)`; `=1`: `g = s(t−s1)/dx`.
  s1/s2 = stress at the boundary/next-inner cell; rows use σ_dd, σ_(t1 d), σ_(t2 d).
- **Natural open boundaries**: traction defaults to 0 → homogeneous Neumann automatically.
- **PBC**: periodic directions use the wrap-around jump-aware differences
  (`first_derivative_with_jump_conditions_and_pbc` equivalents) and get no face override.
- **Airbox**: C=0 cells + `elasticMaskNP=0`; all divisions guarded like `nan_to_num`
  (`melas_safediv`); requires `IterationDepthNP >= 1`, as in the paper.

## 7. Jump conditions

- C/B table exactly as paper Table I (C rows [C11,C66,C55]/[C66,C22,C44]/[C55,C44,C33];
  B holds the coefficient-weighted transversal gradients **minus** the sigma_m component
  with Voigt map x→(0,5,4), y→(5,1,3), z→(4,3,2)).
- Discrete forward/backward differences: uniform-grid specialization of Eqs. (36)/(37)
  (mumax has no irregular mesh):
  `g_fwd = (2C_{i+1}(u_{i+1}−u_i) + dx(Bl_{i+1}−Br_i))/(dx(C_i+C_{i+1}))`, mirrored for
  g_bkw; centered = mean; one-sided at open boundaries; optional 3-pt boundary stencil.
- Second derivative `d/dx(C du/dx)`: pair-based form of `_2nd_derivative_*` with
  `C_avg = 2C_iC_{i±1}/denom`, `jump = −(Br−Bl)/denom`, result divided by dx once
  (`a/dx_denom`), homogeneous-Neumann built in at open boundaries, wrap under PBC.
- Mixed derivatives: jump-aware first derivative of `C·(∂u)` with jump coefficient C and
  B = 0, including the 3-pt boundary override when `BoundaryNodesNP=3`
  (`_mixed_derivative` equivalent).
- Iteration: B initialized from magnetic stress only; then `IterationDepthNP` times:
  harmonic-mean (stiffness-weighted, guarded) displacement-gradient part added and the
  u gradients recomputed; default depth 1 (magnum.np default).
- Magnetic interface values: `m± = m ± dx/2·grad m` with exchange-jump-aware gradients
  (C→Aex, B=0), computed only inside the magnetic window; eps_m(m±) → sigma_m(m±) → B.
  **Limitation (paper Sec. II.B):** derived from the exchange boundary integrals only —
  invalid if other boundary field terms (e.g. DMI) modify the magnetic boundary condition.

## 8. Differences from the old mumax-ME implementation

The old solver (9/12) keeps: linearized B1/B2 magnetoelastic field, central differences
without interface jump conditions (its strain kernel even lacks the z-direction), fixed-step
RK4 with ad-hoc error weighting, `useBoundaries` free-boundary machinery. The NP solver
replaces all of that: thermodynamically consistent field, Table-I jump conditions with
iterative B construction, traction-consistent Neumann boundaries with 3 boundary-node
schemes, airbox support, RKF45 with per-variable tolerances. The two share only the state
storage (M/U/DU), parameters Eta/Rho/C11/C12/C44 and the Dirichlet machinery.

## 9. Differences from magnum.np (all intentional, justified)

1. **Uniform grids only** — mumax has no irregular mesh; all formulas are the uniform-dx
   specialization (they reduce exactly).
2. **Velocity instead of momentum as state** — `du` (mumax-ME naming preserved);
   p = rho·du exposed as `pNP`. Identical dynamics.
3. **eta convention** — force = −eta·du with eta in kg/(s·m³) (existing mumax-ME unit);
   `eta_magnumnp = eta_mumax/rho`. Documented in the manual.
4. **m normalized after every RK stage** (mumax convention for all its solvers);
   magnum.np normalizes only after the full step. Both preserve |m|=1 to solver order.
5. **Error norm** — global per-variable norms `max|err|/(atol + rtol·max|x|)` instead of
   magnum.np's pointwise norm; same atol/rtol structure, standard mumax practice
   (pointwise norms would need an extra reduction kernel; conservative difference).
6. **Field-term strain consistency** — the magnetoelastic field uses the same jump-aware
   strain (with `BoundaryNodesNP` accuracy and the cached stress of the current RHS
   evaluation) as the elastic force. magnum.np's field term calls `epsilon()` separately
   with default (first-order-boundary) settings — ours is the more consistent choice and
   avoids recomputing the pipeline twice per stage.
7. **Traction only on outer non-periodic faces** — magnum.np allows interior `PlaneBC`
   planes; interior free surfaces are covered by the airbox method. (Natural + user
   override merging behaves like `_update_neumann_bcs` for the outer faces.)
8. **C given as 9 regionwise scalars** (block Voigt form) instead of a per-cell 6×6
   tensor — same practical scope as the paper/`C_sym="cubic"`; `RotateC`-style anisotropic
   full tensors are out of scope.
9. **Stiffness-weighted B prefactor** at the r-interface uses the local cell's C
   (matches magnum.np: only the harmonic mean is rolled, the prefactor `C12*dyuy_xr` uses
   the unrolled C — verified line-by-line).
10. **Single-cell dimensions** (n=1, non-PBC): no Neumann override is applied (magnum.np
    applies two opposing natural planes whose contributions cancel to the same zero).

## 10. Manual validation checklist (not run here — GPU-heavy)

1. **Static FM/NM stack** (paper Sec. III.A, `demos/.../strain_jump`): 1D Ni/Al stack,
   PBC y,z, Dirichlet right, free left; relax with high alpha/eta. Expect piecewise u per
   Eq. (57), `normStressNP` xx constant per material and (σ−σm)x̂ continuous;
   `normStrainNP` from the jump-aware gradients must NOT show the interface artifact a
   plain 3-pt stencil produces. Port of the magnum.np heat-flux unit test
   (`llg_with_le_test.py::test_jump_condition`) with C11-only + lambda source recommended.
2. **Bulk shear wave in Ni** (Sec. III.B, `demos/.../bulk_shear_mode`): n=(N,1,1),
   pbc=(0,1,1), traction t_z = u0·C44·k·sin(ωt) on the x⁻ face via `tractionNP`
   (RenderFunctionLimitX to cells 0..1), Dirichlet u=0 at the last cell, m ∥ ŷ vs ẑ with
   h_ext per Eq. (61). Expect coupling only for m ∥ ẑ; energy balance
   ∫t·du dS − d(E+U_el+T_el)/dt = `GilbertDissNP` (eta=0).
3. **Bulk eigenmodes** (unit test): pbc=(1,1,1), u = u0(sin kx, 0, cos kx), du = 0:
   `dduNP` must equal −ω²u with ω = k·sqrt(C11/rho) (x comp) and k·sqrt(C44/rho) (z comp).
4. **Rayleigh open boundary** (Sec. III.C): compare `BoundaryNodesNP = 1/2/3` vs airbox
   (C=0 layer + `IterationDepthNP=1`) for the fitted RSAW mode; bn=3 and airbox should
   show the low-error behavior of Fig. 9 at 200–300 nodes/wavelength.
5. **Thin-film RSAW / thick-film SHSAW** (Secs. III.D/E) and the attached
   `NiStripesRelaxed_dyn.py`: expressible via the setup in `sanity_check_NP.mx3` scaled up
   (do not run by default).

Cheap smoke test: `sanity_check_NP.mx3` (repo root).

## 11. Known limitations

- Block-diagonal stiffness only (Eq. 24); no trigonal/monoclinic/triclinic C.
- Interior traction planes unsupported (airbox covers interior free surfaces).
- Magnetic interface values assume exchange-only boundary terms (no DMI correction).
- `StepRegion` (solver regions) is not implemented for the NP solver (fatal error message).
- Body-force excitation `force_density` (Bf) of the old solver is not included in the NP
  right-hand side (magnum.np has no equivalent); use `tractionNP` or Dirichlet driving.
- Temperature: `B_therm` handling in `SetEffectiveField` is broken independently of this
  work (see BUFFER_LEAK_REPORT.md #2) — thermal field is currently not added there.
- Single-precision (float32) like all of mumax; u ~ 1e-12 m scales are fine, but avoid
  meshes where |u| would underflow below ~1e-30.

## 12. Buffer-leak findings (magnetic path)

See `BUFFER_LEAK_REPORT.md`: (1) confirmed pool-buffer leak in `SetEffectiveFieldRegion`
(B_ext.SliceRegion never recycled — since fixed in the working tree), (2) `B_therm.Slice()`
boolean misinterpreted → thermal field silently dropped, (3) unconditional
`Recycle(Geometry.Slice())` hazard in region steppers, (4) wasteful `ku1 = v0` rebinding in
the old coupled RK4.

## Build notes

- PTX for cc 50–80 compiled with CUDA 12.8 (`cuda/build_melas_np.sh`; `-U_GNU_SOURCE`
  works around the CUDA-12/glibc-2.41 header clash; `-std=c++03` from the old Makefile is
  incompatible with that toolchain and not used for the new kernels).
- `go build ./cuda/ ./engine/ ./cmd/...` passes (the unrelated `goptuna/dashboard` embed
  error pre-exists).
- The heavy validation example was **not** run (as instructed); the implementation was
  audited twice against magnum.np instead. Audit pass 1 caught and fixed a second-derivative
  denominator error (a/dx² → a/dx); audit pass 2 caught and fixed the Dirichlet
  time-ordering in the stepper and verified every formula, sign and wrapper argument order.
