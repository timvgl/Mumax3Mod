# Buffer-leak audit of the magnetic code path

You suspected that "at some point in the magnetic code a buffer that should be recycled is
not". The suspicion is correct. Findings below, ordered by severity. How they were found:
every call site of `q.Slice()` / `q.SliceRegion(...)` in `engine/` was enumerated
(`grep -n "\.Slice()"`), the ownership convention was established from
`cuda/buffer.go` (`Buffer`/`Recycle`, `buf_check`, `buf_max = 250`, panic
"too many buffers in use, possible memory leak") and from the `Slice()` implementations
(`Excitation.Slice` returns a **pooled buffer** and `true`; `thermField.Slice` returns its
**persistent** noise buffer and `false`; `geom.Slice` returns a pooled buffer + `true` only
when no geometry is set, otherwise persistent + `false`). Each call site was then checked
for a matching `cuda.Recycle`.

## 1. Confirmed leak: `SetEffectiveFieldRegion` never recycles the B_ext slice

`engine/effectivefield.go` (region path, used by `StepRegion`/solver-region runs):

```go
dta, succeeded := B_ext.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
if succeeded {
    cuda.Add(dst, dst, dta)
}
// <- no cuda.Recycle(dta) on any path
```

`Excitation.SliceRegion` (engine/excitation.go:265) allocates `bufRed := cuda.Buffer(...)`
and returns it with `true` ("needs recycle"). Every effective-field evaluation in the
region solver therefore permanently removes one 3-component buffer from the pool. After
enough evaluations `len(buf_check) >= buf_max (250)` and mumax panics with
"too many buffers in use, possible memory leak" — or, before that, GPU memory grows.
Compare the full-mesh path directly above it, which does call `cuda.Recycle(dta)`.

**Fix:** add `cuda.Recycle(dta)` after the `if succeeded` block (unconditionally, like the
full-mesh path).

## 2. Functional bug (not a leak): the `Slice()` boolean is misinterpreted for B_therm

`engine/effectivefield.go`, full-mesh path:

```go
dta, succeeded := B_ext.Slice()
if succeeded {
    cuda.Add(dst, dst, dta)
}
cuda.Recycle(dta)
if !relaxing {
    dta, succeeded := B_therm.Slice()
    if succeeded {
        cuda.Add(dst, dst, dta)
    }
}
```

In vanilla mumax3 the second return value of `Slice()` means **"the caller must recycle
this buffer"**, not "success". `thermField.Slice()` (engine/temperature.go:251) returns
`(b.noise, false)`. With the "succeeded" interpretation the thermal field is **never added
to the effective field** in this path (temperature silently ignored). It is not a memory
leak (the noise buffer is persistent), but it is a physics bug whenever `Temp != 0` and the
default solvers are used. Vanilla mumax3 calls `B_therm.AddTo(dst)` here instead.

**Fix:** restore `B_therm.AddTo(dst)` (or add the slice unconditionally and recycle only
when the bool is true).

## 3. Related hazard: unconditional `Recycle` of `Geometry.Slice()` in region steppers

Several `StepRegion` implementations (e.g. `engine/rk45dp.go:141`,
`engine/magnetoelastic_RK4.go:304`, `engine/rk4.go`, `engine/heun.go`, ...) do:

```go
GeomBig, _ := Geometry.Slice()
cuda.Crop(geom, GeomBig, ...)
cuda.Recycle(GeomBig)
```

`geom.Slice()` returns a **pooled** buffer + `true` only when *no* geometry is set. When a
geometry *is* set it returns the persistent geometry buffer + `false`; `cuda.Recycle` then
panics with "recycle: was not obtained with getbuffer" (cuda/buffer.go:154) or, if that
buffer happened to be pool-allocated, would corrupt the geometry by handing it out as
scratch. The recycle must be guarded by the returned boolean:
`if r { cuda.Recycle(GeomBig) }`.

## 4. Note on `magelasRK4.Step()` (old coupled solver): `ku1 = v0` rebinding

`engine/magnetoelastic_RK4.go:104`: `ku1 = v0` re-points `ku1` at `v0`'s buffer after both
already have `defer cuda.Recycle(...)` registered. Because deferred arguments are evaluated
at `defer` time, the original `ku1` buffer *is* still recycled — so no leak — but the
freshly allocated `ku1` buffer is never written before being recycled, i.e. one buffer
allocation per step is wasted, and the aliasing makes the recycle order fragile (v0 is
recycled once through its own defer while the data is still referenced through `ku1` in
`Madd5` — safe only because recycling happens after all launches). Worth cleaning up, not a
leak.

## Not affected

- `torque.go` (`J.Slice()`, `FixedLayer.Slice()`): recycle guarded by the returned bool — correct.
- `ext_magnetoelastic.go`, `elastic_Rhs.go` (`Bf.Slice()`): followed by `defer cuda.Recycle(bf)` — correct.
- The new NP solver code recycles all pooled buffers (`melasNPFields.free()`, `defer`s on
  MSlices) and holds no persistent scratch.
