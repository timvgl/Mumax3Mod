package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for cbrtGovaluate kernel
var cbrtGovaluate_code cu.Function

// Stores the arguments for cbrtGovaluate kernel invocation
type cbrtGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for cbrtGovaluate kernel invocation
var cbrtGovaluate_args cbrtGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	cbrtGovaluate_args.argptr[0] = unsafe.Pointer(&cbrtGovaluate_args.arg_value)
	cbrtGovaluate_args.argptr[1] = unsafe.Pointer(&cbrtGovaluate_args.arg_N)
}

// Wrapper for cbrtGovaluate CUDA kernel, asynchronous.
func k_cbrtGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("cbrtGovaluate")
	}

	cbrtGovaluate_args.Lock()
	defer cbrtGovaluate_args.Unlock()

	if cbrtGovaluate_code == 0 {
		cbrtGovaluate_code = fatbinLoad(cbrtGovaluate_map, "cbrtGovaluate")
	}

	cbrtGovaluate_args.arg_value = value
	cbrtGovaluate_args.arg_N = N

	args := cbrtGovaluate_args.argptr[:]
	cu.LaunchKernel(cbrtGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("cbrtGovaluate")
	}
}

// maps compute capability on PTX code for cbrtGovaluate kernel.
var cbrtGovaluate_map = map[int]string{0: "",
	50: cbrtGovaluate_ptx_50,
	52: cbrtGovaluate_ptx_52,
	53: cbrtGovaluate_ptx_53,
	60: cbrtGovaluate_ptx_60,
	61: cbrtGovaluate_ptx_61,
	62: cbrtGovaluate_ptx_62,
	70: cbrtGovaluate_ptx_70,
	72: cbrtGovaluate_ptx_72,
	75: cbrtGovaluate_ptx_75,
	80: cbrtGovaluate_ptx_80}

// cbrtGovaluate PTX code for various compute capabilities.
const (
	cbrtGovaluate_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
	cbrtGovaluate_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	cbrtGovaluate

.visible .entry cbrtGovaluate(
	.param .u64 cbrtGovaluate_param_0,
	.param .u32 cbrtGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [cbrtGovaluate_param_0];
	ld.param.u32 	%r2, [cbrtGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	lg2.approx.f32 	%f3, %f2;
	mul.f32 	%f4, %f3, 0f3EAAAAAB;
	ex2.approx.ftz.f32 	%f5, %f4;
	mul.f32 	%f6, %f5, %f5;
	rcp.approx.ftz.f32 	%f7, %f6;
	neg.f32 	%f8, %f2;
	fma.rn.f32 	%f9, %f7, %f8, %f5;
	mov.f32 	%f10, 0fBEAAAAAB;
	fma.rn.f32 	%f11, %f9, %f10, %f5;
	setp.lt.f32 	%p2, %f1, 0f00000000;
	neg.f32 	%f12, %f11;
	selp.f32 	%f13, %f12, %f11, %p2;
	add.f32 	%f14, %f1, %f1;
	setp.eq.f32 	%p3, %f14, %f1;
	selp.f32 	%f15, %f14, %f13, %p3;
	st.global.f32 	[%rd4], %f15;

$L__BB0_2:
	ret;

}

`
)
