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

// CUDA handle for coshGovaluate kernel
var coshGovaluate_code cu.Function

// Stores the arguments for coshGovaluate kernel invocation
type coshGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for coshGovaluate kernel invocation
var coshGovaluate_args coshGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	coshGovaluate_args.argptr[0] = unsafe.Pointer(&coshGovaluate_args.arg_value)
	coshGovaluate_args.argptr[1] = unsafe.Pointer(&coshGovaluate_args.arg_N)
}

// Wrapper for coshGovaluate CUDA kernel, asynchronous.
func k_coshGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("coshGovaluate")
	}

	coshGovaluate_args.Lock()
	defer coshGovaluate_args.Unlock()

	if coshGovaluate_code == 0 {
		coshGovaluate_code = fatbinLoad(coshGovaluate_map, "coshGovaluate")
	}

	coshGovaluate_args.arg_value = value
	coshGovaluate_args.arg_N = N

	args := coshGovaluate_args.argptr[:]
	cu.LaunchKernel(coshGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("coshGovaluate")
	}
}

// maps compute capability on PTX code for coshGovaluate kernel.
var coshGovaluate_map = map[int]string{0: "",
	50: coshGovaluate_ptx_50,
	52: coshGovaluate_ptx_52,
	53: coshGovaluate_ptx_53,
	60: coshGovaluate_ptx_60,
	61: coshGovaluate_ptx_61,
	62: coshGovaluate_ptx_62,
	70: coshGovaluate_ptx_70,
	72: coshGovaluate_ptx_72,
	75: coshGovaluate_ptx_75,
	80: coshGovaluate_ptx_80}

// coshGovaluate PTX code for various compute capabilities.
const (
	coshGovaluate_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
	coshGovaluate_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	coshGovaluate

.visible .entry coshGovaluate(
	.param .u64 coshGovaluate_param_0,
	.param .u32 coshGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [coshGovaluate_param_0];
	ld.param.u32 	%r2, [coshGovaluate_param_1];
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
	mov.f32 	%f3, 0f3FB8AA3B;
	mul.rn.f32 	%f4, %f2, %f3;
	cvt.rzi.f32.f32 	%f5, %f4;
	abs.f32 	%f6, %f5;
	setp.gt.f32 	%p2, %f6, 0f42FC0000;
	mov.b32 	%r9, %f5;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f7, %r11;
	selp.f32 	%f8, %f7, %f5, %p2;
	mov.f32 	%f9, 0fBF317218;
	fma.rn.f32 	%f10, %f8, %f9, %f2;
	mov.f32 	%f11, 0f3102E308;
	fma.rn.f32 	%f12, %f8, %f11, %f10;
	mul.f32 	%f13, %f12, 0f3FB8AA3B;
	add.f32 	%f14, %f8, 0f4B40007D;
	mov.b32 	%r12, %f14;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f15, %r13;
	ex2.approx.ftz.f32 	%f16, %f13;
	mul.f32 	%f17, %f16, %f15;
	mov.f32 	%f18, 0f3E000000;
	div.approx.f32 	%f19, %f18, %f17;
	mov.f32 	%f20, 0f40000000;
	fma.rn.f32 	%f21, %f20, %f17, %f19;
	setp.ge.f32 	%p3, %f2, 0f42B40000;
	selp.f32 	%f22, 0f7F800000, %f21, %p3;
	st.global.f32 	[%rd4], %f22;

$L__BB0_2:
	ret;

}

`
)
