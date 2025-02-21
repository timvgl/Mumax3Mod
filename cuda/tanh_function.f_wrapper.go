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

// CUDA handle for tanhGovaluate kernel
var tanhGovaluate_code cu.Function

// Stores the arguments for tanhGovaluate kernel invocation
type tanhGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for tanhGovaluate kernel invocation
var tanhGovaluate_args tanhGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	tanhGovaluate_args.argptr[0] = unsafe.Pointer(&tanhGovaluate_args.arg_value)
	tanhGovaluate_args.argptr[1] = unsafe.Pointer(&tanhGovaluate_args.arg_N)
}

// Wrapper for tanhGovaluate CUDA kernel, asynchronous.
func k_tanhGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("tanhGovaluate")
	}

	tanhGovaluate_args.Lock()
	defer tanhGovaluate_args.Unlock()

	if tanhGovaluate_code == 0 {
		tanhGovaluate_code = fatbinLoad(tanhGovaluate_map, "tanhGovaluate")
	}

	tanhGovaluate_args.arg_value = value
	tanhGovaluate_args.arg_N = N

	args := tanhGovaluate_args.argptr[:]
	cu.LaunchKernel(tanhGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("tanhGovaluate")
	}
}

// maps compute capability on PTX code for tanhGovaluate kernel.
var tanhGovaluate_map = map[int]string{0: "",
	50: tanhGovaluate_ptx_50,
	52: tanhGovaluate_ptx_52,
	53: tanhGovaluate_ptx_53,
	60: tanhGovaluate_ptx_60,
	61: tanhGovaluate_ptx_61,
	62: tanhGovaluate_ptx_62,
	70: tanhGovaluate_ptx_70,
	72: tanhGovaluate_ptx_72,
	75: tanhGovaluate_ptx_75,
	80: tanhGovaluate_ptx_80}

// tanhGovaluate PTX code for various compute capabilities.
const (
	tanhGovaluate_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
	tanhGovaluate_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	tanhGovaluate

.visible .entry tanhGovaluate(
	.param .u64 tanhGovaluate_param_0,
	.param .u32 tanhGovaluate_param_1
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [tanhGovaluate_param_0];
	ld.param.u32 	%r2, [tanhGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.ltu.f32 	%p2, %f2, 0f3F19999A;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mul.f32 	%f14, %f1, %f1;
	mov.f32 	%f15, 0fBD563CAE;
	mov.f32 	%f16, 0f3C80F082;
	fma.rn.f32 	%f17, %f16, %f14, %f15;
	mov.f32 	%f18, 0f3E085941;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mov.f32 	%f20, 0fBEAAA9ED;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f00000000;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	fma.rn.f32 	%f24, %f23, %f1, %f1;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mul.f32 	%f6, %f2, 0f4038AA3B;
	ex2.approx.ftz.f32 	%f7, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	rcp.approx.ftz.f32 	%f10, %f8;
	mov.f32 	%f11, 0fC0000000;
	fma.rn.f32 	%f12, %f10, %f11, %f9;
	setp.ge.f32 	%p3, %f2, 0f41102CB4;
	selp.f32 	%f13, 0f3F800000, %f12, %p3;
	mov.b32 	%r9, %f13;
	mov.b32 	%r10, %f1;
	and.b32  	%r11, %r10, -2147483648;
	or.b32  	%r12, %r11, %r9;
	mov.b32 	%f24, %r12;

$L__BB0_4:
	st.global.f32 	[%rd1], %f24;

$L__BB0_5:
	ret;

}

`
)
