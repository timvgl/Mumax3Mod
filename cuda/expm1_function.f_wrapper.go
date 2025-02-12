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

// CUDA handle for expm1Govaluate kernel
var expm1Govaluate_code cu.Function

// Stores the arguments for expm1Govaluate kernel invocation
type expm1Govaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for expm1Govaluate kernel invocation
var expm1Govaluate_args expm1Govaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	expm1Govaluate_args.argptr[0] = unsafe.Pointer(&expm1Govaluate_args.arg_value)
	expm1Govaluate_args.argptr[1] = unsafe.Pointer(&expm1Govaluate_args.arg_N)
}

// Wrapper for expm1Govaluate CUDA kernel, asynchronous.
func k_expm1Govaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("expm1Govaluate")
	}

	expm1Govaluate_args.Lock()
	defer expm1Govaluate_args.Unlock()

	if expm1Govaluate_code == 0 {
		expm1Govaluate_code = fatbinLoad(expm1Govaluate_map, "expm1Govaluate")
	}

	expm1Govaluate_args.arg_value = value
	expm1Govaluate_args.arg_N = N

	args := expm1Govaluate_args.argptr[:]
	cu.LaunchKernel(expm1Govaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("expm1Govaluate")
	}
}

// maps compute capability on PTX code for expm1Govaluate kernel.
var expm1Govaluate_map = map[int]string{0: "",
	50: expm1Govaluate_ptx_50,
	52: expm1Govaluate_ptx_52,
	53: expm1Govaluate_ptx_53,
	60: expm1Govaluate_ptx_60,
	61: expm1Govaluate_ptx_61,
	62: expm1Govaluate_ptx_62,
	70: expm1Govaluate_ptx_70,
	72: expm1Govaluate_ptx_72,
	75: expm1Govaluate_ptx_75,
	80: expm1Govaluate_ptx_80}

// expm1Govaluate PTX code for various compute capabilities.
const (
	expm1Govaluate_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
	expm1Govaluate_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	expm1Govaluate

.visible .entry expm1Govaluate(
	.param .u64 expm1Govaluate_param_0,
	.param .u32 expm1Govaluate_param_1
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<33>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [expm1Govaluate_param_0];
	ld.param.u32 	%r2, [expm1Govaluate_param_1];
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
	mul.f32 	%f2, %f1, 0f3FB8AA3B;
	cvt.rni.f32.f32 	%f3, %f2;
	abs.f32 	%f4, %f1;
	setp.lt.f32 	%p2, %f4, 0f3ED1EB85;
	selp.f32 	%f5, 0f00000000, %f3, %p2;
	neg.f32 	%f6, %f5;
	mov.f32 	%f7, 0f3F317200;
	fma.rn.f32 	%f8, %f6, %f7, %f1;
	mov.f32 	%f9, 0f35BFBE8E;
	fma.rn.f32 	%f10, %f6, %f9, %f8;
	setp.eq.f32 	%p3, %f5, 0f43000000;
	add.f32 	%f11, %f5, 0fBF800000;
	selp.f32 	%f12, %f11, %f5, %p3;
	mov.f32 	%f13, 0f3C095663;
	mov.f32 	%f14, 0f3AB5EBE6;
	fma.rn.f32 	%f15, %f14, %f10, %f13;
	mov.f32 	%f16, 0f3D2AABE3;
	fma.rn.f32 	%f17, %f15, %f10, %f16;
	mov.f32 	%f18, 0f3E2AA9F6;
	fma.rn.f32 	%f19, %f17, %f10, %f18;
	mov.f32 	%f20, 0f3EFFFFFE;
	fma.rn.f32 	%f21, %f19, %f10, %f20;
	mul.f32 	%f22, %f10, %f21;
	fma.rn.f32 	%f23, %f22, %f10, %f10;
	ex2.approx.f32 	%f24, %f12;
	add.f32 	%f25, %f24, 0fBF800000;
	fma.rn.f32 	%f26, %f23, %f24, %f25;
	add.f32 	%f27, %f26, %f26;
	selp.f32 	%f28, %f27, %f26, %p3;
	setp.gt.f32 	%p4, %f12, 0f43000000;
	selp.f32 	%f29, 0f7F800000, %f28, %p4;
	setp.lt.f32 	%p5, %f12, 0fC1C80000;
	selp.f32 	%f30, 0fBF800000, %f29, %p5;
	setp.eq.f32 	%p6, %f1, 0f00000000;
	add.f32 	%f31, %f1, %f1;
	selp.f32 	%f32, %f31, %f30, %p6;
	st.global.f32 	[%rd4], %f32;

$L__BB0_2:
	ret;

}

`
)
