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

// CUDA handle for ldexpGovaluate1X3 kernel
var ldexpGovaluate1X3_code cu.Function

// Stores the arguments for ldexpGovaluate1X3 kernel invocation
type ldexpGovaluate1X3_args_t struct {
	arg_output unsafe.Pointer
	arg_input2 float32
	arg_input  unsafe.Pointer
	arg_N      int
	argptr     [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for ldexpGovaluate1X3 kernel invocation
var ldexpGovaluate1X3_args ldexpGovaluate1X3_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	ldexpGovaluate1X3_args.argptr[0] = unsafe.Pointer(&ldexpGovaluate1X3_args.arg_output)
	ldexpGovaluate1X3_args.argptr[1] = unsafe.Pointer(&ldexpGovaluate1X3_args.arg_input2)
	ldexpGovaluate1X3_args.argptr[2] = unsafe.Pointer(&ldexpGovaluate1X3_args.arg_input)
	ldexpGovaluate1X3_args.argptr[3] = unsafe.Pointer(&ldexpGovaluate1X3_args.arg_N)
}

// Wrapper for ldexpGovaluate1X3 CUDA kernel, asynchronous.
func k_ldexpGovaluate1X3_async(output unsafe.Pointer, input2 float32, input unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("ldexpGovaluate1X3")
	}

	ldexpGovaluate1X3_args.Lock()
	defer ldexpGovaluate1X3_args.Unlock()

	if ldexpGovaluate1X3_code == 0 {
		ldexpGovaluate1X3_code = fatbinLoad(ldexpGovaluate1X3_map, "ldexpGovaluate1X3")
	}

	ldexpGovaluate1X3_args.arg_output = output
	ldexpGovaluate1X3_args.arg_input2 = input2
	ldexpGovaluate1X3_args.arg_input = input
	ldexpGovaluate1X3_args.arg_N = N

	args := ldexpGovaluate1X3_args.argptr[:]
	cu.LaunchKernel(ldexpGovaluate1X3_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("ldexpGovaluate1X3")
	}
}

// maps compute capability on PTX code for ldexpGovaluate1X3 kernel.
var ldexpGovaluate1X3_map = map[int]string{0: "",
	50: ldexpGovaluate1X3_ptx_50,
	52: ldexpGovaluate1X3_ptx_52,
	53: ldexpGovaluate1X3_ptx_53,
	60: ldexpGovaluate1X3_ptx_60,
	61: ldexpGovaluate1X3_ptx_61,
	62: ldexpGovaluate1X3_ptx_62,
	70: ldexpGovaluate1X3_ptx_70,
	72: ldexpGovaluate1X3_ptx_72,
	75: ldexpGovaluate1X3_ptx_75,
	80: ldexpGovaluate1X3_ptx_80}

// ldexpGovaluate1X3 PTX code for various compute capabilities.
const (
	ldexpGovaluate1X3_ptx_50 = `
.version 8.4
.target sm_50
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_52 = `
.version 8.4
.target sm_52
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_53 = `
.version 8.4
.target sm_53
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_60 = `
.version 8.4
.target sm_60
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_61 = `
.version 8.4
.target sm_61
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_62 = `
.version 8.4
.target sm_62
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_70 = `
.version 8.4
.target sm_70
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_72 = `
.version 8.4
.target sm_72
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_75 = `
.version 8.4
.target sm_75
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
	ldexpGovaluate1X3_ptx_80 = `
.version 8.4
.target sm_80
.address_size 64

	// .globl	ldexpGovaluate1X3

.visible .entry ldexpGovaluate1X3(
	.param .u64 ldexpGovaluate1X3_param_0,
	.param .f32 ldexpGovaluate1X3_param_1,
	.param .u64 ldexpGovaluate1X3_param_2,
	.param .u32 ldexpGovaluate1X3_param_3
)
{
	.reg .pred 	%p<10>;
	.reg .f32 	%f<25>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [ldexpGovaluate1X3_param_0];
	ld.param.f32 	%f7, [ldexpGovaluate1X3_param_1];
	ld.param.u64 	%rd3, [ldexpGovaluate1X3_param_2];
	ld.param.u32 	%r4, [ldexpGovaluate1X3_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %nctaid.x;
	mov.u32 	%r7, %ctaid.x;
	mad.lo.s32 	%r8, %r5, %r6, %r7;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	setp.ge.s32 	%p1, %r1, %r4;
	@%p1 bra 	$L__BB0_9;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f8, [%rd6];
	cvt.rzi.s32.f32 	%r2, %f8;
	abs.f32 	%f1, %f7;
	setp.eq.f32 	%p2, %f1, 0f00000000;
	setp.eq.f32 	%p3, %f1, 0f7F800000;
	or.pred  	%p4, %p2, %p3;
	setp.eq.s32 	%p5, %r2, 0;
	or.pred  	%p6, %p5, %p4;
	@%p6 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_2;

$L__BB0_7:
	setp.gt.f32 	%p9, %f1, 0f00000000;
	add.f32 	%f23, %f7, %f7;
	selp.f32 	%f24, %f7, %f23, %p9;
	bra.uni 	$L__BB0_8;

$L__BB0_2:
	abs.s32 	%r3, %r2;
	setp.lt.s32 	%p7, %r3, 126;
	@%p7 bra 	$L__BB0_6;
	bra.uni 	$L__BB0_3;

$L__BB0_6:
	cvt.rn.f32.s32 	%f21, %r2;
	ex2.approx.ftz.f32 	%f22, %f21;
	mul.f32 	%f24, %f22, %f7;
	bra.uni 	$L__BB0_8;

$L__BB0_3:
	setp.lt.s32 	%p8, %r3, 252;
	@%p8 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_4;

$L__BB0_5:
	shr.u32 	%r16, %r2, 31;
	add.s32 	%r17, %r2, %r16;
	shr.s32 	%r18, %r17, 1;
	sub.s32 	%r19, %r2, %r18;
	cvt.rn.f32.s32 	%f16, %r18;
	ex2.approx.ftz.f32 	%f17, %f16;
	mul.f32 	%f18, %f17, %f7;
	cvt.rn.f32.s32 	%f19, %r19;
	ex2.approx.ftz.f32 	%f20, %f19;
	mul.f32 	%f24, %f18, %f20;
	bra.uni 	$L__BB0_8;

$L__BB0_4:
	shr.s32 	%r11, %r2, 31;
	shr.u32 	%r12, %r11, 30;
	add.s32 	%r13, %r2, %r12;
	shr.s32 	%r14, %r13, 2;
	cvt.rn.f32.s32 	%f9, %r14;
	ex2.approx.ftz.f32 	%f10, %f9;
	mad.lo.s32 	%r15, %r14, -3, %r2;
	mul.f32 	%f11, %f10, %f7;
	mul.f32 	%f12, %f10, %f11;
	mul.f32 	%f13, %f10, %f12;
	cvt.rn.f32.s32 	%f14, %r15;
	ex2.approx.ftz.f32 	%f15, %f14;
	mul.f32 	%f24, %f15, %f13;

$L__BB0_8:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f24;

$L__BB0_9:
	ret;

}

`
)
