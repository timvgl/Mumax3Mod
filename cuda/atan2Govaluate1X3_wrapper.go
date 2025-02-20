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

// CUDA handle for atan2Govaluate1X3 kernel
var atan2Govaluate1X3_code cu.Function

// Stores the arguments for atan2Govaluate1X3 kernel invocation
type atan2Govaluate1X3_args_t struct {
	arg_output unsafe.Pointer
	arg_input2 float32
	arg_input  unsafe.Pointer
	arg_N      int
	argptr     [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for atan2Govaluate1X3 kernel invocation
var atan2Govaluate1X3_args atan2Govaluate1X3_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	atan2Govaluate1X3_args.argptr[0] = unsafe.Pointer(&atan2Govaluate1X3_args.arg_output)
	atan2Govaluate1X3_args.argptr[1] = unsafe.Pointer(&atan2Govaluate1X3_args.arg_input2)
	atan2Govaluate1X3_args.argptr[2] = unsafe.Pointer(&atan2Govaluate1X3_args.arg_input)
	atan2Govaluate1X3_args.argptr[3] = unsafe.Pointer(&atan2Govaluate1X3_args.arg_N)
}

// Wrapper for atan2Govaluate1X3 CUDA kernel, asynchronous.
func k_atan2Govaluate1X3_async(output unsafe.Pointer, input2 float32, input unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("atan2Govaluate1X3")
	}

	atan2Govaluate1X3_args.Lock()
	defer atan2Govaluate1X3_args.Unlock()

	if atan2Govaluate1X3_code == 0 {
		atan2Govaluate1X3_code = fatbinLoad(atan2Govaluate1X3_map, "atan2Govaluate1X3")
	}

	atan2Govaluate1X3_args.arg_output = output
	atan2Govaluate1X3_args.arg_input2 = input2
	atan2Govaluate1X3_args.arg_input = input
	atan2Govaluate1X3_args.arg_N = N

	args := atan2Govaluate1X3_args.argptr[:]
	cu.LaunchKernel(atan2Govaluate1X3_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("atan2Govaluate1X3")
	}
}

// maps compute capability on PTX code for atan2Govaluate1X3 kernel.
var atan2Govaluate1X3_map = map[int]string{0: "",
	50: atan2Govaluate1X3_ptx_50,
	52: atan2Govaluate1X3_ptx_52,
	53: atan2Govaluate1X3_ptx_53,
	60: atan2Govaluate1X3_ptx_60,
	61: atan2Govaluate1X3_ptx_61,
	62: atan2Govaluate1X3_ptx_62,
	70: atan2Govaluate1X3_ptx_70,
	72: atan2Govaluate1X3_ptx_72,
	75: atan2Govaluate1X3_ptx_75,
	80: atan2Govaluate1X3_ptx_80}

// atan2Govaluate1X3 PTX code for various compute capabilities.
const (
	atan2Govaluate1X3_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
	atan2Govaluate1X3_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	atan2Govaluate1X3

.visible .entry atan2Govaluate1X3(
	.param .u64 atan2Govaluate1X3_param_0,
	.param .f32 atan2Govaluate1X3_param_1,
	.param .u64 atan2Govaluate1X3_param_2,
	.param .u32 atan2Govaluate1X3_param_3
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<36>;
	.reg .b32 	%r<25>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [atan2Govaluate1X3_param_0];
	ld.param.f32 	%f8, [atan2Govaluate1X3_param_1];
	ld.param.u64 	%rd3, [atan2Govaluate1X3_param_2];
	ld.param.u32 	%r2, [atan2Govaluate1X3_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	abs.f32 	%f2, %f1;
	setp.eq.f32 	%p2, %f2, 0f00000000;
	abs.f32 	%f3, %f8;
	setp.eq.f32 	%p3, %f3, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r19, %f1;
	shr.s32 	%r20, %r19, 31;
	and.b32  	%r21, %r20, 1078530011;
	mov.b32 	%r22, %f8;
	and.b32  	%r23, %r22, -2147483648;
	or.b32  	%r24, %r21, %r23;
	mov.b32 	%f35, %r24;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f2, 0f7F800000;
	setp.eq.f32 	%p6, %f3, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r14, %f1;
	setp.lt.s32 	%p11, %r14, 0;
	selp.b32 	%r15, 1075235812, 1061752795, %p11;
	mov.b32 	%r16, %f8;
	and.b32  	%r17, %r16, -2147483648;
	or.b32  	%r18, %r15, %r17;
	mov.b32 	%f35, %r18;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f9, %f3, %f2;
	min.f32 	%f10, %f3, %f2;
	div.rn.f32 	%f11, %f10, %f9;
	mul.rn.f32 	%f12, %f11, %f11;
	mov.f32 	%f13, 0fC0B59883;
	mov.f32 	%f14, 0fBF52C7EA;
	fma.rn.f32 	%f15, %f12, %f14, %f13;
	mov.f32 	%f16, 0fC0D21907;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mul.f32 	%f18, %f12, %f17;
	mul.f32 	%f19, %f11, %f18;
	add.f32 	%f20, %f12, 0f41355DC0;
	mov.f32 	%f21, 0f41E6BD60;
	fma.rn.f32 	%f22, %f20, %f12, %f21;
	mov.f32 	%f23, 0f419D92C8;
	fma.rn.f32 	%f24, %f22, %f12, %f23;
	rcp.rn.f32 	%f25, %f24;
	fma.rn.f32 	%f26, %f19, %f25, %f11;
	mov.f32 	%f27, 0f3FC90FDB;
	sub.f32 	%f28, %f27, %f26;
	setp.gt.f32 	%p8, %f3, %f2;
	selp.f32 	%f29, %f28, %f26, %p8;
	mov.b32 	%r9, %f1;
	setp.lt.s32 	%p9, %r9, 0;
	mov.f32 	%f30, 0f40490FDB;
	sub.f32 	%f31, %f30, %f29;
	selp.f32 	%f32, %f31, %f29, %p9;
	mov.b32 	%r10, %f32;
	mov.b32 	%r11, %f8;
	and.b32  	%r12, %r11, -2147483648;
	or.b32  	%r13, %r12, %r10;
	mov.b32 	%f33, %r13;
	add.f32 	%f34, %f2, %f3;
	setp.le.f32 	%p10, %f34, 0f7F800000;
	selp.f32 	%f35, %f33, %f34, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	st.global.f32 	[%rd9], %f35;

$L__BB0_7:
	ret;

}

`
)
