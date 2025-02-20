package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"sync"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
)

// CUDA handle for complexToPolar kernel
var complexToPolar_code cu.Function

// Stores the arguments for complexToPolar kernel invocation
type complexToPolar_args_t struct {
	arg_output unsafe.Pointer
	arg_input  unsafe.Pointer
	arg_N      int
	argptr     [3]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for complexToPolar kernel invocation
var complexToPolar_args complexToPolar_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	complexToPolar_args.argptr[0] = unsafe.Pointer(&complexToPolar_args.arg_output)
	complexToPolar_args.argptr[1] = unsafe.Pointer(&complexToPolar_args.arg_input)
	complexToPolar_args.argptr[2] = unsafe.Pointer(&complexToPolar_args.arg_N)
}

// Wrapper for complexToPolar CUDA kernel, asynchronous.
func k_complexToPolar_async(output unsafe.Pointer, input unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("complexToPolar")
	}

	complexToPolar_args.Lock()
	defer complexToPolar_args.Unlock()

	if complexToPolar_code == 0 {
		complexToPolar_code = fatbinLoad(complexToPolar_map, "complexToPolar")
	}

	complexToPolar_args.arg_output = output
	complexToPolar_args.arg_input = input
	complexToPolar_args.arg_N = N

	args := complexToPolar_args.argptr[:]
	cu.LaunchKernel(complexToPolar_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("complexToPolar")
	}
}

// maps compute capability on PTX code for complexToPolar kernel.
var complexToPolar_map = map[int]string{0: "",
	50: complexToPolar_ptx_50,
	52: complexToPolar_ptx_52,
	53: complexToPolar_ptx_53,
	60: complexToPolar_ptx_60,
	61: complexToPolar_ptx_61,
	62: complexToPolar_ptx_62,
	70: complexToPolar_ptx_70,
	72: complexToPolar_ptx_72,
	75: complexToPolar_ptx_75,
	80: complexToPolar_ptx_80}

// complexToPolar PTX code for various compute capabilities.
const (
	complexToPolar_ptx_50 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_50
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_52 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_52
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_53 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_53
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_60 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_60
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_61 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_61
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_62 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_62
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_70 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_70
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_72 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_72
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_75 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_75
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
	complexToPolar_ptx_80 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_80
.address_size 64

	// .globl	complexToPolar

.visible .entry complexToPolar(
	.param .u64 complexToPolar_param_0,
	.param .u64 complexToPolar_param_1,
	.param .u32 complexToPolar_param_2
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [complexToPolar_param_0];
	ld.param.u64 	%rd3, [complexToPolar_param_1];
	ld.param.u32 	%r2, [complexToPolar_param_2];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	shl.b32 	%r9, %r1, 1;
	cvt.s64.s32 	%rd1, %r9;
	cvta.to.global.u64 	%rd4, %rd3;
	mul.wide.s32 	%rd5, %r9, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.f32 	%f1, [%rd6];
	ld.global.nc.f32 	%f2, [%rd6+4];
	mul.f32 	%f10, %f2, %f2;
	fma.rn.f32 	%f3, %f1, %f1, %f10;
	abs.f32 	%f4, %f1;
	abs.f32 	%f5, %f2;
	setp.eq.f32 	%p2, %f4, 0f00000000;
	setp.eq.f32 	%p3, %f5, 0f00000000;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_5;
	bra.uni 	$L__BB0_2;

$L__BB0_5:
	mov.b32 	%r20, %f1;
	shr.s32 	%r21, %r20, 31;
	and.b32  	%r22, %r21, 1078530011;
	mov.b32 	%r23, %f2;
	and.b32  	%r24, %r23, -2147483648;
	or.b32  	%r25, %r22, %r24;
	mov.b32 	%f38, %r25;
	bra.uni 	$L__BB0_6;

$L__BB0_2:
	setp.eq.f32 	%p5, %f4, 0f7F800000;
	setp.eq.f32 	%p6, %f5, 0f7F800000;
	and.pred  	%p7, %p5, %p6;
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_3;

$L__BB0_4:
	mov.b32 	%r15, %f1;
	setp.lt.s32 	%p11, %r15, 0;
	selp.b32 	%r16, 1075235812, 1061752795, %p11;
	mov.b32 	%r17, %f2;
	and.b32  	%r18, %r17, -2147483648;
	or.b32  	%r19, %r16, %r18;
	mov.b32 	%f38, %r19;
	bra.uni 	$L__BB0_6;

$L__BB0_3:
	max.f32 	%f11, %f5, %f4;
	min.f32 	%f12, %f5, %f4;
	div.rn.f32 	%f13, %f12, %f11;
	mul.rn.f32 	%f14, %f13, %f13;
	mov.f32 	%f15, 0fC0B59883;
	mov.f32 	%f16, 0fBF52C7EA;
	fma.rn.f32 	%f17, %f14, %f16, %f15;
	mov.f32 	%f18, 0fC0D21907;
	fma.rn.f32 	%f19, %f17, %f14, %f18;
	mul.f32 	%f20, %f14, %f19;
	mul.f32 	%f21, %f13, %f20;
	add.f32 	%f22, %f14, 0f41355DC0;
	mov.f32 	%f23, 0f41E6BD60;
	fma.rn.f32 	%f24, %f22, %f14, %f23;
	mov.f32 	%f25, 0f419D92C8;
	fma.rn.f32 	%f26, %f24, %f14, %f25;
	rcp.rn.f32 	%f27, %f26;
	fma.rn.f32 	%f28, %f21, %f27, %f13;
	mov.f32 	%f29, 0f3FC90FDB;
	sub.f32 	%f30, %f29, %f28;
	setp.gt.f32 	%p8, %f5, %f4;
	selp.f32 	%f31, %f30, %f28, %p8;
	mov.b32 	%r10, %f1;
	setp.lt.s32 	%p9, %r10, 0;
	mov.f32 	%f32, 0f40490FDB;
	sub.f32 	%f33, %f32, %f31;
	selp.f32 	%f34, %f33, %f31, %p9;
	mov.b32 	%r11, %f34;
	mov.b32 	%r12, %f2;
	and.b32  	%r13, %r12, -2147483648;
	or.b32  	%r14, %r13, %r11;
	mov.b32 	%f35, %r14;
	add.f32 	%f36, %f4, %f5;
	setp.le.f32 	%p10, %f36, 0f7F800000;
	selp.f32 	%f38, %f35, %f36, %p10;

$L__BB0_6:
	cvta.to.global.u64 	%rd7, %rd2;
	shl.b64 	%rd8, %rd1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	sqrt.rn.f32 	%f37, %f3;
	st.global.f32 	[%rd9], %f37;
	st.global.f32 	[%rd9+4], %f38;

$L__BB0_7:
	ret;

}

`
)
