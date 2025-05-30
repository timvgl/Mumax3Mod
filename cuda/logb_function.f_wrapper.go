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

// CUDA handle for logbGovaluate kernel
var logbGovaluate_code cu.Function

// Stores the arguments for logbGovaluate kernel invocation
type logbGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for logbGovaluate kernel invocation
var logbGovaluate_args logbGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	logbGovaluate_args.argptr[0] = unsafe.Pointer(&logbGovaluate_args.arg_value)
	logbGovaluate_args.argptr[1] = unsafe.Pointer(&logbGovaluate_args.arg_N)
}

// Wrapper for logbGovaluate CUDA kernel, asynchronous.
func k_logbGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("logbGovaluate")
	}

	logbGovaluate_args.Lock()
	defer logbGovaluate_args.Unlock()

	if logbGovaluate_code == 0 {
		logbGovaluate_code = fatbinLoad(logbGovaluate_map, "logbGovaluate")
	}

	logbGovaluate_args.arg_value = value
	logbGovaluate_args.arg_N = N

	args := logbGovaluate_args.argptr[:]
	cu.LaunchKernel(logbGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("logbGovaluate")
	}
}

// maps compute capability on PTX code for logbGovaluate kernel.
var logbGovaluate_map = map[int]string{0: "",
	50: logbGovaluate_ptx_50,
	52: logbGovaluate_ptx_52,
	53: logbGovaluate_ptx_53,
	60: logbGovaluate_ptx_60,
	61: logbGovaluate_ptx_61,
	62: logbGovaluate_ptx_62,
	70: logbGovaluate_ptx_70,
	72: logbGovaluate_ptx_72,
	75: logbGovaluate_ptx_75,
	80: logbGovaluate_ptx_80}

// logbGovaluate PTX code for various compute capabilities.
const (
	logbGovaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
	logbGovaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	logbGovaluate

.visible .entry logbGovaluate(
	.param .u64 logbGovaluate_param_0,
	.param .u32 logbGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<15>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [logbGovaluate_param_0];
	ld.param.u32 	%r3, [logbGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f5, %f1;
	mov.b32 	%r2, %f5;
	setp.lt.u32 	%p2, %r2, 8388608;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	clz.b32 	%r12, %r2;
	mov.u32 	%r13, -118;
	sub.s32 	%r14, %r13, %r12;
	cvt.rn.f32.s32 	%f8, %r14;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f9, 0fFF800000, %f8, %p4;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	shr.u32 	%r10, %r2, 23;
	add.s32 	%r11, %r10, -127;
	cvt.rn.f32.s32 	%f6, %r11;
	mul.f32 	%f7, %f1, %f1;
	setp.gt.u32 	%p3, %r2, 2139095039;
	selp.f32 	%f9, %f7, %f6, %p3;

$L__BB0_4:
	st.global.f32 	[%rd1], %f9;

$L__BB0_5:
	ret;

}

`
)
