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

// CUDA handle for real kernel
var real_code cu.Function

// Stores the arguments for real kernel invocation
type real_args_t struct {
	arg_output unsafe.Pointer
	arg_input  unsafe.Pointer
	arg_N      int
	argptr     [3]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for real kernel invocation
var real_args real_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	real_args.argptr[0] = unsafe.Pointer(&real_args.arg_output)
	real_args.argptr[1] = unsafe.Pointer(&real_args.arg_input)
	real_args.argptr[2] = unsafe.Pointer(&real_args.arg_N)
}

// Wrapper for real CUDA kernel, asynchronous.
func k_real_async(output unsafe.Pointer, input unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("real")
	}

	real_args.Lock()
	defer real_args.Unlock()

	if real_code == 0 {
		real_code = fatbinLoad(real_map, "real")
	}

	real_args.arg_output = output
	real_args.arg_input = input
	real_args.arg_N = N

	args := real_args.argptr[:]
	cu.LaunchKernel(real_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("real")
	}
}

// maps compute capability on PTX code for real kernel.
var real_map = map[int]string{0: "",
	50: real_ptx_50,
	52: real_ptx_52,
	53: real_ptx_53,
	60: real_ptx_60,
	61: real_ptx_61,
	62: real_ptx_62,
	70: real_ptx_70,
	72: real_ptx_72,
	75: real_ptx_75,
	80: real_ptx_80}

// real PTX code for various compute capabilities.
const (
	real_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	real_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	real

.visible .entry real(
	.param .u64 real_param_0,
	.param .u64 real_param_1,
	.param .u32 real_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [real_param_0];
	ld.param.u64 	%rd2, [real_param_1];
	ld.param.u32 	%r2, [real_param_2];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shl.b32 	%r9, %r1, 1;
	mul.wide.s32 	%rd4, %r9, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
)
