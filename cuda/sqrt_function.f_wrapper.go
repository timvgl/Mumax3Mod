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

// CUDA handle for sqrtGovaluate kernel
var sqrtGovaluate_code cu.Function

// Stores the arguments for sqrtGovaluate kernel invocation
type sqrtGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for sqrtGovaluate kernel invocation
var sqrtGovaluate_args sqrtGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	sqrtGovaluate_args.argptr[0] = unsafe.Pointer(&sqrtGovaluate_args.arg_value)
	sqrtGovaluate_args.argptr[1] = unsafe.Pointer(&sqrtGovaluate_args.arg_N)
}

// Wrapper for sqrtGovaluate CUDA kernel, asynchronous.
func k_sqrtGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("sqrtGovaluate")
	}

	sqrtGovaluate_args.Lock()
	defer sqrtGovaluate_args.Unlock()

	if sqrtGovaluate_code == 0 {
		sqrtGovaluate_code = fatbinLoad(sqrtGovaluate_map, "sqrtGovaluate")
	}

	sqrtGovaluate_args.arg_value = value
	sqrtGovaluate_args.arg_N = N

	args := sqrtGovaluate_args.argptr[:]
	cu.LaunchKernel(sqrtGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("sqrtGovaluate")
	}
}

// maps compute capability on PTX code for sqrtGovaluate kernel.
var sqrtGovaluate_map = map[int]string{0: "",
	50: sqrtGovaluate_ptx_50,
	52: sqrtGovaluate_ptx_52,
	53: sqrtGovaluate_ptx_53,
	60: sqrtGovaluate_ptx_60,
	61: sqrtGovaluate_ptx_61,
	62: sqrtGovaluate_ptx_62,
	70: sqrtGovaluate_ptx_70,
	72: sqrtGovaluate_ptx_72,
	75: sqrtGovaluate_ptx_75,
	80: sqrtGovaluate_ptx_80}

// sqrtGovaluate PTX code for various compute capabilities.
const (
	sqrtGovaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	sqrtGovaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	sqrtGovaluate

.visible .entry sqrtGovaluate(
	.param .u64 sqrtGovaluate_param_0,
	.param .u32 sqrtGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [sqrtGovaluate_param_0];
	ld.param.u32 	%r2, [sqrtGovaluate_param_1];
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
	sqrt.rn.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
)
