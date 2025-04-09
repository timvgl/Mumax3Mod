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

// CUDA handle for exp2Govaluate kernel
var exp2Govaluate_code cu.Function

// Stores the arguments for exp2Govaluate kernel invocation
type exp2Govaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for exp2Govaluate kernel invocation
var exp2Govaluate_args exp2Govaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	exp2Govaluate_args.argptr[0] = unsafe.Pointer(&exp2Govaluate_args.arg_value)
	exp2Govaluate_args.argptr[1] = unsafe.Pointer(&exp2Govaluate_args.arg_N)
}

// Wrapper for exp2Govaluate CUDA kernel, asynchronous.
func k_exp2Govaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("exp2Govaluate")
	}

	exp2Govaluate_args.Lock()
	defer exp2Govaluate_args.Unlock()

	if exp2Govaluate_code == 0 {
		exp2Govaluate_code = fatbinLoad(exp2Govaluate_map, "exp2Govaluate")
	}

	exp2Govaluate_args.arg_value = value
	exp2Govaluate_args.arg_N = N

	args := exp2Govaluate_args.argptr[:]
	cu.LaunchKernel(exp2Govaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("exp2Govaluate")
	}
}

// maps compute capability on PTX code for exp2Govaluate kernel.
var exp2Govaluate_map = map[int]string{0: "",
	50: exp2Govaluate_ptx_50,
	52: exp2Govaluate_ptx_52,
	53: exp2Govaluate_ptx_53,
	60: exp2Govaluate_ptx_60,
	61: exp2Govaluate_ptx_61,
	62: exp2Govaluate_ptx_62,
	70: exp2Govaluate_ptx_70,
	72: exp2Govaluate_ptx_72,
	75: exp2Govaluate_ptx_75,
	80: exp2Govaluate_ptx_80}

// exp2Govaluate PTX code for various compute capabilities.
const (
	exp2Govaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	exp2Govaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	exp2Govaluate

.visible .entry exp2Govaluate(
	.param .u64 exp2Govaluate_param_0,
	.param .u32 exp2Govaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [exp2Govaluate_param_0];
	ld.param.u32 	%r2, [exp2Govaluate_param_1];
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
	ex2.approx.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
)
