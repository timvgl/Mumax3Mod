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

// CUDA handle for normGovaluate kernel
var normGovaluate_code cu.Function

// Stores the arguments for normGovaluate kernel invocation
type normGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for normGovaluate kernel invocation
var normGovaluate_args normGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	normGovaluate_args.argptr[0] = unsafe.Pointer(&normGovaluate_args.arg_value)
	normGovaluate_args.argptr[1] = unsafe.Pointer(&normGovaluate_args.arg_N)
}

// Wrapper for normGovaluate CUDA kernel, asynchronous.
func k_normGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("normGovaluate")
	}

	normGovaluate_args.Lock()
	defer normGovaluate_args.Unlock()

	if normGovaluate_code == 0 {
		normGovaluate_code = fatbinLoad(normGovaluate_map, "normGovaluate")
	}

	normGovaluate_args.arg_value = value
	normGovaluate_args.arg_N = N

	args := normGovaluate_args.argptr[:]
	cu.LaunchKernel(normGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("normGovaluate")
	}
}

// maps compute capability on PTX code for normGovaluate kernel.
var normGovaluate_map = map[int]string{0: "",
	50: normGovaluate_ptx_50,
	52: normGovaluate_ptx_52,
	53: normGovaluate_ptx_53,
	60: normGovaluate_ptx_60,
	61: normGovaluate_ptx_61,
	62: normGovaluate_ptx_62,
	70: normGovaluate_ptx_70,
	72: normGovaluate_ptx_72,
	75: normGovaluate_ptx_75,
	80: normGovaluate_ptx_80}

// normGovaluate PTX code for various compute capabilities.
const (
	normGovaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	normGovaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	normGovaluate

.visible .entry normGovaluate(
	.param .u64 normGovaluate_param_0,
	.param .u32 normGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [normGovaluate_param_0];
	ld.param.u32 	%r2, [normGovaluate_param_1];
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
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
)
