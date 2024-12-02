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

// CUDA handle for scale kernel
var scale_code cu.Function

// Stores the arguments for scale kernel invocation
type scale_args_t struct {
	arg_x      unsafe.Pointer
	arg_scale  float32
	arg_offset float32
	arg_N      int
	argptr     [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for scale kernel invocation
var scale_args scale_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	scale_args.argptr[0] = unsafe.Pointer(&scale_args.arg_x)
	scale_args.argptr[1] = unsafe.Pointer(&scale_args.arg_scale)
	scale_args.argptr[2] = unsafe.Pointer(&scale_args.arg_offset)
	scale_args.argptr[3] = unsafe.Pointer(&scale_args.arg_N)
}

// Wrapper for scale CUDA kernel, asynchronous.
func k_scale_async(x unsafe.Pointer, scale float32, offset float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("scale")
	}

	scale_args.Lock()
	defer scale_args.Unlock()

	if scale_code == 0 {
		scale_code = fatbinLoad(scale_map, "scale")
	}

	scale_args.arg_x = x
	scale_args.arg_scale = scale
	scale_args.arg_offset = offset
	scale_args.arg_N = N

	args := scale_args.argptr[:]
	cu.LaunchKernel(scale_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("scale")
	}
}

// maps compute capability on PTX code for scale kernel.
var scale_map = map[int]string{0: "",
	50: scale_ptx_50,
	52: scale_ptx_52,
	53: scale_ptx_53,
	60: scale_ptx_60,
	61: scale_ptx_61,
	62: scale_ptx_62,
	70: scale_ptx_70,
	72: scale_ptx_72,
	75: scale_ptx_75,
	80: scale_ptx_80}

// scale PTX code for various compute capabilities.
const (
	scale_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
	scale_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	scale

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .f32 scale_param_1,
	.param .f32 scale_param_2,
	.param .u32 scale_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.f32 	%f1, [scale_param_1];
	ld.param.f32 	%f2, [scale_param_2];
	ld.param.u32 	%r2, [scale_param_3];
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
	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd4], %f4;

$L__BB0_2:
	ret;

}

`
)
