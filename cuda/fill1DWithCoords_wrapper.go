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

// CUDA handle for fill1DWithCoords kernel
var fill1DWithCoords_code cu.Function

// Stores the arguments for fill1DWithCoords kernel invocation
type fill1DWithCoords_args_t struct {
	arg_dst    unsafe.Pointer
	arg_factor float32
	arg_N      int
	argptr     [3]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for fill1DWithCoords kernel invocation
var fill1DWithCoords_args fill1DWithCoords_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	fill1DWithCoords_args.argptr[0] = unsafe.Pointer(&fill1DWithCoords_args.arg_dst)
	fill1DWithCoords_args.argptr[1] = unsafe.Pointer(&fill1DWithCoords_args.arg_factor)
	fill1DWithCoords_args.argptr[2] = unsafe.Pointer(&fill1DWithCoords_args.arg_N)
}

// Wrapper for fill1DWithCoords CUDA kernel, asynchronous.
func k_fill1DWithCoords_async(dst unsafe.Pointer, factor float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("fill1DWithCoords")
	}

	fill1DWithCoords_args.Lock()
	defer fill1DWithCoords_args.Unlock()

	if fill1DWithCoords_code == 0 {
		fill1DWithCoords_code = fatbinLoad(fill1DWithCoords_map, "fill1DWithCoords")
	}

	fill1DWithCoords_args.arg_dst = dst
	fill1DWithCoords_args.arg_factor = factor
	fill1DWithCoords_args.arg_N = N

	args := fill1DWithCoords_args.argptr[:]
	cu.LaunchKernel(fill1DWithCoords_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("fill1DWithCoords")
	}
}

// maps compute capability on PTX code for fill1DWithCoords kernel.
var fill1DWithCoords_map = map[int]string{0: "",
	50: fill1DWithCoords_ptx_50,
	52: fill1DWithCoords_ptx_52,
	53: fill1DWithCoords_ptx_53,
	60: fill1DWithCoords_ptx_60,
	61: fill1DWithCoords_ptx_61,
	62: fill1DWithCoords_ptx_62,
	70: fill1DWithCoords_ptx_70,
	72: fill1DWithCoords_ptx_72,
	75: fill1DWithCoords_ptx_75,
	80: fill1DWithCoords_ptx_80}

// fill1DWithCoords PTX code for various compute capabilities.
const (
	fill1DWithCoords_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
	fill1DWithCoords_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	fill1DWithCoords

.visible .entry fill1DWithCoords(
	.param .u64 fill1DWithCoords_param_0,
	.param .f32 fill1DWithCoords_param_1,
	.param .u32 fill1DWithCoords_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [fill1DWithCoords_param_0];
	ld.param.f32 	%f1, [fill1DWithCoords_param_1];
	ld.param.u32 	%r2, [fill1DWithCoords_param_2];
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
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f2, %f1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f3;

$L__BB0_2:
	ret;

}

`
)
