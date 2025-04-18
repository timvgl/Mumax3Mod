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

// CUDA handle for truncGovaluate kernel
var truncGovaluate_code cu.Function

// Stores the arguments for truncGovaluate kernel invocation
type truncGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for truncGovaluate kernel invocation
var truncGovaluate_args truncGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	truncGovaluate_args.argptr[0] = unsafe.Pointer(&truncGovaluate_args.arg_value)
	truncGovaluate_args.argptr[1] = unsafe.Pointer(&truncGovaluate_args.arg_N)
}

// Wrapper for truncGovaluate CUDA kernel, asynchronous.
func k_truncGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("truncGovaluate")
	}

	truncGovaluate_args.Lock()
	defer truncGovaluate_args.Unlock()

	if truncGovaluate_code == 0 {
		truncGovaluate_code = fatbinLoad(truncGovaluate_map, "truncGovaluate")
	}

	truncGovaluate_args.arg_value = value
	truncGovaluate_args.arg_N = N

	args := truncGovaluate_args.argptr[:]
	cu.LaunchKernel(truncGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("truncGovaluate")
	}
}

// maps compute capability on PTX code for truncGovaluate kernel.
var truncGovaluate_map = map[int]string{0: "",
	50: truncGovaluate_ptx_50,
	52: truncGovaluate_ptx_52,
	53: truncGovaluate_ptx_53,
	60: truncGovaluate_ptx_60,
	61: truncGovaluate_ptx_61,
	62: truncGovaluate_ptx_62,
	70: truncGovaluate_ptx_70,
	72: truncGovaluate_ptx_72,
	75: truncGovaluate_ptx_75,
	80: truncGovaluate_ptx_80}

// truncGovaluate PTX code for various compute capabilities.
const (
	truncGovaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
	truncGovaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	truncGovaluate

.visible .entry truncGovaluate(
	.param .u64 truncGovaluate_param_0,
	.param .u32 truncGovaluate_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [truncGovaluate_param_0];
	ld.param.u32 	%r2, [truncGovaluate_param_1];
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
	cvt.rzi.f32.f32 	%f2, %f1;
	st.global.f32 	[%rd4], %f2;

$L__BB0_2:
	ret;

}

`
)
