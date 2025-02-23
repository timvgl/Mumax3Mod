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

// CUDA handle for mulGovaluate3X1 kernel
var mulGovaluate3X1_code cu.Function

// Stores the arguments for mulGovaluate3X1 kernel invocation
type mulGovaluate3X1_args_t struct {
	arg_output unsafe.Pointer
	arg_input  unsafe.Pointer
	arg_input2 float32
	arg_N      int
	argptr     [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for mulGovaluate3X1 kernel invocation
var mulGovaluate3X1_args mulGovaluate3X1_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	mulGovaluate3X1_args.argptr[0] = unsafe.Pointer(&mulGovaluate3X1_args.arg_output)
	mulGovaluate3X1_args.argptr[1] = unsafe.Pointer(&mulGovaluate3X1_args.arg_input)
	mulGovaluate3X1_args.argptr[2] = unsafe.Pointer(&mulGovaluate3X1_args.arg_input2)
	mulGovaluate3X1_args.argptr[3] = unsafe.Pointer(&mulGovaluate3X1_args.arg_N)
}

// Wrapper for mulGovaluate3X1 CUDA kernel, asynchronous.
func k_mulGovaluate3X1_async(output unsafe.Pointer, input unsafe.Pointer, input2 float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("mulGovaluate3X1")
	}

	mulGovaluate3X1_args.Lock()
	defer mulGovaluate3X1_args.Unlock()

	if mulGovaluate3X1_code == 0 {
		mulGovaluate3X1_code = fatbinLoad(mulGovaluate3X1_map, "mulGovaluate3X1")
	}

	mulGovaluate3X1_args.arg_output = output
	mulGovaluate3X1_args.arg_input = input
	mulGovaluate3X1_args.arg_input2 = input2
	mulGovaluate3X1_args.arg_N = N

	args := mulGovaluate3X1_args.argptr[:]
	cu.LaunchKernel(mulGovaluate3X1_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("mulGovaluate3X1")
	}
}

// maps compute capability on PTX code for mulGovaluate3X1 kernel.
var mulGovaluate3X1_map = map[int]string{0: "",
	50: mulGovaluate3X1_ptx_50,
	52: mulGovaluate3X1_ptx_52,
	53: mulGovaluate3X1_ptx_53,
	60: mulGovaluate3X1_ptx_60,
	61: mulGovaluate3X1_ptx_61,
	62: mulGovaluate3X1_ptx_62,
	70: mulGovaluate3X1_ptx_70,
	72: mulGovaluate3X1_ptx_72,
	75: mulGovaluate3X1_ptx_75,
	80: mulGovaluate3X1_ptx_80}

// mulGovaluate3X1 PTX code for various compute capabilities.
const (
	mulGovaluate3X1_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	mulGovaluate3X1_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	mulGovaluate3X1

.visible .entry mulGovaluate3X1(
	.param .u64 mulGovaluate3X1_param_0,
	.param .u64 mulGovaluate3X1_param_1,
	.param .f32 mulGovaluate3X1_param_2,
	.param .u32 mulGovaluate3X1_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [mulGovaluate3X1_param_0];
	ld.param.u64 	%rd2, [mulGovaluate3X1_param_1];
	ld.param.f32 	%f1, [mulGovaluate3X1_param_2];
	ld.param.u32 	%r2, [mulGovaluate3X1_param_3];
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
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f2, [%rd5];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
)
