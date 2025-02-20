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

// CUDA handle for divGovaluate1X3 kernel
var divGovaluate1X3_code cu.Function

// Stores the arguments for divGovaluate1X3 kernel invocation
type divGovaluate1X3_args_t struct {
	arg_output unsafe.Pointer
	arg_input2 float32
	arg_input  unsafe.Pointer
	arg_N      int
	argptr     [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for divGovaluate1X3 kernel invocation
var divGovaluate1X3_args divGovaluate1X3_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	divGovaluate1X3_args.argptr[0] = unsafe.Pointer(&divGovaluate1X3_args.arg_output)
	divGovaluate1X3_args.argptr[1] = unsafe.Pointer(&divGovaluate1X3_args.arg_input2)
	divGovaluate1X3_args.argptr[2] = unsafe.Pointer(&divGovaluate1X3_args.arg_input)
	divGovaluate1X3_args.argptr[3] = unsafe.Pointer(&divGovaluate1X3_args.arg_N)
}

// Wrapper for divGovaluate1X3 CUDA kernel, asynchronous.
func k_divGovaluate1X3_async(output unsafe.Pointer, input2 float32, input unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("divGovaluate1X3")
	}

	divGovaluate1X3_args.Lock()
	defer divGovaluate1X3_args.Unlock()

	if divGovaluate1X3_code == 0 {
		divGovaluate1X3_code = fatbinLoad(divGovaluate1X3_map, "divGovaluate1X3")
	}

	divGovaluate1X3_args.arg_output = output
	divGovaluate1X3_args.arg_input2 = input2
	divGovaluate1X3_args.arg_input = input
	divGovaluate1X3_args.arg_N = N

	args := divGovaluate1X3_args.argptr[:]
	cu.LaunchKernel(divGovaluate1X3_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("divGovaluate1X3")
	}
}

// maps compute capability on PTX code for divGovaluate1X3 kernel.
var divGovaluate1X3_map = map[int]string{0: "",
	50: divGovaluate1X3_ptx_50,
	52: divGovaluate1X3_ptx_52,
	53: divGovaluate1X3_ptx_53,
	60: divGovaluate1X3_ptx_60,
	61: divGovaluate1X3_ptx_61,
	62: divGovaluate1X3_ptx_62,
	70: divGovaluate1X3_ptx_70,
	72: divGovaluate1X3_ptx_72,
	75: divGovaluate1X3_ptx_75,
	80: divGovaluate1X3_ptx_80}

// divGovaluate1X3 PTX code for various compute capabilities.
const (
	divGovaluate1X3_ptx_50 = `
.version 8.4
.target sm_50
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_52 = `
.version 8.4
.target sm_52
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_53 = `
.version 8.4
.target sm_53
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_60 = `
.version 8.4
.target sm_60
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_61 = `
.version 8.4
.target sm_61
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_62 = `
.version 8.4
.target sm_62
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_70 = `
.version 8.4
.target sm_70
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_72 = `
.version 8.4
.target sm_72
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_75 = `
.version 8.4
.target sm_75
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate1X3_ptx_80 = `
.version 8.4
.target sm_80
.address_size 64

	// .globl	divGovaluate1X3

.visible .entry divGovaluate1X3(
	.param .u64 divGovaluate1X3_param_0,
	.param .f32 divGovaluate1X3_param_1,
	.param .u64 divGovaluate1X3_param_2,
	.param .u32 divGovaluate1X3_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [divGovaluate1X3_param_0];
	ld.param.f32 	%f1, [divGovaluate1X3_param_1];
	ld.param.u64 	%rd2, [divGovaluate1X3_param_2];
	ld.param.u32 	%r2, [divGovaluate1X3_param_3];
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
	div.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}

`
)
