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

// CUDA handle for extractYSlice kernel
var extractYSlice_code cu.Function

// Stores the arguments for extractYSlice kernel invocation
type extractYSlice_args_t struct {
	arg_output unsafe.Pointer
	arg_input  unsafe.Pointer
	arg_X      int
	arg_Y      int
	arg_Z      int
	arg_x      int
	arg_z      int
	argptr     [7]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for extractYSlice kernel invocation
var extractYSlice_args extractYSlice_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	extractYSlice_args.argptr[0] = unsafe.Pointer(&extractYSlice_args.arg_output)
	extractYSlice_args.argptr[1] = unsafe.Pointer(&extractYSlice_args.arg_input)
	extractYSlice_args.argptr[2] = unsafe.Pointer(&extractYSlice_args.arg_X)
	extractYSlice_args.argptr[3] = unsafe.Pointer(&extractYSlice_args.arg_Y)
	extractYSlice_args.argptr[4] = unsafe.Pointer(&extractYSlice_args.arg_Z)
	extractYSlice_args.argptr[5] = unsafe.Pointer(&extractYSlice_args.arg_x)
	extractYSlice_args.argptr[6] = unsafe.Pointer(&extractYSlice_args.arg_z)
}

// Wrapper for extractYSlice CUDA kernel, asynchronous.
func k_extractYSlice_async(output unsafe.Pointer, input unsafe.Pointer, X int, Y int, Z int, x int, z int, key string, cfg *config) {
	if Synchronous { // debug
		SyncFFT_T(key)
		timer.Start("extractYSlice" + key)
	}

	extractYSlice_args.Lock()
	defer extractYSlice_args.Unlock()

	if extractYSlice_code == 0 {
		extractYSlice_code = fatbinLoad(extractYSlice_map, "extractYSlice")
	}

	extractYSlice_args.arg_output = output
	extractYSlice_args.arg_input = input
	extractYSlice_args.arg_X = X
	extractYSlice_args.arg_Y = Y
	extractYSlice_args.arg_Z = Z
	extractYSlice_args.arg_x = x
	extractYSlice_args.arg_z = z

	args := extractYSlice_args.argptr[:]
	cu.LaunchKernel(extractYSlice_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, Get_Stream(key), args)

	if Synchronous { // debug
		SyncFFT_T(key)
		timer.Stop("extractYSlice" + key)
	}
}

// maps compute capability on PTX code for extractYSlice kernel.
var extractYSlice_map = map[int]string{0: "",
	50: extractYSlice_ptx_50,
	52: extractYSlice_ptx_52,
	53: extractYSlice_ptx_53,
	60: extractYSlice_ptx_60,
	61: extractYSlice_ptx_61,
	62: extractYSlice_ptx_62,
	70: extractYSlice_ptx_70,
	72: extractYSlice_ptx_72,
	75: extractYSlice_ptx_75,
	80: extractYSlice_ptx_80}

// extractYSlice PTX code for various compute capabilities.
const (
	extractYSlice_ptx_50 = `
.version 8.4
.target sm_50
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_52 = `
.version 8.4
.target sm_52
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_53 = `
.version 8.4
.target sm_53
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_60 = `
.version 8.4
.target sm_60
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_61 = `
.version 8.4
.target sm_61
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_62 = `
.version 8.4
.target sm_62
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_70 = `
.version 8.4
.target sm_70
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_72 = `
.version 8.4
.target sm_72
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_75 = `
.version 8.4
.target sm_75
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
	extractYSlice_ptx_80 = `
.version 8.4
.target sm_80
.address_size 64

	// .globl	extractYSlice

.visible .entry extractYSlice(
	.param .u64 extractYSlice_param_0,
	.param .u64 extractYSlice_param_1,
	.param .u32 extractYSlice_param_2,
	.param .u32 extractYSlice_param_3,
	.param .u32 extractYSlice_param_4,
	.param .u32 extractYSlice_param_5,
	.param .u32 extractYSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractYSlice_param_0];
	ld.param.u64 	%rd2, [extractYSlice_param_1];
	ld.param.u32 	%r2, [extractYSlice_param_3];
	ld.param.u32 	%r3, [extractYSlice_param_4];
	ld.param.u32 	%r4, [extractYSlice_param_5];
	ld.param.u32 	%r5, [extractYSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r1;
	mad.lo.s32 	%r10, %r9, %r3, %r5;
	mul.wide.s32 	%rd4, %r10, 4;
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
