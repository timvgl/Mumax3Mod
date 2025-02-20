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

// CUDA handle for extractZSlice kernel
var extractZSlice_code cu.Function

// Stores the arguments for extractZSlice kernel invocation
type extractZSlice_args_t struct {
	arg_output unsafe.Pointer
	arg_input  unsafe.Pointer
	arg_X      int
	arg_Y      int
	arg_Z      int
	arg_x      int
	arg_y      int
	argptr     [7]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for extractZSlice kernel invocation
var extractZSlice_args extractZSlice_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	extractZSlice_args.argptr[0] = unsafe.Pointer(&extractZSlice_args.arg_output)
	extractZSlice_args.argptr[1] = unsafe.Pointer(&extractZSlice_args.arg_input)
	extractZSlice_args.argptr[2] = unsafe.Pointer(&extractZSlice_args.arg_X)
	extractZSlice_args.argptr[3] = unsafe.Pointer(&extractZSlice_args.arg_Y)
	extractZSlice_args.argptr[4] = unsafe.Pointer(&extractZSlice_args.arg_Z)
	extractZSlice_args.argptr[5] = unsafe.Pointer(&extractZSlice_args.arg_x)
	extractZSlice_args.argptr[6] = unsafe.Pointer(&extractZSlice_args.arg_y)
}

// Wrapper for extractZSlice CUDA kernel, asynchronous.
func k_extractZSlice_async(output unsafe.Pointer, input unsafe.Pointer, X int, Y int, Z int, x int, y int, key string, cfg *config) {
	if Synchronous { // debug
		SyncFFT_T(key)
		timer.Start("extractZSlice" + key)
	}

	extractZSlice_args.Lock()
	defer extractZSlice_args.Unlock()

	if extractZSlice_code == 0 {
		extractZSlice_code = fatbinLoad(extractZSlice_map, "extractZSlice")
	}

	extractZSlice_args.arg_output = output
	extractZSlice_args.arg_input = input
	extractZSlice_args.arg_X = X
	extractZSlice_args.arg_Y = Y
	extractZSlice_args.arg_Z = Z
	extractZSlice_args.arg_x = x
	extractZSlice_args.arg_y = y

	args := extractZSlice_args.argptr[:]
	cu.LaunchKernel(extractZSlice_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, Get_Stream(key), args)

	if Synchronous { // debug
		SyncFFT_T(key)
		timer.Stop("extractZSlice" + key)
	}
}

// maps compute capability on PTX code for extractZSlice kernel.
var extractZSlice_map = map[int]string{0: "",
	50: extractZSlice_ptx_50,
	52: extractZSlice_ptx_52,
	53: extractZSlice_ptx_53,
	60: extractZSlice_ptx_60,
	61: extractZSlice_ptx_61,
	62: extractZSlice_ptx_62,
	70: extractZSlice_ptx_70,
	72: extractZSlice_ptx_72,
	75: extractZSlice_ptx_75,
	80: extractZSlice_ptx_80}

// extractZSlice PTX code for various compute capabilities.
const (
	extractZSlice_ptx_50 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_50
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_52 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_52
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_53 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_53
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_60 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_60
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_61 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_61
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_62 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_62
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_70 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_70
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_72 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_72
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_75 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_75
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
	extractZSlice_ptx_80 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_80
.address_size 64

	// .globl	extractZSlice

.visible .entry extractZSlice(
	.param .u64 extractZSlice_param_0,
	.param .u64 extractZSlice_param_1,
	.param .u32 extractZSlice_param_2,
	.param .u32 extractZSlice_param_3,
	.param .u32 extractZSlice_param_4,
	.param .u32 extractZSlice_param_5,
	.param .u32 extractZSlice_param_6
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [extractZSlice_param_0];
	ld.param.u64 	%rd2, [extractZSlice_param_1];
	ld.param.u32 	%r2, [extractZSlice_param_3];
	ld.param.u32 	%r3, [extractZSlice_param_4];
	ld.param.u32 	%r4, [extractZSlice_param_5];
	ld.param.u32 	%r5, [extractZSlice_param_6];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	mad.lo.s32 	%r9, %r4, %r2, %r5;
	mad.lo.s32 	%r10, %r9, %r3, %r1;
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
