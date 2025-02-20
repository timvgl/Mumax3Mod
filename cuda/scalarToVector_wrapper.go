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

// CUDA handle for scalarToVector kernel
var scalarToVector_code cu.Function

// Stores the arguments for scalarToVector kernel invocation
type scalarToVector_args_t struct {
	arg_dstx  unsafe.Pointer
	arg_dsty  unsafe.Pointer
	arg_dstz  unsafe.Pointer
	arg_a_    unsafe.Pointer
	arg_a_mul float32
	arg_b_    unsafe.Pointer
	arg_b_mul float32
	arg_c_    unsafe.Pointer
	arg_c_mul float32
	arg_Nx    int
	arg_Ny    int
	arg_Nz    int
	arg_PBC   byte
	argptr    [13]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for scalarToVector kernel invocation
var scalarToVector_args scalarToVector_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	scalarToVector_args.argptr[0] = unsafe.Pointer(&scalarToVector_args.arg_dstx)
	scalarToVector_args.argptr[1] = unsafe.Pointer(&scalarToVector_args.arg_dsty)
	scalarToVector_args.argptr[2] = unsafe.Pointer(&scalarToVector_args.arg_dstz)
	scalarToVector_args.argptr[3] = unsafe.Pointer(&scalarToVector_args.arg_a_)
	scalarToVector_args.argptr[4] = unsafe.Pointer(&scalarToVector_args.arg_a_mul)
	scalarToVector_args.argptr[5] = unsafe.Pointer(&scalarToVector_args.arg_b_)
	scalarToVector_args.argptr[6] = unsafe.Pointer(&scalarToVector_args.arg_b_mul)
	scalarToVector_args.argptr[7] = unsafe.Pointer(&scalarToVector_args.arg_c_)
	scalarToVector_args.argptr[8] = unsafe.Pointer(&scalarToVector_args.arg_c_mul)
	scalarToVector_args.argptr[9] = unsafe.Pointer(&scalarToVector_args.arg_Nx)
	scalarToVector_args.argptr[10] = unsafe.Pointer(&scalarToVector_args.arg_Ny)
	scalarToVector_args.argptr[11] = unsafe.Pointer(&scalarToVector_args.arg_Nz)
	scalarToVector_args.argptr[12] = unsafe.Pointer(&scalarToVector_args.arg_PBC)
}

// Wrapper for scalarToVector CUDA kernel, asynchronous.
func k_scalarToVector_async(dstx unsafe.Pointer, dsty unsafe.Pointer, dstz unsafe.Pointer, a_ unsafe.Pointer, a_mul float32, b_ unsafe.Pointer, b_mul float32, c_ unsafe.Pointer, c_mul float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("scalarToVector")
	}

	scalarToVector_args.Lock()
	defer scalarToVector_args.Unlock()

	if scalarToVector_code == 0 {
		scalarToVector_code = fatbinLoad(scalarToVector_map, "scalarToVector")
	}

	scalarToVector_args.arg_dstx = dstx
	scalarToVector_args.arg_dsty = dsty
	scalarToVector_args.arg_dstz = dstz
	scalarToVector_args.arg_a_ = a_
	scalarToVector_args.arg_a_mul = a_mul
	scalarToVector_args.arg_b_ = b_
	scalarToVector_args.arg_b_mul = b_mul
	scalarToVector_args.arg_c_ = c_
	scalarToVector_args.arg_c_mul = c_mul
	scalarToVector_args.arg_Nx = Nx
	scalarToVector_args.arg_Ny = Ny
	scalarToVector_args.arg_Nz = Nz
	scalarToVector_args.arg_PBC = PBC

	args := scalarToVector_args.argptr[:]
	cu.LaunchKernel(scalarToVector_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("scalarToVector")
	}
}

// maps compute capability on PTX code for scalarToVector kernel.
var scalarToVector_map = map[int]string{0: "",
	50: scalarToVector_ptx_50,
	52: scalarToVector_ptx_52,
	53: scalarToVector_ptx_53,
	60: scalarToVector_ptx_60,
	61: scalarToVector_ptx_61,
	62: scalarToVector_ptx_62,
	70: scalarToVector_ptx_70,
	72: scalarToVector_ptx_72,
	75: scalarToVector_ptx_75,
	80: scalarToVector_ptx_80}

// scalarToVector PTX code for various compute capabilities.
const (
	scalarToVector_ptx_50 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_50
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_52 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_52
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_53 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_53
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_60 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_60
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_61 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_61
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_62 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_62
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_70 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_70
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_72 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_72
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_75 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_75
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
	scalarToVector_ptx_80 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_80
.address_size 64

	// .globl	scalarToVector

.visible .entry scalarToVector(
	.param .u64 scalarToVector_param_0,
	.param .u64 scalarToVector_param_1,
	.param .u64 scalarToVector_param_2,
	.param .u64 scalarToVector_param_3,
	.param .f32 scalarToVector_param_4,
	.param .u64 scalarToVector_param_5,
	.param .f32 scalarToVector_param_6,
	.param .u64 scalarToVector_param_7,
	.param .f32 scalarToVector_param_8,
	.param .u32 scalarToVector_param_9,
	.param .u32 scalarToVector_param_10,
	.param .u32 scalarToVector_param_11,
	.param .u8 scalarToVector_param_12
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<16>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<26>;


	ld.param.u64 	%rd2, [scalarToVector_param_0];
	ld.param.u64 	%rd3, [scalarToVector_param_1];
	ld.param.u64 	%rd4, [scalarToVector_param_2];
	ld.param.u64 	%rd5, [scalarToVector_param_3];
	ld.param.f32 	%f13, [scalarToVector_param_4];
	ld.param.u64 	%rd6, [scalarToVector_param_5];
	ld.param.f32 	%f14, [scalarToVector_param_6];
	ld.param.u64 	%rd7, [scalarToVector_param_7];
	ld.param.f32 	%f15, [scalarToVector_param_8];
	ld.param.u32 	%r5, [scalarToVector_param_9];
	ld.param.u32 	%r6, [scalarToVector_param_10];
	ld.param.u32 	%r7, [scalarToVector_param_11];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_8;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd5, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd5;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f10, [%rd10];
	mul.f32 	%f13, %f10, %f13;

$L__BB0_3:
	cvt.s64.s32 	%rd1, %r4;
	cvta.to.global.u64 	%rd11, %rd2;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	st.global.f32 	[%rd13], %f13;
	setp.eq.s64 	%p7, %rd6, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd14, %rd6;
	shl.b64 	%rd15, %rd1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.f32 	%f11, [%rd16];
	mul.f32 	%f14, %f11, %f14;

$L__BB0_5:
	cvta.to.global.u64 	%rd17, %rd3;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	st.global.f32 	[%rd19], %f14;
	setp.eq.s64 	%p8, %rd7, 0;
	@%p8 bra 	$L__BB0_7;

	cvta.to.global.u64 	%rd20, %rd7;
	add.s64 	%rd22, %rd20, %rd18;
	ld.global.nc.f32 	%f12, [%rd22];
	mul.f32 	%f15, %f12, %f15;

$L__BB0_7:
	cvta.to.global.u64 	%rd23, %rd4;
	add.s64 	%rd25, %rd23, %rd18;
	st.global.f32 	[%rd25], %f15;

$L__BB0_8:
	ret;

}

`
)
