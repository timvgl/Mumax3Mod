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

// CUDA handle for limitToGeometry kernel
var limitToGeometry_code cu.Function

// Stores the arguments for limitToGeometry kernel invocation
type limitToGeometry_args_t struct {
	arg_vx  unsafe.Pointer
	arg_vy  unsafe.Pointer
	arg_vz  unsafe.Pointer
	arg_vol unsafe.Pointer
	arg_N   int
	argptr  [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for limitToGeometry kernel invocation
var limitToGeometry_args limitToGeometry_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	limitToGeometry_args.argptr[0] = unsafe.Pointer(&limitToGeometry_args.arg_vx)
	limitToGeometry_args.argptr[1] = unsafe.Pointer(&limitToGeometry_args.arg_vy)
	limitToGeometry_args.argptr[2] = unsafe.Pointer(&limitToGeometry_args.arg_vz)
	limitToGeometry_args.argptr[3] = unsafe.Pointer(&limitToGeometry_args.arg_vol)
	limitToGeometry_args.argptr[4] = unsafe.Pointer(&limitToGeometry_args.arg_N)
}

// Wrapper for limitToGeometry CUDA kernel, asynchronous.
func k_limitToGeometry_async(vx unsafe.Pointer, vy unsafe.Pointer, vz unsafe.Pointer, vol unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("limitToGeometry")
	}

	limitToGeometry_args.Lock()
	defer limitToGeometry_args.Unlock()

	if limitToGeometry_code == 0 {
		limitToGeometry_code = fatbinLoad(limitToGeometry_map, "limitToGeometry")
	}

	limitToGeometry_args.arg_vx = vx
	limitToGeometry_args.arg_vy = vy
	limitToGeometry_args.arg_vz = vz
	limitToGeometry_args.arg_vol = vol
	limitToGeometry_args.arg_N = N

	args := limitToGeometry_args.argptr[:]
	cu.LaunchKernel(limitToGeometry_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("limitToGeometry")
	}
}

// maps compute capability on PTX code for limitToGeometry kernel.
var limitToGeometry_map = map[int]string{0: "",
	50: limitToGeometry_ptx_50,
	52: limitToGeometry_ptx_52,
	53: limitToGeometry_ptx_53,
	60: limitToGeometry_ptx_60,
	61: limitToGeometry_ptx_61,
	62: limitToGeometry_ptx_62,
	70: limitToGeometry_ptx_70,
	72: limitToGeometry_ptx_72,
	75: limitToGeometry_ptx_75,
	80: limitToGeometry_ptx_80}

// limitToGeometry PTX code for various compute capabilities.
const (
	limitToGeometry_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
	limitToGeometry_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	limitToGeometry

.visible .entry limitToGeometry(
	.param .u64 limitToGeometry_param_0,
	.param .u64 limitToGeometry_param_1,
	.param .u64 limitToGeometry_param_2,
	.param .u64 limitToGeometry_param_3,
	.param .u32 limitToGeometry_param_4
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<11>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd1, [limitToGeometry_param_0];
	ld.param.u64 	%rd2, [limitToGeometry_param_1];
	ld.param.u64 	%rd3, [limitToGeometry_param_2];
	ld.param.u64 	%rd4, [limitToGeometry_param_3];
	ld.param.u32 	%r2, [limitToGeometry_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_5;

	setp.eq.s64 	%p2, %rd4, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd5, %rd4;
	mul.wide.s32 	%rd6, %r1, 4;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.nc.f32 	%f10, [%rd7];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f10, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd8, %rd1;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f5, %f10, %f4;
	cvta.to.global.u64 	%rd11, %rd2;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.f32 	%f6, [%rd12];
	mul.f32 	%f7, %f10, %f6;
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.f32 	%f8, [%rd14];
	mul.f32 	%f9, %f10, %f8;
	st.global.f32 	[%rd10], %f5;
	st.global.f32 	[%rd12], %f7;
	st.global.f32 	[%rd14], %f9;

$L__BB0_5:
	ret;

}

`
)
