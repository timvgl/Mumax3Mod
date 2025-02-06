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

// CUDA handle for normalize kernel
var normalize_code cu.Function

// Stores the arguments for normalize kernel invocation
type normalize_args_t struct {
	arg_vx  unsafe.Pointer
	arg_vy  unsafe.Pointer
	arg_vz  unsafe.Pointer
	arg_vol unsafe.Pointer
	arg_N   int
	argptr  [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for normalize kernel invocation
var normalize_args normalize_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	normalize_args.argptr[0] = unsafe.Pointer(&normalize_args.arg_vx)
	normalize_args.argptr[1] = unsafe.Pointer(&normalize_args.arg_vy)
	normalize_args.argptr[2] = unsafe.Pointer(&normalize_args.arg_vz)
	normalize_args.argptr[3] = unsafe.Pointer(&normalize_args.arg_vol)
	normalize_args.argptr[4] = unsafe.Pointer(&normalize_args.arg_N)
}

// Wrapper for normalize CUDA kernel, asynchronous.
func k_normalize_async(vx unsafe.Pointer, vy unsafe.Pointer, vz unsafe.Pointer, vol unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("normalize")
	}

	normalize_args.Lock()
	defer normalize_args.Unlock()

	if normalize_code == 0 {
		normalize_code = fatbinLoad(normalize_map, "normalize")
	}

	normalize_args.arg_vx = vx
	normalize_args.arg_vy = vy
	normalize_args.arg_vz = vz
	normalize_args.arg_vol = vol
	normalize_args.arg_N = N

	args := normalize_args.argptr[:]
	cu.LaunchKernel(normalize_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("normalize")
	}
}

// maps compute capability on PTX code for normalize kernel.
var normalize_map = map[int]string{0: "",
	53: normalize_ptx_53,
	60: normalize_ptx_60,
	61: normalize_ptx_61,
	62: normalize_ptx_62,
	70: normalize_ptx_70,
	72: normalize_ptx_72,
	75: normalize_ptx_75,
	80: normalize_ptx_80}

// normalize PTX code for various compute capabilities.
const (
	normalize_ptx_53 = `
.version 8.4
.target sm_53
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_60 = `
.version 8.4
.target sm_60
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_61 = `
.version 8.4
.target sm_61
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_62 = `
.version 8.4
.target sm_62
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_70 = `
.version 8.4
.target sm_70
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_72 = `
.version 8.4
.target sm_72
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_75 = `
.version 8.4
.target sm_75
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
	normalize_ptx_80 = `
.version 8.4
.target sm_80
.address_size 64

	// .globl	normalize

.visible .entry normalize(
	.param .u64 normalize_param_0,
	.param .u64 normalize_param_1,
	.param .u64 normalize_param_2,
	.param .u64 normalize_param_3,
	.param .u32 normalize_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<15>;


	ld.param.u64 	%rd4, [normalize_param_0];
	ld.param.u64 	%rd5, [normalize_param_1];
	ld.param.u64 	%rd6, [normalize_param_2];
	ld.param.u64 	%rd7, [normalize_param_3];
	ld.param.u32 	%r2, [normalize_param_4];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_7;

	setp.eq.s64 	%p2, %rd7, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f20, [%rd10];
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	mov.f32 	%f20, 0f3F800000;

$L__BB0_4:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd1, %rd11, %rd12;
	ld.global.f32 	%f11, [%rd1];
	mul.f32 	%f3, %f20, %f11;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd2, %rd13, %rd12;
	ld.global.f32 	%f12, [%rd2];
	mul.f32 	%f4, %f20, %f12;
	cvta.to.global.u64 	%rd14, %rd6;
	add.s64 	%rd3, %rd14, %rd12;
	ld.global.f32 	%f13, [%rd3];
	mul.f32 	%f5, %f20, %f13;
	mul.f32 	%f14, %f4, %f4;
	fma.rn.f32 	%f15, %f3, %f3, %f14;
	fma.rn.f32 	%f16, %f5, %f5, %f15;
	sqrt.rn.f32 	%f6, %f16;
	setp.eq.f32 	%p3, %f6, 0f00000000;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	rcp.rn.f32 	%f21, %f6;

$L__BB0_6:
	mul.f32 	%f17, %f3, %f21;
	st.global.f32 	[%rd1], %f17;
	mul.f32 	%f18, %f4, %f21;
	st.global.f32 	[%rd2], %f18;
	mul.f32 	%f19, %f5, %f21;
	st.global.f32 	[%rd3], %f19;

$L__BB0_7:
	ret;

}

`
)
