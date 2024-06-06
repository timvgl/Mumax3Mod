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

// CUDA handle for lltorque2 kernel
var lltorque2_code cu.Function

// Stores the arguments for lltorque2 kernel invocation
type lltorque2_args_t struct {
	arg_tx        unsafe.Pointer
	arg_ty        unsafe.Pointer
	arg_tz        unsafe.Pointer
	arg_mx        unsafe.Pointer
	arg_my        unsafe.Pointer
	arg_mz        unsafe.Pointer
	arg_hx        unsafe.Pointer
	arg_hy        unsafe.Pointer
	arg_hz        unsafe.Pointer
	arg_alpha_    unsafe.Pointer
	arg_alpha_mul float32
	arg_N         int
	argptr        [12]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for lltorque2 kernel invocation
var lltorque2_args lltorque2_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	lltorque2_args.argptr[0] = unsafe.Pointer(&lltorque2_args.arg_tx)
	lltorque2_args.argptr[1] = unsafe.Pointer(&lltorque2_args.arg_ty)
	lltorque2_args.argptr[2] = unsafe.Pointer(&lltorque2_args.arg_tz)
	lltorque2_args.argptr[3] = unsafe.Pointer(&lltorque2_args.arg_mx)
	lltorque2_args.argptr[4] = unsafe.Pointer(&lltorque2_args.arg_my)
	lltorque2_args.argptr[5] = unsafe.Pointer(&lltorque2_args.arg_mz)
	lltorque2_args.argptr[6] = unsafe.Pointer(&lltorque2_args.arg_hx)
	lltorque2_args.argptr[7] = unsafe.Pointer(&lltorque2_args.arg_hy)
	lltorque2_args.argptr[8] = unsafe.Pointer(&lltorque2_args.arg_hz)
	lltorque2_args.argptr[9] = unsafe.Pointer(&lltorque2_args.arg_alpha_)
	lltorque2_args.argptr[10] = unsafe.Pointer(&lltorque2_args.arg_alpha_mul)
	lltorque2_args.argptr[11] = unsafe.Pointer(&lltorque2_args.arg_N)
}

// Wrapper for lltorque2 CUDA kernel, asynchronous.
func k_lltorque2_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, hx unsafe.Pointer, hy unsafe.Pointer, hz unsafe.Pointer, alpha_ unsafe.Pointer, alpha_mul float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("lltorque2")
	}

	lltorque2_args.Lock()
	defer lltorque2_args.Unlock()

	if lltorque2_code == 0 {
		lltorque2_code = fatbinLoad(lltorque2_map, "lltorque2")
	}

	lltorque2_args.arg_tx = tx
	lltorque2_args.arg_ty = ty
	lltorque2_args.arg_tz = tz
	lltorque2_args.arg_mx = mx
	lltorque2_args.arg_my = my
	lltorque2_args.arg_mz = mz
	lltorque2_args.arg_hx = hx
	lltorque2_args.arg_hy = hy
	lltorque2_args.arg_hz = hz
	lltorque2_args.arg_alpha_ = alpha_
	lltorque2_args.arg_alpha_mul = alpha_mul
	lltorque2_args.arg_N = N

	args := lltorque2_args.argptr[:]
	cu.LaunchKernel(lltorque2_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("lltorque2")
	}
}

// maps compute capability on PTX code for lltorque2 kernel.
var lltorque2_map = map[int]string{0: "",
	35: lltorque2_ptx_35,
	37: lltorque2_ptx_37,
	50: lltorque2_ptx_50,
	52: lltorque2_ptx_52,
	53: lltorque2_ptx_53,
	60: lltorque2_ptx_60,
	61: lltorque2_ptx_61,
	62: lltorque2_ptx_62,
	70: lltorque2_ptx_70,
	80: lltorque2_ptx_80}

// lltorque2 PTX code for various compute capabilities.
const (
	lltorque2_ptx_35 = `
.version 7.4
.target sm_35
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_37 = `
.version 7.4
.target sm_37
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_50 = `
.version 7.4
.target sm_50
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_52 = `
.version 7.4
.target sm_52
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_53 = `
.version 7.4
.target sm_53
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_60 = `
.version 7.4
.target sm_60
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_61 = `
.version 7.4
.target sm_61
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_62 = `
.version 7.4
.target sm_62
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_70 = `
.version 7.4
.target sm_70
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
	lltorque2_ptx_80 = `
.version 7.4
.target sm_80
.address_size 64

	// .globl	lltorque2

.visible .entry lltorque2(
	.param .u64 lltorque2_param_0,
	.param .u64 lltorque2_param_1,
	.param .u64 lltorque2_param_2,
	.param .u64 lltorque2_param_3,
	.param .u64 lltorque2_param_4,
	.param .u64 lltorque2_param_5,
	.param .u64 lltorque2_param_6,
	.param .u64 lltorque2_param_7,
	.param .u64 lltorque2_param_8,
	.param .u64 lltorque2_param_9,
	.param .f32 lltorque2_param_10,
	.param .u32 lltorque2_param_11
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<39>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<34>;


	ld.param.u64 	%rd1, [lltorque2_param_0];
	ld.param.u64 	%rd2, [lltorque2_param_1];
	ld.param.u64 	%rd3, [lltorque2_param_2];
	ld.param.u64 	%rd4, [lltorque2_param_3];
	ld.param.u64 	%rd5, [lltorque2_param_4];
	ld.param.u64 	%rd6, [lltorque2_param_5];
	ld.param.u64 	%rd7, [lltorque2_param_6];
	ld.param.u64 	%rd8, [lltorque2_param_7];
	ld.param.u64 	%rd9, [lltorque2_param_8];
	ld.param.u64 	%rd10, [lltorque2_param_9];
	ld.param.f32 	%f38, [lltorque2_param_10];
	ld.param.u32 	%r2, [lltorque2_param_11];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r1, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	cvta.to.global.u64 	%rd14, %rd5;
	add.s64 	%rd15, %rd14, %rd12;
	ld.global.nc.f32 	%f2, [%rd15];
	cvta.to.global.u64 	%rd16, %rd6;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f3, [%rd17];
	cvta.to.global.u64 	%rd18, %rd7;
	add.s64 	%rd19, %rd18, %rd12;
	ld.global.nc.f32 	%f4, [%rd19];
	cvta.to.global.u64 	%rd20, %rd8;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f5, [%rd21];
	cvta.to.global.u64 	%rd22, %rd9;
	add.s64 	%rd23, %rd22, %rd12;
	ld.global.nc.f32 	%f6, [%rd23];
	setp.eq.s64 	%p2, %rd10, 0;
	@%p2 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd24, %rd10;
	add.s64 	%rd26, %rd24, %rd12;
	ld.global.nc.f32 	%f10, [%rd26];
	mul.f32 	%f38, %f10, %f38;

$L__BB0_3:
	mul.f32 	%f11, %f3, %f5;
	mul.f32 	%f12, %f2, %f6;
	sub.f32 	%f13, %f12, %f11;
	mul.f32 	%f14, %f1, %f6;
	mul.f32 	%f15, %f3, %f4;
	sub.f32 	%f16, %f15, %f14;
	mul.f32 	%f17, %f2, %f4;
	mul.f32 	%f18, %f1, %f5;
	sub.f32 	%f19, %f18, %f17;
	fma.rn.f32 	%f20, %f38, %f38, 0f3F800000;
	mov.f32 	%f21, 0fBF800000;
	div.rn.f32 	%f22, %f21, %f20;
	mul.f32 	%f23, %f2, %f19;
	mul.f32 	%f24, %f3, %f16;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f3, %f13;
	mul.f32 	%f27, %f1, %f19;
	sub.f32 	%f28, %f26, %f27;
	mul.f32 	%f29, %f1, %f16;
	mul.f32 	%f30, %f2, %f13;
	sub.f32 	%f31, %f29, %f30;
	fma.rn.f32 	%f32, %f25, %f38, %f13;
	fma.rn.f32 	%f33, %f28, %f38, %f16;
	fma.rn.f32 	%f34, %f31, %f38, %f19;
	mul.f32 	%f35, %f32, %f22;
	mul.f32 	%f36, %f33, %f22;
	mul.f32 	%f37, %f34, %f22;
	cvta.to.global.u64 	%rd27, %rd1;
	add.s64 	%rd29, %rd27, %rd12;
	st.global.f32 	[%rd29], %f35;
	cvta.to.global.u64 	%rd30, %rd2;
	add.s64 	%rd31, %rd30, %rd12;
	st.global.f32 	[%rd31], %f36;
	cvta.to.global.u64 	%rd32, %rd3;
	add.s64 	%rd33, %rd32, %rd12;
	st.global.f32 	[%rd33], %f37;

$L__BB0_4:
	ret;

}

`
)
