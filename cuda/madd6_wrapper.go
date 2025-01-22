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

// CUDA handle for madd6 kernel
var madd6_code cu.Function

// Stores the arguments for madd6 kernel invocation
type madd6_args_t struct {
	arg_dst  unsafe.Pointer
	arg_src1 unsafe.Pointer
	arg_fac1 float32
	arg_src2 unsafe.Pointer
	arg_fac2 float32
	arg_src3 unsafe.Pointer
	arg_fac3 float32
	arg_src4 unsafe.Pointer
	arg_fac4 float32
	arg_src5 unsafe.Pointer
	arg_fac5 float32
	arg_src6 unsafe.Pointer
	arg_fac6 float32
	arg_N    int
	argptr   [14]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for madd6 kernel invocation
var madd6_args madd6_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	madd6_args.argptr[0] = unsafe.Pointer(&madd6_args.arg_dst)
	madd6_args.argptr[1] = unsafe.Pointer(&madd6_args.arg_src1)
	madd6_args.argptr[2] = unsafe.Pointer(&madd6_args.arg_fac1)
	madd6_args.argptr[3] = unsafe.Pointer(&madd6_args.arg_src2)
	madd6_args.argptr[4] = unsafe.Pointer(&madd6_args.arg_fac2)
	madd6_args.argptr[5] = unsafe.Pointer(&madd6_args.arg_src3)
	madd6_args.argptr[6] = unsafe.Pointer(&madd6_args.arg_fac3)
	madd6_args.argptr[7] = unsafe.Pointer(&madd6_args.arg_src4)
	madd6_args.argptr[8] = unsafe.Pointer(&madd6_args.arg_fac4)
	madd6_args.argptr[9] = unsafe.Pointer(&madd6_args.arg_src5)
	madd6_args.argptr[10] = unsafe.Pointer(&madd6_args.arg_fac5)
	madd6_args.argptr[11] = unsafe.Pointer(&madd6_args.arg_src6)
	madd6_args.argptr[12] = unsafe.Pointer(&madd6_args.arg_fac6)
	madd6_args.argptr[13] = unsafe.Pointer(&madd6_args.arg_N)
}

// Wrapper for madd6 CUDA kernel, asynchronous.
func k_madd6_async(dst unsafe.Pointer, src1 unsafe.Pointer, fac1 float32, src2 unsafe.Pointer, fac2 float32, src3 unsafe.Pointer, fac3 float32, src4 unsafe.Pointer, fac4 float32, src5 unsafe.Pointer, fac5 float32, src6 unsafe.Pointer, fac6 float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("madd6")
	}

	madd6_args.Lock()
	defer madd6_args.Unlock()

	if madd6_code == 0 {
		madd6_code = fatbinLoad(madd6_map, "madd6")
	}

	madd6_args.arg_dst = dst
	madd6_args.arg_src1 = src1
	madd6_args.arg_fac1 = fac1
	madd6_args.arg_src2 = src2
	madd6_args.arg_fac2 = fac2
	madd6_args.arg_src3 = src3
	madd6_args.arg_fac3 = fac3
	madd6_args.arg_src4 = src4
	madd6_args.arg_fac4 = fac4
	madd6_args.arg_src5 = src5
	madd6_args.arg_fac5 = fac5
	madd6_args.arg_src6 = src6
	madd6_args.arg_fac6 = fac6
	madd6_args.arg_N = N

	args := madd6_args.argptr[:]
	cu.LaunchKernel(madd6_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("madd6")
	}
}

// maps compute capability on PTX code for madd6 kernel.
var madd6_map = map[int]string{0: "",
	50: madd6_ptx_50,
	52: madd6_ptx_52,
	53: madd6_ptx_53,
	60: madd6_ptx_60,
	61: madd6_ptx_61,
	62: madd6_ptx_62,
	70: madd6_ptx_70,
	72: madd6_ptx_72,
	75: madd6_ptx_75,
	80: madd6_ptx_80}

// madd6 PTX code for various compute capabilities.
const (
	madd6_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
	madd6_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	madd6

.visible .entry madd6(
	.param .u64 madd6_param_0,
	.param .u64 madd6_param_1,
	.param .f32 madd6_param_2,
	.param .u64 madd6_param_3,
	.param .f32 madd6_param_4,
	.param .u64 madd6_param_5,
	.param .f32 madd6_param_6,
	.param .u64 madd6_param_7,
	.param .f32 madd6_param_8,
	.param .u64 madd6_param_9,
	.param .f32 madd6_param_10,
	.param .u64 madd6_param_11,
	.param .f32 madd6_param_12,
	.param .u32 madd6_param_13
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<19>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [madd6_param_0];
	ld.param.u64 	%rd2, [madd6_param_1];
	ld.param.f32 	%f1, [madd6_param_2];
	ld.param.u64 	%rd3, [madd6_param_3];
	ld.param.f32 	%f2, [madd6_param_4];
	ld.param.u64 	%rd4, [madd6_param_5];
	ld.param.f32 	%f3, [madd6_param_6];
	ld.param.u64 	%rd5, [madd6_param_7];
	ld.param.f32 	%f4, [madd6_param_8];
	ld.param.u64 	%rd6, [madd6_param_9];
	ld.param.f32 	%f5, [madd6_param_10];
	ld.param.u64 	%rd7, [madd6_param_11];
	ld.param.f32 	%f6, [madd6_param_12];
	ld.param.u32 	%r2, [madd6_param_13];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd2;
	mul.wide.s32 	%rd9, %r1, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f7, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f8, [%rd12];
	mul.f32 	%f9, %f8, %f2;
	fma.rn.f32 	%f10, %f7, %f1, %f9;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd9;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f3, %f10;
	cvta.to.global.u64 	%rd15, %rd5;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f13, [%rd16];
	fma.rn.f32 	%f14, %f13, %f4, %f12;
	cvta.to.global.u64 	%rd17, %rd6;
	add.s64 	%rd18, %rd17, %rd9;
	ld.global.nc.f32 	%f15, [%rd18];
	fma.rn.f32 	%f16, %f15, %f5, %f14;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f17, [%rd20];
	fma.rn.f32 	%f18, %f17, %f6, %f16;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f18;

$L__BB0_2:
	ret;

}

`
)
