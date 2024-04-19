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

// CUDA handle for pointwise_div kernel
var pointwise_div_code cu.Function

// Stores the arguments for pointwise_div kernel invocation
type pointwise_div_args_t struct {
	arg_dst unsafe.Pointer
	arg_a   unsafe.Pointer
	arg_b   unsafe.Pointer
	arg_N   int
	argptr  [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for pointwise_div kernel invocation
var pointwise_div_args pointwise_div_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	pointwise_div_args.argptr[0] = unsafe.Pointer(&pointwise_div_args.arg_dst)
	pointwise_div_args.argptr[1] = unsafe.Pointer(&pointwise_div_args.arg_a)
	pointwise_div_args.argptr[2] = unsafe.Pointer(&pointwise_div_args.arg_b)
	pointwise_div_args.argptr[3] = unsafe.Pointer(&pointwise_div_args.arg_N)
}

// Wrapper for pointwise_div CUDA kernel, asynchronous.
func k_pointwise_div_async(dst unsafe.Pointer, a unsafe.Pointer, b unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("pointwise_div")
	}

	pointwise_div_args.Lock()
	defer pointwise_div_args.Unlock()

	if pointwise_div_code == 0 {
		pointwise_div_code = fatbinLoad(pointwise_div_map, "pointwise_div")
	}

	pointwise_div_args.arg_dst = dst
	pointwise_div_args.arg_a = a
	pointwise_div_args.arg_b = b
	pointwise_div_args.arg_N = N

	args := pointwise_div_args.argptr[:]
	cu.LaunchKernel(pointwise_div_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("pointwise_div")
	}
}

// maps compute capability on PTX code for pointwise_div kernel.
var pointwise_div_map = map[int]string{0: "",
	35: pointwise_div_ptx_35,
	37: pointwise_div_ptx_37,
	50: pointwise_div_ptx_50,
	52: pointwise_div_ptx_52,
	53: pointwise_div_ptx_53,
	60: pointwise_div_ptx_60,
	61: pointwise_div_ptx_61,
	62: pointwise_div_ptx_62,
	70: pointwise_div_ptx_70}

// pointwise_div PTX code for various compute capabilities.
const (
	pointwise_div_ptx_35 = `
.version 7.7
.target sm_35
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_37 = `
.version 7.7
.target sm_37
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_50 = `
.version 7.7
.target sm_50
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_52 = `
.version 7.7
.target sm_52
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_53 = `
.version 7.7
.target sm_53
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_60 = `
.version 7.7
.target sm_60
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_61 = `
.version 7.7
.target sm_61
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_62 = `
.version 7.7
.target sm_62
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
	pointwise_div_ptx_70 = `
.version 7.7
.target sm_70
.address_size 64

	// .globl	pointwise_div

.visible .entry pointwise_div(
	.param .u64 pointwise_div_param_0,
	.param .u64 pointwise_div_param_1,
	.param .u64 pointwise_div_param_2,
	.param .u32 pointwise_div_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd3, [pointwise_div_param_0];
	ld.param.u64 	%rd4, [pointwise_div_param_1];
	ld.param.u64 	%rd5, [pointwise_div_param_2];
	ld.param.u32 	%r2, [pointwise_div_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd6, %rd5;
	cvt.s64.s32 	%rd1, %r1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	setp.neu.f32 	%p2, %f1, 0f00000000;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd2, %rd9, %rd7;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	cvta.to.global.u64 	%rd10, %rd4;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	div.rn.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd2], %f3;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	mov.u32 	%r9, 0;
	st.global.u32 	[%rd2], %r9;

$L__BB0_4:
	ret;

}

`
)
