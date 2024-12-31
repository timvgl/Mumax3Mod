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

// CUDA handle for regionaddv kernel
var regionaddv_code cu.Function

// Stores the arguments for regionaddv kernel invocation
type regionaddv_args_t struct {
	arg_dstx    unsafe.Pointer
	arg_dsty    unsafe.Pointer
	arg_dstz    unsafe.Pointer
	arg_LUTx    unsafe.Pointer
	arg_LUTy    unsafe.Pointer
	arg_LUTz    unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_N       int
	argptr      [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for regionaddv kernel invocation
var regionaddv_args regionaddv_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	regionaddv_args.argptr[0] = unsafe.Pointer(&regionaddv_args.arg_dstx)
	regionaddv_args.argptr[1] = unsafe.Pointer(&regionaddv_args.arg_dsty)
	regionaddv_args.argptr[2] = unsafe.Pointer(&regionaddv_args.arg_dstz)
	regionaddv_args.argptr[3] = unsafe.Pointer(&regionaddv_args.arg_LUTx)
	regionaddv_args.argptr[4] = unsafe.Pointer(&regionaddv_args.arg_LUTy)
	regionaddv_args.argptr[5] = unsafe.Pointer(&regionaddv_args.arg_LUTz)
	regionaddv_args.argptr[6] = unsafe.Pointer(&regionaddv_args.arg_regions)
	regionaddv_args.argptr[7] = unsafe.Pointer(&regionaddv_args.arg_N)
}

// Wrapper for regionaddv CUDA kernel, asynchronous.
func k_regionaddv_async(dstx unsafe.Pointer, dsty unsafe.Pointer, dstz unsafe.Pointer, LUTx unsafe.Pointer, LUTy unsafe.Pointer, LUTz unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("regionaddv")
	}

	regionaddv_args.Lock()
	defer regionaddv_args.Unlock()

	if regionaddv_code == 0 {
		regionaddv_code = fatbinLoad(regionaddv_map, "regionaddv")
	}

	regionaddv_args.arg_dstx = dstx
	regionaddv_args.arg_dsty = dsty
	regionaddv_args.arg_dstz = dstz
	regionaddv_args.arg_LUTx = LUTx
	regionaddv_args.arg_LUTy = LUTy
	regionaddv_args.arg_LUTz = LUTz
	regionaddv_args.arg_regions = regions
	regionaddv_args.arg_N = N

	args := regionaddv_args.argptr[:]
	cu.LaunchKernel(regionaddv_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("regionaddv")
	}
}

// maps compute capability on PTX code for regionaddv kernel.
var regionaddv_map = map[int]string{0: "",
	50: regionaddv_ptx_50,
	52: regionaddv_ptx_52,
	53: regionaddv_ptx_53,
	60: regionaddv_ptx_60,
	61: regionaddv_ptx_61,
	62: regionaddv_ptx_62,
	70: regionaddv_ptx_70,
	72: regionaddv_ptx_72,
	75: regionaddv_ptx_75,
	80: regionaddv_ptx_80}

// regionaddv PTX code for various compute capabilities.
const (
	regionaddv_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
	regionaddv_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	regionaddv

.visible .entry regionaddv(
	.param .u64 regionaddv_param_0,
	.param .u64 regionaddv_param_1,
	.param .u64 regionaddv_param_2,
	.param .u64 regionaddv_param_3,
	.param .u64 regionaddv_param_4,
	.param .u64 regionaddv_param_5,
	.param .u64 regionaddv_param_6,
	.param .u32 regionaddv_param_7
)
{
	.reg .pred 	%p&lt;2&gt;;
	.reg .b16 	%rs&lt;2&gt;;
	.reg .f32 	%f&lt;10&gt;;
	.reg .b32 	%r&lt;11&gt;;
	.reg .b64 	%rd&lt;25&gt;;


	ld.param.u64 	%rd1, [regionaddv_param_0];
	ld.param.u64 	%rd2, [regionaddv_param_1];
	ld.param.u64 	%rd3, [regionaddv_param_2];
	ld.param.u64 	%rd4, [regionaddv_param_3];
	ld.param.u64 	%rd5, [regionaddv_param_4];
	ld.param.u64 	%rd6, [regionaddv_param_5];
	ld.param.u64 	%rd7, [regionaddv_param_6];
	ld.param.u32 	%r2, [regionaddv_param_7];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd8, %rd7;
	cvt.s64.s32 	%rd9, %r1;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.u8 	%rs1, [%rd10];
	cvta.to.global.u64 	%rd11, %rd4;
	cvt.u32.u16 	%r9, %rs1;
	and.b32  	%r10, %r9, 255;
	mul.wide.u32 	%rd12, %r10, 4;
	add.s64 	%rd13, %rd11, %rd12;
	cvta.to.global.u64 	%rd14, %rd1;
	mul.wide.s32 	%rd15, %r1, 4;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd13];
	add.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd16], %f3;
	cvta.to.global.u64 	%rd17, %rd5;
	add.s64 	%rd18, %rd17, %rd12;
	cvta.to.global.u64 	%rd19, %rd2;
	add.s64 	%rd20, %rd19, %rd15;
	ld.global.f32 	%f4, [%rd20];
	ld.global.nc.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f4;
	st.global.f32 	[%rd20], %f6;
	cvta.to.global.u64 	%rd21, %rd6;
	add.s64 	%rd22, %rd21, %rd12;
	cvta.to.global.u64 	%rd23, %rd3;
	add.s64 	%rd24, %rd23, %rd15;
	ld.global.f32 	%f7, [%rd24];
	ld.global.nc.f32 	%f8, [%rd22];
	add.f32 	%f9, %f8, %f7;
	st.global.f32 	[%rd24], %f9;

$L__BB0_2:
	ret;

}

`
)
