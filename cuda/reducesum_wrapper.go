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

// CUDA handle for reducesum kernel
var reducesum_code cu.Function

// Stores the arguments for reducesum kernel invocation
type reducesum_args_t struct {
	arg_src     unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducesum kernel invocation
var reducesum_args reducesum_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducesum_args.argptr[0] = unsafe.Pointer(&reducesum_args.arg_src)
	reducesum_args.argptr[1] = unsafe.Pointer(&reducesum_args.arg_dst)
	reducesum_args.argptr[2] = unsafe.Pointer(&reducesum_args.arg_initVal)
	reducesum_args.argptr[3] = unsafe.Pointer(&reducesum_args.arg_n)
}

// Wrapper for reducesum CUDA kernel, asynchronous.
func k_reducesum_async(src unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducesum")
	}

	reducesum_args.Lock()
	defer reducesum_args.Unlock()

	if reducesum_code == 0 {
		reducesum_code = fatbinLoad(reducesum_map, "reducesum")
	}

	reducesum_args.arg_src = src
	reducesum_args.arg_dst = dst
	reducesum_args.arg_initVal = initVal
	reducesum_args.arg_n = n

	args := reducesum_args.argptr[:]
	cu.LaunchKernel(reducesum_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducesum")
	}
}

// maps compute capability on PTX code for reducesum kernel.
var reducesum_map = map[int]string{0: "",
	35: reducesum_ptx_35,
	37: reducesum_ptx_37,
	50: reducesum_ptx_50,
	52: reducesum_ptx_52,
	53: reducesum_ptx_53,
	60: reducesum_ptx_60,
	61: reducesum_ptx_61,
	62: reducesum_ptx_62,
	70: reducesum_ptx_70}

// reducesum PTX code for various compute capabilities.
const (
	reducesum_ptx_35 = `
.version 7.7
.target sm_35
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_37 = `
.version 7.7
.target sm_37
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_50 = `
.version 7.7
.target sm_50
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_52 = `
.version 7.7
.target sm_52
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_53 = `
.version 7.7
.target sm_53
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_60 = `
.version 7.7
.target sm_60
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_61 = `
.version 7.7
.target sm_61
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_62 = `
.version 7.7
.target sm_62
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
	reducesum_ptx_70 = `
.version 7.7
.target sm_70
.address_size 64

	// .globl	reducesum

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .f32 reducesum_param_2,
	.param .u32 reducesum_param_3
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<46>;
	.reg .b32 	%r<37>;
	.reg .b64 	%rd<17>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducesumE5sdata[2048];

	ld.param.u64 	%rd8, [reducesum_param_0];
	ld.param.u64 	%rd7, [reducesum_param_1];
	ld.param.f32 	%f45, [reducesum_param_2];
	ld.param.u32 	%r17, [reducesum_param_3];
	cvta.to.global.u64 	%rd1, %rd8;
	mov.u32 	%r36, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r34, %r18, %r36, %r2;
	mov.u32 	%r19, %nctaid.x;
	mul.lo.s32 	%r4, %r19, %r36;
	setp.ge.s32 	%p1, %r34, %r17;
	@%p1 bra 	$L__BB0_7;

	add.s32 	%r20, %r4, %r17;
	add.s32 	%r21, %r34, %r4;
	not.b32 	%r22, %r21;
	add.s32 	%r23, %r20, %r22;
	div.u32 	%r5, %r23, %r4;
	add.s32 	%r24, %r5, 1;
	and.b32  	%r33, %r24, 3;
	setp.eq.s32 	%p2, %r33, 0;
	@%p2 bra 	$L__BB0_4;

	mul.wide.s32 	%rd9, %r34, 4;
	add.s64 	%rd16, %rd1, %rd9;
	mul.wide.s32 	%rd3, %r4, 4;

$L__BB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd16];
	add.f32 	%f45, %f45, %f10;
	add.s32 	%r34, %r34, %r4;
	add.s64 	%rd16, %rd16, %rd3;
	add.s32 	%r33, %r33, -1;
	setp.ne.s32 	%p3, %r33, 0;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	setp.lt.u32 	%p4, %r5, 3;
	@%p4 bra 	$L__BB0_7;

	mul.wide.s32 	%rd6, %r4, 4;

$L__BB0_6:
	mul.wide.s32 	%rd10, %r34, 4;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.f32 	%f11, [%rd11];
	add.f32 	%f12, %f45, %f11;
	add.s64 	%rd12, %rd11, %rd6;
	ld.global.nc.f32 	%f13, [%rd12];
	add.f32 	%f14, %f12, %f13;
	add.s32 	%r25, %r34, %r4;
	add.s32 	%r26, %r25, %r4;
	add.s64 	%rd13, %rd12, %rd6;
	ld.global.nc.f32 	%f15, [%rd13];
	add.f32 	%f16, %f14, %f15;
	add.s32 	%r27, %r26, %r4;
	add.s64 	%rd14, %rd13, %rd6;
	ld.global.nc.f32 	%f17, [%rd14];
	add.f32 	%f45, %f16, %f17;
	add.s32 	%r34, %r27, %r4;
	setp.lt.s32 	%p5, %r34, %r17;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	shl.b32 	%r28, %r2, 2;
	mov.u32 	%r29, _ZZ9reducesumE5sdata;
	add.s32 	%r14, %r29, %r28;
	st.shared.f32 	[%r14], %f45;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r36, 66;
	@%p6 bra 	$L__BB0_12;

$L__BB0_9:
	shr.u32 	%r16, %r36, 1;
	setp.ge.u32 	%p7, %r2, %r16;
	@%p7 bra 	$L__BB0_11;

	ld.shared.f32 	%f18, [%r14];
	shl.b32 	%r30, %r16, 2;
	add.s32 	%r31, %r14, %r30;
	ld.shared.f32 	%f19, [%r31];
	add.f32 	%f20, %f18, %f19;
	st.shared.f32 	[%r14], %f20;

$L__BB0_11:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r36, 131;
	mov.u32 	%r36, %r16;
	@%p8 bra 	$L__BB0_9;

$L__BB0_12:
	setp.gt.s32 	%p9, %r2, 31;
	@%p9 bra 	$L__BB0_14;

	ld.volatile.shared.f32 	%f21, [%r14];
	ld.volatile.shared.f32 	%f22, [%r14+128];
	add.f32 	%f23, %f21, %f22;
	st.volatile.shared.f32 	[%r14], %f23;
	ld.volatile.shared.f32 	%f24, [%r14+64];
	ld.volatile.shared.f32 	%f25, [%r14];
	add.f32 	%f26, %f25, %f24;
	st.volatile.shared.f32 	[%r14], %f26;
	ld.volatile.shared.f32 	%f27, [%r14+32];
	ld.volatile.shared.f32 	%f28, [%r14];
	add.f32 	%f29, %f28, %f27;
	st.volatile.shared.f32 	[%r14], %f29;
	ld.volatile.shared.f32 	%f30, [%r14+16];
	ld.volatile.shared.f32 	%f31, [%r14];
	add.f32 	%f32, %f31, %f30;
	st.volatile.shared.f32 	[%r14], %f32;
	ld.volatile.shared.f32 	%f33, [%r14+8];
	ld.volatile.shared.f32 	%f34, [%r14];
	add.f32 	%f35, %f34, %f33;
	st.volatile.shared.f32 	[%r14], %f35;
	ld.volatile.shared.f32 	%f36, [%r14+4];
	ld.volatile.shared.f32 	%f37, [%r14];
	add.f32 	%f38, %f37, %f36;
	st.volatile.shared.f32 	[%r14], %f38;

$L__BB0_14:
	setp.ne.s32 	%p10, %r2, 0;
	@%p10 bra 	$L__BB0_16;

	ld.shared.f32 	%f39, [_ZZ9reducesumE5sdata];
	cvta.to.global.u64 	%rd15, %rd7;
	atom.global.add.f32 	%f40, [%rd15], %f39;

$L__BB0_16:
	ret;

}

`
)
