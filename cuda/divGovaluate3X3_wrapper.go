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

// CUDA handle for divGovaluate3X3 kernel
var divGovaluate3X3_code cu.Function

// Stores the arguments for divGovaluate3X3 kernel invocation
type divGovaluate3X3_args_t struct {
	arg_out unsafe.Pointer
	arg_a   unsafe.Pointer
	arg_b   unsafe.Pointer
	arg_Nx  int
	arg_Ny  int
	arg_Nz  int
	arg_aNx int
	arg_aNy int
	arg_aNz int
	arg_bNx int
	arg_bNy int
	arg_bNz int
	argptr  [12]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for divGovaluate3X3 kernel invocation
var divGovaluate3X3_args divGovaluate3X3_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	divGovaluate3X3_args.argptr[0] = unsafe.Pointer(&divGovaluate3X3_args.arg_out)
	divGovaluate3X3_args.argptr[1] = unsafe.Pointer(&divGovaluate3X3_args.arg_a)
	divGovaluate3X3_args.argptr[2] = unsafe.Pointer(&divGovaluate3X3_args.arg_b)
	divGovaluate3X3_args.argptr[3] = unsafe.Pointer(&divGovaluate3X3_args.arg_Nx)
	divGovaluate3X3_args.argptr[4] = unsafe.Pointer(&divGovaluate3X3_args.arg_Ny)
	divGovaluate3X3_args.argptr[5] = unsafe.Pointer(&divGovaluate3X3_args.arg_Nz)
	divGovaluate3X3_args.argptr[6] = unsafe.Pointer(&divGovaluate3X3_args.arg_aNx)
	divGovaluate3X3_args.argptr[7] = unsafe.Pointer(&divGovaluate3X3_args.arg_aNy)
	divGovaluate3X3_args.argptr[8] = unsafe.Pointer(&divGovaluate3X3_args.arg_aNz)
	divGovaluate3X3_args.argptr[9] = unsafe.Pointer(&divGovaluate3X3_args.arg_bNx)
	divGovaluate3X3_args.argptr[10] = unsafe.Pointer(&divGovaluate3X3_args.arg_bNy)
	divGovaluate3X3_args.argptr[11] = unsafe.Pointer(&divGovaluate3X3_args.arg_bNz)
}

// Wrapper for divGovaluate3X3 CUDA kernel, asynchronous.
func k_divGovaluate3X3_async(out unsafe.Pointer, a unsafe.Pointer, b unsafe.Pointer, Nx int, Ny int, Nz int, aNx int, aNy int, aNz int, bNx int, bNy int, bNz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("divGovaluate3X3")
	}

	divGovaluate3X3_args.Lock()
	defer divGovaluate3X3_args.Unlock()

	if divGovaluate3X3_code == 0 {
		divGovaluate3X3_code = fatbinLoad(divGovaluate3X3_map, "divGovaluate3X3")
	}

	divGovaluate3X3_args.arg_out = out
	divGovaluate3X3_args.arg_a = a
	divGovaluate3X3_args.arg_b = b
	divGovaluate3X3_args.arg_Nx = Nx
	divGovaluate3X3_args.arg_Ny = Ny
	divGovaluate3X3_args.arg_Nz = Nz
	divGovaluate3X3_args.arg_aNx = aNx
	divGovaluate3X3_args.arg_aNy = aNy
	divGovaluate3X3_args.arg_aNz = aNz
	divGovaluate3X3_args.arg_bNx = bNx
	divGovaluate3X3_args.arg_bNy = bNy
	divGovaluate3X3_args.arg_bNz = bNz

	args := divGovaluate3X3_args.argptr[:]
	cu.LaunchKernel(divGovaluate3X3_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("divGovaluate3X3")
	}
}

// maps compute capability on PTX code for divGovaluate3X3 kernel.
var divGovaluate3X3_map = map[int]string{0: "",
	50: divGovaluate3X3_ptx_50,
	52: divGovaluate3X3_ptx_52,
	53: divGovaluate3X3_ptx_53,
	60: divGovaluate3X3_ptx_60,
	61: divGovaluate3X3_ptx_61,
	62: divGovaluate3X3_ptx_62,
	70: divGovaluate3X3_ptx_70,
	72: divGovaluate3X3_ptx_72,
	75: divGovaluate3X3_ptx_75,
	80: divGovaluate3X3_ptx_80}

// divGovaluate3X3 PTX code for various compute capabilities.
const (
	divGovaluate3X3_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
	divGovaluate3X3_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	divGovaluate3X3

.visible .entry divGovaluate3X3(
	.param .u64 divGovaluate3X3_param_0,
	.param .u64 divGovaluate3X3_param_1,
	.param .u64 divGovaluate3X3_param_2,
	.param .u32 divGovaluate3X3_param_3,
	.param .u32 divGovaluate3X3_param_4,
	.param .u32 divGovaluate3X3_param_5,
	.param .u32 divGovaluate3X3_param_6,
	.param .u32 divGovaluate3X3_param_7,
	.param .u32 divGovaluate3X3_param_8,
	.param .u32 divGovaluate3X3_param_9,
	.param .u32 divGovaluate3X3_param_10,
	.param .u32 divGovaluate3X3_param_11
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<34>;
	.reg .b64 	%rd<13>;


	ld.param.u64 	%rd1, [divGovaluate3X3_param_0];
	ld.param.u64 	%rd2, [divGovaluate3X3_param_1];
	ld.param.u64 	%rd3, [divGovaluate3X3_param_2];
	ld.param.u32 	%r4, [divGovaluate3X3_param_3];
	ld.param.u32 	%r5, [divGovaluate3X3_param_4];
	ld.param.u32 	%r12, [divGovaluate3X3_param_5];
	ld.param.u32 	%r6, [divGovaluate3X3_param_6];
	ld.param.u32 	%r7, [divGovaluate3X3_param_7];
	ld.param.u32 	%r8, [divGovaluate3X3_param_8];
	ld.param.u32 	%r9, [divGovaluate3X3_param_9];
	ld.param.u32 	%r10, [divGovaluate3X3_param_10];
	ld.param.u32 	%r11, [divGovaluate3X3_param_11];
	mov.u32 	%r13, %ctaid.x;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %tid.x;
	mad.lo.s32 	%r1, %r13, %r14, %r15;
	mov.u32 	%r16, %ntid.y;
	mov.u32 	%r17, %ctaid.y;
	mov.u32 	%r18, %tid.y;
	mad.lo.s32 	%r2, %r17, %r16, %r18;
	mov.u32 	%r19, %ntid.z;
	mov.u32 	%r20, %ctaid.z;
	mov.u32 	%r21, %tid.z;
	mad.lo.s32 	%r3, %r20, %r19, %r21;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r12;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mad.lo.s32 	%r22, %r3, %r5, %r2;
	mad.lo.s32 	%r23, %r22, %r4, %r1;
	setp.eq.s32 	%p6, %r6, 1;
	selp.b32 	%r24, 0, %r1, %p6;
	setp.eq.s32 	%p7, %r7, 1;
	selp.b32 	%r25, 0, %r2, %p7;
	setp.eq.s32 	%p8, %r8, 1;
	selp.b32 	%r26, 0, %r3, %p8;
	mad.lo.s32 	%r27, %r26, %r7, %r25;
	mad.lo.s32 	%r28, %r27, %r6, %r24;
	setp.eq.s32 	%p9, %r9, 1;
	selp.b32 	%r29, 0, %r1, %p9;
	setp.eq.s32 	%p10, %r10, 1;
	selp.b32 	%r30, 0, %r2, %p10;
	setp.eq.s32 	%p11, %r11, 1;
	selp.b32 	%r31, 0, %r3, %p11;
	mad.lo.s32 	%r32, %r31, %r10, %r30;
	mad.lo.s32 	%r33, %r32, %r9, %r29;
	mul.wide.s32 	%rd5, %r28, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	mul.wide.s32 	%rd8, %r33, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	ld.global.f32 	%f2, [%rd6];
	div.rn.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd10, %rd1;
	mul.wide.s32 	%rd11, %r23, 4;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.f32 	[%rd12], %f3;

$L__BB0_2:
	ret;

}

`
)
