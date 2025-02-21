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

// CUDA handle for fftshift3D_partial kernel
var fftshift3D_partial_code cu.Function

// Stores the arguments for fftshift3D_partial kernel invocation
type fftshift3D_partial_args_t struct {
	arg_data_out unsafe.Pointer
	arg_data_in  unsafe.Pointer
	arg_Nx       int
	arg_Ny       int
	arg_Nz       int
	argptr       [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for fftshift3D_partial kernel invocation
var fftshift3D_partial_args fftshift3D_partial_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	fftshift3D_partial_args.argptr[0] = unsafe.Pointer(&fftshift3D_partial_args.arg_data_out)
	fftshift3D_partial_args.argptr[1] = unsafe.Pointer(&fftshift3D_partial_args.arg_data_in)
	fftshift3D_partial_args.argptr[2] = unsafe.Pointer(&fftshift3D_partial_args.arg_Nx)
	fftshift3D_partial_args.argptr[3] = unsafe.Pointer(&fftshift3D_partial_args.arg_Ny)
	fftshift3D_partial_args.argptr[4] = unsafe.Pointer(&fftshift3D_partial_args.arg_Nz)
}

// Wrapper for fftshift3D_partial CUDA kernel, asynchronous.
func k_fftshift3D_partial_async(data_out unsafe.Pointer, data_in unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("fftshift3D_partial")
	}

	fftshift3D_partial_args.Lock()
	defer fftshift3D_partial_args.Unlock()

	if fftshift3D_partial_code == 0 {
		fftshift3D_partial_code = fatbinLoad(fftshift3D_partial_map, "fftshift3D_partial")
	}

	fftshift3D_partial_args.arg_data_out = data_out
	fftshift3D_partial_args.arg_data_in = data_in
	fftshift3D_partial_args.arg_Nx = Nx
	fftshift3D_partial_args.arg_Ny = Ny
	fftshift3D_partial_args.arg_Nz = Nz

	args := fftshift3D_partial_args.argptr[:]
	cu.LaunchKernel(fftshift3D_partial_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("fftshift3D_partial")
	}
}

// maps compute capability on PTX code for fftshift3D_partial kernel.
var fftshift3D_partial_map = map[int]string{0: "",
	50: fftshift3D_partial_ptx_50,
	52: fftshift3D_partial_ptx_52,
	53: fftshift3D_partial_ptx_53,
	60: fftshift3D_partial_ptx_60,
	61: fftshift3D_partial_ptx_61,
	62: fftshift3D_partial_ptx_62,
	70: fftshift3D_partial_ptx_70,
	72: fftshift3D_partial_ptx_72,
	75: fftshift3D_partial_ptx_75,
	80: fftshift3D_partial_ptx_80}

// fftshift3D_partial PTX code for various compute capabilities.
const (
	fftshift3D_partial_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	fftshift3D_partial_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	fftshift3D_partial

.visible .entry fftshift3D_partial(
	.param .u64 fftshift3D_partial_param_0,
	.param .u64 fftshift3D_partial_param_1,
	.param .u32 fftshift3D_partial_param_2,
	.param .u32 fftshift3D_partial_param_3,
	.param .u32 fftshift3D_partial_param_4
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [fftshift3D_partial_param_0];
	ld.param.u64 	%rd2, [fftshift3D_partial_param_1];
	ld.param.u32 	%r4, [fftshift3D_partial_param_2];
	ld.param.u32 	%r5, [fftshift3D_partial_param_3];
	ld.param.u32 	%r6, [fftshift3D_partial_param_4];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r11, %r10, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r14, %r13, %r15;
	setp.ge.s32 	%p1, %r1, %r4;
	setp.ge.s32 	%p2, %r2, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	shr.u32 	%r16, %r5, 31;
	add.s32 	%r17, %r5, %r16;
	shr.s32 	%r18, %r17, 1;
	add.s32 	%r19, %r18, %r2;
	rem.s32 	%r20, %r19, %r5;
	shr.u32 	%r21, %r6, 31;
	add.s32 	%r22, %r6, %r21;
	shr.s32 	%r23, %r22, 1;
	add.s32 	%r24, %r23, %r3;
	rem.s32 	%r25, %r24, %r6;
	mad.lo.s32 	%r26, %r3, %r5, %r2;
	mad.lo.s32 	%r27, %r26, %r4, %r1;
	mad.lo.s32 	%r28, %r25, %r5, %r20;
	mad.lo.s32 	%r29, %r28, %r4, %r1;
	mul.wide.s32 	%rd4, %r27, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r29, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
)
