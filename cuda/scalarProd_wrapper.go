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

// CUDA handle for scalarProd kernel
var scalarProd_code cu.Function

// Stores the arguments for scalarProd kernel invocation
type scalarProd_args_t struct {
	arg_res unsafe.Pointer
	arg_ax  unsafe.Pointer
	arg_ay  unsafe.Pointer
	arg_az  unsafe.Pointer
	arg_bx  unsafe.Pointer
	arg_by  unsafe.Pointer
	arg_bz  unsafe.Pointer
	arg_Nx  int
	arg_Ny  int
	arg_Nz  int
	argptr  [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for scalarProd kernel invocation
var scalarProd_args scalarProd_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	scalarProd_args.argptr[0] = unsafe.Pointer(&scalarProd_args.arg_res)
	scalarProd_args.argptr[1] = unsafe.Pointer(&scalarProd_args.arg_ax)
	scalarProd_args.argptr[2] = unsafe.Pointer(&scalarProd_args.arg_ay)
	scalarProd_args.argptr[3] = unsafe.Pointer(&scalarProd_args.arg_az)
	scalarProd_args.argptr[4] = unsafe.Pointer(&scalarProd_args.arg_bx)
	scalarProd_args.argptr[5] = unsafe.Pointer(&scalarProd_args.arg_by)
	scalarProd_args.argptr[6] = unsafe.Pointer(&scalarProd_args.arg_bz)
	scalarProd_args.argptr[7] = unsafe.Pointer(&scalarProd_args.arg_Nx)
	scalarProd_args.argptr[8] = unsafe.Pointer(&scalarProd_args.arg_Ny)
	scalarProd_args.argptr[9] = unsafe.Pointer(&scalarProd_args.arg_Nz)
}

// Wrapper for scalarProd CUDA kernel, asynchronous.
func k_scalarProd_async(res unsafe.Pointer, ax unsafe.Pointer, ay unsafe.Pointer, az unsafe.Pointer, bx unsafe.Pointer, by unsafe.Pointer, bz unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("scalarProd")
	}

	scalarProd_args.Lock()
	defer scalarProd_args.Unlock()

	if scalarProd_code == 0 {
		scalarProd_code = fatbinLoad(scalarProd_map, "scalarProd")
	}

	scalarProd_args.arg_res = res
	scalarProd_args.arg_ax = ax
	scalarProd_args.arg_ay = ay
	scalarProd_args.arg_az = az
	scalarProd_args.arg_bx = bx
	scalarProd_args.arg_by = by
	scalarProd_args.arg_bz = bz
	scalarProd_args.arg_Nx = Nx
	scalarProd_args.arg_Ny = Ny
	scalarProd_args.arg_Nz = Nz

	args := scalarProd_args.argptr[:]
	cu.LaunchKernel(scalarProd_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("scalarProd")
	}
}

// maps compute capability on PTX code for scalarProd kernel.
var scalarProd_map = map[int]string{0: "",
	50: scalarProd_ptx_50,
	52: scalarProd_ptx_52,
	53: scalarProd_ptx_53,
	60: scalarProd_ptx_60,
	61: scalarProd_ptx_61,
	62: scalarProd_ptx_62,
	70: scalarProd_ptx_70,
	72: scalarProd_ptx_72,
	75: scalarProd_ptx_75,
	80: scalarProd_ptx_80}

// scalarProd PTX code for various compute capabilities.
const (
	scalarProd_ptx_50 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_50
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_52 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_52
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_53 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_53
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_60 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_60
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_61 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_61
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_62 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_62
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_70 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_70
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_72 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_72
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_75 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_75
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
	scalarProd_ptx_80 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_80
.address_size 64

	// .globl	scalarProd

.visible .entry scalarProd(
	.param .u64 scalarProd_param_0,
	.param .u64 scalarProd_param_1,
	.param .u64 scalarProd_param_2,
	.param .u64 scalarProd_param_3,
	.param .u64 scalarProd_param_4,
	.param .u64 scalarProd_param_5,
	.param .u64 scalarProd_param_6,
	.param .u32 scalarProd_param_7,
	.param .u32 scalarProd_param_8,
	.param .u32 scalarProd_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<10>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<23>;


	ld.param.u64 	%rd1, [scalarProd_param_0];
	ld.param.u64 	%rd2, [scalarProd_param_1];
	ld.param.u64 	%rd3, [scalarProd_param_2];
	ld.param.u64 	%rd4, [scalarProd_param_3];
	ld.param.u64 	%rd5, [scalarProd_param_4];
	ld.param.u64 	%rd6, [scalarProd_param_5];
	ld.param.u64 	%rd7, [scalarProd_param_6];
	ld.param.u32 	%r4, [scalarProd_param_7];
	ld.param.u32 	%r5, [scalarProd_param_8];
	ld.param.u32 	%r6, [scalarProd_param_9];
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

	cvta.to.global.u64 	%rd8, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd9, %r17, 4;
	add.s64 	%rd10, %rd8, %rd9;
	cvta.to.global.u64 	%rd11, %rd5;
	add.s64 	%rd12, %rd11, %rd9;
	ld.global.nc.f32 	%f1, [%rd12];
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd13, %rd3;
	add.s64 	%rd14, %rd13, %rd9;
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd9;
	ld.global.nc.f32 	%f3, [%rd16];
	ld.global.nc.f32 	%f4, [%rd14];
	mul.f32 	%f5, %f4, %f3;
	fma.rn.f32 	%f6, %f2, %f1, %f5;
	cvta.to.global.u64 	%rd17, %rd4;
	add.s64 	%rd18, %rd17, %rd9;
	cvta.to.global.u64 	%rd19, %rd7;
	add.s64 	%rd20, %rd19, %rd9;
	ld.global.nc.f32 	%f7, [%rd20];
	ld.global.nc.f32 	%f8, [%rd18];
	fma.rn.f32 	%f9, %f8, %f7, %f6;
	cvta.to.global.u64 	%rd21, %rd1;
	add.s64 	%rd22, %rd21, %rd9;
	st.global.f32 	[%rd22], %f9;

$L__BB0_2:
	ret;

}

`
)
