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

// CUDA handle for KineticEnergy kernel
var KineticEnergy_code cu.Function

// Stores the arguments for KineticEnergy kernel invocation
type KineticEnergy_args_t struct {
	arg_energy unsafe.Pointer
	arg_dux    unsafe.Pointer
	arg_duy    unsafe.Pointer
	arg_duz    unsafe.Pointer
	arg_rho    unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	argptr     [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for KineticEnergy kernel invocation
var KineticEnergy_args KineticEnergy_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	KineticEnergy_args.argptr[0] = unsafe.Pointer(&KineticEnergy_args.arg_energy)
	KineticEnergy_args.argptr[1] = unsafe.Pointer(&KineticEnergy_args.arg_dux)
	KineticEnergy_args.argptr[2] = unsafe.Pointer(&KineticEnergy_args.arg_duy)
	KineticEnergy_args.argptr[3] = unsafe.Pointer(&KineticEnergy_args.arg_duz)
	KineticEnergy_args.argptr[4] = unsafe.Pointer(&KineticEnergy_args.arg_rho)
	KineticEnergy_args.argptr[5] = unsafe.Pointer(&KineticEnergy_args.arg_Nx)
	KineticEnergy_args.argptr[6] = unsafe.Pointer(&KineticEnergy_args.arg_Ny)
	KineticEnergy_args.argptr[7] = unsafe.Pointer(&KineticEnergy_args.arg_Nz)
}

// Wrapper for KineticEnergy CUDA kernel, asynchronous.
func k_KineticEnergy_async(energy unsafe.Pointer, dux unsafe.Pointer, duy unsafe.Pointer, duz unsafe.Pointer, rho unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("KineticEnergy")
	}

	KineticEnergy_args.Lock()
	defer KineticEnergy_args.Unlock()

	if KineticEnergy_code == 0 {
		KineticEnergy_code = fatbinLoad(KineticEnergy_map, "KineticEnergy")
	}

	KineticEnergy_args.arg_energy = energy
	KineticEnergy_args.arg_dux = dux
	KineticEnergy_args.arg_duy = duy
	KineticEnergy_args.arg_duz = duz
	KineticEnergy_args.arg_rho = rho
	KineticEnergy_args.arg_Nx = Nx
	KineticEnergy_args.arg_Ny = Ny
	KineticEnergy_args.arg_Nz = Nz

	args := KineticEnergy_args.argptr[:]
	cu.LaunchKernel(KineticEnergy_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("KineticEnergy")
	}
}

// maps compute capability on PTX code for KineticEnergy kernel.
var KineticEnergy_map = map[int]string{0: "",
	50: KineticEnergy_ptx_50,
	52: KineticEnergy_ptx_52,
	53: KineticEnergy_ptx_53,
	60: KineticEnergy_ptx_60,
	61: KineticEnergy_ptx_61,
	62: KineticEnergy_ptx_62,
	70: KineticEnergy_ptx_70,
	72: KineticEnergy_ptx_72,
	75: KineticEnergy_ptx_75,
	80: KineticEnergy_ptx_80}

// KineticEnergy PTX code for various compute capabilities.
const (
	KineticEnergy_ptx_50 = `
.version 8.4
.target sm_50
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_52 = `
.version 8.4
.target sm_52
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_53 = `
.version 8.4
.target sm_53
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_60 = `
.version 8.4
.target sm_60
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_61 = `
.version 8.4
.target sm_61
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_62 = `
.version 8.4
.target sm_62
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_70 = `
.version 8.4
.target sm_70
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_72 = `
.version 8.4
.target sm_72
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_75 = `
.version 8.4
.target sm_75
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
	KineticEnergy_ptx_80 = `
.version 8.4
.target sm_80
.address_size 64

	// .globl	KineticEnergy

.visible .entry KineticEnergy(
	.param .u64 KineticEnergy_param_0,
	.param .u64 KineticEnergy_param_1,
	.param .u64 KineticEnergy_param_2,
	.param .u64 KineticEnergy_param_3,
	.param .u64 KineticEnergy_param_4,
	.param .u32 KineticEnergy_param_5,
	.param .u32 KineticEnergy_param_6,
	.param .u32 KineticEnergy_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<18>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [KineticEnergy_param_0];
	ld.param.u64 	%rd2, [KineticEnergy_param_1];
	ld.param.u64 	%rd3, [KineticEnergy_param_2];
	ld.param.u64 	%rd4, [KineticEnergy_param_3];
	ld.param.u64 	%rd5, [KineticEnergy_param_4];
	ld.param.u32 	%r4, [KineticEnergy_param_5];
	ld.param.u32 	%r5, [KineticEnergy_param_6];
	ld.param.u32 	%r6, [KineticEnergy_param_7];
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

	cvta.to.global.u64 	%rd6, %rd2;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvta.to.global.u64 	%rd7, %rd5;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvt.f64.f32 	%fd1, %f1;
	mul.f64 	%fd2, %fd1, 0d3FE0000000000000;
	add.s64 	%rd10, %rd6, %rd8;
	ld.global.nc.f32 	%f2, [%rd10];
	cvta.to.global.u64 	%rd11, %rd3;
	add.s64 	%rd12, %rd11, %rd8;
	ld.global.nc.f32 	%f3, [%rd12];
	mul.f32 	%f4, %f3, %f3;
	fma.rn.f32 	%f5, %f2, %f2, %f4;
	cvta.to.global.u64 	%rd13, %rd4;
	add.s64 	%rd14, %rd13, %rd8;
	ld.global.nc.f32 	%f6, [%rd14];
	fma.rn.f32 	%f7, %f6, %f6, %f5;
	cvt.f64.f32 	%fd3, %f7;
	mul.f64 	%fd4, %fd2, %fd3;
	cvt.rn.f32.f64 	%f8, %fd4;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd8;
	st.global.f32 	[%rd16], %f8;

$L__BB0_2:
	ret;

}

`
)
