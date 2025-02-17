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

// CUDA handle for Normstress kernel
var Normstress_code cu.Function

// Stores the arguments for Normstress kernel invocation
type Normstress_args_t struct {
	arg_sx     unsafe.Pointer
	arg_sy     unsafe.Pointer
	arg_sz     unsafe.Pointer
	arg_ex     unsafe.Pointer
	arg_ey     unsafe.Pointer
	arg_ez     unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	arg_C1_    unsafe.Pointer
	arg_C1_mul float32
	arg_C2_    unsafe.Pointer
	arg_C2_mul float32
	argptr     [13]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for Normstress kernel invocation
var Normstress_args Normstress_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	Normstress_args.argptr[0] = unsafe.Pointer(&Normstress_args.arg_sx)
	Normstress_args.argptr[1] = unsafe.Pointer(&Normstress_args.arg_sy)
	Normstress_args.argptr[2] = unsafe.Pointer(&Normstress_args.arg_sz)
	Normstress_args.argptr[3] = unsafe.Pointer(&Normstress_args.arg_ex)
	Normstress_args.argptr[4] = unsafe.Pointer(&Normstress_args.arg_ey)
	Normstress_args.argptr[5] = unsafe.Pointer(&Normstress_args.arg_ez)
	Normstress_args.argptr[6] = unsafe.Pointer(&Normstress_args.arg_Nx)
	Normstress_args.argptr[7] = unsafe.Pointer(&Normstress_args.arg_Ny)
	Normstress_args.argptr[8] = unsafe.Pointer(&Normstress_args.arg_Nz)
	Normstress_args.argptr[9] = unsafe.Pointer(&Normstress_args.arg_C1_)
	Normstress_args.argptr[10] = unsafe.Pointer(&Normstress_args.arg_C1_mul)
	Normstress_args.argptr[11] = unsafe.Pointer(&Normstress_args.arg_C2_)
	Normstress_args.argptr[12] = unsafe.Pointer(&Normstress_args.arg_C2_mul)
}

// Wrapper for Normstress CUDA kernel, asynchronous.
func k_Normstress_async(sx unsafe.Pointer, sy unsafe.Pointer, sz unsafe.Pointer, ex unsafe.Pointer, ey unsafe.Pointer, ez unsafe.Pointer, Nx int, Ny int, Nz int, C1_ unsafe.Pointer, C1_mul float32, C2_ unsafe.Pointer, C2_mul float32, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("Normstress")
	}

	Normstress_args.Lock()
	defer Normstress_args.Unlock()

	if Normstress_code == 0 {
		Normstress_code = fatbinLoad(Normstress_map, "Normstress")
	}

	Normstress_args.arg_sx = sx
	Normstress_args.arg_sy = sy
	Normstress_args.arg_sz = sz
	Normstress_args.arg_ex = ex
	Normstress_args.arg_ey = ey
	Normstress_args.arg_ez = ez
	Normstress_args.arg_Nx = Nx
	Normstress_args.arg_Ny = Ny
	Normstress_args.arg_Nz = Nz
	Normstress_args.arg_C1_ = C1_
	Normstress_args.arg_C1_mul = C1_mul
	Normstress_args.arg_C2_ = C2_
	Normstress_args.arg_C2_mul = C2_mul

	args := Normstress_args.argptr[:]
	cu.LaunchKernel(Normstress_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("Normstress")
	}
}

// maps compute capability on PTX code for Normstress kernel.
var Normstress_map = map[int]string{0: "",
	50: Normstress_ptx_50,
	52: Normstress_ptx_52,
	53: Normstress_ptx_53,
	60: Normstress_ptx_60,
	61: Normstress_ptx_61,
	62: Normstress_ptx_62,
	70: Normstress_ptx_70,
	72: Normstress_ptx_72,
	75: Normstress_ptx_75,
	80: Normstress_ptx_80}

// Normstress PTX code for various compute capabilities.
const (
	Normstress_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
	Normstress_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	Normstress

.visible .entry Normstress(
	.param .u64 Normstress_param_0,
	.param .u64 Normstress_param_1,
	.param .u64 Normstress_param_2,
	.param .u64 Normstress_param_3,
	.param .u64 Normstress_param_4,
	.param .u64 Normstress_param_5,
	.param .u32 Normstress_param_6,
	.param .u32 Normstress_param_7,
	.param .u32 Normstress_param_8,
	.param .u64 Normstress_param_9,
	.param .f32 Normstress_param_10,
	.param .u64 Normstress_param_11,
	.param .f32 Normstress_param_12
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<23>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<28>;


	ld.param.u64 	%rd1, [Normstress_param_0];
	ld.param.u64 	%rd2, [Normstress_param_1];
	ld.param.u64 	%rd3, [Normstress_param_2];
	ld.param.u64 	%rd4, [Normstress_param_3];
	ld.param.u64 	%rd5, [Normstress_param_4];
	ld.param.u64 	%rd6, [Normstress_param_5];
	ld.param.u32 	%r5, [Normstress_param_6];
	ld.param.u32 	%r6, [Normstress_param_7];
	ld.param.u32 	%r7, [Normstress_param_8];
	ld.param.u64 	%rd7, [Normstress_param_9];
	ld.param.f32 	%f21, [Normstress_param_10];
	ld.param.u64 	%rd8, [Normstress_param_11];
	ld.param.f32 	%f22, [Normstress_param_12];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r12, %r11, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r15, %r14, %r16;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_6;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	$L__BB0_3;

	cvta.to.global.u64 	%rd9, %rd7;
	mul.wide.s32 	%rd10, %r4, 4;
	add.s64 	%rd11, %rd9, %rd10;
	ld.global.nc.f32 	%f7, [%rd11];
	mul.f32 	%f21, %f7, %f21;

$L__BB0_3:
	setp.eq.s64 	%p7, %rd8, 0;
	@%p7 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd12, %rd8;
	mul.wide.s32 	%rd13, %r4, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f8, [%rd14];
	mul.f32 	%f22, %f8, %f22;

$L__BB0_5:
	cvta.to.global.u64 	%rd15, %rd4;
	mul.wide.s32 	%rd16, %r4, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f9, [%rd17];
	cvta.to.global.u64 	%rd18, %rd5;
	add.s64 	%rd19, %rd18, %rd16;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd16;
	ld.global.nc.f32 	%f10, [%rd21];
	ld.global.nc.f32 	%f11, [%rd19];
	add.f32 	%f12, %f11, %f10;
	mul.f32 	%f13, %f22, %f12;
	fma.rn.f32 	%f14, %f21, %f9, %f13;
	cvta.to.global.u64 	%rd22, %rd1;
	add.s64 	%rd23, %rd22, %rd16;
	st.global.f32 	[%rd23], %f14;
	add.f32 	%f15, %f9, %f10;
	mul.f32 	%f16, %f22, %f15;
	fma.rn.f32 	%f17, %f21, %f11, %f16;
	cvta.to.global.u64 	%rd24, %rd2;
	add.s64 	%rd25, %rd24, %rd16;
	st.global.f32 	[%rd25], %f17;
	add.f32 	%f18, %f9, %f11;
	mul.f32 	%f19, %f22, %f18;
	fma.rn.f32 	%f20, %f21, %f10, %f19;
	cvta.to.global.u64 	%rd26, %rd3;
	add.s64 	%rd27, %rd26, %rd16;
	st.global.f32 	[%rd27], %f20;

$L__BB0_6:
	ret;

}

`
)
