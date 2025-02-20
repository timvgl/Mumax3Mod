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

// CUDA handle for setTheta kernel
var setTheta_code cu.Function

// Stores the arguments for setTheta kernel invocation
type setTheta_args_t struct {
	arg_theta unsafe.Pointer
	arg_mz    unsafe.Pointer
	arg_Nx    int
	arg_Ny    int
	arg_Nz    int
	argptr    [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for setTheta kernel invocation
var setTheta_args setTheta_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	setTheta_args.argptr[0] = unsafe.Pointer(&setTheta_args.arg_theta)
	setTheta_args.argptr[1] = unsafe.Pointer(&setTheta_args.arg_mz)
	setTheta_args.argptr[2] = unsafe.Pointer(&setTheta_args.arg_Nx)
	setTheta_args.argptr[3] = unsafe.Pointer(&setTheta_args.arg_Ny)
	setTheta_args.argptr[4] = unsafe.Pointer(&setTheta_args.arg_Nz)
}

// Wrapper for setTheta CUDA kernel, asynchronous.
func k_setTheta_async(theta unsafe.Pointer, mz unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("setTheta")
	}

	setTheta_args.Lock()
	defer setTheta_args.Unlock()

	if setTheta_code == 0 {
		setTheta_code = fatbinLoad(setTheta_map, "setTheta")
	}

	setTheta_args.arg_theta = theta
	setTheta_args.arg_mz = mz
	setTheta_args.arg_Nx = Nx
	setTheta_args.arg_Ny = Ny
	setTheta_args.arg_Nz = Nz

	args := setTheta_args.argptr[:]
	cu.LaunchKernel(setTheta_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("setTheta")
	}
}

// maps compute capability on PTX code for setTheta kernel.
var setTheta_map = map[int]string{0: "",
	50: setTheta_ptx_50,
	52: setTheta_ptx_52,
	53: setTheta_ptx_53,
	60: setTheta_ptx_60,
	61: setTheta_ptx_61,
	62: setTheta_ptx_62,
	70: setTheta_ptx_70,
	72: setTheta_ptx_72,
	75: setTheta_ptx_75,
	80: setTheta_ptx_80}

// setTheta PTX code for various compute capabilities.
const (
	setTheta_ptx_50 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_50
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_52 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_52
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_53 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_53
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_60 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_60
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_61 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_61
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_62 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_62
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_70 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_70
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_72 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_72
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_75 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_75
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
	setTheta_ptx_80 = `
<<<<<<< HEAD
.version 8.5
=======
.version 8.4
>>>>>>> origin/region_solver
.target sm_80
.address_size 64

	// .globl	setTheta

.visible .entry setTheta(
	.param .u64 setTheta_param_0,
	.param .u64 setTheta_param_1,
	.param .u32 setTheta_param_2,
	.param .u32 setTheta_param_3,
	.param .u32 setTheta_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<22>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [setTheta_param_0];
	ld.param.u64 	%rd2, [setTheta_param_1];
	ld.param.u32 	%r4, [setTheta_param_2];
	ld.param.u32 	%r5, [setTheta_param_3];
	ld.param.u32 	%r6, [setTheta_param_4];
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
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd4, %r17, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.nc.f32 	%f1, [%rd5];
	abs.f32 	%f2, %f1;
	neg.f32 	%f3, %f2;
	mov.f32 	%f4, 0f3F000000;
	fma.rn.f32 	%f5, %f4, %f3, %f4;
	rsqrt.approx.ftz.f32 	%f6, %f5;
	mul.f32 	%f7, %f5, %f6;
	mul.f32 	%f8, %f6, 0f3F000000;
	neg.f32 	%f9, %f7;
	fma.rn.f32 	%f10, %f9, %f8, %f4;
	fma.rn.f32 	%f11, %f7, %f10, %f7;
	setp.eq.f32 	%p6, %f2, 0f3F800000;
	selp.f32 	%f12, 0f00000000, %f11, %p6;
	setp.gt.f32 	%p7, %f2, 0f3F0F5C29;
	selp.f32 	%f13, %f12, %f2, %p7;
	mov.b32 	%r18, %f13;
	mov.b32 	%r19, %f1;
	and.b32  	%r20, %r19, -2147483648;
	or.b32  	%r21, %r20, %r18;
	mov.b32 	%f14, %r21;
	mul.f32 	%f15, %f14, %f14;
	mov.f32 	%f16, 0f3C8B1ABB;
	mov.f32 	%f17, 0f3D10ECEF;
	fma.rn.f32 	%f18, %f17, %f15, %f16;
	mov.f32 	%f19, 0f3CFC028C;
	fma.rn.f32 	%f20, %f18, %f15, %f19;
	mov.f32 	%f21, 0f3D372139;
	fma.rn.f32 	%f22, %f20, %f15, %f21;
	mov.f32 	%f23, 0f3D9993DB;
	fma.rn.f32 	%f24, %f22, %f15, %f23;
	mov.f32 	%f25, 0f3E2AAAC6;
	fma.rn.f32 	%f26, %f24, %f15, %f25;
	mul.f32 	%f27, %f26, %f15;
	fma.rn.f32 	%f28, %f27, %f14, %f14;
	neg.f32 	%f29, %f28;
	selp.f32 	%f30, %f28, %f29, %p7;
	mov.f32 	%f31, 0f3FD774EB;
	mov.f32 	%f32, 0f3F6EE581;
	fma.rn.f32 	%f33, %f32, %f31, %f30;
	setp.gt.f32 	%p8, %f1, 0f3F0F5C29;
	selp.f32 	%f34, %f28, %f33, %p8;
	add.f32 	%f35, %f34, %f34;
	selp.f32 	%f36, %f35, %f34, %p7;
	cvta.to.global.u64 	%rd6, %rd1;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f36;

$L__BB0_2:
	ret;

}

`
)
