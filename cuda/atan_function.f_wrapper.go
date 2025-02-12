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

// CUDA handle for atanGovaluate kernel
var atanGovaluate_code cu.Function

// Stores the arguments for atanGovaluate kernel invocation
type atanGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for atanGovaluate kernel invocation
var atanGovaluate_args atanGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	atanGovaluate_args.argptr[0] = unsafe.Pointer(&atanGovaluate_args.arg_value)
	atanGovaluate_args.argptr[1] = unsafe.Pointer(&atanGovaluate_args.arg_N)
}

// Wrapper for atanGovaluate CUDA kernel, asynchronous.
func k_atanGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("atanGovaluate")
	}

	atanGovaluate_args.Lock()
	defer atanGovaluate_args.Unlock()

	if atanGovaluate_code == 0 {
		atanGovaluate_code = fatbinLoad(atanGovaluate_map, "atanGovaluate")
	}

	atanGovaluate_args.arg_value = value
	atanGovaluate_args.arg_N = N

	args := atanGovaluate_args.argptr[:]
	cu.LaunchKernel(atanGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("atanGovaluate")
	}
}

// maps compute capability on PTX code for atanGovaluate kernel.
var atanGovaluate_map = map[int]string{0: "",
	50: atanGovaluate_ptx_50,
	52: atanGovaluate_ptx_52,
	53: atanGovaluate_ptx_53,
	60: atanGovaluate_ptx_60,
	61: atanGovaluate_ptx_61,
	62: atanGovaluate_ptx_62,
	70: atanGovaluate_ptx_70,
	72: atanGovaluate_ptx_72,
	75: atanGovaluate_ptx_75,
	80: atanGovaluate_ptx_80}

// atanGovaluate PTX code for various compute capabilities.
const (
	atanGovaluate_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
	atanGovaluate_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	atanGovaluate

.visible .entry atanGovaluate(
	.param .u64 atanGovaluate_param_0,
	.param .u32 atanGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<31>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [atanGovaluate_param_0];
	ld.param.u32 	%r2, [atanGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	abs.f32 	%f2, %f1;
	setp.leu.f32 	%p2, %f2, 0f3F800000;
	setp.gt.f32 	%p3, %f2, 0f3F800000;
	rcp.approx.ftz.f32 	%f6, %f2;
	selp.f32 	%f7, %f6, %f2, %p3;
	mul.f32 	%f8, %f7, %f7;
	mov.f32 	%f9, 0fBC6BE14F;
	mov.f32 	%f10, 0f3B2090AA;
	fma.rn.f32 	%f11, %f10, %f8, %f9;
	mov.f32 	%f12, 0f3D23397E;
	fma.rn.f32 	%f13, %f11, %f8, %f12;
	mov.f32 	%f14, 0fBD948A7A;
	fma.rn.f32 	%f15, %f13, %f8, %f14;
	mov.f32 	%f16, 0f3DD76B21;
	fma.rn.f32 	%f17, %f15, %f8, %f16;
	mov.f32 	%f18, 0fBE111E88;
	fma.rn.f32 	%f19, %f17, %f8, %f18;
	mov.f32 	%f20, 0f3E4CAF60;
	fma.rn.f32 	%f21, %f19, %f8, %f20;
	mov.f32 	%f22, 0fBEAAAA27;
	fma.rn.f32 	%f23, %f21, %f8, %f22;
	mul.f32 	%f24, %f8, %f23;
	fma.rn.f32 	%f30, %f24, %f7, %f7;
	@%p2 bra 	$L__BB0_3;

	neg.f32 	%f25, %f30;
	mov.f32 	%f26, 0f3FD774EB;
	mov.f32 	%f27, 0f3F6EE581;
	fma.rn.f32 	%f30, %f27, %f26, %f25;

$L__BB0_3:
	mov.b32 	%r9, %f1;
	and.b32  	%r10, %r9, -2147483648;
	mov.b32 	%r11, %f30;
	or.b32  	%r12, %r10, %r11;
	mov.b32 	%f28, %r12;
	setp.le.f32 	%p4, %f2, 0f7F800000;
	selp.f32 	%f29, %f28, %f30, %p4;
	st.global.f32 	[%rd1], %f29;

$L__BB0_4:
	ret;

}

`
)
