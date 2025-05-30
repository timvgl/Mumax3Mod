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

// CUDA handle for erfcGovaluate kernel
var erfcGovaluate_code cu.Function

// Stores the arguments for erfcGovaluate kernel invocation
type erfcGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for erfcGovaluate kernel invocation
var erfcGovaluate_args erfcGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	erfcGovaluate_args.argptr[0] = unsafe.Pointer(&erfcGovaluate_args.arg_value)
	erfcGovaluate_args.argptr[1] = unsafe.Pointer(&erfcGovaluate_args.arg_N)
}

// Wrapper for erfcGovaluate CUDA kernel, asynchronous.
func k_erfcGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("erfcGovaluate")
	}

	erfcGovaluate_args.Lock()
	defer erfcGovaluate_args.Unlock()

	if erfcGovaluate_code == 0 {
		erfcGovaluate_code = fatbinLoad(erfcGovaluate_map, "erfcGovaluate")
	}

	erfcGovaluate_args.arg_value = value
	erfcGovaluate_args.arg_N = N

	args := erfcGovaluate_args.argptr[:]
	cu.LaunchKernel(erfcGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("erfcGovaluate")
	}
}

// maps compute capability on PTX code for erfcGovaluate kernel.
var erfcGovaluate_map = map[int]string{0: "",
	50: erfcGovaluate_ptx_50,
	52: erfcGovaluate_ptx_52,
	53: erfcGovaluate_ptx_53,
	60: erfcGovaluate_ptx_60,
	61: erfcGovaluate_ptx_61,
	62: erfcGovaluate_ptx_62,
	70: erfcGovaluate_ptx_70,
	72: erfcGovaluate_ptx_72,
	75: erfcGovaluate_ptx_75,
	80: erfcGovaluate_ptx_80}

// erfcGovaluate PTX code for various compute capabilities.
const (
	erfcGovaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
	erfcGovaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	erfcGovaluate

.visible .entry erfcGovaluate(
	.param .u64 erfcGovaluate_param_0,
	.param .u32 erfcGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<67>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [erfcGovaluate_param_0];
	ld.param.u32 	%r2, [erfcGovaluate_param_1];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	abs.f32 	%f2, %f1;
	add.f32 	%f3, %f2, 0fC0800000;
	mov.f32 	%f4, 0fC0800000;
	add.f32 	%f5, %f2, 0f40800000;
	rcp.approx.ftz.f32 	%f6, %f5;
	mul.rn.f32 	%f7, %f3, %f6;
	add.f32 	%f8, %f7, 0f3F800000;
	mov.f32 	%f9, 0f3F800000;
	fma.rn.f32 	%f10, %f4, %f8, %f2;
	neg.f32 	%f11, %f7;
	fma.rn.f32 	%f12, %f11, %f2, %f10;
	fma.rn.f32 	%f13, %f6, %f12, %f7;
	mov.f32 	%f14, 0f3BE6E05B;
	mov.f32 	%f15, 0f3A69A091;
	fma.rn.f32 	%f16, %f15, %f13, %f14;
	mov.f32 	%f17, 0fBC81FB4B;
	fma.rn.f32 	%f18, %f16, %f13, %f17;
	mov.f32 	%f19, 0f3D15373B;
	fma.rn.f32 	%f20, %f18, %f13, %f19;
	mov.f32 	%f21, 0fBD887C5A;
	fma.rn.f32 	%f22, %f20, %f13, %f21;
	mov.f32 	%f23, 0f3DC021D5;
	fma.rn.f32 	%f24, %f22, %f13, %f23;
	mov.f32 	%f25, 0fBDCED424;
	fma.rn.f32 	%f26, %f24, %f13, %f25;
	mov.f32 	%f27, 0f3D8B74DE;
	fma.rn.f32 	%f28, %f26, %f13, %f27;
	mov.f32 	%f29, 0f3C7BF170;
	fma.rn.f32 	%f30, %f28, %f13, %f29;
	mov.f32 	%f31, 0fBE0EF8D4;
	fma.rn.f32 	%f32, %f30, %f13, %f31;
	mov.f32 	%f33, 0f3F9DD2C9;
	fma.rn.f32 	%f34, %f32, %f13, %f33;
	mov.f32 	%f35, 0f40000000;
	fma.rn.f32 	%f36, %f35, %f2, %f9;
	rcp.approx.ftz.f32 	%f37, %f36;
	mul.rn.f32 	%f38, %f34, %f37;
	mul.f32 	%f39, %f38, 0fC0000000;
	fma.rn.f32 	%f40, %f2, %f39, %f34;
	sub.f32 	%f41, %f40, %f38;
	fma.rn.f32 	%f42, %f41, %f37, %f38;
	mul.f32 	%f43, %f2, %f2;
	neg.f32 	%f44, %f43;
	mov.f32 	%f45, 0f3FB8AA3B;
	mul.rn.f32 	%f46, %f44, %f45;
	cvt.rzi.f32.f32 	%f47, %f46;
	abs.f32 	%f48, %f47;
	setp.gt.f32 	%p2, %f48, 0f42FC0000;
	mov.b32 	%r9, %f47;
	and.b32  	%r10, %r9, -2147483648;
	or.b32  	%r11, %r10, 1123811328;
	mov.b32 	%f49, %r11;
	selp.f32 	%f50, %f49, %f47, %p2;
	mov.f32 	%f51, 0fBF317218;
	fma.rn.f32 	%f52, %f50, %f51, %f44;
	mov.f32 	%f53, 0f3102E308;
	fma.rn.f32 	%f54, %f50, %f53, %f52;
	mul.f32 	%f55, %f54, 0f3FB8AA3B;
	add.f32 	%f56, %f50, 0f4B40007F;
	mov.b32 	%r12, %f56;
	shl.b32 	%r13, %r12, 23;
	mov.b32 	%f57, %r13;
	ex2.approx.ftz.f32 	%f58, %f55;
	mul.f32 	%f59, %f58, %f57;
	neg.f32 	%f60, %f2;
	fma.rn.f32 	%f61, %f60, %f2, %f43;
	fma.rn.f32 	%f62, %f59, %f61, %f59;
	mul.f32 	%f63, %f42, %f62;
	setp.gt.f32 	%p3, %f2, 0f4120E148;
	selp.f32 	%f64, 0f00000000, %f63, %p3;
	setp.lt.f32 	%p4, %f1, 0f00000000;
	sub.f32 	%f65, %f35, %f64;
	selp.f32 	%f66, %f65, %f64, %p4;
	st.global.f32 	[%rd4], %f66;

$L__BB0_2:
	ret;

}

`
)
