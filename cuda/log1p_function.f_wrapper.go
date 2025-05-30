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

// CUDA handle for log1pGovaluate kernel
var log1pGovaluate_code cu.Function

// Stores the arguments for log1pGovaluate kernel invocation
type log1pGovaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for log1pGovaluate kernel invocation
var log1pGovaluate_args log1pGovaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	log1pGovaluate_args.argptr[0] = unsafe.Pointer(&log1pGovaluate_args.arg_value)
	log1pGovaluate_args.argptr[1] = unsafe.Pointer(&log1pGovaluate_args.arg_N)
}

// Wrapper for log1pGovaluate CUDA kernel, asynchronous.
func k_log1pGovaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("log1pGovaluate")
	}

	log1pGovaluate_args.Lock()
	defer log1pGovaluate_args.Unlock()

	if log1pGovaluate_code == 0 {
		log1pGovaluate_code = fatbinLoad(log1pGovaluate_map, "log1pGovaluate")
	}

	log1pGovaluate_args.arg_value = value
	log1pGovaluate_args.arg_N = N

	args := log1pGovaluate_args.argptr[:]
	cu.LaunchKernel(log1pGovaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("log1pGovaluate")
	}
}

// maps compute capability on PTX code for log1pGovaluate kernel.
var log1pGovaluate_map = map[int]string{0: "",
	50: log1pGovaluate_ptx_50,
	52: log1pGovaluate_ptx_52,
	53: log1pGovaluate_ptx_53,
	60: log1pGovaluate_ptx_60,
	61: log1pGovaluate_ptx_61,
	62: log1pGovaluate_ptx_62,
	70: log1pGovaluate_ptx_70,
	72: log1pGovaluate_ptx_72,
	75: log1pGovaluate_ptx_75,
	80: log1pGovaluate_ptx_80}

// log1pGovaluate PTX code for various compute capabilities.
const (
	log1pGovaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
	log1pGovaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	log1pGovaluate

.visible .entry log1pGovaluate(
	.param .u64 log1pGovaluate_param_0,
	.param .u32 log1pGovaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<40>;
	.reg .b32 	%r<16>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log1pGovaluate_param_0];
	ld.param.u32 	%r3, [log1pGovaluate_param_1];
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %nctaid.x;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	$L__BB0_6;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd1, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd1];
	mov.f32 	%f7, 0f3F800000;
	add.rz.f32 	%f8, %f1, %f7;
	mov.b32 	%r10, %f8;
	add.s32 	%r11, %r10, -1061158912;
	and.b32  	%r12, %r11, -8388608;
	mov.b32 	%r2, %f1;
	sub.s32 	%r13, %r2, %r12;
	mov.b32 	%f9, %r13;
	mov.u32 	%r14, 1082130432;
	sub.s32 	%r15, %r14, %r12;
	mov.b32 	%f10, %r15;
	mov.f32 	%f11, 0fBF800000;
	mov.f32 	%f12, 0f3E800000;
	fma.rn.f32 	%f13, %f12, %f10, %f11;
	add.f32 	%f14, %f13, %f9;
	cvt.rn.f32.s32 	%f15, %r12;
	mul.f32 	%f16, %f15, 0f34000000;
	mov.f32 	%f17, 0f3DD80012;
	mov.f32 	%f18, 0fBD39BF78;
	fma.rn.f32 	%f19, %f18, %f14, %f17;
	mov.f32 	%f20, 0fBE0778E0;
	fma.rn.f32 	%f21, %f19, %f14, %f20;
	mov.f32 	%f22, 0f3E146475;
	fma.rn.f32 	%f23, %f21, %f14, %f22;
	mov.f32 	%f24, 0fBE2A68DD;
	fma.rn.f32 	%f25, %f23, %f14, %f24;
	mov.f32 	%f26, 0f3E4CAF9E;
	fma.rn.f32 	%f27, %f25, %f14, %f26;
	mov.f32 	%f28, 0fBE800042;
	fma.rn.f32 	%f29, %f27, %f14, %f28;
	mov.f32 	%f30, 0f3EAAAAE6;
	fma.rn.f32 	%f31, %f29, %f14, %f30;
	mov.f32 	%f32, 0fBF000000;
	fma.rn.f32 	%f33, %f31, %f14, %f32;
	mul.f32 	%f34, %f14, %f33;
	fma.rn.f32 	%f35, %f34, %f14, %f14;
	mov.f32 	%f36, 0f3F317218;
	fma.rn.f32 	%f38, %f16, %f36, %f35;
	setp.lt.u32 	%p2, %r2, 2139095040;
	@%p2 bra 	$L__BB0_5;

	setp.lt.s32 	%p3, %r2, -1082130431;
	@%p3 bra 	$L__BB0_4;

	mov.f32 	%f37, 0f7F800000;
	fma.rn.f32 	%f38, %f1, %f37, %f37;

$L__BB0_4:
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f38, 0f80000000, %f38, %p4;

$L__BB0_5:
	st.global.f32 	[%rd1], %f38;

$L__BB0_6:
	ret;

}

`
)
