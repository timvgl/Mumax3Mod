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

// CUDA handle for log10Govaluate kernel
var log10Govaluate_code cu.Function

// Stores the arguments for log10Govaluate kernel invocation
type log10Govaluate_args_t struct {
	arg_value unsafe.Pointer
	arg_N     int
	argptr    [2]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for log10Govaluate kernel invocation
var log10Govaluate_args log10Govaluate_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	log10Govaluate_args.argptr[0] = unsafe.Pointer(&log10Govaluate_args.arg_value)
	log10Govaluate_args.argptr[1] = unsafe.Pointer(&log10Govaluate_args.arg_N)
}

// Wrapper for log10Govaluate CUDA kernel, asynchronous.
func k_log10Govaluate_async(value unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("log10Govaluate")
	}

	log10Govaluate_args.Lock()
	defer log10Govaluate_args.Unlock()

	if log10Govaluate_code == 0 {
		log10Govaluate_code = fatbinLoad(log10Govaluate_map, "log10Govaluate")
	}

	log10Govaluate_args.arg_value = value
	log10Govaluate_args.arg_N = N

	args := log10Govaluate_args.argptr[:]
	cu.LaunchKernel(log10Govaluate_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("log10Govaluate")
	}
}

// maps compute capability on PTX code for log10Govaluate kernel.
var log10Govaluate_map = map[int]string{0: "",
	50: log10Govaluate_ptx_50,
	52: log10Govaluate_ptx_52,
	53: log10Govaluate_ptx_53,
	60: log10Govaluate_ptx_60,
	61: log10Govaluate_ptx_61,
	62: log10Govaluate_ptx_62,
	70: log10Govaluate_ptx_70,
	72: log10Govaluate_ptx_72,
	75: log10Govaluate_ptx_75,
	80: log10Govaluate_ptx_80}

// log10Govaluate PTX code for various compute capabilities.
const (
	log10Govaluate_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
	log10Govaluate_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	log10Govaluate

.visible .entry log10Govaluate(
	.param .u64 log10Govaluate_param_0,
	.param .u32 log10Govaluate_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [log10Govaluate_param_0];
	ld.param.u32 	%r2, [log10Govaluate_param_1];
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
	ld.global.f32 	%f5, [%rd1];
	setp.lt.f32 	%p2, %f5, 0f00800000;
	mul.f32 	%f6, %f5, 0f4B000000;
	selp.f32 	%f1, %f6, %f5, %p2;
	selp.f32 	%f7, 0fC1B80000, 0f00000000, %p2;
	mov.b32 	%r9, %f1;
	add.s32 	%r10, %r9, -1059760811;
	and.b32  	%r11, %r10, -8388608;
	sub.s32 	%r12, %r9, %r11;
	mov.b32 	%f8, %r12;
	cvt.rn.f32.s32 	%f9, %r11;
	mov.f32 	%f10, 0f34000000;
	fma.rn.f32 	%f11, %f9, %f10, %f7;
	add.f32 	%f12, %f8, 0fBF800000;
	mov.f32 	%f13, 0f3E1039F6;
	mov.f32 	%f14, 0fBE055027;
	fma.rn.f32 	%f15, %f14, %f12, %f13;
	mov.f32 	%f16, 0fBDF8CDCC;
	fma.rn.f32 	%f17, %f15, %f12, %f16;
	mov.f32 	%f18, 0f3E0F2955;
	fma.rn.f32 	%f19, %f17, %f12, %f18;
	mov.f32 	%f20, 0fBE2AD8B9;
	fma.rn.f32 	%f21, %f19, %f12, %f20;
	mov.f32 	%f22, 0f3E4CED0B;
	fma.rn.f32 	%f23, %f21, %f12, %f22;
	mov.f32 	%f24, 0fBE7FFF22;
	fma.rn.f32 	%f25, %f23, %f12, %f24;
	mov.f32 	%f26, 0f3EAAAA78;
	fma.rn.f32 	%f27, %f25, %f12, %f26;
	mov.f32 	%f28, 0fBF000000;
	fma.rn.f32 	%f29, %f27, %f12, %f28;
	mul.f32 	%f30, %f12, %f29;
	fma.rn.f32 	%f31, %f30, %f12, %f12;
	mov.f32 	%f32, 0f3F317218;
	fma.rn.f32 	%f36, %f11, %f32, %f31;
	setp.lt.u32 	%p3, %r9, 2139095040;
	@%p3 bra 	$L__BB0_3;

	mov.f32 	%f33, 0f7F800000;
	fma.rn.f32 	%f36, %f1, %f33, %f33;

$L__BB0_3:
	mul.f32 	%f34, %f36, 0f3EDE5BD9;
	setp.eq.f32 	%p4, %f1, 0f00000000;
	selp.f32 	%f35, 0fFF800000, %f34, %p4;
	st.global.f32 	[%rd1], %f35;

$L__BB0_4:
	ret;

}

`
)
