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

// CUDA handle for RelMaxVecDiff kernel
var RelMaxVecDiff_code cu.Function

// Stores the arguments for RelMaxVecDiff kernel invocation
type RelMaxVecDiff_args_t struct {
	arg_out unsafe.Pointer
	arg_x1  unsafe.Pointer
	arg_y1  unsafe.Pointer
	arg_z1  unsafe.Pointer
	arg_x2  unsafe.Pointer
	arg_y2  unsafe.Pointer
	arg_z2  unsafe.Pointer
	arg_Nx  int
	arg_Ny  int
	arg_Nz  int
	argptr  [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for RelMaxVecDiff kernel invocation
var RelMaxVecDiff_args RelMaxVecDiff_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	RelMaxVecDiff_args.argptr[0] = unsafe.Pointer(&RelMaxVecDiff_args.arg_out)
	RelMaxVecDiff_args.argptr[1] = unsafe.Pointer(&RelMaxVecDiff_args.arg_x1)
	RelMaxVecDiff_args.argptr[2] = unsafe.Pointer(&RelMaxVecDiff_args.arg_y1)
	RelMaxVecDiff_args.argptr[3] = unsafe.Pointer(&RelMaxVecDiff_args.arg_z1)
	RelMaxVecDiff_args.argptr[4] = unsafe.Pointer(&RelMaxVecDiff_args.arg_x2)
	RelMaxVecDiff_args.argptr[5] = unsafe.Pointer(&RelMaxVecDiff_args.arg_y2)
	RelMaxVecDiff_args.argptr[6] = unsafe.Pointer(&RelMaxVecDiff_args.arg_z2)
	RelMaxVecDiff_args.argptr[7] = unsafe.Pointer(&RelMaxVecDiff_args.arg_Nx)
	RelMaxVecDiff_args.argptr[8] = unsafe.Pointer(&RelMaxVecDiff_args.arg_Ny)
	RelMaxVecDiff_args.argptr[9] = unsafe.Pointer(&RelMaxVecDiff_args.arg_Nz)
}

// Wrapper for RelMaxVecDiff CUDA kernel, asynchronous.
func k_RelMaxVecDiff_async(out unsafe.Pointer, x1 unsafe.Pointer, y1 unsafe.Pointer, z1 unsafe.Pointer, x2 unsafe.Pointer, y2 unsafe.Pointer, z2 unsafe.Pointer, Nx int, Ny int, Nz int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("RelMaxVecDiff")
	}

	RelMaxVecDiff_args.Lock()
	defer RelMaxVecDiff_args.Unlock()

	if RelMaxVecDiff_code == 0 {
		RelMaxVecDiff_code = fatbinLoad(RelMaxVecDiff_map, "RelMaxVecDiff")
	}

	RelMaxVecDiff_args.arg_out = out
	RelMaxVecDiff_args.arg_x1 = x1
	RelMaxVecDiff_args.arg_y1 = y1
	RelMaxVecDiff_args.arg_z1 = z1
	RelMaxVecDiff_args.arg_x2 = x2
	RelMaxVecDiff_args.arg_y2 = y2
	RelMaxVecDiff_args.arg_z2 = z2
	RelMaxVecDiff_args.arg_Nx = Nx
	RelMaxVecDiff_args.arg_Ny = Ny
	RelMaxVecDiff_args.arg_Nz = Nz

	args := RelMaxVecDiff_args.argptr[:]
	cu.LaunchKernel(RelMaxVecDiff_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("RelMaxVecDiff")
	}
}

// maps compute capability on PTX code for RelMaxVecDiff kernel.
var RelMaxVecDiff_map = map[int]string{0: "",
	50: RelMaxVecDiff_ptx_50,
	52: RelMaxVecDiff_ptx_52,
	53: RelMaxVecDiff_ptx_53,
	60: RelMaxVecDiff_ptx_60,
	61: RelMaxVecDiff_ptx_61,
	62: RelMaxVecDiff_ptx_62,
	70: RelMaxVecDiff_ptx_70,
	72: RelMaxVecDiff_ptx_72,
	75: RelMaxVecDiff_ptx_75,
	80: RelMaxVecDiff_ptx_80}

// RelMaxVecDiff PTX code for various compute capabilities.
const (
	RelMaxVecDiff_ptx_50 = `
.version 8.2
.target sm_50
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_52 = `
.version 8.2
.target sm_52
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_53 = `
.version 8.2
.target sm_53
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_60 = `
.version 8.2
.target sm_60
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_61 = `
.version 8.2
.target sm_61
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_62 = `
.version 8.2
.target sm_62
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_70 = `
.version 8.2
.target sm_70
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_72 = `
.version 8.2
.target sm_72
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_75 = `
.version 8.2
.target sm_75
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
	RelMaxVecDiff_ptx_80 = `
.version 8.2
.target sm_80
.address_size 64

	// .globl	RelMaxVecDiff

.visible .entry RelMaxVecDiff(
	.param .u64 RelMaxVecDiff_param_0,
	.param .u64 RelMaxVecDiff_param_1,
	.param .u64 RelMaxVecDiff_param_2,
	.param .u64 RelMaxVecDiff_param_3,
	.param .u64 RelMaxVecDiff_param_4,
	.param .u64 RelMaxVecDiff_param_5,
	.param .u64 RelMaxVecDiff_param_6,
	.param .u32 RelMaxVecDiff_param_7,
	.param .u32 RelMaxVecDiff_param_8,
	.param .u32 RelMaxVecDiff_param_9
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [RelMaxVecDiff_param_0];
	ld.param.u32 	%r4, [RelMaxVecDiff_param_7];
	ld.param.u32 	%r5, [RelMaxVecDiff_param_8];
	ld.param.u32 	%r6, [RelMaxVecDiff_param_9];
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

	cvta.to.global.u64 	%rd2, %rd1;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	mul.wide.s32 	%rd3, %r17, 4;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u32 	%r18, 1084227584;
	st.global.u32 	[%rd4], %r18;

$L__BB0_2:
	ret;

}

`
)
