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

// CUDA handle for CopyPartKernel kernel
var CopyPartKernel_code cu.Function

// Stores the arguments for CopyPartKernel kernel invocation
type CopyPartKernel_args_t struct {
	arg_dst        unsafe.Pointer
	arg_src        unsafe.Pointer
	arg_xStart_src int
	arg_yStart_src int
	arg_zStart_src int
	arg_fStart_src int
	arg_xCount     int
	arg_yCount     int
	arg_zCount     int
	arg_fCount     int
	arg_xStart_dst int
	arg_yStart_dst int
	arg_zStart_dst int
	arg_fStart_dst int
	arg_src_dim_x  int
	arg_src_dim_y  int
	arg_src_dim_z  int
	arg_dst_dim_x  int
	arg_dst_dim_y  int
	arg_dst_dim_z  int
	argptr         [20]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for CopyPartKernel kernel invocation
var CopyPartKernel_args CopyPartKernel_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	CopyPartKernel_args.argptr[0] = unsafe.Pointer(&CopyPartKernel_args.arg_dst)
	CopyPartKernel_args.argptr[1] = unsafe.Pointer(&CopyPartKernel_args.arg_src)
	CopyPartKernel_args.argptr[2] = unsafe.Pointer(&CopyPartKernel_args.arg_xStart_src)
	CopyPartKernel_args.argptr[3] = unsafe.Pointer(&CopyPartKernel_args.arg_yStart_src)
	CopyPartKernel_args.argptr[4] = unsafe.Pointer(&CopyPartKernel_args.arg_zStart_src)
	CopyPartKernel_args.argptr[5] = unsafe.Pointer(&CopyPartKernel_args.arg_fStart_src)
	CopyPartKernel_args.argptr[6] = unsafe.Pointer(&CopyPartKernel_args.arg_xCount)
	CopyPartKernel_args.argptr[7] = unsafe.Pointer(&CopyPartKernel_args.arg_yCount)
	CopyPartKernel_args.argptr[8] = unsafe.Pointer(&CopyPartKernel_args.arg_zCount)
	CopyPartKernel_args.argptr[9] = unsafe.Pointer(&CopyPartKernel_args.arg_fCount)
	CopyPartKernel_args.argptr[10] = unsafe.Pointer(&CopyPartKernel_args.arg_xStart_dst)
	CopyPartKernel_args.argptr[11] = unsafe.Pointer(&CopyPartKernel_args.arg_yStart_dst)
	CopyPartKernel_args.argptr[12] = unsafe.Pointer(&CopyPartKernel_args.arg_zStart_dst)
	CopyPartKernel_args.argptr[13] = unsafe.Pointer(&CopyPartKernel_args.arg_fStart_dst)
	CopyPartKernel_args.argptr[14] = unsafe.Pointer(&CopyPartKernel_args.arg_src_dim_x)
	CopyPartKernel_args.argptr[15] = unsafe.Pointer(&CopyPartKernel_args.arg_src_dim_y)
	CopyPartKernel_args.argptr[16] = unsafe.Pointer(&CopyPartKernel_args.arg_src_dim_z)
	CopyPartKernel_args.argptr[17] = unsafe.Pointer(&CopyPartKernel_args.arg_dst_dim_x)
	CopyPartKernel_args.argptr[18] = unsafe.Pointer(&CopyPartKernel_args.arg_dst_dim_y)
	CopyPartKernel_args.argptr[19] = unsafe.Pointer(&CopyPartKernel_args.arg_dst_dim_z)
}

// Wrapper for CopyPartKernel CUDA kernel, asynchronous.
func k_CopyPartKernel_async(dst unsafe.Pointer, src unsafe.Pointer, xStart_src int, yStart_src int, zStart_src int, fStart_src int, xCount int, yCount int, zCount int, fCount int, xStart_dst int, yStart_dst int, zStart_dst int, fStart_dst int, src_dim_x int, src_dim_y int, src_dim_z int, dst_dim_x int, dst_dim_y int, dst_dim_z int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("CopyPartKernel")
	}

	CopyPartKernel_args.Lock()
	defer CopyPartKernel_args.Unlock()

	if CopyPartKernel_code == 0 {
		CopyPartKernel_code = fatbinLoad(CopyPartKernel_map, "CopyPartKernel")
	}

	CopyPartKernel_args.arg_dst = dst
	CopyPartKernel_args.arg_src = src
	CopyPartKernel_args.arg_xStart_src = xStart_src
	CopyPartKernel_args.arg_yStart_src = yStart_src
	CopyPartKernel_args.arg_zStart_src = zStart_src
	CopyPartKernel_args.arg_fStart_src = fStart_src
	CopyPartKernel_args.arg_xCount = xCount
	CopyPartKernel_args.arg_yCount = yCount
	CopyPartKernel_args.arg_zCount = zCount
	CopyPartKernel_args.arg_fCount = fCount
	CopyPartKernel_args.arg_xStart_dst = xStart_dst
	CopyPartKernel_args.arg_yStart_dst = yStart_dst
	CopyPartKernel_args.arg_zStart_dst = zStart_dst
	CopyPartKernel_args.arg_fStart_dst = fStart_dst
	CopyPartKernel_args.arg_src_dim_x = src_dim_x
	CopyPartKernel_args.arg_src_dim_y = src_dim_y
	CopyPartKernel_args.arg_src_dim_z = src_dim_z
	CopyPartKernel_args.arg_dst_dim_x = dst_dim_x
	CopyPartKernel_args.arg_dst_dim_y = dst_dim_y
	CopyPartKernel_args.arg_dst_dim_z = dst_dim_z

	args := CopyPartKernel_args.argptr[:]
	cu.LaunchKernel(CopyPartKernel_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("CopyPartKernel")
	}
}

// maps compute capability on PTX code for CopyPartKernel kernel.
var CopyPartKernel_map = map[int]string{0: "",
	50: CopyPartKernel_ptx_50,
	52: CopyPartKernel_ptx_52,
	53: CopyPartKernel_ptx_53,
	60: CopyPartKernel_ptx_60,
	61: CopyPartKernel_ptx_61,
	62: CopyPartKernel_ptx_62,
	70: CopyPartKernel_ptx_70,
	72: CopyPartKernel_ptx_72,
	75: CopyPartKernel_ptx_75,
	80: CopyPartKernel_ptx_80}

// CopyPartKernel PTX code for various compute capabilities.
const (
	CopyPartKernel_ptx_50 = `
.version 8.5
.target sm_50
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_52 = `
.version 8.5
.target sm_52
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_53 = `
.version 8.5
.target sm_53
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_60 = `
.version 8.5
.target sm_60
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_61 = `
.version 8.5
.target sm_61
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_62 = `
.version 8.5
.target sm_62
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_70 = `
.version 8.5
.target sm_70
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_72 = `
.version 8.5
.target sm_72
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_75 = `
.version 8.5
.target sm_75
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
	CopyPartKernel_ptx_80 = `
.version 8.5
.target sm_80
.address_size 64

	// .globl	CopyPartKernel

.visible .entry CopyPartKernel(
	.param .u64 CopyPartKernel_param_0,
	.param .u64 CopyPartKernel_param_1,
	.param .u32 CopyPartKernel_param_2,
	.param .u32 CopyPartKernel_param_3,
	.param .u32 CopyPartKernel_param_4,
	.param .u32 CopyPartKernel_param_5,
	.param .u32 CopyPartKernel_param_6,
	.param .u32 CopyPartKernel_param_7,
	.param .u32 CopyPartKernel_param_8,
	.param .u32 CopyPartKernel_param_9,
	.param .u32 CopyPartKernel_param_10,
	.param .u32 CopyPartKernel_param_11,
	.param .u32 CopyPartKernel_param_12,
	.param .u32 CopyPartKernel_param_13,
	.param .u32 CopyPartKernel_param_14,
	.param .u32 CopyPartKernel_param_15,
	.param .u32 CopyPartKernel_param_16,
	.param .u32 CopyPartKernel_param_17,
	.param .u32 CopyPartKernel_param_18,
	.param .u32 CopyPartKernel_param_19
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<52>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [CopyPartKernel_param_0];
	ld.param.u64 	%rd2, [CopyPartKernel_param_1];
	ld.param.u32 	%r2, [CopyPartKernel_param_2];
	ld.param.u32 	%r3, [CopyPartKernel_param_3];
	ld.param.u32 	%r4, [CopyPartKernel_param_4];
	ld.param.u32 	%r5, [CopyPartKernel_param_5];
	ld.param.u32 	%r6, [CopyPartKernel_param_6];
	ld.param.u32 	%r7, [CopyPartKernel_param_7];
	ld.param.u32 	%r8, [CopyPartKernel_param_8];
	ld.param.u32 	%r19, [CopyPartKernel_param_9];
	ld.param.u32 	%r9, [CopyPartKernel_param_10];
	ld.param.u32 	%r10, [CopyPartKernel_param_11];
	ld.param.u32 	%r11, [CopyPartKernel_param_12];
	ld.param.u32 	%r12, [CopyPartKernel_param_13];
	ld.param.u32 	%r13, [CopyPartKernel_param_14];
	ld.param.u32 	%r14, [CopyPartKernel_param_15];
	ld.param.u32 	%r15, [CopyPartKernel_param_16];
	ld.param.u32 	%r16, [CopyPartKernel_param_17];
	ld.param.u32 	%r17, [CopyPartKernel_param_18];
	ld.param.u32 	%r18, [CopyPartKernel_param_19];
	mul.lo.s32 	%r20, %r7, %r6;
	mul.lo.s32 	%r21, %r20, %r8;
	mul.lo.s32 	%r22, %r21, %r19;
	mov.u32 	%r23, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r25, %ctaid.x;
	mad.lo.s32 	%r26, %r24, %r23, %r25;
	mov.u32 	%r27, %ntid.x;
	mov.u32 	%r28, %tid.x;
	mad.lo.s32 	%r1, %r26, %r27, %r28;
	setp.ge.s32 	%p1, %r1, %r22;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd2;
	div.s32 	%r29, %r1, %r6;
	div.s32 	%r30, %r29, %r7;
	mul.lo.s32 	%r31, %r30, %r7;
	sub.s32 	%r32, %r29, %r31;
	div.s32 	%r33, %r30, %r8;
	mul.lo.s32 	%r34, %r33, %r8;
	sub.s32 	%r35, %r30, %r34;
	add.s32 	%r36, %r33, %r5;
	add.s32 	%r37, %r35, %r4;
	mad.lo.s32 	%r38, %r36, %r15, %r37;
	add.s32 	%r39, %r32, %r3;
	mad.lo.s32 	%r40, %r38, %r14, %r39;
	mul.lo.s32 	%r41, %r29, %r6;
	sub.s32 	%r42, %r1, %r41;
	add.s32 	%r43, %r42, %r2;
	mad.lo.s32 	%r44, %r40, %r13, %r43;
	add.s32 	%r45, %r33, %r12;
	add.s32 	%r46, %r35, %r11;
	mad.lo.s32 	%r47, %r45, %r18, %r46;
	add.s32 	%r48, %r32, %r10;
	mad.lo.s32 	%r49, %r47, %r17, %r48;
	add.s32 	%r50, %r42, %r9;
	mad.lo.s32 	%r51, %r49, %r16, %r50;
	mul.wide.s32 	%rd4, %r44, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f1, [%rd5];
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r51, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f1;

$L__BB0_2:
	ret;

}

`
)
