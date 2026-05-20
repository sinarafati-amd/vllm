import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl


# -----------------------------------------------------------------------------
# Simple utility helpers.
# -----------------------------------------------------------------------------


@gluon.jit
def _pid_grid(pid: int, num_pid_m: int, num_pid_n: int, group_size_m: gl.constexpr = 1):
    if group_size_m == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        group_size_m_runtime = min(num_pid_m - first_pid_m, group_size_m)
        gl.assume(group_size_m_runtime >= 0)
        pid_m = first_pid_m + pid % group_size_m_runtime
        pid_n = pid % num_pid_in_group // group_size_m_runtime
    return pid_m, pid_n


@gluon.jit
def _remap_xcd(pid, grid_mn, num_xcds: gl.constexpr = 8):
    # Keep the same XCD swizzle as the original Triton kernel.  These exact
    # shapes are small along M and wide along N, so preserving balanced XCD
    # distribution helps avoid lopsided scheduling.
    pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
    tall_xcds = grid_mn % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@gluon.jit
def _remap_xcd_160(pid):
    # Exact remap for the scored qkv kernel grid: 160 CTAs over 8 XCDs.
    xcd = pid % 8
    local_pid = pid // 8
    return xcd * 20 + local_pid


@gluon.jit
def _remap_xcd_180(pid):
    # Exact remap for the scored output-projection kernel grid: 180 CTAs over
    # 8 XCDs -> four XCDs get 23 CTAs, four get 22.
    xcd = pid % 8
    local_pid = pid // 8
    if xcd < 4:
        return xcd * 23 + local_pid
    return 92 + (xcd - 4) * 22 + local_pid



@gluon.jit
def _issue_async_k64_nomask(
    a_smem,
    b_smem,
    a_ptr,
    b_ptr,
    a_offsets,
    b_offsets,
    B_CACHE: gl.constexpr,
):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_smem,
        a_ptr,
        a_offsets,
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b_smem,
        b_ptr,
        b_offsets,
        cache_modifier=B_CACHE,
    )



@gluon.jit
def _issue_async_k128_pair_nomask(
    a0_smem,
    b0_smem,
    a1_smem,
    b1_smem,
    a_ptr,
    b_ptr,
    a_offsets,
    b_offsets,
    stride_ak,
    stride_bk,
    B_CACHE: gl.constexpr,
):
    _issue_async_k64_nomask(
        a0_smem,
        b0_smem,
        a_ptr,
        b_ptr,
        a_offsets,
        b_offsets,
        B_CACHE,
    )
    _issue_async_k64_nomask(
        a1_smem,
        b1_smem,
        a_ptr + 64 * stride_ak,
        b_ptr + 64 * stride_bk,
        a_offsets,
        b_offsets,
        B_CACHE,
    )



@gluon.jit
def _gemm_bf16_cdna4_qkv_exact_k64_s5_nocache_bias(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
):
    # Exact scored-QKV kernel retuned around a deeper K64 pipeline.
    #
    # Compared to the prior four-stage K64 kernel, this version keeps five K64
    # slices resident and disables cache promotion on the streamed B tiles.
    # On MI355 this reduced the number of waits in the hot loop and was more
    # robustly faster on the scored decode-QKV shape.
    BLOCK_M: gl.constexpr = 32
    BLOCK_N: gl.constexpr = 64
    BLOCK_K: gl.constexpr = 64
    NUM_K_TILES: gl.constexpr = 45
    STRIDE_AM: gl.constexpr = 2880
    STRIDE_BN: gl.constexpr = 2880
    STRIDE_CM: gl.constexpr = 5120
    B_CACHE: gl.constexpr = ""

    a_copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    b_copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    shared_a_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8,
        per_phase=1,
        max_phase=8,
        order=[1, 0],
    )
    shared_b_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8,
        per_phase=1,
        max_phase=8,
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[2, 2],
        tiles_per_warp=[1, 2],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=8,
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=8,
    )

    pid = _remap_xcd_160(gl.program_id(axis=0))
    pid_m = pid % 2
    pid_n = pid // 2

    rows_copy = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=a_copy_layout))
    cols_copy = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=b_copy_layout))
    k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(dim=0, parent=a_copy_layout))
    k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(dim=1, parent=b_copy_layout))

    rows_out = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=mfma_layout))
    cols_out = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=mfma_layout))

    a_base = a_ptr + pid_m * BLOCK_M * STRIDE_AM
    b_base = b_ptr + pid_n * BLOCK_N * STRIDE_BN
    a_offsets = rows_copy[:, None] * STRIDE_AM + k_a[None, :]
    b_offsets = k_b[:, None] + cols_copy[None, :] * STRIDE_BN

    # Prefetch bias once so its latency overlaps the long reduction over K.
    bias_vals = gl.amd.cdna4.buffer_load(bias_ptr, cols_out, cache="").to(gl.float32)

    a_smem0 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem1 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem2 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem3 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem4 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    b_smem0 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem1 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem2 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem3 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem4 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    _issue_async_k64_nomask(a_smem0, b_smem0, a_base, b_base, a_offsets, b_offsets, B_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    _issue_async_k64_nomask(a_smem1, b_smem1, a_base + 64, b_base + 64, a_offsets, b_offsets, B_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    _issue_async_k64_nomask(a_smem2, b_smem2, a_base + 128, b_base + 128, a_offsets, b_offsets, B_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    _issue_async_k64_nomask(a_smem3, b_smem3, a_base + 192, b_base + 192, a_offsets, b_offsets, B_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    _issue_async_k64_nomask(a_smem4, b_smem4, a_base + 256, b_base + 256, a_offsets, b_offsets, B_CACHE)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(4)

    for ki in gl.static_range(0, NUM_K_TILES):
        stage = ki % 5
        k_future = (ki + 5) * 64
        if stage == 0:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem0, layout=dot_a_layout)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem0, layout=dot_b_layout)
            if ki + 5 < NUM_K_TILES:
                _issue_async_k64_nomask(a_smem0, b_smem0, a_base + k_future, b_base + k_future, a_offsets, b_offsets, B_CACHE)
                gl.amd.cdna4.async_copy.commit_group()
        elif stage == 1:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem1, layout=dot_a_layout)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem1, layout=dot_b_layout)
            if ki + 5 < NUM_K_TILES:
                _issue_async_k64_nomask(a_smem1, b_smem1, a_base + k_future, b_base + k_future, a_offsets, b_offsets, B_CACHE)
                gl.amd.cdna4.async_copy.commit_group()
        elif stage == 2:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem2, layout=dot_a_layout)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem2, layout=dot_b_layout)
            if ki + 5 < NUM_K_TILES:
                _issue_async_k64_nomask(a_smem2, b_smem2, a_base + k_future, b_base + k_future, a_offsets, b_offsets, B_CACHE)
                gl.amd.cdna4.async_copy.commit_group()
        elif stage == 3:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem3, layout=dot_a_layout)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem3, layout=dot_b_layout)
            if ki + 5 < NUM_K_TILES:
                _issue_async_k64_nomask(a_smem3, b_smem3, a_base + k_future, b_base + k_future, a_offsets, b_offsets, B_CACHE)
                gl.amd.cdna4.async_copy.commit_group()
        else:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem4, layout=dot_a_layout)
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem4, layout=dot_b_layout)
            if ki + 5 < NUM_K_TILES:
                _issue_async_k64_nomask(a_smem4, b_smem4, a_base + k_future, b_base + k_future, a_offsets, b_offsets, B_CACHE)
                gl.amd.cdna4.async_copy.commit_group()

        acc = gl.amd.cdna4.mfma(a, b, acc)

        if ki + 5 < NUM_K_TILES:
            gl.amd.cdna4.async_copy.wait_group(4)
        elif ki + 4 < NUM_K_TILES:
            gl.amd.cdna4.async_copy.wait_group(3)
        elif ki + 3 < NUM_K_TILES:
            gl.amd.cdna4.async_copy.wait_group(2)
        elif ki + 2 < NUM_K_TILES:
            gl.amd.cdna4.async_copy.wait_group(1)
        elif ki + 1 < NUM_K_TILES:
            gl.amd.cdna4.async_copy.wait_group(0)

    acc += bias_vals[None, :]
    c = acc.to(c_ptr.type.element_ty)
    c_offsets = rows_out[:, None] * STRIDE_CM + cols_out[None, :]
    gl.amd.cdna4.buffer_store(c, c_ptr, c_offsets)



@gluon.jit
def _gemm_bf16_cdna4_out_exact_interleave_bias(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
):
    # Exact kernel for the scored attention-output projection shape:
    #   M=64, N=2880, K=4096, bias always present, dense row-major inputs.
    #
    # The key optimization is a tighter K128 schedule: rather than loading the
    # two K64 slices of the current K128 stage, then issuing both replacement
    # copies, then executing both MFMA operations, we stream the stage as
    #   load slice 0 -> issue replacement 0 -> mfma 0 -> load slice 1
    #   -> issue replacement 1 -> commit -> mfma 1
    # This drops the accumulator-side register pressure and improved the scored
    # output-projection latency by several percent on MI355.
    BLOCK_M: gl.constexpr = 32
    BLOCK_N: gl.constexpr = 32
    BLOCK_K: gl.constexpr = 64
    NUM_K_PAIRS: gl.constexpr = 32
    STRIDE_AM: gl.constexpr = 4096
    STRIDE_BN: gl.constexpr = 4096
    STRIDE_CM: gl.constexpr = 2880

    a_copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    b_copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    shared_a_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8,
        per_phase=1,
        max_phase=8,
        order=[1, 0],
    )
    shared_b_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8,
        per_phase=1,
        max_phase=8,
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[2, 2],
        tiles_per_warp=[1, 1],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=8,
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=8,
    )

    pid = _remap_xcd_180(gl.program_id(axis=0))
    pid_m = pid % 2
    pid_n = pid // 2

    rows_copy = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=a_copy_layout))
    cols_copy = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=b_copy_layout))
    k_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(dim=0, parent=a_copy_layout))
    k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(dim=1, parent=b_copy_layout))

    rows_out = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=mfma_layout))
    cols_out = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=mfma_layout))

    a_base = a_ptr + pid_m * BLOCK_M * STRIDE_AM
    b_base = b_ptr + pid_n * BLOCK_N * STRIDE_BN
    a_offsets = rows_copy[:, None] * STRIDE_AM + k_a[None, :]
    b_offsets = k_b[:, None] + cols_copy[None, :] * STRIDE_BN

    a_smem00 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem01 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    b_smem00 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem01 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)

    a_smem10 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem11 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    b_smem10 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem11 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)

    a_smem20 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    a_smem21 = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a_layout)
    b_smem20 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)
    b_smem21 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b_layout)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    _issue_async_k128_pair_nomask(
        a_smem00,
        b_smem00,
        a_smem01,
        b_smem01,
        a_base,
        b_base,
        a_offsets,
        b_offsets,
        1,
        1,
        ".ca",
    )
    gl.amd.cdna4.async_copy.commit_group()
    _issue_async_k128_pair_nomask(
        a_smem10,
        b_smem10,
        a_smem11,
        b_smem11,
        a_base + 128,
        b_base + 128,
        a_offsets,
        b_offsets,
        1,
        1,
        ".ca",
    )
    gl.amd.cdna4.async_copy.commit_group()
    _issue_async_k128_pair_nomask(
        a_smem20,
        b_smem20,
        a_smem21,
        b_smem21,
        a_base + 256,
        b_base + 256,
        a_offsets,
        b_offsets,
        1,
        1,
        ".ca",
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(2)

    for pair_idx in gl.static_range(0, NUM_K_PAIRS):
        stage = pair_idx % 3
        k_future = (pair_idx + 3) * 128
        if stage == 0:
            a0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem00, layout=dot_a_layout)
            b0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem00, layout=dot_b_layout)
            if pair_idx + 3 < NUM_K_PAIRS:
                _issue_async_k64_nomask(a_smem00, b_smem00, a_base + k_future, b_base + k_future, a_offsets, b_offsets, ".ca")
            acc = gl.amd.cdna4.mfma(a0, b0, acc)
            a1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem01, layout=dot_a_layout)
            b1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem01, layout=dot_b_layout)
            if pair_idx + 3 < NUM_K_PAIRS:
                _issue_async_k64_nomask(a_smem01, b_smem01, a_base + k_future + 64, b_base + k_future + 64, a_offsets, b_offsets, ".ca")
                gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma(a1, b1, acc)
        elif stage == 1:
            a0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem10, layout=dot_a_layout)
            b0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem10, layout=dot_b_layout)
            if pair_idx + 3 < NUM_K_PAIRS:
                _issue_async_k64_nomask(a_smem10, b_smem10, a_base + k_future, b_base + k_future, a_offsets, b_offsets, ".ca")
            acc = gl.amd.cdna4.mfma(a0, b0, acc)
            a1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem11, layout=dot_a_layout)
            b1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem11, layout=dot_b_layout)
            if pair_idx + 3 < NUM_K_PAIRS:
                _issue_async_k64_nomask(a_smem11, b_smem11, a_base + k_future + 64, b_base + k_future + 64, a_offsets, b_offsets, ".ca")
                gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma(a1, b1, acc)
        else:
            a0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem20, layout=dot_a_layout)
            b0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem20, layout=dot_b_layout)
            if pair_idx + 3 < NUM_K_PAIRS:
                _issue_async_k64_nomask(a_smem20, b_smem20, a_base + k_future, b_base + k_future, a_offsets, b_offsets, ".ca")
            acc = gl.amd.cdna4.mfma(a0, b0, acc)
            a1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a_smem21, layout=dot_a_layout)
            b1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b_smem21, layout=dot_b_layout)
            if pair_idx + 3 < NUM_K_PAIRS:
                _issue_async_k64_nomask(a_smem21, b_smem21, a_base + k_future + 64, b_base + k_future + 64, a_offsets, b_offsets, ".ca")
                gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma(a1, b1, acc)

        if pair_idx + 3 < NUM_K_PAIRS:
            gl.amd.cdna4.async_copy.wait_group(2)
        elif pair_idx + 2 < NUM_K_PAIRS:
            gl.amd.cdna4.async_copy.wait_group(1)
        elif pair_idx + 1 < NUM_K_PAIRS:
            gl.amd.cdna4.async_copy.wait_group(0)

    bias_vals = gl.amd.cdna4.buffer_load(bias_ptr, cols_out, cache=".ca").to(gl.float32)
    acc += bias_vals[None, :]
    c = acc.to(c_ptr.type.element_ty)
    c_offsets = rows_out[:, None] * STRIDE_CM + cols_out[None, :]
    gl.amd.cdna4.buffer_store(c, c_ptr, c_offsets)




def _valid_generated_call(
    lhs: object,
    weight: object,
    bias: object | None,
    dtype: object | None,
    activation: str | None,
    skip_reduce: bool,
) -> bool:
    if bias is None or dtype not in (None, torch.bfloat16) or activation not in (None, "") or skip_reduce:
        return False
    if not (
        getattr(lhs, "is_cuda", False)
        and getattr(weight, "is_cuda", False)
        and getattr(bias, "is_cuda", False)
    ):
        return False
    if not (
        getattr(lhs, "dtype", None) == torch.bfloat16
        and getattr(weight, "dtype", None) == torch.bfloat16
        and getattr(bias, "dtype", None) == torch.bfloat16
    ):
        return False
    if getattr(lhs, "ndim", None) != 2 or getattr(weight, "ndim", None) != 2 or getattr(bias, "ndim", None) != 1:
        return False
    _m, k = lhs.shape  # type: ignore[attr-defined]
    n, wk = weight.shape  # type: ignore[attr-defined]
    return k == wk and bias.shape[0] == n  # type: ignore[index,union-attr]


def _valid_generated_y(y: object | None, expected_shape: tuple[int, int]) -> bool:
    if y is None:
        return True
    return (
        getattr(y, "is_cuda", False)
        and getattr(y, "dtype", None) == torch.bfloat16
        and getattr(y, "ndim", None) == 2
        and tuple(y.shape) == expected_shape  # type: ignore[attr-defined]
    )


def _is_exact_contiguous_m64(lhs: object, weight: object, bias: object | None) -> tuple[bool, int, int, int]:
    if bias is None:
        return False, 0, 0, 0
    m, k = lhs.shape  # type: ignore[attr-defined]
    n, _ = weight.shape  # type: ignore[attr-defined]
    m = int(m); n = int(n); k = int(k)
    exact = m == 64 and ((n, k) == (5120, 2880) or (n, k) == (2880, 4096))
    contiguous = (
        lhs.stride() == (k, 1)  # type: ignore[attr-defined]
        and weight.stride() == (k, 1)  # type: ignore[attr-defined]
        and bias.stride() == (1,)  # type: ignore[attr-defined]
    )
    return exact and contiguous, m, n, k


def generated_gemm_a16w16(
    lhs: object,
    weight: object,
    bias: object | None = None,
    dtype: object | None = torch.bfloat16,
    y: object | None = None,
    config: dict[str, object] | None = None,
    activation: str | None = None,
    skip_reduce: bool = False,
) -> object:
    """Optimized exact-M64 replacement for GPT-OSS BF16 attention GEMMs under Triton 3.7."""
    if not _valid_generated_call(lhs, weight, bias, dtype, activation, skip_reduce):
        return NotImplemented
    expected_shape = (int(lhs.shape[0]), int(weight.shape[0]))  # type: ignore[index,attr-defined]
    if not _valid_generated_y(y, expected_shape):
        return NotImplemented

    use_specialized, _m, n, k = _is_exact_contiguous_m64(lhs, weight, bias)
    if use_specialized:
        out = y if y is not None else torch.empty(expected_shape, dtype=torch.bfloat16, device=lhs.device)  # type: ignore[attr-defined]
        weight_t = weight.T  # type: ignore[attr-defined]
        if (n, k) == (5120, 2880):
            _gemm_bf16_cdna4_qkv_exact_k64_s5_nocache_bias[(160,)](
                lhs,
                weight_t,
                bias,
                out,
                num_warps=4,
                num_stages=2,
                waves_per_eu=8,
                default_dot_input_precision="bf16x3",
                schedule_hint="attention",
                matrix_instr_nonkdim=32,
                allow_flush_denorm=False,
                sanitize_overflow=True,
            )
            return out
        _gemm_bf16_cdna4_out_exact_interleave_bias[(180,)](
            lhs,
            weight_t,
            bias,
            out,
            num_warps=4,
            num_stages=4,
            waves_per_eu=4,
            default_dot_input_precision="ieee",
            schedule_hint="none",
            matrix_instr_nonkdim=32,
            kpack=1,
            allow_flush_denorm=True,
            sanitize_overflow=False,
        )
        return out

    # Keep this generated kernel as a pure exact-shape specialization.
    # The reintegration wrapper handles fallback to the original aiter implementation.
    return NotImplemented
