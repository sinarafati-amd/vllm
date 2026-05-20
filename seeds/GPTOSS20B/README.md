# GPT-OSS-20B MI300 fused MXFP4 + OAI SwiGLU stage-1 kernel

Target upstream: [`vllm-project/vllm`](https://github.com/vllm-project/vllm) ·
Staging fork: [`sinarafati-amd/vllm`](https://github.com/sinarafati-amd/vllm) ·
Hardware: AMD MI300X (gfx9 / CDNA3)

This seed proposes a narrow, opt-in MI300 specialization for the **stage-1
matmul of the GPT-OSS Triton-kernels MoE expert**. On `ISL=OSL=1024` it
produces a geomean `+10.6%` `req/s` / `-9.6%` `TPOT` improvement across
concurrencies `4–64`, all without touching the non-MI300 / non-GPT-OSS
paths.

| CONC | `req/s` | `tok/s` | interactivity | TPOT |
|-----:|--------:|--------:|--------------:|-----:|
|    4 | +11.6 % | +11.6 % | +11.1 % | -10.0 % |
|   16 |  +5.4 % |  +5.4 % |  +4.8 % |  -4.6 % |
|   32 | +16.6 % | +16.6 % | +18.2 % | -15.4 % |
|   64 |  +9.3 % |  +9.3 % |  +8.8 % |  -8.1 % |
| **geomean** | **+10.6 %** | **+10.6 %** | **+10.6 %** | **-9.6 %** |

Kernel-level at `CONC=64`: baseline `221.35 µs`, fused `176.36 µs`
(`1.255x`, `-20.33 %`).

## Why this exists

The first matmul of `triton_kernel_fused_experts` in
`vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`
writes a `2 * intermediate_size`-wide preactivation into the
`intermediate_cache`, then a fused-activation step re-reads it and
applies OAI SwiGLU (`alpha=1.702`, `limit=7.0`). For tiny ragged routed
slices on MI300 the write+reread dominates the launch — and this stage
is roughly `35 %` of inference time at decode.

The new kernel folds both work items into one Triton launch:

* the packed-MXFP4 GEMM is issued via `tl.dot_scaled` with
  `rhs_k_pack=True` (no manual unpack / dequantize),
* the `2 * BLOCK_N` preactivation is computed only in registers,
* OAI SwiGLU (`alpha=1.702`, `limit=7.0`, clipping, `s * (linear + 1)`)
  is applied in-kernel,
* the result is written straight into the
  `(M * topk, N // 2)` view of `intermediate_cache`.

The dispatch is driven by the existing ragged expert block schedule
(`metadata.slice_sizes` / `metadata.slice_offs` /
`metadata.block_schedule(BLOCK_M)`), and packed `-1` schedule entries
are masked for CUDA-graph safety.

## PR plan

The seed is structured as a two-commit series so each layer can be
reviewed independently:

1. **`0001-add-mi300-fused-mxfp4-swiglu-stage1-kernel.patch`** —
   adds the new module
   `vllm/model_executor/layers/fused_moe/experts/mi300_optimized_swiglu_candidate_1.py`.
   No call sites are touched. The module is self-contained and the
   guard `maybe_run_swiglu_stage1` returns `False` outside a strict
   shape / dtype / quant / activation / routed-row envelope, so even
   if someone imports it directly there is no observable behavior
   change without the call-site change in commit 2.

2. **`0002-dispatch-stage1-to-mi300-fused-kernel-when-guarded.patch`** —
   wires `maybe_run_swiglu_stage1` in front of the existing stage-1
   `matmul_ogs` call in `triton_kernel_fused_experts`. When the guard
   returns `False` the call falls through to the original
   `matmul_ogs` + `fused_activation` path with bit-identical numerics.

Both patches apply cleanly against `upstream/main`. The combined
unified diff is also provided as `full_series.diff`.

## Surface area

```text
vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py   # +28 / -7
vllm/model_executor/layers/fused_moe/experts/mi300_optimized_swiglu_candidate_1.py   # +331 / -0 (new)
```

Total diff: 2 files changed, 359 insertions, 7 deletions.

### Guarded fast-path (call-site change)

The single touched block inside `triton_kernel_fused_experts` is the
stage-1 `matmul_ogs` invocation. The patch wraps it in:

```python
used_mi300_swiglu_stage1 = maybe_run_swiglu_stage1(
    hidden_states,
    w1,
    routing_data,
    gather_indx,
    quant_config.w1_precision,
    quant_config.w1_bias,
    intermediate_cache.view(M * topk, N // 2),
    apply_router_weight_on_input=apply_router_weight_on_input,
    swiglu_alpha=swiglu_alpha,
    swiglu_limit=swiglu_limit,
)

if not used_mi300_swiglu_stage1:
    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )
```

Stage-2 (`matmul_ogs` into `output_tensor`) is **not** modified.

### Guard envelope

`maybe_run_swiglu_stage1` returns `True` only when **all** of the
following hold; any deviation falls back to the existing path with
zero numerical drift:

| Condition | Required value |
|---|---|
| `hidden_states.dtype` | `torch.bfloat16` |
| `hidden_states.ndim` | `2` |
| `hidden_states.shape[1]` | `3072` (GPT-OSS hidden dim) |
| `swiglu_alpha`, `swiglu_limit` | `1.702`, `7.0` |
| `apply_router_weight_on_input` | `False` |
| `gather_indx.src_indx.dtype` | `torch.int32` |
| `gather_indx.src_indx.numel()` | one of `{4, 8, 16, 32, 64, 96, …, 480}` |
| routes per token | `4` |
| `bias.shape` | `(32, 6144)`, `bias.dtype == torch.float32` |
| `w1.shape` | `(32, 3072, 6144)` with packed MXFP4 storage |
| `weight_scale.shape` | `(32, 96, 6144)` (E2M1 scales) |
| `expected_tokens_per_expt` | `<= 128` |
| `out.shape`, `out.dtype` | `(numel, 3072)`, `torch.bfloat16` |

The constraints are tight on purpose: this is a hot-path shim for
GPT-OSS-20B on MI300, not a general activation/matmul fusion.

## Container install / quick A/B

For users who don't want to rebuild vLLM, `install_shim.py` patches the
installed `gpt_oss_triton_kernels_moe.py` in place and drops the new
module next to it. The patch is bracketed with
`# PR_PUNDIT_MI300_SWIGLU_BEGIN` / `# PR_PUNDIT_MI300_SWIGLU_END` so it
is trivial to detect and remove.

```bash
# Install (creates a backup, drops the kernel module, patches the call site).
python install_shim.py install

# Inspect.
python install_shim.py status

# Restore.
python install_shim.py restore
```

The `0001-*.patch` / `0002-*.patch` files in this folder are the
upstream-style review-ready unified diffs; `install_shim.py` is just a
convenience for serving containers.

## Files in this seed

| Path | Purpose |
|---|---|
| `README.md` | This document. |
| `0001-add-mi300-fused-mxfp4-swiglu-stage1-kernel.patch` | Adds the new kernel module. Reviewable in isolation. |
| `0002-dispatch-stage1-to-mi300-fused-kernel-when-guarded.patch` | Wires the guard in front of the existing stage-1 `matmul_ogs`. |
| `full_series.diff` | Combined unified diff of the two patches. |
| `mi300_optimized_swiglu_candidate_1.py` | Standalone copy of the new kernel module for container installs. |
| `install_shim.py` | Install / restore the patch in a running container. |
