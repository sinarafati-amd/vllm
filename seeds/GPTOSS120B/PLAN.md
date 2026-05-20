# PR Series Plan — GPT-OSS MI355X `gemm_a16w16` M=64 Fast Path

> Hand-authored fallback because the pr-pundit MCP planner stage failed
> server-side (see `PIPELINE_REPORT.md`). This plan follows the same
> output shape `plan_pr_series` would have produced and cites the
> relevant rules from `upstream_rules_vllm-project_vllm.json`.

- **Upstream:** `https://github.com/vllm-project/vllm`
- **Staging fork:** `https://github.com/sinarafati-amd/vllm`
- **Seed branch:** `pr-pundit-seed-gptoss120b-gemm-a16w16` (also on `main`
  of the staging fork at `seeds/GPTOSS120B/`)
- **Hardware target:** AMD MI355X (gfx950 / CDNA4), via Gluon under
  Triton 3.7 (Artemis wheel)
- **Models:** GPT-OSS-20B and GPT-OSS-120B
- **Measured (CONC=64, three independent containers per row):**
  - 20B: +2.93% output tok/s geomean, -2.15% TTFT geomean
  - 120B: +2.46% output tok/s geomean, -6.09% TTFT geomean
  - Combined: +2.70% output tok/s, -4.14% TTFT

## Upstream integration anchor

In current upstream vLLM the natural call site already exists:

```113:158:vllm/model_executor/layers/utils.py
def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx9, on_gfx950

    n = x.numel() / x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    import math

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return gemm_a16w16(x, weight, bias)
```

The existing predicate already selects the two GPT-OSS shapes the seed
optimizes (`m=5120,k=2880` and `m=2880,k=4096`). The plan is therefore to
add a thin Gluon fast path inside vLLM that runs **before**
`aiter.gemm_a16w16` for `n == 64` only, and degrades to the existing aiter
call for everything else — preserving the current default behavior bit
for bit.

## Series shape (4 PRs, layered)

Each PR is independently reviewable and revertable. The series is ordered
so each merge leaves `main` green and behaves identically to today's
behavior until the final wire-up PR flips the dispatch on.

### PR 1 — Add MI355X exact-shape Gluon GEMM kernel (no caller changes)

**Title:** `[Kernel][ROCm] Add MI355X exact-M64 BF16 GEMM Gluon kernel for GPT-OSS attention shapes`

**Branch (staging fork):** `kernel/mi355-gemm-a16w16-m64`

**Files (new only):**
- `vllm/model_executor/layers/quantization/utils/aiter_gemm_a16w16_mi355_m64.py`
  Carries the two Gluon kernels (`_gemm_bf16_cdna4_qkv_exact_k64_s5_nocache_bias`
  and `_gemm_bf16_cdna4_out_exact_interleave_bias`), the helpers
  (`_pid_grid`, `_remap_xcd_*`, `_issue_async_k64_nomask`,
  `_issue_async_k128_pair_nomask`), and the gated entry function
  `generated_gemm_a16w16(lhs, weight, bias, ...) -> Tensor | NotImplemented`.
  Identical content to `seeds/GPTOSS120B/gemm_a16w16_mi355_m64_v1.py`
  with the module docstring updated for vLLM (Apache-2.0 SPDX header,
  link to PR, link to gpt-oss issue).

**Why this lands first / is safe:** the module is dead code at import
time — nothing calls `generated_gemm_a16w16` until PR 3 wires it in.
Adding it is purely additive.

**Rules to self-check** (`upstream_rules_vllm-project_vllm.json`):
- Triton kernels must mask potentially negative indices before pointer
  arithmetic (rule at line 11713) — the kernel relies on
  `gl.amd.cdna4.buffer_load` bounds for masking; add comments at each
  buffer_load explaining the implicit guard.
- Use `TORCH_CHECK` / Python assertions over raw asserts in CUDA/HIP
  kernels (rule at line 12718) — N/A, this is Python.
- When adding temporary upstream-library workarounds (Triton 3.7 / Gluon
  experimental imports), include a `TODO` comment referencing the
  upstream Triton version that will make the import path non-experimental
  (rule at line 7551). Add a header comment:
  `# TODO(triton>=3.8): drop triton.experimental.gluon once Gluon graduates.`
- Add explicit validation for computed values feeding pointer arithmetic
  (rule at line 6351) — keep `_valid_generated_call`, `_valid_generated_y`,
  `_is_exact_contiguous_m64` exactly as-is; they already enforce dtype,
  ndim, contiguity, and shape.

**Tests added:**
- `tests/kernels/quantization/test_mi355_gemm_a16w16_m64.py`
  - `@pytest.mark.skipif(not (current_platform.is_rocm() and on_gfx950()), reason="MI355X-only")`
  - Two parametrized tests for the exact `(M=64, N=5120, K=2880)` and
    `(M=64, N=2880, K=4096)` shapes:
      1. Numerical: random BF16 inputs, compare `generated_gemm_a16w16`
         against `torch.nn.functional.linear(x, w, bias)` with
         `torch.testing.assert_close(atol=8e-3, rtol=8e-3)`.
      2. Rejection: pass `M=63`, `N=5121`, non-contiguous, fp16,
         missing-bias, and `skip_reduce=True` cases and assert the entry
         returns `NotImplemented` for each.
  - Also place an `eye(K)` deterministic-init test variant per rule at
    line 2240 (compilation/correctness tests should include both
    deterministic AND random init).

**Commit message:**
```
[Kernel][ROCm] Add MI355X exact-M64 BF16 GEMM Gluon kernel for GPT-OSS

Adds two Gluon kernels under
vllm/model_executor/layers/quantization/utils/aiter_gemm_a16w16_mi355_m64.py
specialized for the GPT-OSS decode-time linear projections:

  - QKV-family:                M=64, N=5120, K=2880  (s5, nocache_bias)
  - output-projection family:  M=64, N=2880, K=4096  (s4, interleave_bias)

Both shapes are exact-contiguous BF16 with bias and require
Triton >= 3.7 / Gluon under MI355X (gfx950 / CDNA4). The entry function
`generated_gemm_a16w16` is gated to return NotImplemented for any input
that does not exactly match shape, stride, dtype, or activation
constraints, so this commit by itself is dead code until the dispatcher
in a follow-up commit wires it in.

No behavior change.

Hardware: AMD MI355X
Triton: >=3.7 (Gluon experimental)
```

**PR description (use this verbatim):**
```
## Summary
- Adds a new helper module `aiter_gemm_a16w16_mi355_m64.py` with two
  exact-shape Gluon kernels for GPT-OSS QKV-family and out-proj-family
  BF16 GEMMs at the decode shape `M=64`.
- Pure addition: nothing calls the new entry point yet. Default behavior
  unchanged.
- Follow-up PR adds the call site in
  `vllm/model_executor/layers/utils.py:rocm_unquantized_gemm_impl`.

## Why
On MI355X with the Artemis Triton 3.7 wheel, the new kernels reduce
end-to-end GPT-OSS decode latency:

| Model | ISL/OSL | output tok/s delta | TTFT delta |
|---|---|---:|---:|
| 20B   | 1024/1024 | +4.43% | -7.02% |
| 20B   | 1024/8192 | +3.17% | -8.16% |
| 20B   | 8192/1024 | +1.97% | +0.98% |
| 20B   | 8192/8192 | +2.19% | +6.32% |
| 120B  | 1024/1024 | +3.72% | -10.42% |
| 120B  | 1024/8192 | +1.38% | -12.34% |
| 120B  | 8192/1024 | +2.00% | -1.25% |
| 120B  | 8192/8192 | +2.76% | +0.29% |

Geomean across all rows: **+2.70% output tok/s, -4.14% TTFT** at
CONC=64, three independent containers per row.

## Scope
- New file only.
- No changes to dispatchers, no env-var flips, no public APIs touched.
- Falls back through `NotImplemented` for any shape outside the two
  GPT-OSS targets, so this kernel can never be reached without the
  follow-up wiring PR.

## Test plan
- [ ] Run `pytest tests/kernels/quantization/test_mi355_gemm_a16w16_m64.py`
      on an MI355X box with Triton 3.7 — exact-shape numerical match
      vs `torch.nn.functional.linear` (atol/rtol = 8e-3) and rejection
      tests for each off-target case.
- [ ] CI: `tests/kernels/quantization/` job continues to skip cleanly
      on non-ROCm/non-gfx950 runners (rule at line 14001).

## Follow-ups
- PR 2: standalone reintegration installer for environments that cannot
        rebuild vLLM but can monkey-patch installed aiter modules.
- PR 3: wire the kernel into `rocm_unquantized_gemm_impl` (env-gated).
- PR 4: docs + microbenchmark + CI benchmark registration.
```

---

### PR 2 — Standalone reintegration installer script (out-of-tree usage)

**Title:** `[ROCm] Add standalone installer for MI355X gemm_a16w16 M=64 fast path on prebuilt aiter`

**Branch:** `tools/mi355-gemm-a16w16-m64-installer`

**Why this is a separate PR:** the installer monkey-patches the installed
`aiter.ops.triton.gemm.basic.gemm_a16w16` module at runtime. It is
useful in serving containers that ship a pinned aiter wheel and cannot
rebuild vLLM. Decoupling it from PR 1/3 keeps the in-tree change
review-only-on-vLLM-internals.

**Files (new only):**
- `tools/rocm/install_mi355_gemm_a16w16_m64.py` — a verbatim cleanup of
  `seeds/GPTOSS120B/reintegrate_gemm_a16w16_m64_v1.py`:
  - rename to a `tools/rocm/`-namespaced filename so it does not look
    like an importable module,
  - add Apache-2.0 SPDX header,
  - add `argparse` examples in the docstring (already present),
  - keep `install`, `restore`, `status` subcommands,
  - keep `PATCH_BEGIN` / `PATCH_END` sentinel pattern,
  - **Add**: print a structured deprecation warning when the in-tree
    vLLM kernel (PR 1) is already present — detect by `importlib.util.find_spec("vllm.model_executor.layers.quantization.utils.aiter_gemm_a16w16_mi355_m64")` and warn that the in-tree dispatch is preferred.

**Files referenced (must not be re-shipped here):**
- The generated kernel is **NOT** re-shipped in this PR; the installer
  reads the path passed via `--generated-file` (default to PR 1's
  module path inside the installed vLLM package).

**Tests added:**
- `tests/tools/test_install_mi355_gemm_a16w16_m64.py`
  - `monkeypatch.syspath_prepend` + a fake `aiter.ops.triton.gemm.basic.gemm_a16w16`
    target module written into `tmp_path`, then assert:
      - `install` action writes the backup and footer,
      - `restore` removes both,
      - `status` reports correct flags,
      - running `install` twice is idempotent (uses backup, not double-patches).
  - Pure-Python test, no GPU required.

**Rules to self-check:**
- Rule at line 4630 (promote weight-processing methods to module scope):
  N/A — this installer is a one-shot tool, not a layer.
- Rule at line 8856 (vendor-specific workarounds need TODO removal
  conditions): add `# TODO: remove this installer once the in-tree
  kernel ships in all supported aiter releases (>= <pin>).`
- Rule at line 6460 (no device-specific logic in shell scripts) — this
  is Python, fine.

**Commit message:**
```
[ROCm] Add standalone installer for MI355X gemm_a16w16 M=64 fast path

Adds tools/rocm/install_mi355_gemm_a16w16_m64.py. The installer wraps
the system-installed `aiter.ops.triton.gemm.basic.gemm_a16w16` module
with a footer that loads the MI355X exact-M64 BF16 Gluon kernel
(added in <PR 1 link>) and dispatches to it for matching shapes,
falling back to the original aiter implementation otherwise.

Subcommands: install, restore, status. Idempotent — running install
twice uses the backup as the base each time. Emits a deprecation
notice when the in-tree kernel from <PR 1 link> is importable, since
that path is preferred over the runtime monkey-patch.
```

---

### PR 3 — Wire the MI355X kernel into `rocm_unquantized_gemm_impl` (default-off, env-gated)

**Title:** `[ROCm][GPT-OSS] Dispatch M=64 GPT-OSS BF16 GEMMs through MI355X Gluon kernel`

**Branch:** `dispatch/mi355-gemm-a16w16-m64`

**Files modified:**
- `vllm/envs.py` — add a new env var, default to **off** for the first
  release:
  ```python
  VLLM_ROCM_USE_MI355X_M64_GEMM: bool = False
  ```
  and the corresponding lambda in the env table around line 951
  (mirror the pattern used by `VLLM_ROCM_USE_AITER_TRITON_GEMM`).
- `vllm/_aiter_ops.py` — add a class flag and `is_mi355x_m64_gemm_enabled`
  classmethod beside `is_triton_gemm_enabled` (around line 1018).
- `vllm/model_executor/layers/utils.py` — inside
  `rocm_unquantized_gemm_impl`, **before** the existing
  `if use_aiter_triton_gemm(...)` branch, add:
  ```python
  if (
      rocm_aiter_ops.is_mi355x_m64_gemm_enabled()
      and on_gfx950()
      and n == 64
      and x.dtype == torch.bfloat16
      and (
          (m == 5120 and k == 2880)
          or (m == 2880 and k == 4096)
      )
  ):
      from vllm.model_executor.layers.quantization.utils.aiter_gemm_a16w16_mi355_m64 import (
          generated_gemm_a16w16,
      )

      x_view = x.reshape(-1, x.size(-1)) if x.ndim > 2 else x
      out = generated_gemm_a16w16(x_view, weight, bias)
      if out is not NotImplemented:
          return out.reshape(*x.shape[:-1], weight.shape[0])
  ```
- Add an informative `logger.warning_once` the first time the dispatch
  is skipped because of a Triton-version mismatch (per rule at line
  5797: "Add informative warning or error messages when attention
  backend features are unavailable or when fallback behavior occurs").

**No new files.** No model code changed.

**Rules to self-check:**
- Rule at line 5714 ("Enable performance-validated optimizations by
  default for new GPU architectures when benchmarks demonstrate clear
  benefits") — we _do_ have measured benefits, but this is a Triton
  3.7 / Artemis-only path and many MI355X deployments still ship a
  Triton 3.5 wheel; defaulting to **off** for the first release is
  more conservative. Flip the default to `True` in a small follow-up
  once the Triton 3.7 wheel is the standard in the release-test
  containers.
- Rule at line 1626 ("When adding a new engine argument… ensure it is
  properly propagated"): the new env var has no EngineArgs surface, so
  this does not apply. Just route it via `envs.VLLM_ROCM_USE_MI355X_M64_GEMM`.
- Rule at line 980 ("Use dictionary-based dispatch instead of if-else
  chains"): keep the new branch minimal — a single `if (...)` is fine
  because we are gating on platform AND shape AND dtype, but if the
  list of supported shapes grows beyond two we should refactor into a
  `_MI355X_M64_SHAPES = {(5120, 2880), (2880, 4096)}` set lookup.
- Rule at line 11713 ("Always mask potentially negative indices… in
  Triton kernels") — verify that `_remap_xcd_*` helpers in PR 1 never
  feed negative `pid` into a `buffer_load`. They do not, but the gate
  is re-checked here for completeness.
- Rule at line 5797 ("informative warning when backends fall back") —
  emit `logger.warning_once` when the env is on but
  `triton.experimental.gluon` is unimportable.

**Tests added:**
- `tests/kernels/quantization/test_rocm_skinny_gemms.py` — extend the
  existing file (rule at line 5087 prefers adding to existing files
  over creating new ones for overlapping scope) with:
  - a new parameter set `NKM_FACTORS_MI355X_M64 = [(5120, 2880, 64), (2880, 4096, 64)]`,
  - a `@pytest.mark.skipif(not on_gfx950() or not envs.VLLM_ROCM_USE_MI355X_M64_GEMM, ...)` guard,
  - a wrapper test that calls `rocm_unquantized_gemm_impl` with and
    without the env var and asserts numerical equivalence to
    `torch.nn.functional.linear`.
- A negative test: with the env on but `m=64, n=5120, k=2879`, assert
  the fall-through to the aiter `gemm_a16w16` path (i.e., the new
  branch returns `NotImplemented` and the function continues).

**Commit message:**
```
[ROCm][GPT-OSS] Dispatch M=64 GPT-OSS BF16 GEMMs through MI355X Gluon kernel

Adds a new env var VLLM_ROCM_USE_MI355X_M64_GEMM (default off) and a
gated branch at the top of rocm_unquantized_gemm_impl that routes the
two GPT-OSS exact decode shapes:

  - QKV-family:                M=64, N=5120, K=2880
  - output-projection family:  M=64, N=2880, K=4096

through the MI355X Gluon kernel added in <PR 1 link>. Falls through to
the existing aiter.gemm_a16w16 path for every other shape, dtype,
contiguity, or architecture, preserving the current default behavior
bit for bit when the env var is off.

Default-off because the new kernel requires the Artemis Triton 3.7
(Gluon) wheel, which is not yet the default in all release containers.
A follow-up will flip the default once Triton 3.7 ships everywhere.

Measured CONC=64 geomean on MI355X:
  GPT-OSS-20B:  +2.93% output tok/s, -2.15% TTFT
  GPT-OSS-120B: +2.46% output tok/s, -6.09% TTFT
```

---

### PR 4 — Docs, microbenchmark, and CI benchmark registration

**Title:** `[Docs][Benchmark] Document MI355X GPT-OSS GEMM fast path and add microbenchmark`

**Branch:** `docs/mi355-gemm-a16w16-m64`

**Files (new):**
- `benchmarks/kernels/benchmark_mi355_gemm_a16w16_m64.py` — a single
  file modeled on the existing `benchmarks/kernels/benchmark_*.py`
  layout (e.g., `benchmark_grouped_gemm_cutlass.py`):
  - `argparse` with `--mode {generated,aiter,torch}`, `--dtype bf16`,
    `--shape {qkv,outproj,all}`, `--num-iters`,
  - timing loop with `torch.cuda.synchronize` and `torch.cuda.Event`
    on ROCm,
  - print rows like `shape=qkv mode=generated us=176.3 tok/s=...`.
- `.buildkite/run_benchmarks.sh` — append a row that runs
  `benchmark_mi355_gemm_a16w16_m64.py --mode generated --mode aiter`
  on MI355X agents only. Use `--append` and a per-GPU context name per
  rule at line 4238 ("use unique context names per GPU type or job…
  to prevent annotation overwrites").

**Files (modified):**
- `docs/source/features/quantization/index.md` (or the ROCm features
  doc — locate by `rg -l 'VLLM_ROCM_USE_AITER_TRITON_GEMM' docs`) —
  add a small section "MI355X GPT-OSS M=64 GEMM fast path" describing
  the env var, the shape gating, the Triton 3.7 / Gluon requirement,
  and a link to the new microbenchmark.
- `vllm/envs.py` — make sure the env-var docstring block lists
  `VLLM_ROCM_USE_MI355X_M64_GEMM` next to its sibling ROCm flags.

**Rules to self-check:**
- Rule at line 1554 ("update the CI benchmark runner script when adding
  new benchmark scripts") — this PR is exactly that.
- Rule at line 4238 (unique context names) — already addressed above.
- Rule at line 4397 ("consolidate metadata args into `--metadata key=value`")
  — keep the benchmark CLI focused on functional knobs, not metadata.
- Rule at line 13681 ("specify exact package source URL when documenting
  platform-specific installation instructions") — when the docs section
  mentions Triton 3.7, link to the Artemis wheel index URL, not a
  generic `pip install triton`.
- Rule at line 2012 ("avoid maintaining static lists of supported
  models/shapes in docs; link to source of truth") — in the docs
  section, link to `vllm/model_executor/layers/utils.py:use_aiter_triton_gemm`
  rather than re-stating the supported shape set.

**Commit message:**
```
[Docs][Benchmark] Document MI355X GPT-OSS GEMM fast path and add microbenchmark

Adds benchmarks/kernels/benchmark_mi355_gemm_a16w16_m64.py to compare
the in-tree MI355X Gluon kernel (PR 1) against the existing aiter
gemm_a16w16 baseline and a torch.nn.functional.linear reference for
the two GPT-OSS decode shapes.

Registers the script in .buildkite/run_benchmarks.sh with a per-GPU
context name so multiple Buildkite jobs do not overwrite each other's
annotations.

Documents the new VLLM_ROCM_USE_MI355X_M64_GEMM env var, the Triton
3.7 / Gluon requirement, and the shape gating, with links back to the
predicate in vllm/model_executor/layers/utils.py.
```

## Tracking issue (open this before any PR is pushed)

**Title:** `[Tracking][ROCm] MI355X GPT-OSS BF16 GEMM M=64 fast path`

**Body (copy/paste):**
```
Tracking the upstream landing of a narrow MI355X (gfx950 / CDNA4 / Triton 3.7 Gluon) fast path for the two GPT-OSS decode-time linear-projection BF16 GEMMs:

  - QKV-family:                M=64, N=5120, K=2880
  - output-projection family:  M=64, N=2880, K=4096

Every other shape, dtype, contiguity, or architecture falls through
unchanged to the existing `aiter.ops.triton.gemm.basic.gemm_a16w16`
path, gated by `VLLM_ROCM_USE_MI355X_M64_GEMM` (default off in PR 3).

### Series

- [ ] PR 1: `[Kernel][ROCm] Add MI355X exact-M64 BF16 GEMM Gluon kernel for GPT-OSS attention shapes`
- [ ] PR 2: `[ROCm] Add standalone installer for MI355X gemm_a16w16 M=64 fast path on prebuilt aiter`
- [ ] PR 3: `[ROCm][GPT-OSS] Dispatch M=64 GPT-OSS BF16 GEMMs through MI355X Gluon kernel`
- [ ] PR 4: `[Docs][Benchmark] Document MI355X GPT-OSS GEMM fast path and add microbenchmark`

### Performance (CONC=64, MI355X CaaS containers, three independent containers per row)

| Model | ISL | OSL | Baseline tok/s | Optimized tok/s | Δ tok/s | Baseline TTFT (ms) | Optimized TTFT (ms) | Δ TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS-20B | 1024 | 1024 | 11791.6 ± 104.4 | 12313.7 ± 244.7 | +4.43% | 252.7 ± 31.6 | 234.9 ± 140.4 | -7.02% |
| GPT-OSS-20B | 1024 | 8192 | 10857.9 ± 123.5 | 11202.1 ± 75.9 | +3.17% | 259.0 ± 91.8 | 237.8 ± 70.3 | -8.16% |
| GPT-OSS-20B | 8192 | 1024 | 6283.0 ± 147.0 | 6406.8 ± 100.6 | +1.97% | 623.3 ± 73.1 | 629.4 ± 90.8 | +0.98% |
| GPT-OSS-20B | 8192 | 8192 | 7912.0 ± 91.3 | 8085.2 ± 103.9 | +2.19% | 585.6 ± 22.0 | 622.6 ± 51.0 | +6.32% |
| GPT-OSS-120B | 1024 | 1024 | 5810.9 ± 20.7 | 6027.0 ± 67.3 | +3.72% | 313.9 ± 104.3 | 281.2 ± 75.1 | -10.42% |
| GPT-OSS-120B | 1024 | 8192 | 5563.0 ± 36.3 | 5639.9 ± 412.0 | +1.38% | 368.4 ± 80.1 | 322.9 ± 18.7 | -12.34% |
| GPT-OSS-120B | 8192 | 1024 | 3641.3 ± 64.0 | 3714.0 ± 117.6 | +2.00% | 796.4 ± 61.0 | 786.5 ± 75.4 | -1.25% |
| GPT-OSS-120B | 8192 | 8192 | 4556.3 ± 121.1 | 4682.2 ± 37.1 | +2.76% | 802.3 ± 135.3 | 804.6 ± 35.3 | +0.29% |

Geomean deltas:
  - GPT-OSS-20B:  +2.93% tok/s, -2.15% TTFT
  - GPT-OSS-120B: +2.46% tok/s, -6.09% TTFT
  - Combined:     +2.70% tok/s, -4.14% TTFT

Seed branch on staging fork:
https://github.com/sinarafati-amd/vllm/tree/main/seeds/GPTOSS120B

Hardware: AMD MI355X (gfx950 / CDNA4)
Triton: >= 3.7 (Gluon experimental, Artemis wheel)
```

## Per-PR push checklist (use with `PUSH_INSTRUCTIONS.md`)

For each PR above:

1. Branch off `upstream/main` (re-fetch first).
2. Apply only that PR's file set.
3. Run `tests/kernels/quantization/test_*` locally on MI355X for PR 1/3.
4. Run `pre-commit run --files <files...>` per upstream contribution guide.
5. Push to `origin` on `sinarafati-amd/vllm`.
6. Open the PR against `vllm-project/vllm:main` with the description above.
7. Link the tracking issue.

## Open questions for the human reviewer

1. Should PR 2 (the installer) be a vLLM PR at all, or should it live in
   a separate ops repo? The current `tools/` directory in vLLM does host
   operational scripts, but the script monkey-patches an external
   package (`aiter`) — some maintainers may push back on that. If so,
   drop PR 2 from the series and ship the installer in
   `ROCm/aiter`-side instead.
2. Default for `VLLM_ROCM_USE_MI355X_M64_GEMM` — proposal: ship **off**
   in PR 3 and flip to **on** in a separate follow-up once the Triton
   3.7 wheel is the standard in CI containers (rule at line 5714 vs the
   "no Triton 3.5 breakage" pragmatics).
3. Whether to up-port the same kernel template to the other GPT-OSS
   shapes already in `use_aiter_triton_gemm` (`(128, 2880)`,
   `(640, 2880)`, `(2880, 512)`). Not in scope for this series; record
   as a TODO in the tracking issue.
