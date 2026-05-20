# Push Instructions — Staging Repo `sinarafati-amd/vllm`

> The pr-pundit MCP would have produced these as `gh` commands inline in
> `get_plan()`. Since that pipeline crashed (see `PIPELINE_REPORT.md`),
> these are hand-built. They work against the staging fork
> `github.com/sinarafati-amd/vllm` and target upstream
> `github.com/vllm-project/vllm`.

## Prereqs (one-time)

```bash
# already true in this workspace:
#   /home/sirafati/sina_workspace/vllm
#     - origin   -> git@github.com:sinarafati-amd/vllm.git
#     - upstream -> https://github.com/vllm-project/vllm.git
#   SSH key:  ~/.ssh/id_ed25519
#   No `gh auth login` yet; do this once before opening PRs:
gh auth login --hostname github.com --git-protocol ssh
```

## Refresh upstream

```bash
cd /home/sirafati/sina_workspace/vllm
git fetch upstream main
```

## Seed branch (already pushed)

```bash
# branch on origin:
#   pr-pundit-seed-gptoss120b-gemm-a16w16
#   (and the same content lives on origin/main under seeds/GPTOSS120B/)
# nothing to do
```

## PR 1 — `kernel/mi355-gemm-a16w16-m64`

```bash
git checkout -b kernel/mi355-gemm-a16w16-m64 upstream/main

# 1) Create the new module file (Apache-2.0 SPDX header on top):
mkdir -p vllm/model_executor/layers/quantization/utils
cp seeds/GPTOSS120B/gemm_a16w16_mi355_m64_v1.py \
   vllm/model_executor/layers/quantization/utils/aiter_gemm_a16w16_mi355_m64.py
# Then prepend the SPDX header and a TODO comment for triton>=3.8:
#   # SPDX-License-Identifier: Apache-2.0
#   # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#   # TODO(triton>=3.8): drop triton.experimental.gluon once Gluon graduates.

# 2) Add the test:
#   tests/kernels/quantization/test_mi355_gemm_a16w16_m64.py
#   (skeleton in PLAN.md, PR 1 Tests section)

git add vllm/model_executor/layers/quantization/utils/aiter_gemm_a16w16_mi355_m64.py
git add tests/kernels/quantization/test_mi355_gemm_a16w16_m64.py
pre-commit run --files \
  vllm/model_executor/layers/quantization/utils/aiter_gemm_a16w16_mi355_m64.py \
  tests/kernels/quantization/test_mi355_gemm_a16w16_m64.py

git commit  # use PLAN.md PR 1 commit message
git push -u origin kernel/mi355-gemm-a16w16-m64

gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head sinarafati-amd:kernel/mi355-gemm-a16w16-m64 \
  --title "[Kernel][ROCm] Add MI355X exact-M64 BF16 GEMM Gluon kernel for GPT-OSS attention shapes" \
  --body-file - <<'EOF'
<paste PR 1 description from PLAN.md verbatim>
EOF
```

## PR 2 — `tools/mi355-gemm-a16w16-m64-installer`

```bash
git checkout -b tools/mi355-gemm-a16w16-m64-installer upstream/main

mkdir -p tools/rocm
cp seeds/GPTOSS120B/reintegrate_gemm_a16w16_m64_v1.py \
   tools/rocm/install_mi355_gemm_a16w16_m64.py
# Prepend SPDX header. Add the deprecation-notice block from PLAN.md PR 2.

# Add pure-python test:
#   tests/tools/test_install_mi355_gemm_a16w16_m64.py

git add tools/rocm/install_mi355_gemm_a16w16_m64.py \
        tests/tools/test_install_mi355_gemm_a16w16_m64.py
pre-commit run --files tools/rocm/install_mi355_gemm_a16w16_m64.py \
                       tests/tools/test_install_mi355_gemm_a16w16_m64.py
git commit  # use PLAN.md PR 2 commit message
git push -u origin tools/mi355-gemm-a16w16-m64-installer

gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head sinarafati-amd:tools/mi355-gemm-a16w16-m64-installer \
  --title "[ROCm] Add standalone installer for MI355X gemm_a16w16 M=64 fast path on prebuilt aiter" \
  --body-file - <<'EOF'
<paste PR 2 description (you'll write this from PLAN.md PR 2)>
EOF
```

## PR 3 — `dispatch/mi355-gemm-a16w16-m64`

```bash
# IMPORTANT: do not stack on PR 1; open this *after* PR 1 lands so the
# import path exists upstream. While waiting, you can stage it on top of
# PR 1's branch in the fork for end-to-end testing.

git checkout -b dispatch/mi355-gemm-a16w16-m64 upstream/main  # rebase later if needed

# 1) vllm/envs.py    — add VLLM_ROCM_USE_MI355X_M64_GEMM (default False)
#                       in the dataclass and the env table.
# 2) vllm/_aiter_ops.py — add is_mi355x_m64_gemm_enabled classmethod + flag.
# 3) vllm/model_executor/layers/utils.py — insert the gated branch above
#    the existing `if use_aiter_triton_gemm(...)` block, per PLAN.md.
# 4) tests/kernels/quantization/test_rocm_skinny_gemms.py — extend with
#    NKM_FACTORS_MI355X_M64 set and a wrapper-level test.

git add vllm/envs.py vllm/_aiter_ops.py vllm/model_executor/layers/utils.py \
        tests/kernels/quantization/test_rocm_skinny_gemms.py
pre-commit run --files <same files>
git commit  # use PLAN.md PR 3 commit message
git push -u origin dispatch/mi355-gemm-a16w16-m64

gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head sinarafati-amd:dispatch/mi355-gemm-a16w16-m64 \
  --title "[ROCm][GPT-OSS] Dispatch M=64 GPT-OSS BF16 GEMMs through MI355X Gluon kernel" \
  --body-file - <<'EOF'
<paste PR 3 description from PLAN.md verbatim, link PR 1>
EOF
```

## PR 4 — `docs/mi355-gemm-a16w16-m64`

```bash
git checkout -b docs/mi355-gemm-a16w16-m64 upstream/main

# 1) benchmarks/kernels/benchmark_mi355_gemm_a16w16_m64.py  (new)
# 2) .buildkite/run_benchmarks.sh                            (modified)
# 3) docs/...(ROCm features page)                            (modified)
# 4) vllm/envs.py docstring block lists the env var          (modified)

git add benchmarks/kernels/benchmark_mi355_gemm_a16w16_m64.py \
        .buildkite/run_benchmarks.sh \
        docs/... vllm/envs.py
pre-commit run --files <same files>
git commit  # use PLAN.md PR 4 commit message
git push -u origin docs/mi355-gemm-a16w16-m64

gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head sinarafati-amd:docs/mi355-gemm-a16w16-m64 \
  --title "[Docs][Benchmark] Document MI355X GPT-OSS GEMM fast path and add microbenchmark" \
  --body-file - <<'EOF'
<paste PR 4 description from PLAN.md verbatim, link PR 1 and PR 3>
EOF
```

## Tracking issue (open before any PR is pushed)

```bash
gh issue create \
  --repo vllm-project/vllm \
  --title "[Tracking][ROCm] MI355X GPT-OSS BF16 GEMM M=64 fast path" \
  --label "tracking-issue,rocm,gpt-oss,performance" \
  --body-file - <<'EOF'
<paste tracking issue body from PLAN.md verbatim>
EOF
```

## After merging upstream

```bash
git checkout main
git pull upstream main
git push origin main
# Delete merged branches both locally and on origin.
```

## Verify locally before each `gh pr create`

```bash
# linting + formatting
pre-commit run --files <changed files>

# targeted tests
pytest tests/kernels/quantization/test_mi355_gemm_a16w16_m64.py -q   # PR 1 / PR 3
pytest tests/tools/test_install_mi355_gemm_a16w16_m64.py -q          # PR 2

# microbenchmark sanity (PR 4)
python benchmarks/kernels/benchmark_mi355_gemm_a16w16_m64.py --shape all --mode generated --num-iters 100
python benchmarks/kernels/benchmark_mi355_gemm_a16w16_m64.py --shape all --mode aiter     --num-iters 100
```
