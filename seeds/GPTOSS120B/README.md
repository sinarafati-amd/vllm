# GPT-OSS MI355X `gemm_a16w16` M=64 specialization

This change packages a kernel specialization for manual integration into vLLM/aiter serving containers.

The generated kernel is intentionally narrow. It only handles the decode-time
attention GEMM shapes observed as hot for GPT-OSS on MI355X:

- `M=64, N=5120, K=2880` for the QKV-family projection.
- `M=64, N=2880, K=4096` for the output-projection-family GEMM.

All other calls fall back to the original `aiter` implementation through the
reintegration wrapper.

## Files

- `gpt_oss_gemm_a16w16/gemm_a16w16_mi355_m64_v1.py`
  contains the cleaned generated Gluon/Triton kernels and the exact-shape
  adapter function `generated_gemm_a16w16`.
- `gpt_oss_gemm_a16w16/gemm_a16w16_mi355_m64_v1.py` is a standalone script
  that patches the installed `aiter.ops.triton.gemm.basic.gemm_a16w16` module.

## Reintegration

Install the Artemis Triton 3.7 wheel in the serving container first. Then run:

```bash
python gpt_oss_gemm_a16w16/reintegrate_gemm_a16w16_m64_v1.py install
```

The script resolves the installed `aiter` module, writes the generated kernel
next to it, backs up the original module, and appends a small wrapper around
`gemm_a16w16`. The wrapper calls the specialized kernel when it returns a real tensor;
if the generated adapter returns `NotImplemented`, it calls the original
implementation.

Useful commands:

```bash
# Inspect patch state.
python gpt_oss_gemm_a16w16/reintegrate_gemm_a16w16_m64_v1.py status

# Restore the original module and remove the generated module copy.
python gpt_oss_gemm_a16w16/reintegrate_gemm_a16w16_m64_v1.py restore

# Patch an explicit module path instead of resolving the import.
python gpt_oss_gemm_a16w16/reintegrate_gemm_a16w16_m64_v1.py install \
  --target-path /usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gemm/basic/gemm_a16w16.py
```

The patch should be installed before vLLM imports `aiter`.

## Measurement setup

Measurements used MI355X CaaS containers, `CONC=64`, fixed `ISL/OSL` pairs,
and three independent containers per baseline and candidate row. The baseline
was the stable system `aiter` path. An unpatched Triton 3.7 baseline was not
used because it failed vLLM engine initialization for 20B with a Triton shared
memory `OutOfResources` error in the default GEMM path.

Optimized runs used this generated kernel with the exact-M64 wrapper.

| Model | ISL | OSL | Baseline output tok/s | Optimized output tok/s | Output delta | Baseline TTFT ms | Optimized TTFT ms | TTFT delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS-20B | 1024 | 1024 | 11791.6 ± 104.4 | 12313.7 ± 244.7 | +4.43% | 252.7 ± 31.6 | 234.9 ± 140.4 | -7.02% |
| GPT-OSS-20B | 1024 | 8192 | 10857.9 ± 123.5 | 11202.1 ± 75.9 | +3.17% | 259.0 ± 91.8 | 237.8 ± 70.3 | -8.16% |
| GPT-OSS-20B | 8192 | 1024 | 6283.0 ± 147.0 | 6406.8 ± 100.6 | +1.97% | 623.3 ± 73.1 | 629.4 ± 90.8 | +0.98% |
| GPT-OSS-20B | 8192 | 8192 | 7912.0 ± 91.3 | 8085.2 ± 103.9 | +2.19% | 585.6 ± 22.0 | 622.6 ± 51.0 | +6.32% |
| GPT-OSS-120B | 1024 | 1024 | 5810.9 ± 20.7 | 6027.0 ± 67.3 | +3.72% | 313.9 ± 104.3 | 281.2 ± 75.1 | -10.42% |
| GPT-OSS-120B | 1024 | 8192 | 5563.0 ± 36.3 | 5639.9 ± 412.0 | +1.38% | 368.4 ± 80.1 | 322.9 ± 18.7 | -12.34% |
| GPT-OSS-120B | 8192 | 1024 | 3641.3 ± 64.0 | 3714.0 ± 117.6 | +2.00% | 796.4 ± 61.0 | 786.5 ± 75.4 | -1.25% |
| GPT-OSS-120B | 8192 | 8192 | 4556.3 ± 121.1 | 4682.2 ± 37.1 | +2.76% | 802.3 ± 135.3 | 804.6 ± 35.3 | +0.29% |

Geomean deltas by model:

| Model | Output tok/s geomean delta | TTFT geomean delta |
|---|---:|---:|
| GPT-OSS-20B | +2.93% | -2.15% |
| GPT-OSS-120B | +2.46% | -6.09% |
| Combined | +2.70% | -4.14% |