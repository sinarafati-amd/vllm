gpt-oss-20b kernel candidate for MI300s targeting the stage-1 MOE for row counts in [4, 480]. This is targeting the _matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu kernel which is ~35% of the model inference time. This is an early candidate written in Triton; we expect we could de even better in Gluon and with more compute. Still, this optimized kernel shows lots of interesting, nontrivial changes including fusion of the swiglu. Our baselines on ISL=OSL=1024 show significant improvements in e2e latency:

CONC      req/s    tok/s    interactivity    TPOT
4        +11.6%   +11.6%       +11.1%      -10.0%
16        +5.4%    +5.4%        +4.8%       -4.6%
32       +16.6%   +16.6%       +18.2%      -15.4%
64        +9.3%    +9.3%        +8.8%       -8.1%
---------------------------------------------------
geomean  +10.6%   +10.6%       +10.6%       -9.6%

*results obtained via 5 independent runs for both the baseline and optimized at each concurrency


This kernel is a narrow MI300 specialization for GPT-OSS stage-1 MoE:

`bf16 hidden_states + MXFP4 w1/scales + routed gather + bias + OAI SwiGLU -> bf16 routed output`

The main optimizations:

1. **Fuses the expensive stage-1 path**
   - Baseline does the first MoE matmul into a `6144`-wide preactivation, then applies SwiGLU.
   - optimized computes the `2 * 3072` preactivation only inside registers, applies SwiGLU in-kernel, and writes the final `3072` bf16 output.
   - This removes a large intermediate write/read.

2. **Uses MI300’s packed FP4 fast path**
   - It uses `tl.dot_scaled` over packed MXFP4 weights/scales rather than manually unpacking/dequantizing.
   - Live reintegration uses `rhs_k_pack=True`, which is the important packed-weight path.

3. **Specializes hard to tiny ragged MoE slices**
   - The distinctive sample6c tiling is:
     - `BLOCK_M=16`
     - `BLOCK_N=64`
     - `BLOCK_K=256`
     - `num_warps=8`
     - `num_stages=2`
   - In the live variant, we only use this for `gather_rows < 512`; larger cases fall back.

4. **Uses the ragged expert block schedule directly**
   - The kernel launches over scheduled expert-row blocks instead of trying to make the operation generic.
   - It masks padded `-1` schedule entries for CUDA-graph safety.

5. **Keeps GPT-OSS correctness details**
   - Adds stage-1 bias.
   - Uses OAI SwiGLU semantics: `alpha=1.702`, `limit=7.0`, clipping, and `s * (linear + 1)`.
   - Converts live routed indices back to token rows via `src_indx // 4`.
   - Guards exact shape/path so other activations or shapes stay on baseline.

Kernel-level result at `CONC=64`: baseline target row was `221.35 µs`; optimized replacement was `176.36 µs`, a `1.255x` kernel speedup / `20.33%` reduction. It is interesting because the win is not just "better tile size"; it is a fused, packed-MXFP4, ragged-MoE-specific stage-1 replacement with an aggressive `BLOCK_K=256` tiny-row policy.