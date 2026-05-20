# Qwen3 fused shared expert gate

Qwen3-Next-80b shared-expert gate path fusion

Summary: Today, vLLM does that work as three separate GPU steps: compute a scalar gate from the token row, apply sigmoid, then multiply the shared-expert output by that scalar. The patch replaces those with one specialized Triton kernel that performs the same row-local work in a single pass.

Throughput gain: preliminary results suggest this provides a meaningful throughput boost across concurrencies and ISL/OSL combos of 2.5 - 4%. However, we are rerunning a more detailed analysis across all sweeps.
At concurrency=16:
1024/1024: 1243.82 -> 1293.60 tok/s (+4.00%)
1024/8192: 996.55 -> 1033.11 tok/s (+3.67%)
8192/1024: 538.68 -> 552.50 tok/s (+2.57%)
8192/8192: 663.87 -> 681.92 tok/s (+2.72%)

Details: please find details of this change, including an easy patch you can use to run it yourself, in the :thread:

Re-integration path: I recommend your team makes a PR directly into vllm with this fix.

my thoughts: this is an interesting but nontrivial kernel fusion that will provide wins for both qwen2 and qwen3. While this patch is specific to the qwen2_moe.py path (which qwen3-80b seems to use), it can be applied equally to the qwen3_moe.py path as well. Therefore I expect we might also see wins in Qwen3.5 397B A17B / Qwen3.5 122B A10B / Qwen3.5 35B A3B if they use the same vllm qwen3 moe path.

## Context

`Qwen3-Next-80B-A3B-Instruct-FP8` reaches its shared expert through vLLM's
`Qwen2MoeMLP` implementation in `vllm.model_executor.models.qwen2_moe`.
The stock path computes a learned scalar gate after the shared-expert MLP:

```python
if self.expert_gate is not None:
    out = F.sigmoid(self.expert_gate(x)[0]) * out
```

For Qwen3-Next, `x` and `out` have width `2048`, while `expert_gate.weight`
has shape `[1, 2048]`. The stock expression is effectively three GPU steps:

1. a skinny `2048 -> 1` linear,
2. a sigmoid over `[num_tokens, 1]`,
3. a broadcast multiply into `[num_tokens, 2048]`.

The specialization below fuses that row-local work into one Triton kernel:
load one row of `x`, reduce it against the single gate-weight row, apply
sigmoid, then multiply the corresponding row of `out`.

On MI355, using the real shared-gate shapes observed for Qwen3-Next
(`1024x2048`, `7177x2048`, `8192x2048`), the fused helper was `1.49-1.76x`
faster than the Torch reference. In the full `ISL=1024`, `OSL=1024`
throughput sweep, it improved mean output throughput from `1801.48` to
`1885.34 tok/s` across concurrencies `4, 16, 32, 64` (`+4.65%`).

## Reference

The equivalent unfused reference is:

```python
import torch
import torch.nn.functional as F


def reference_shared_expert_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return F.sigmoid(F.linear(x, weight)) * out
```

## Optimized implementation

Add the import near the other vLLM imports:

```python
from vllm.triton_utils import tl, triton
```

Then add this helper to `vllm/model_executor/models/qwen2_moe.py` after
`logger = init_logger(__name__)`:

```python


@triton.jit
def _fused_shared_expert_gate_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    y_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * weight, axis=0))

    out = tl.load(out_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * K + offsets, out * gate, mask=mask)


def fused_shared_expert_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if (
        x.ndim != 2
        or out.ndim != 2
        or weight.ndim != 2
        or weight.shape[0] != 1
        or x.shape != out.shape
        or weight.shape[1] != x.shape[1]
    ):
        return F.sigmoid(F.linear(x, weight)) * out

    y = torch.empty_like(out)
    _fused_shared_expert_gate_kernel[(x.shape[0],)](
        x,
        weight,
        out,
        y,
        K=x.shape[1],
        BLOCK_K=triton.next_power_of_2(x.shape[1]),
        num_warps=8,
    )
    return y
```

Then replace the shared-expert gate call inside `Qwen2MoeMLP.forward`:

```python
if self.expert_gate is not None:
    out = fused_shared_expert_gate(x, self.expert_gate.weight, out)
```

## Exact installation

The script below applies the specialization to the currently installed vLLM
package, creates a one-time backup next to the target file, and clears the
module `__pycache__`. It was written against vLLM `0.19.1` and intentionally
fails loudly if the expected anchors move.

```bash
python3 - <<'PY'
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

module_name = "vllm.model_executor.models.qwen2_moe"
spec = importlib.util.find_spec(module_name)
if spec is None or spec.origin is None:
    raise SystemExit(f"could not resolve {module_name}")

target = Path(spec.origin).resolve()
backup = target.with_suffix(target.suffix + ".before_qwen3_shared_gate_fusion")
if not backup.exists():
    shutil.copy2(target, backup)

text = target.read_text()

import_anchor = "from vllm.sequence import IntermediateTensors\n"
import_line = "from vllm.triton_utils import tl, triton\n"
if import_line not in text:
    if import_anchor not in text:
        raise SystemExit("import anchor not found")
    text = text.replace(import_anchor, import_anchor + import_line, 1)

helper_marker = "logger = init_logger(__name__)\n\n\n"
helper = '''logger = init_logger(__name__)\n\n\n@triton.jit\ndef _fused_shared_expert_gate_kernel(\n    x_ptr,\n    weight_ptr,\n    out_ptr,\n    y_ptr,\n    K: tl.constexpr,\n    BLOCK_K: tl.constexpr,\n):\n    row = tl.program_id(0)\n    offsets = tl.arange(0, BLOCK_K)\n    mask = offsets < K\n\n    x = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)\n    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)\n    gate = tl.sigmoid(tl.sum(x * weight, axis=0))\n\n    out = tl.load(out_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)\n    tl.store(y_ptr + row * K + offsets, out * gate, mask=mask)\n\n\ndef fused_shared_expert_gate(\n    x: torch.Tensor,\n    weight: torch.Tensor,\n    out: torch.Tensor,\n) -> torch.Tensor:\n    if (\n        x.ndim != 2\n        or out.ndim != 2\n        or weight.ndim != 2\n        or weight.shape[0] != 1\n        or x.shape != out.shape\n        or weight.shape[1] != x.shape[1]\n    ):\n        return F.sigmoid(F.linear(x, weight)) * out\n\n    y = torch.empty_like(out)\n    _fused_shared_expert_gate_kernel[(x.shape[0],)](\n        x,\n        weight,\n        out,\n        y,\n        K=x.shape[1],\n        BLOCK_K=triton.next_power_of_2(x.shape[1]),\n        num_warps=8,\n    )\n    return y\n\n\n'''
if "def fused_shared_expert_gate(" not in text:
    if helper_marker not in text:
        raise SystemExit("helper insertion marker not found")
    text = text.replace(helper_marker, helper, 1)

old_call = "            out = F.sigmoid(self.expert_gate(x)[0]) * out\n"
new_call = "            out = fused_shared_expert_gate(x, self.expert_gate.weight, out)\n"
if new_call not in text:
    if old_call not in text:
        raise SystemExit("shared expert gate call site not found")
    text = text.replace(old_call, new_call, 1)

target.write_text(text)
pycache = target.parent / "__pycache__"
if pycache.exists():
    shutil.rmtree(pycache)

print(f"installed into: {target}")
print(f"backup saved at: {backup}")
PY
```

## Quick correctness check

Run this on a GPU after installation:

```bash
python3 - <<'PY'
import torch
import torch.nn.functional as F

from vllm.model_executor.models.qwen2_moe import fused_shared_expert_gate

torch.manual_seed(0)
x = torch.randn((1024, 2048), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((1, 2048), device="cuda", dtype=torch.bfloat16)
out = torch.randn((1024, 2048), device="cuda", dtype=torch.bfloat16)

expected = F.sigmoid(F.linear(x, weight)) * out
actual = fused_shared_expert_gate(x, weight, out)
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, atol=3.125e-2, rtol=2e-2)
print("ok")
PY
```