#!/usr/bin/env python3
"""Install or restore the GPT-OSS-20B MI300 fused MXFP4 + SwiGLU stage-1
specialization in an existing vLLM serving container.

This script is intentionally standalone so it can be copied into a
serving container and executed before vLLM imports the GPT-OSS Triton-
kernels MoE module. It performs two operations:

1. Copies the standalone Triton module
   ``mi300_optimized_swiglu_candidate_1.py`` next to the installed
   ``vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe``.
2. Patches that module to import ``maybe_run_swiglu_stage1`` and insert
   a guarded fast-path in front of the existing stage-1 ``matmul_ogs``
   call, identical to the upstream-style change in
   ``0002-dispatch-stage1-to-mi300-fused-kernel-when-guarded.patch``.

Default install:

    python install_shim.py install

Restore the original module:

    python install_shim.py restore

Show patch status:

    python install_shim.py status
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

TARGET_MODULE = (
    "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe"
)
BACKUP_SUFFIX = ".pr_pundit_mi300_swiglu.bak"
GENERATED_MODULE_FILE = "mi300_optimized_swiglu_candidate_1.py"
PATCH_BEGIN = "# PR_PUNDIT_MI300_SWIGLU_BEGIN"
PATCH_END = "# PR_PUNDIT_MI300_SWIGLU_END"

IMPORT_NEEDLE = (
    "from vllm.utils.import_utils import has_triton_kernels\n"
)
IMPORT_REPLACEMENT = (
    IMPORT_NEEDLE
    + PATCH_BEGIN + "\n"
    + "from .mi300_optimized_swiglu_candidate_1 import "
    "maybe_run_swiglu_stage1\n"
    + PATCH_END + "\n"
)

STAGE1_OLD = (
    "    matmul_ogs(\n"
    "        hidden_states,\n"
    "        w1,\n"
    "        quant_config.w1_bias,\n"
    "        routing_data,\n"
    "        gather_indx=gather_indx,\n"
    "        precision_config=quant_config.w1_precision,\n"
    "        gammas=gammas if apply_router_weight_on_input else None,\n"
    "        fused_activation=act,\n"
    "        y=intermediate_cache,\n"
    "    )\n"
)
STAGE1_NEW = (
    PATCH_BEGIN + "\n"
    "    if not maybe_run_swiglu_stage1(\n"
    "        hidden_states,\n"
    "        w1,\n"
    "        routing_data,\n"
    "        gather_indx,\n"
    "        quant_config.w1_precision,\n"
    "        quant_config.w1_bias,\n"
    "        intermediate_cache.view(M * topk, N // 2),\n"
    "        apply_router_weight_on_input=apply_router_weight_on_input,\n"
    "        swiglu_alpha=swiglu_alpha,\n"
    "        swiglu_limit=swiglu_limit,\n"
    "    ):\n"
    "        matmul_ogs(\n"
    "            hidden_states,\n"
    "            w1,\n"
    "            quant_config.w1_bias,\n"
    "            routing_data,\n"
    "            gather_indx=gather_indx,\n"
    "            precision_config=quant_config.w1_precision,\n"
    "            gammas=gammas if apply_router_weight_on_input else None,\n"
    "            fused_activation=act,\n"
    "            y=intermediate_cache,\n"
    "        )\n"
    + PATCH_END + "\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("install", "restore", "status"),
        help="install the shim, restore the backup, or print patch status",
    )
    parser.add_argument("--target-module", default=TARGET_MODULE)
    parser.add_argument(
        "--target-path",
        default=None,
        help=(
            "Patch this file directly instead of resolving --target-module. "
            "Useful when the installed package lives in an unusual location."
        ),
    )
    parser.add_argument(
        "--generated-file",
        type=Path,
        default=Path(__file__).resolve().parent / GENERATED_MODULE_FILE,
        help=f"Path to {GENERATED_MODULE_FILE}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    return parser.parse_args()


def resolve_target_path(target_module: str, target_path: str | None) -> Path:
    if target_path:
        return Path(target_path).resolve()
    spec = importlib.util.find_spec(target_module)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"cannot resolve target module: {target_module} "
            "(is vLLM importable in this Python environment?)"
        )
    return Path(spec.origin).resolve()


def install(args: argparse.Namespace) -> dict:
    target = resolve_target_path(args.target_module, args.target_path)
    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    text = target.read_text()
    already = PATCH_BEGIN in text

    if not already:
        if backup.exists():
            shutil.copy2(backup, target)
            text = target.read_text()
        else:
            shutil.copy2(target, backup)

        if IMPORT_NEEDLE not in text:
            raise RuntimeError(
                "could not find import insertion point "
                f"({IMPORT_NEEDLE!r}) in {target}; vLLM version mismatch?"
            )
        if STAGE1_OLD not in text:
            raise RuntimeError(
                "could not find stage-1 matmul_ogs block in "
                f"{target}; vLLM version mismatch?"
            )
        text = text.replace(IMPORT_NEEDLE, IMPORT_REPLACEMENT, 1)
        text = text.replace(STAGE1_OLD, STAGE1_NEW, 1)
        target.write_text(text)

    generated_src = args.generated_file
    if not generated_src.is_file():
        raise RuntimeError(
            f"generated kernel file not found: {generated_src}"
        )
    generated_dst = target.parent / GENERATED_MODULE_FILE
    shutil.copy2(generated_src, generated_dst)

    pycache = target.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    return {
        "action": "install",
        "target": str(target),
        "backup": str(backup),
        "generated": str(generated_dst),
        "patch_already_present": already,
    }


def restore(args: argparse.Namespace) -> dict:
    target = resolve_target_path(args.target_module, args.target_path)
    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    restored = False
    if backup.exists():
        shutil.copy2(backup, target)
        restored = True

    generated_dst = target.parent / GENERATED_MODULE_FILE
    if generated_dst.is_file():
        generated_dst.unlink()

    pycache = target.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    return {
        "action": "restore",
        "target": str(target),
        "restored_from_backup": restored,
    }


def status(args: argparse.Namespace) -> dict:
    target = resolve_target_path(args.target_module, args.target_path)
    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    text = target.read_text() if target.exists() else ""
    return {
        "action": "status",
        "target": str(target),
        "backup_exists": backup.exists(),
        "patch_present": PATCH_BEGIN in text,
        "generated_present": (target.parent / GENERATED_MODULE_FILE).exists(),
    }


def main() -> None:
    args = parse_args()
    handlers = {"install": install, "restore": restore, "status": status}
    result = handlers[args.action](args)
    if args.json:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
    else:
        for k, v in result.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
