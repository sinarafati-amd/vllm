#!/usr/bin/env python3
"""Install or restore the GPT-OSS MI355X M=64 GEMM specialization.

This script is intentionally standalone so it can be copied into a serving
container and executed before vLLM imports aiter. It patches the installed
Python module that defines ``aiter.ops.triton.gemm.basic.gemm_a16w16`` and
wraps ``gemm_a16w16`` with the generated exact-shape specialization.

Default install:

    python reintegrate_gemm_a16w16_m64.py install

Restore the original module:

    python reintegrate_gemm_a16w16_m64.py restore

The generated kernel requires the Artemis Triton 3.7 wheel in the runtime where
vLLM imports the patched module.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

TARGET_MODULE = "aiter.ops.triton.gemm.basic.gemm_a16w16"
TARGET_SYMBOL = "gemm_a16w16"
GENERATED_SYMBOL = "generated_gemm_a16w16"
GENERATED_MODULE_NAME = "__artemis_gemm_a16w16_m64_top2p224"
PATCH_BEGIN = "# ARTEMIS_GEMM_A16W16_M64_PATCH_BEGIN"
PATCH_END = "# ARTEMIS_GEMM_A16W16_M64_PATCH_END"

DEFAULT_GENERATED_FILE = (
    Path(__file__).resolve().parent
    / "gemm_a16w16_mi355_m64_v1.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("install", "restore", "status"),
        help="install the wrapper, restore the backup, or print patch status",
    )
    parser.add_argument("--target-module", default=TARGET_MODULE)
    parser.add_argument(
        "--target-path",
        default=None,
        help="Patch this module file directly instead of resolving --target-module.",
    )
    parser.add_argument("--target-symbol", default=TARGET_SYMBOL)
    parser.add_argument("--generated-symbol", default=GENERATED_SYMBOL)
    parser.add_argument("--generated-module-name", default=GENERATED_MODULE_NAME)
    parser.add_argument(
        "--generated-file",
        type=Path,
        default=DEFAULT_GENERATED_FILE,
        help="Path to gemm_a16w16_mi355_m64_v1.py.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    return parser.parse_args()


def resolve_target_path(target_module: str, target_path: str | None) -> Path:
    if target_path:
        path = Path(target_path).resolve()
    else:
        spec = importlib.util.find_spec(target_module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve target module: {target_module}")
        path = Path(spec.origin).resolve()
    if path.suffix != ".py":
        raise RuntimeError(f"target must be a Python module, got: {path}")
    if not path.exists():
        raise RuntimeError(f"target path does not exist: {path}")
    return path


def backup_path_for(installed_path: Path) -> Path:
    return installed_path.with_suffix(installed_path.suffix + ".artemis_m64.bak")


def generated_path_for(installed_path: Path, generated_module_name: str) -> Path:
    return installed_path.parent / f"{generated_module_name}.py"


def strip_existing_patch(text: str) -> str:
    if PATCH_BEGIN not in text:
        return text.rstrip() + "\n"
    before, _sep, rest = text.partition(PATCH_BEGIN)
    if PATCH_END in rest:
        _patch, _sep, after = rest.partition(PATCH_END)
        return (before.rstrip() + "\n" + after.lstrip()).rstrip() + "\n"
    return before.rstrip() + "\n"


def build_patch_footer(
    *,
    generated_module_path: Path,
    target_symbol: str,
    generated_symbol: str,
) -> str:
    # Keep this footer dependency-free. It is appended to the installed aiter
    # module and runs when that module is imported by vLLM.
    return f'''

{PATCH_BEGIN}
def __artemis_gemm_a16w16_m64_patch_install():
    import importlib.util as __artemis_importlib_util

    __artemis_generated_path = {str(generated_module_path)!r}
    __artemis_target_symbol = {target_symbol!r}
    __artemis_generated_symbol = {generated_symbol!r}
    __artemis_spec = __artemis_importlib_util.spec_from_file_location(
        "__artemis_gemm_a16w16_m64_module",
        __artemis_generated_path,
    )
    if __artemis_spec is None or __artemis_spec.loader is None:
        raise RuntimeError(f"could not load generated GEMM module: {{__artemis_generated_path}}")
    __artemis_module = __artemis_importlib_util.module_from_spec(__artemis_spec)
    __artemis_spec.loader.exec_module(__artemis_module)

    __artemis_generated = getattr(__artemis_module, __artemis_generated_symbol)
    __artemis_original = globals()[__artemis_target_symbol]

    def __artemis_wrapped_gemm_a16w16(*args, **kwargs):
        replacement = __artemis_generated(*args, **kwargs)
        if replacement is NotImplemented:
            return __artemis_original(*args, **kwargs)
        return replacement

    __artemis_wrapped_gemm_a16w16.__name__ = getattr(
        __artemis_original,
        "__name__",
        __artemis_target_symbol,
    )
    __artemis_wrapped_gemm_a16w16.__doc__ = getattr(__artemis_original, "__doc__", None)
    __artemis_wrapped_gemm_a16w16.__module__ = getattr(__artemis_original, "__module__", __name__)
    globals()[__artemis_target_symbol] = __artemis_wrapped_gemm_a16w16


__artemis_gemm_a16w16_m64_patch_install()
{PATCH_END}
'''.rstrip() + "\n"


def clear_pycache(installed_path: Path) -> None:
    pycache = installed_path.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)


def install(args: argparse.Namespace) -> dict[str, object]:
    installed_path = resolve_target_path(args.target_module, args.target_path)
    generated_file = args.generated_file.resolve()
    if not generated_file.exists():
        raise RuntimeError(f"generated kernel file does not exist: {generated_file}")

    backup_path = backup_path_for(installed_path)
    generated_module_path = generated_path_for(installed_path, args.generated_module_name)

    if backup_path.exists():
        base_text = backup_path.read_text()
        shutil.copy2(backup_path, installed_path)
    else:
        base_text = installed_path.read_text()
        shutil.copy2(installed_path, backup_path)

    generated_module_path.write_text(generated_file.read_text())
    footer = build_patch_footer(
        generated_module_path=generated_module_path,
        target_symbol=args.target_symbol,
        generated_symbol=args.generated_symbol,
    )
    installed_path.write_text(strip_existing_patch(base_text).rstrip() + footer)
    clear_pycache(installed_path)

    return status_payload(args, installed_path=installed_path)


def restore(args: argparse.Namespace) -> dict[str, object]:
    installed_path = resolve_target_path(args.target_module, args.target_path)
    backup_path = backup_path_for(installed_path)
    generated_module_path = generated_path_for(installed_path, args.generated_module_name)
    if backup_path.exists():
        shutil.copy2(backup_path, installed_path)
        backup_path.unlink()
    else:
        installed_path.write_text(strip_existing_patch(installed_path.read_text()))
    generated_module_path.unlink(missing_ok=True)
    clear_pycache(installed_path)
    return status_payload(args, installed_path=installed_path)


def status_payload(args: argparse.Namespace, *, installed_path: Path | None = None) -> dict[str, object]:
    installed_path = installed_path or resolve_target_path(args.target_module, args.target_path)
    backup_path = backup_path_for(installed_path)
    generated_module_path = generated_path_for(installed_path, args.generated_module_name)
    text = installed_path.read_text() if installed_path.exists() else ""
    return {
        "target_module": args.target_module,
        "target_path": str(installed_path),
        "target_symbol": args.target_symbol,
        "generated_file": str(args.generated_file.resolve()),
        "generated_module_path": str(generated_module_path),
        "backup_path": str(backup_path),
        "patch_installed": PATCH_BEGIN in text and PATCH_END in text,
        "backup_exists": backup_path.exists(),
        "generated_module_exists": generated_module_path.exists(),
    }


def main() -> int:
    args = parse_args()
    if args.action == "install":
        payload = install(args)
    elif args.action == "restore":
        payload = restore(args)
    else:
        payload = status_payload(args)

    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise