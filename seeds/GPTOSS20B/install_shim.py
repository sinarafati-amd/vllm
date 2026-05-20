# ... put your imports here

@dataclasses.dataclass(frozen=True, slots=True)
class Mi300SwiGLUShimInstall:
    gpt_oss_moe_path: str
    gpt_oss_moe_backup_path: str
    shim_module_path: str


_RUNTIME_SOURCE_PATH = pathlib.Path(__file__).with_name(
    "mi300_optimized_swiglu_candidate_1.py"
)


def _shim_source() -> str:
    return _RUNTIME_SOURCE_PATH.read_text()


async def install_mi300_swiglu_shim(
    session: Any,
    *,
    timeout: int = 60,
) -> Mi300SwiGLUShimInstall:
    package_dir = (
        "/usr/local/lib/python3.12/dist-packages/"
        "vllm/model_executor/layers/fused_moe"
    )
    gpt_oss_moe_path = f"{package_dir}/gpt_oss_triton_kernels_moe.py"
    gpt_oss_moe_backup_path = f"{gpt_oss_moe_path}.codex_mi300_swiglu.bak"
    shim_module_path = f"{package_dir}/mi300_sample6c_swiglu_shim.py"
    stage1_old = (
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
    stage1_new = (
        "    if not maybe_run_sample6c_swiglu_stage1(\n"
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
		)

	UploadFile(
		path=shim_module_path,
		data=_shim_source()
	)

	script = textwrap.dedent(
	f"""
		python - <<'PY'
		import shutil
		from pathlib import Path

path = Path({gpt_oss_moe_path!r})
		backup_path = Path({gpt_oss_moe_backup_path!r})
		if backup_path.exists():
		shutil.copy2(backup_path, path)
			shutil.copy2(path, backup_path)

text = path.read_text()
		import_needle = "from vllm.utils.import_utils import has_triton_kernels\\n"
		import_replacement = (
		import_needle
			+ (
			"from .mi300_optimized_swiglu_candidate_1 import "
				"maybe_run_swiglu_stage1\\n"
				)
			)
		print("MI300_SHIM_IMPORT_NEEDLE_FOUND", import_needle in text)
		if "from .mi300_optimized_swiglu_candidate_1 import maybe_run_swiglu_stage1" not in text:
		if import_needle not in text:
			raise SystemExit("could not find GPT-OSS MoE import insertion point")
				text = text.replace(import_needle, import_replacement, 1)

stage1_old = {stage1_old!r}
		stage1_new = {stage1_new!r}
		print("MI300_SHIM_STAGE1_OLD_FOUND", stage1_old in text)
		if stage1_old not in text:
		raise SystemExit("could not find GPT-OSS stage1 matmul block")
			text = text.replace(stage1_old, stage1_new, 1)

path.write_text(text)
		pycache = path.parent / "__pycache__"
		if pycache.exists():
		shutil.rmtree(pycache)
			PY
		"""
		).strip()

	run_in_container(session, script)