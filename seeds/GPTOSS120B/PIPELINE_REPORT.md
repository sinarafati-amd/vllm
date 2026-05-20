# pr-pundit Pipeline Report — GPT-OSS-120B MI355X gemm_a16w16 M=64

Date: 2026-05-19
Seed: `/home/sirafati/GPTOSS-120B`
Upstream: `https://github.com/vllm-project/vllm`
Staging: `https://github.com/sinarafati-amd/vllm`

## Summary

The `pr-pundit` `plan_pr_series` MCP tool was invoked four times. The seed
was successfully fetched on every attempt that used a GitHub URL, but the
server-side LLM pipeline crashed at the first DSPy stage every time with
the same fatal error:

```
dspy.settings can only be changed by the thread that initially configured it.
```

This is a server-side bug in the pr-pundit MCP service (not a client-side
or seed-side issue). The full plan, commit messages, PR descriptions, and
benchmark scripts that the tool normally returns via `get_plan()` could not
be produced. As a fallback, this report includes a hand-built plan
(`PLAN.md`), the upstream rules dataset that the tool would have used to
judge each PR (`upstream_rules_vllm-project_vllm.json`), and explicit
manual push instructions (`PUSH_INSTRUCTIONS.md`).

## Pipeline runs

| Run ID     | Plan ID    | Seed URL                                                                                            | Result      | Where it died                       |
|------------|------------|-----------------------------------------------------------------------------------------------------|-------------|-------------------------------------|
| `c97646e4` | `f56a4111` | `/home/sirafati/GPTOSS-120B` (local path)                                                           | error       | `seed_fetched` (path not visible to remote MCP) |
| `6f3af608` | `aea17b90` | `https://github.com/sinarafati-amd/vllm/tree/pr-pundit-seed-gptoss120b-gemm-a16w16/seeds/GPTOSS120B` | error       | `seed_fetched` (Contents API 404 — branch ref dropped) |
| `175ca875` | `7ed1548f` | `https://github.com/sinarafati-amd/vllm/tree/main/seeds/GPTOSS120B`                                  | error       | `stage_intent_extraction` (DSPy threading) |
| `3ea00d20` | `77632678` | `https://github.com/sinarafati-amd/vllm/tree/main/seeds/GPTOSS120B`                                  | error       | `stage_intent_extraction` (DSPy threading) |
| `0a6e89ae` | `fbfdd646` | `https://github.com/sinarafati-amd/vllm/tree/main/seeds/GPTOSS120B`                                  | error       | `stage_intent_extraction` (DSPy threading) |
| `652478cd` | `56bb2ac2` | `https://github.com/sinarafati-amd/vllm/tree/main/seeds/GPTOSS120B` + `target_tier=fast-adoption`    | error       | `stage_intent_extraction` (DSPy threading) |

The last four runs all advanced past the seed-fetch step (the server
reported `n_file_edits=2, has_readme=true`), so the seed is correctly
shaped for the tool. The crash is reliably at the first LLM call.

## What did succeed

- **`get_rules("https://github.com/vllm-project/vllm")`** — returned the
  full 557-rule merge-rules dataset (~1.87 MB). Saved to
  `upstream_rules_vllm-project_vllm.json` in this folder. The hand plan
  below cites the relevant rule IDs/numbers so PR authors can self-check.
- **Seed ingestion** — the server confirmed both kernel files plus
  README were parsed correctly as 2 file edits, so when the server bug
  is fixed the same seed can be re-submitted unchanged.

## Suggested next steps for the pr-pundit maintainer

The DSPy fault means `dspy.configure()` was called on a thread that no
longer owns the settings object when the planner stage runs. Likely fix
sites:

- Move `dspy.configure(...)` into the same worker thread that calls the
  planner module, OR wrap each stage entry point with
  `with dspy.settings.context(...):` per DSPy's documented
  thread-safe configuration pattern.
- Initialize DSPy lazily in the worker, not at module import on the
  HTTP server's main thread.

Once fixed, re-run with:

```bash
# from the IDE agent's MCP context
plan_pr_series(
    seed_url="https://github.com/sinarafati-amd/vllm/tree/main/seeds/GPTOSS120B",
    upstream_repo_url="https://github.com/vllm-project/vllm",
    staging_repo_url="https://github.com/sinarafati-amd/vllm",
)
```

The seed on `main` of the staging fork is at commit
`7f4335aef` (`seed: GPT-OSS-120B MI355X gemm_a16w16 M=64 specialization`)
and is reachable both via the tree URL above and via
`https://api.github.com/repos/sinarafati-amd/vllm/contents/seeds/GPTOSS120B`
(returns 200).
