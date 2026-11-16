# Execution scripts

This folder contains the OpenAI Agents SDK runner, the direct LLM baseline
runner, and the small support modules they import. Both runners use the
included `../data/satd_dataset.xlsx` workbook (`samples` sheet) by default.

Install the execution dependencies with `pip install -r execution/requirements.txt`.
Provide API credentials through environment variables; do not put credentials
in this package. The LLM runner accepts `OPENAI_API_KEY` or
`OPENROUTER_API_KEY`. The agent runner uses OpenRouter by default and accepts
`OPENROUTER_API_KEY`; pass `--model-provider openai` to use `OPENAI_API_KEY`.

## LLM baseline

From the package root, run:

```powershell
python execution/llm/run_llm_fix.py --dataset data/satd_dataset.xlsx --sheet samples --output-dir execution_runs/llm --models gpt-4o-2024-08-06 --templates NoExplain
```

## Agent

The agent runner clones repositories referenced by the dataset when they are
not already present. Set `SATD_REPOS_ROOT` to a location for those clones, or
pass `--repos-root`. Clones are not included in this package. The runner creates
temporary isolated worktrees under `execution_runs/worktrees` and writes its
logs, trajectories, and result files under `execution_runs/agent`.

From the package root, run:

```powershell
$env:SATD_REPOS_ROOT = 'D:\satd-repositories'
python execution/agent_fix/OpenAI_agent/openai_agents_satd_fix.py --dataset data/satd_dataset.xlsx --sheet samples --repos-root $env:SATD_REPOS_ROOT --output-dir execution_runs/agent --worktrees-dir execution_runs/worktrees --model gpt-4o-2024-08-06
```

These commands make live model/API calls and create run artifacts. The package
includes the analysis inputs and results separately; rerunning either system
may produce different model responses and outcomes.
