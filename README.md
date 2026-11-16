# Why Does a Repository-Level Coding Agent Fail to Repay Self-Admitted Technical Debt? An Empirical Study of Agent Trajectories

This repository contains the replication package for the paper. It includes the sampled SATD dataset, evaluation inputs, recorded agent trajectories, analysis scripts, and reported results.

## Repository structure

```text
satd-fix/
├── LICENSE
├── README.md
├── requirements.txt
├── data/
├── execution/
│   ├── agent_fix/
│   ├── llm/
│   ├── requirements.txt
│   └── README.md
├── RQ1/
│   ├── data/
│   ├── results/
│   └── scripts/
├── RQ2/
│   ├── results/
│   └── scripts/
├── RQ3/
│   ├── data/
│   ├── results/
│   └── scripts/
└── RQ4/
    ├── results/
    └── scripts/
```

Analysis scripts are organized under each research question; execution scripts are under `execution/`.

## Package contents

- `data/`: the 500-instance dataset, agent and LLM evaluation workbooks, and recorded trajectories.
- `execution/`: scripts for running the repository-level coding agent and method-level LLM baseline.
- `RQ1/`–`RQ4/`: research-question data, results, and analysis scripts.
- `requirements.txt`: Python dependencies for the analysis scripts.

## Running the agent and LLM baseline

From the repository root, install the execution dependencies and set your API key. Do not store credentials in the repository.

```powershell
python -m pip install -r execution/requirements.txt
$env:OPENROUTER_API_KEY = '<your-api-key>'
```

Run the method-level LLM baseline:

```powershell
python execution/llm/run_llm_fix.py --dataset data/satd_dataset.xlsx --sheet samples --output-dir execution_runs/llm --models gpt-4o-2024-08-06 --templates NoExplain
```

For the repository-level agent, set the location for repository clones and run:

```powershell
$env:SATD_REPOS_ROOT = 'D:\satd-repositories'
python execution/agent_fix/OpenAI_agent/openai_agents_satd_fix.py --dataset data/satd_dataset.xlsx --sheet samples --repos-root $env:SATD_REPOS_ROOT --output-dir execution_runs/agent --worktrees-dir execution_runs/worktrees --model gpt-4o-2024-08-06
```

Repository clones are not included in this package. Both commands make live model API calls and may produce different outputs; existing evaluation results are included under `RQ1/`–`RQ4/`.

