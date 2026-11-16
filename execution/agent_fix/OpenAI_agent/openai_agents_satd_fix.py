"""Run the OpenAI SDK agent in pre-fix repositories and record repairs and trajectories."""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from agents import Agent, ModelSettings, MultiProvider, RunConfig, RunHooks, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded


AGENT_FIX_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BASELINE_DIR.parent
DATASET_PATH = PROJECT_DIR / "data" / "satd_dataset.xlsx"
DATASET_SHEET = "samples"
OUTPUT_DIR = PROJECT_DIR / "execution_runs" / "agent"
DEFAULT_REPOS_ROOT = Path(os.getenv("SATD_REPOS_ROOT", PROJECT_DIR / "external_repositories"))
AGENT_NAME = "OpenAIAgentsSATDAgent"
TEMPLATE = "OpenAIAgents"

sys.path.insert(0, str(AGENT_FIX_DIR))
sys.path.insert(0, str(BASELINE_DIR))
from SATD_tool import get_deleted_and_inserted_lines, save_to_json
import fix_satd_eval
from helpers import (
    agent_from_git,
    clean_agent_error,
    clean_worktree,
    compact_json,
    excel_safe_df,
    method_diff,
    load_project_env,
    normalize_dataset_for_runner,
    render_prompt_template,
    repo_dir,
    setup_execution_logger,
    target_method_matches_dataset_before,
    validate_runner_dataset,
    write_satd_metadata,
    write_jsonl,
)
from repo_tools import RepoToolkit
from git_history_isolation import prepare_isolated_worktree
from satd_diagnostics import (
    analyze_evidence_use,
    collect_satd_diagnostics,
    diagnostic_prompt_block,
    diagnostic_trajectory_step,
    evidence_use_trajectory_step,
)

PROJECT_ENV_PATH = PROJECT_DIR / ".env"
CONF_ENV_PATH = PROJECT_DIR / "conf" / ".env"
load_project_env(PROJECT_ENV_PATH)
load_project_env(CONF_ENV_PATH)
DEFAULT_MODEL = "gpt-4o-2024-08-06"


SYSTEM_PROMPT = "Follow the user task and work only inside the provided repository worktree."


def command_slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "value"


def json_safe(value, depth=0, max_depth=10):
    if depth > max_depth:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v, depth + 1, max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, depth + 1, max_depth) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(), depth + 1, max_depth)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            "__type__": type(value).__name__,
            **{
                str(k): json_safe(v, depth + 1, max_depth)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            },
        }
    return repr(value)

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def render_task(row, worktree, diagnostic=None, include_diagnostic=False):
    prompt = render_prompt_template(
        "main_prompt.yaml", "main_template",
        repo_path=str(worktree),
        file_path=str(row["url_file_path"]),
        method_name=str(row.get("method_name", "")),
        satd_comment=str(row.get("SATD_comment", "")),
        method_before_repayment=str(row["containing_method_before_repayment"]),
    )
    if include_diagnostic and diagnostic:
        prompt += diagnostic_prompt_block(diagnostic)
    return prompt


def safe_join(worktree, relative_path):
    root = Path(worktree).resolve()
    target = (root / str(relative_path).replace("\\", "/")).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Path escapes worktree: {relative_path}")
    return target


def compact_text(text, limit=20000):
    text = str(text or "")
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated: {len(text)} chars total]"


def make_command_tool(worktree, command_timeout):
    """Create the controlled command tool."""
    toolkit = RepoToolkit(worktree, command_timeout)

    @function_tool
    def run_command(command: str, timeout_seconds: int = 60) -> str:
        """Run a shell command in the repository worktree."""
        return toolkit.run_command(command, timeout_seconds)

    return run_command


def make_function_tools(worktree, command_timeout):
    """Wrap repository operations as agent-visible function tools."""
    toolkit = RepoToolkit(worktree, command_timeout)

    @function_tool
    def list_directory(path: str = ".") -> str:
        """List files and directories under a path inside the repository worktree."""
        return toolkit.list_directory(path)

    @function_tool
    def explore_worktree(path: str = ".", max_depth: int = 2, max_entries: int = 200) -> str:
        """Summarize the worktree structure using ignore-aware traversal."""
        return toolkit.explore_worktree(path, max_depth, max_entries)

    @function_tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 300) -> str:
        """Read a text file from the repository worktree."""
        return toolkit.read_file(path, start_line, max_lines)

    @function_tool
    def write_file(path: str, content: str) -> str:
        """Write a complete text file inside the repository worktree."""
        return toolkit.write_file(path, content)

    @function_tool
    def replace_in_file(path: str, old: str, new: str, count: int = 1) -> str:
        """Replace text in a file inside the repository worktree."""
        return toolkit.replace_in_file(path, old, new, count)

    @function_tool
    def search_text(pattern: str, path_glob: str = "", max_results: int = 50) -> str:
        """Search repository text using ripgrep when available."""
        return toolkit.search_text(pattern, path_glob, max_results)

    @function_tool
    def read_target_method(path: str, method_name: str, max_lines: int = 260) -> str:
        """Read a target function/method using lizard when available, with ast fallback for Python."""
        return toolkit.read_target_method(path, method_name, max_lines)

    @function_tool
    def inspect_satd_context(path: str, satd_comment: str, context_lines: int = 25) -> str:
        """Locate the SATD comment and report the surrounding method/context."""
        return toolkit.inspect_satd_context(path, satd_comment, context_lines)

    @function_tool
    def get_git_status() -> str:
        """Return git status for the current repository worktree."""
        return toolkit.git_status()

    @function_tool
    def get_git_diff(max_chars: int = 40000) -> str:
        """Return the current repository diff for patch inspection."""
        return toolkit.git_diff(max_chars)

    return [
        explore_worktree,
        list_directory,
        read_file,
        write_file,
        replace_in_file,
        search_text,
        read_target_method,
        inspect_satd_context,
        get_git_status,
        get_git_diff,
        make_command_tool(worktree, command_timeout),
    ]


class RunLogger:
    def __init__(self, path, rand_index):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rand_index = rand_index
        self.events = []
        self.handle = self.path.open("w", encoding="utf-8")

    def event(self, event_name, **payload):
        safe_payload = json_safe(payload)
        for reserved in ("event", "rand_index", "timestamp"):
            safe_payload.pop(reserved, None)
        record = {
            "event": event_name,
            "rand_index": self.rand_index,
            "timestamp": now_iso(),
            **safe_payload,
        }
        self.events.append(record)
        json.dump(record, self.handle, ensure_ascii=False, default=str)
        self.handle.write("\n")
        self.handle.flush()
        return record

    def close(self):
        self.handle.close()


def should_snapshot_tool_diff(tool_name):
    """Only snapshot full diffs for tools that can plausibly modify files."""
    name = str(tool_name or "").lower()
    if "git_diff" in name or "get_git_diff" in name:
        return False
    mutating_terms = (
        "write",
        "edit",
        "replace",
        "delete",
        "move",
        "rename",
        "create",
        "run_command",
        "shell",
        "bash",
    )
    return any(term in name for term in mutating_terms)


class LoggingHooks(RunHooks):
    def __init__(self, run_logger, worktree):
        self.log = run_logger
        self.worktree = worktree

    async def on_agent_start(self, context, agent):
        self.log.event("agent_start", agent=getattr(agent, "name", ""))

    async def on_agent_end(self, context, agent, output):
        self.log.event("agent_end", agent=getattr(agent, "name", ""), output=output)

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.log.event(
            "llm_start",
            agent=getattr(agent, "name", ""),
            system_prompt=system_prompt,
            input_items=input_items,
        )

    async def on_llm_end(self, context, agent, response):
        self.log.event("llm_end", agent=getattr(agent, "name", ""), response=response)

    async def on_tool_start(self, context, agent, tool):
        tool_name = getattr(context, "tool_name", "") or getattr(tool, "name", "")
        tool_args = getattr(context, "tool_arguments", None)
        self.log.event(
            "tool_start",
            agent=getattr(agent, "name", ""),
            tool=tool_name,
            tool_call_id=getattr(context, "tool_call_id", ""),
            arguments=tool_args,
        )

    async def on_tool_end(self, context, agent, tool, result):
        tool_name = getattr(context, "tool_name", "") or getattr(tool, "name", "")
        snapshot_diff = should_snapshot_tool_diff(tool_name)
        self.log.event(
            "tool_end",
            agent=getattr(agent, "name", ""),
            tool=tool_name,
            tool_call_id=getattr(context, "tool_call_id", ""),
            result=compact_text(result, limit=30000),
            git_status_after=git_status_safe(self.worktree),
            git_diff_after_included=snapshot_diff,
            git_diff_after=git_diff_safe(self.worktree) if snapshot_diff else "",
        )

    async def on_handoff(self, context, from_agent, to_agent):
        self.log.event(
            "handoff",
            from_agent=getattr(from_agent, "name", ""),
            to_agent=getattr(to_agent, "name", ""),
        )


def git_status_safe(worktree):
    try:
        completed = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=20,
        )
        return (completed.stdout or "").rstrip("\r\n")
    except Exception as exc:
        return f"git_status_error: {clean_agent_error(exc)}"


def git_diff_safe(worktree):
    try:
        completed = subprocess.run(
            ["git", "-C", str(worktree), "diff", "--binary"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
        )
        return compact_text(completed.stdout or "", limit=40000)
    except Exception as exc:
        return f"git_diff_error: {clean_agent_error(exc)}"


def result_events(result):
    return {
        "final_output": json_safe(getattr(result, "final_output", "")),
        "to_input_list": json_safe(result.to_input_list()) if hasattr(result, "to_input_list") else [],
        "new_items": json_safe(getattr(result, "new_items", [])),
        "raw_responses": json_safe(getattr(result, "raw_responses", [])),
        "last_agent": getattr(getattr(result, "last_agent", None), "name", ""),
        "usage": json_safe(getattr(result, "usage", {})),
    }


def hook_events_to_tool_steps(events):
    steps = []
    for event in events or []:
        if event.get("event") != "tool_start":
            continue
        tool = str(event.get("tool", ""))
        arguments = event.get("arguments", "")
        steps.append({
            "event": "tool_use",
            "tool": tool,
            "action": f"{tool} {json.dumps(arguments, ensure_ascii=False, default=str)}",
            "tool_input": arguments,
        })
    return steps


def make_model_settings(args):
    extra_body = None
    if args.reasoning_effort:
        extra_body = {"reasoning": {"effort": args.reasoning_effort}}
    return ModelSettings(
        temperature=args.temperature,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        extra_body=extra_body,
    )


def make_model_provider(args):
    if args.model_provider == "openrouter":
        api_key = os.getenv(args.openrouter_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"OpenRouter selected but {args.openrouter_api_key_env} is not set."
            )
        return MultiProvider(
            openai_api_key=api_key,
            openai_base_url=args.openrouter_base_url,
            openai_use_responses=False,
            openai_prefix_mode="model_id",
            unknown_prefix_mode="model_id",
        )
    return MultiProvider(openai_use_responses=True)


def make_sdk_run_config(args):
    return RunConfig(
        model_provider=make_model_provider(args),
        workflow_name=f"{AGENT_NAME}:{TEMPLATE}",
        trace_metadata={
            "agent": AGENT_NAME,
            "template": TEMPLATE,
            "model_provider": args.model_provider,
        },
    )


async def run_openai_agent(task_text, worktree, args, row_log):
    try:
        import agents  
    except Exception as exc:
        raise RuntimeError("OpenAI Agents SDK is not installed. Install with: python -m pip install openai-agents") from exc

    model_settings = make_model_settings(args)
    sdk_run_config = make_sdk_run_config(args)
    hooks = LoggingHooks(row_log, worktree)
    row_log.event("user_prompt", content=task_text)
    row_log.event("tool_backend", resolved="function")
    row_log.event(
        "model_provider",
        provider=args.model_provider,
        base_url=args.openrouter_base_url if args.model_provider == "openrouter" else "",
        api_key_env=args.openrouter_api_key_env if args.model_provider == "openrouter" else "",
        api_key_present=bool(os.getenv(args.openrouter_api_key_env, "").strip())
        if args.model_provider == "openrouter" else bool(os.getenv("OPENAI_API_KEY", "").strip()),
    )
    try:
        agent = Agent(
            name="satd_repair",
            model=args.model,
            instructions=SYSTEM_PROMPT,
            tools=make_function_tools(worktree, args.command_timeout),
            model_settings=model_settings,
        )
        result = await Runner.run(
            agent, task_text, max_turns=args.max_turns,
            hooks=hooks, run_config=sdk_run_config
        )
        payload = result_events(result)
        row_log.event("run_result", **payload)
        return {
            "returncode": 0,
            "events": row_log.events,
            "final_output": str(payload.get("final_output", "")),
            **payload,
        }
    except MaxTurnsExceeded:
        row_log.event("max_turns_exceeded", max_turns=args.max_turns)
        return {
            "returncode": 2,
            "events": row_log.events,
            "final_output": "",
            "to_input_list": [],
            "new_items": [],
            "raw_responses": [],
            "usage": {},
        }


def write_trajectory(path, row, args, baseline, task_text, diagnostic, run_result, evidence, agent):
    trajectory = []
    if diagnostic:
        trajectory.append(diagnostic_trajectory_step(diagnostic, len(trajectory), attempt=0))
    for event in run_result.get("events", []):
        trajectory.append({
            "step": len(trajectory),
            "event_index": len(trajectory),
            "timestamp": event.get("timestamp", ""),
            "rand_index": int(row["rand_index"]),
            "attempt": 1,
            "event": event.get("event", ""),
            "response": json.dumps(event.get("output", event.get("final_output", "")), ensure_ascii=False, default=str),
            "thought": "",
            "action": json.dumps({
                "tool": event.get("tool", ""),
                "arguments": event.get("arguments", ""),
            }, ensure_ascii=False, default=str) if event.get("event") in {"tool_start", "tool_end"} else "",
            "observation": json.dumps(event, ensure_ascii=False, default=str),
            "state": json.dumps({
                "git_status_after": event.get("git_status_after", ""),
            }, ensure_ascii=False),
            "query": event.get("input_items", []) if event.get("event") == "llm_start" else [],
            "query_exact": event.get("event") == "llm_start",
        })
    if evidence:
        trajectory.append(evidence_use_trajectory_step(evidence, len(trajectory), attempt=1))

    payload = {
        "environment": "openai_agents_sdk",
        "instance_id": str(int(row["rand_index"])),
        "agent": AGENT_NAME,
        "model": args.model,
        "task": {
            "rand_index": int(row["rand_index"]),
            "repo_slug": str(row.get("repo_slug", "")),
            "language": str(row.get("language", "")),
            "target_file": str(row.get("url_file_path", "")),
            "target_method": str(row.get("method_name", "")),
            "satd_comment": str(row.get("SATD_comment", "")),
        },
        "baseline": baseline,
        "prompt": task_text,
        "diagnostic": diagnostic or {},
        "trajectory": trajectory,
        "sdk_result": {
            "returncode": run_result.get("returncode"),
            "final_output": run_result.get("final_output", ""),
            "to_input_list": run_result.get("to_input_list", []),
            "new_items": run_result.get("new_items", []),
            "raw_responses": run_result.get("raw_responses", []),
            "usage": run_result.get("usage", {}),
        },
        "evidence_use": evidence or {},
        "result": agent,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, default=str, indent=2)


async def process_row(row, args, repos_root, worktrees_root, logger, output_paths):
    rand_index = int(row["rand_index"])
    repo = repo_dir(repos_root, str(row["repo_slug"]))
    # Verify repository isolation before invoking the model.
    worktree, visible_commit, visible_date, fix_date, isolation = prepare_isolated_worktree(
        repo, row, worktrees_root
    )
    baseline = {
        "agent_visible_commit": visible_commit,
        "agent_visible_commit_date": visible_date,
        "fix_commit_date": fix_date,
        "git_history_isolation": isolation,
    }
    if not target_method_matches_dataset_before(worktree, row):
        logger.warning("Baseline target-method mismatch rand_index=%s", rand_index)

    diagnostic, evidence = {}, {}
    task_text = ""
    row_log = None
    run_result = {"returncode": 1, "events": [], "final_output": "", "usage": {}}
    raw = ""
    try:
        if args.satd_diagnostics != "off":
            diagnostic = collect_satd_diagnostics(row, worktree, max_items=args.satd_diagnostic_max_items)
            diagnostic_path = output_paths["diagnostics_dir"] / f"{rand_index}.diagnostic.json"
            diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        else:
            diagnostic_path = ""

        task_text = render_task(
            row, worktree, diagnostic=diagnostic,
            include_diagnostic=args.satd_diagnostics == "prompt",
        )
        task_path = output_paths["tasks_dir"] / f"{rand_index}.md"
        task_path.write_text(task_text, encoding="utf-8")

        activity_path = output_paths["stdout_dir"] / f"{rand_index}.jsonl"
        row_log = RunLogger(activity_path, rand_index)
        if diagnostic:
            row_log.event("satd_diagnostic", **diagnostic)

        run_result = await asyncio.wait_for(
            run_openai_agent(task_text, worktree, args, row_log),
            timeout=args.timeout_seconds,
        )
        raw = "\n".join(json.dumps(event, ensure_ascii=False, default=str) for event in row_log.events)
        agent = agent_from_git(row, run_result.get("final_output", ""), raw, worktree)
        evidence = analyze_evidence_use(
            row, worktree, hook_events_to_tool_steps(run_result.get("events", [])), agent, diagnostic
        )
        if args.satd_diagnostics != "off":
            evidence_path = output_paths["diagnostics_dir"] / f"{rand_index}.evidence_use.attempt1.json"
            evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        else:
            evidence_path = ""

        agent.update({
            "attempts": 1,
            "model_requested": args.model,
            "model_reported": args.model,
            "num_turns_last": "",
            "num_turns_total": "",
            "result_subtype": f"returncode_{run_result.get('returncode')}",
            "usage": run_result.get("usage", {}),
            "final_answer_status": "changed" if agent.get("repayment_status") != "no_change" else "unchanged",
            "openai_agents_returncode": run_result.get("returncode"),
            "openai_agents_activity_path": str(activity_path),
            "task_path": str(task_path),
            "satd_diagnostic": diagnostic,
            "satd_evidence_use": evidence,
            "satd_diagnostic_path": str(diagnostic_path),
            "satd_evidence_use_path": str(evidence_path),
        })
        with output_paths["activity_trace_path"].open("a", encoding="utf-8") as handle:
            for event in row_log.events:
                json.dump(event, handle, ensure_ascii=False, default=str)
                handle.write("\n")
    except Exception as exc:
        raw = f"OPENAI_AGENTS_ERROR: {clean_agent_error(exc)}"
        if row_log is not None:
            row_log.event("openai_agents_error", text=clean_agent_error(exc))
        agent = {
            "repayment_status": "failed",
            "updated_method": "",
            "summary": raw,
            "model_requested": args.model,
            "model_reported": args.model,
            "openai_agents_returncode": 1,
            "satd_diagnostic": diagnostic,
            "satd_evidence_use": evidence,
        }
    finally:
        if row_log is not None:
            row_log.close()
        traj_path = output_paths["trajectories_dir"] / f"{rand_index}.traj"
        agent["trajectory_path"] = str(traj_path)
        write_trajectory(traj_path, row, args, baseline, task_text, diagnostic, run_result, evidence, agent)
        metadata_path = write_satd_metadata(
            output_paths.get("metadata_dir"),
            row,
            args,
            baseline=baseline,
            run_config=output_paths.get("run_config", {}),
            agent=agent,
            extra={"git_history_isolation": isolation},
            paths={
                "task_path": agent.get("task_path", ""),
                "activity_path": agent.get("openai_agents_activity_path", ""),
                "trajectory_path": agent.get("trajectory_path", ""),
                "diagnostic_path": agent.get("satd_diagnostic_path", ""),
                "evidence_use_path": agent.get("satd_evidence_use_path", ""),
                "global_activity_trace_path": str(output_paths.get("activity_trace_path", "")),
            },
        )
        agent["metadata_path"] = metadata_path
        try:
            clean_worktree(worktree)
        except Exception as exc:
            logger.error("Failed to restore worktree rand_index=%s error=%s", rand_index, exc)

    return agent, raw, baseline


def row_result(row, agent, baseline):
    before = str(row["containing_method_before_repayment"])
    generated = str(agent.get("updated_method") or "")
    deleted, inserted = get_deleted_and_inserted_lines(before, generated)
    diagnostic = agent.get("satd_diagnostic", {}) or {}
    evidence = agent.get("satd_evidence_use", {}) or {}
    return {
        "rand_index": int(row["rand_index"]),
        "Model": agent.get("model_requested", ""),
        "model_reported": agent.get("model_reported", ""),
        "repo": row.get("repo_slug", ""),
        "language": row["language"],
        "url_file_path": row.get("url_file_path", ""),
        "method_name": row.get("method_name", ""),
        "fix_commit": row.get("fix_commit", ""),
        "fix_commit_date": baseline["fix_commit_date"],
        "agent_visible_commit": baseline["agent_visible_commit"],
        "agent_visible_commit_date": baseline["agent_visible_commit_date"],
        "git_history_isolation_verified": baseline.get("git_history_isolation", {}).get("verified", False),
        "repayment_status": agent.get("repayment_status", ""),
        "method_before_repayment": before,
        "files_touched": "; ".join(map(str, agent.get("files_touched", []))),
        "original_method_updated": agent.get("original_method_updated", False),
        "comment_only_target_edit": agent.get("comment_only_target_edit", False),
        "extraction_source": agent.get("extraction_source", ""),
        "update_files": "; ".join(map(str, agent.get("update_files", []))),
        "method_level_only": agent.get("method_level_only", True),
        "repayment_scope": agent.get("repayment_scope", ""),
        "attempts": agent.get("attempts", 0),
        "num_turns_last": agent.get("num_turns_last", ""),
        "num_turns_total": agent.get("num_turns_total", ""),
        "result_subtype": agent.get("result_subtype", ""),
        "final_answer_status": agent.get("final_answer_status", ""),
        "usage_json": compact_json(agent.get("usage", {})),
        "summary": agent.get("summary", ""),
        "trajectory_path": agent.get("trajectory_path", ""),
        "openai_agents_activity_path": agent.get("openai_agents_activity_path", ""),
        "satd_diagnostic_path": agent.get("satd_diagnostic_path", ""),
        "satd_evidence_use_path": agent.get("satd_evidence_use_path", ""),
        "diagnostic_debt_type_guess": diagnostic.get("debt_type_guess", ""),
        "diagnostic_repair_scope_guess": diagnostic.get("repair_scope_guess", ""),
        "diagnostic_callers_found": diagnostic.get("callers_found", ""),
        "diagnostic_related_tests_found": diagnostic.get("related_tests_found", ""),
        "evidence_inspected_callers": evidence.get("inspected_callers_or_references", ""),
        "evidence_ran_tests_or_checks": evidence.get("ran_tests_or_checks", ""),
        "evidence_ignored_available_tests": evidence.get("ignored_available_tests", ""),
        "number_of_deleted_lines": len(deleted),
        "number_of_inserted_lines": len(inserted),
    }


def patch_row(row, agent):
    before = str(row["containing_method_before_repayment"])
    generated = str(agent.get("updated_method") or "")
    return {
        "rand_index": int(row["rand_index"]),
        "Model": agent.get("model_requested", ""),
        "repo": row.get("repo_slug", ""),
        "file_path": row.get("url_file_path", ""),
        "method_name": row.get("method_name", ""),
        "patch_scope": agent.get("repayment_scope", ""),
        "before_method": before,
        "generated_method": generated,
        "reference_after_method": row.get("containing_method_after_repayment", ""),
        "method_diff": method_diff(before, generated),
        "files_touched": "; ".join(map(str, agent.get("files_touched", []))),
        "original_method_updated": agent.get("original_method_updated", False),
        "comment_only_target_edit": agent.get("comment_only_target_edit", False),
        "extraction_source": agent.get("extraction_source", ""),
    }


def proposed_file_rows(row, agent):
    rows = []
    for i, item in enumerate(agent.get("proposed_files", []) or [], 1):
        rows.append({
            "rand_index": int(row["rand_index"]),
            "Model": agent.get("model_requested", ""),
            "repo": row.get("repo_slug", ""),
            "file_index": i,
            "path": item.get("path", ""),
            "change_type": item.get("change_type", ""),
            "patch": item.get("patch", ""),
        })
    return rows


def eval_fix(df, output_dir, models, result_rows, patch_rows, agent_rows, proposed_files, run_config=None):
    answers_by_model_template = {}
    for row in agent_rows:
        key = (row.get("Model", ""), row.get("Template", ""))
        answers_by_model_template.setdefault(key, {})[int(row["rand_index"])] = row.get("extracted_code", "")

    original_loader = fix_satd_eval.load_answers_for_model

    def load_current_run_answers(_output_dir, model_name, prompt_template, current_df):
        answers = answers_by_model_template.get((model_name, prompt_template), {})
        valid_ids = set(current_df["rand_index"].astype(int))
        return {k: v for k, v in answers.items() if k in valid_ids}

    fix_satd_eval.load_answers_for_model = load_current_run_answers
    try:
        exact_df, oracle_df = fix_satd_eval.evaluate_exact_match(df, output_dir, models)
        whole_df, diff_df, per_item_whole_df, per_item_diff_df = fix_satd_eval.evaluate_bleu_tables(df, output_dir, models)
    finally:
        fix_satd_eval.load_answers_for_model = original_loader
    baseline_df = fix_satd_eval.evaluate_baseline_input(df)
    llm_answers_df = pd.DataFrame(agent_rows)[
        ["Model", "Agent", "Template", "rand_index", "language",
         "SATD_comment", "method_before_repayment", "answer", "extracted_code"]
    ]
    summary_df = pd.DataFrame([
        {"Metric": "Rows", "Value": len(df)},
        {"Metric": "Python rows", "Value": len(df[df["language"] == "Python"])},
        {"Metric": "Java rows", "Value": len(df[df["language"] == "Java"])},
        {"Metric": "Model", "Value": ", ".join(m for _, m in models)},
        {"Metric": "Agent", "Value": AGENT_NAME},
        {"Metric": "Template", "Value": TEMPLATE},
    ])
    run_config_df = pd.DataFrame(
        [{"Setting": k, "Value": str(v)} for k, v in (run_config or {}).items()]
    )

    model_slug = command_slug(models[0][1]) if models else "model"
    path = output_dir / f"openai_agents_satd_repayment_{model_slug}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    sidecar_dir = output_dir / "excel_full_rows" / path.stem
    write_jsonl(sidecar_dir / "agent_rows.jsonl", agent_rows)
    write_jsonl(sidecar_dir / "result_rows.jsonl", result_rows)
    write_jsonl(sidecar_dir / "patch_rows.jsonl", patch_rows)
    write_jsonl(sidecar_dir / "proposed_files.jsonl", proposed_files)

    sheets = [
        ("run_config", run_config_df),
        ("llm_answers", llm_answers_df), ("exact_match", exact_df),
        ("bleu_whole_code", whole_df), ("baseline_input", baseline_df),
        ("bleu_diff", diff_df), ("per_item_whole", per_item_whole_df),
        ("per_item_diff", per_item_diff_df), ("dataset_summary", summary_df),
        ("openai_agents_answers", pd.DataFrame(agent_rows)),
        ("repayment_results", pd.DataFrame(result_rows)),
        ("patches", pd.DataFrame(patch_rows)),
        ("proposed_files", pd.DataFrame(proposed_files)),
        ("oracle_prompt", oracle_df),
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            excel_safe_df(frame).to_excel(writer, sheet_name=name, index=False)
    return path


async def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_execution_logger()
    logger.info("Starting OpenAI Agents SATD pipeline: model=%s limit=%s", args.model, args.limit)

    fix_satd_eval.PROMPT_TEMPLATES = [TEMPLATE]
    df = fix_satd_eval.load_dataset(args.dataset, sheet_name=args.sheet)
    df = normalize_dataset_for_runner(df)
    validate_runner_dataset(df)
    if args.limit:
        df = df.head(args.limit).copy()
    logger.info("Loaded %s SATD rows", len(df))

    repos_root = Path(args.repos_root)
    worktrees_root = Path(args.worktrees_dir)
    worktrees_root.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    model_slug = command_slug(args.model)
    dataset_slug = command_slug(Path(args.dataset).stem)
    answers_dir = output_dir / "generated_outputs" / model_slug
    answers_dir.mkdir(parents=True, exist_ok=True)
    answers_path = answers_dir / f"answers_{TEMPLATE}_{dataset_slug}_{run_id}_v3.json"

    activity_dir = output_dir / "agent_activity"
    task_dir = output_dir / "task_files" / run_id
    stdout_dir = activity_dir / "stdout" / run_id
    trajectories_dir = output_dir / "trajectories" / model_slug / run_id
    diagnostics_dir = output_dir / "satd_diagnostics" / model_slug / run_id
    metadata_dir = output_dir / "metadata" / model_slug / run_id
    for path in (activity_dir, task_dir, stdout_dir, trajectories_dir, diagnostics_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    activity_trace_path = activity_dir / f"openai_agents_activity_{model_slug}_{run_id}.jsonl"
    run_config = {
        "event": "run_config",
        "model": args.model,
        "model_provider": args.model_provider,
        "openrouter_base_url": args.openrouter_base_url if args.model_provider == "openrouter" else "",
        "openrouter_api_key_env": args.openrouter_api_key_env if args.model_provider == "openrouter" else "",
        "openrouter_api_key_present": bool(os.getenv(args.openrouter_api_key_env, "").strip())
        if args.model_provider == "openrouter" else False,
        "agent": AGENT_NAME,
        "template": TEMPLATE,
        "dataset": str(args.dataset),
        "dataset_slug": dataset_slug,
        "sheet": args.sheet,
        "run_id": run_id,
        "answers_path": str(answers_path),
        "activity_trace_path": str(activity_trace_path),
        "trajectories_dir": str(trajectories_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "metadata_dir": str(metadata_dir),
        "max_turns": args.max_turns,
        "git_history_isolation_required": True,
        "git_history_policy": "pre_fix_ancestors_only",
        "timeout_seconds": args.timeout_seconds,
        "tool_backend": "function",
        "satd_diagnostics": args.satd_diagnostics,
        "timestamp": now_iso(),
    }
    with activity_trace_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False)
        handle.write("\n")

    output_paths = {
        "tasks_dir": task_dir,
        "stdout_dir": stdout_dir,
        "trajectories_dir": trajectories_dir,
        "diagnostics_dir": diagnostics_dir,
        "metadata_dir": metadata_dir,
        "activity_trace_path": activity_trace_path,
        "run_config": run_config,
    }

    answers, result_rows, patches, agent_rows, proposed_files = {}, [], [], [], []
    for _, row in df.iterrows():
        rand_index = int(row["rand_index"])
        logger.info("Running row rand_index=%s repo=%s method=%s",
                    rand_index, row.get("repo_slug", ""), row.get("method_name", ""))
        agent, raw, baseline = await process_row(row, args, repos_root, worktrees_root, logger, output_paths)

        answers[rand_index] = str(agent.get("updated_method") or "")
        result_rows.append(row_result(row, agent, baseline))
        patches.append(patch_row(row, agent))
        proposed_files.extend(proposed_file_rows(row, agent))
        agent_rows.append({
            "Model": args.model,
            "Agent": AGENT_NAME,
            "LLM_Model": args.model,
            "Template": TEMPLATE,
            "rand_index": rand_index,
            "language": row["language"],
            "SATD_comment": row["SATD_comment"],
            "method_before_repayment": str(row["containing_method_before_repayment"]),
            "answer": raw,
            "extracted_code": answers[rand_index],
            "agent_json": json.dumps(agent, ensure_ascii=False, default=str),
            "git_status": agent.get("git_status", ""),
            "git_diff": agent.get("git_diff", ""),
            "trajectory_path": agent.get("trajectory_path", ""),
            "openai_agents_activity_path": agent.get("openai_agents_activity_path", ""),
            "metadata_path": agent.get("metadata_path", ""),
            "satd_diagnostic_path": agent.get("satd_diagnostic_path", ""),
            "satd_evidence_use_path": agent.get("satd_evidence_use_path", ""),
        })
        save_to_json(answers, str(answers_path))
        logger.info("Saved row rand_index=%s status=%s extracted_chars=%s",
                    rand_index, agent.get("repayment_status", ""), len(answers[rand_index]))

    logger.info("Evaluating generated fixes")
    workbook = eval_fix(
        df, output_dir, [(args.model, args.model)],
        result_rows, patches, agent_rows, proposed_files,
        run_config=run_config,
    )
    logger.info("Excel results saved to: %s", workbook)
    print(f"Excel results saved to: {workbook}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--sheet", default=DATASET_SHEET)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--repos-root", default=str(DEFAULT_REPOS_ROOT))
    parser.add_argument("--worktrees-dir", default=str(PROJECT_DIR / "execution_runs" / "worktrees"))
    parser.add_argument("--model", default=os.getenv(
        "OPENROUTER_MODEL", os.getenv("OPENAI_AGENTS_MODEL", DEFAULT_MODEL)
    ), help="Model ID; defaults to the project .env setting or gpt-4o-2024-08-06.")
    parser.add_argument("--model-provider", choices=["openrouter", "openai"],
                        default=os.getenv("OPENAI_AGENTS_MODEL_PROVIDER", "openrouter"),
                        help="Model backend. Default uses OpenRouter via an OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--openrouter-base-url",
                        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY",
                        help="Environment variable containing the OpenRouter API key.")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--command-timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="Rows to process (0 = full dataset).")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--satd-diagnostics", choices=["off", "log", "prompt"], default="log",
                        help="'log' preserves baseline prompts; 'prompt' exposes diagnostic evidence to the agent.")
    parser.add_argument("--satd-diagnostic-max-items", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
