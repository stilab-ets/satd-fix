"""Collect SATD diagnostics and summarize observable evidence use."""

import json
import re
import subprocess
import time
from pathlib import Path

from helpers import (
    extract_updated_method_from_worktree,
    git_changed_files,
    git_diff,
    git_status_porcelain,
    read_worktree_file,
)


CODE_EXTENSIONS = {".py", ".java", ".js", ".ts", ".go", ".rb", ".php", ".cs", ".cpp", ".c", ".h"}
TEST_PATTERNS = ("test", "tests", "spec")
SATD_TERMS = (
    "todo", "fixme", "xxx", "hack", "workaround", "temporary", "temp",
    "technical debt", "debt", "cleanup", "refactor", "deprecated",
)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _compact(text, limit=12000):
    text = str(text or "")
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated: {len(text)} chars total]"


def _safe_rel(path):
    return str(path).replace("\\", "/")


def _run_rg(worktree, args, timeout=15):
    try:
        output = subprocess.check_output(
            ["rg", *args],
            cwd=str(worktree),
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return (output or "").rstrip("\r\n")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _line_number_for_text(file_text, needle):
    needle = str(needle or "").strip()
    if not needle:
        return None
    index = file_text.find(needle)
    if index < 0:
        normalized = re.sub(r"\s+", " ", needle)
        for match in re.finditer(re.escape(normalized[:80]), re.sub(r"\s+", " ", file_text)):
            return file_text[:match.start()].count("\n") + 1
        return None
    return file_text[:index].count("\n") + 1


def _method_line_range(file_text, method_text):
    method_text = str(method_text or "").strip()
    if not method_text:
        return None
    index = file_text.find(method_text)
    if index < 0:
        first = next((line.strip() for line in method_text.splitlines() if line.strip()), "")
        if first:
            index = file_text.find(first)
    if index < 0:
        return None
    start = file_text[:index].count("\n") + 1
    end = start + max(0, len(method_text.splitlines()) - 1)
    return {"start": start, "end": end}


def _parse_rg_lines(output, max_items):
    rows = []
    for line in str(output or "").splitlines():
        parts = line.split(":", 3)
        if len(parts) >= 4 and parts[1].isdigit():
            rows.append({
                "path": _safe_rel(parts[0]),
                "line": int(parts[1]),
                "text": parts[3].strip(),
            })
        elif line.strip():
            rows.append({"path": "", "line": None, "text": line.strip()})
        if len(rows) >= max_items:
            break
    return rows


def _debt_type_guess(satd_comment):
    text = str(satd_comment or "").lower()
    rules = [
        ("test_debt", ("test", "coverage", "mock", "fixture")),
        ("performance_debt", ("slow", "performance", "optimize", "cache", "efficient")),
        ("compatibility_debt", ("compat", "backward", "legacy", "deprecated")),
        ("error_handling_debt", ("exception", "error", "fail", "catch", "raise")),
        ("design_refactoring_debt", ("refactor", "cleanup", "duplicate", "hack", "workaround")),
        ("temporary_workaround", ("todo", "fixme", "temporary", "temp", "later", "for now")),
        ("hardcoded_value", ("hardcoded", "hard-coded", "magic", "constant")),
        ("documentation_debt", ("document", "docs", "comment")),
    ]
    for label, terms in rules:
        if any(term in text for term in terms):
            return label
    return "unspecified_satd"


def _repair_scope_guess(debt_type, callers, callees, related_satd):
    if debt_type in {"design_refactoring_debt", "compatibility_debt"} and len(callers) > 2:
        return "cross-file-or-api"
    if len(callers) > 5 or len(related_satd) > 3:
        return "multi-location"
    if debt_type in {"hardcoded_value", "temporary_workaround", "error_handling_debt"}:
        return "method-or-file"
    return "unknown-or-local"


def collect_satd_diagnostics(row, worktree, max_items=25):
    """Collect pre-repair SATD evidence without modifying the worktree."""
    target_file = str(row.get("url_file_path", ""))
    method_name = str(row.get("method_name", ""))
    satd_comment = str(row.get("SATD_comment", ""))
    method_before = str(row.get("containing_method_before_repayment", ""))
    language = str(row.get("language", ""))

    file_text = read_worktree_file(worktree, target_file)
    target_path = Path(worktree) / target_file
    satd_line = _line_number_for_text(file_text, satd_comment)
    method_range = _method_line_range(file_text, method_before)
    comment_inside_method = (
        bool(method_range and satd_line)
        and method_range["start"] <= satd_line <= method_range["end"]
    )

    method_after_checkout = extract_updated_method_from_worktree(worktree, row)
    method_found = bool(method_after_checkout.strip())

    method_pattern = rf"\b{re.escape(method_name)}\b" if method_name else ""
    caller_rows = []
    if method_pattern:
        caller_output = _run_rg(
            worktree,
            ["-n", "--glob", "!*.md", "--glob", "!*.txt", "--", method_pattern],
        )
        caller_rows = [
            item for item in _parse_rg_lines(caller_output, max_items)
            if item.get("path") and item.get("path") != _safe_rel(target_file)
        ][:max_items]

    test_rows = []
    if method_pattern:
        test_output = _run_rg(
            worktree,
            ["-n", "--glob", "*test*", "--glob", "*Test*", "--", method_pattern],
        )
        test_rows = _parse_rg_lines(test_output, max_items)

    satd_output = _run_rg(
        worktree,
        ["-n", "-i", "--glob", "!*.md", "--glob", "!*.txt", "--",
         "|".join(re.escape(term) for term in SATD_TERMS)],
    )
    related_satd = _parse_rg_lines(satd_output, max_items)

    callees = []
    for name in sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", method_before))):
        if name not in {"if", "for", "while", "switch", "catch", "return", "assert"}:
            callees.append(name)
        if len(callees) >= max_items:
            break

    debt_type = _debt_type_guess(satd_comment)
    diagnostic = {
        "event": "satd_diagnostic",
        "timestamp": _now(),
        "rand_index": int(row["rand_index"]),
        "repo_slug": str(row.get("repo_slug", "")),
        "language": language,
        "target_file": target_file,
        "target_file_exists": target_path.is_file(),
        "target_method": method_name,
        "target_method_found": method_found,
        "satd_comment": satd_comment,
        "satd_comment_line": satd_line,
        "target_method_line_range": method_range,
        "comment_inside_target_method": comment_inside_method,
        "debt_type_guess": debt_type,
        "repair_scope_guess": _repair_scope_guess(debt_type, caller_rows, callees, related_satd),
        "callers_found": len(caller_rows),
        "caller_examples": caller_rows,
        "callees_found": len(callees),
        "callee_examples": callees,
        "related_tests_found": len(test_rows),
        "related_test_examples": test_rows,
        "related_satd_found": len(related_satd),
        "related_satd_examples": related_satd,
        "static_signal": "No direct executable failure collected; SATD evidence is structural/contextual.",
        "worktree_status_before": git_status_porcelain(worktree),
    }
    return diagnostic


def diagnostic_prompt_block(diagnostic):
    """Compact optional prompt addendum for experiments that expose evidence."""
    fields = {
        "debt_type_guess": diagnostic.get("debt_type_guess", ""),
        "repair_scope_guess": diagnostic.get("repair_scope_guess", ""),
        "satd_comment_line": diagnostic.get("satd_comment_line"),
        "target_method_found": diagnostic.get("target_method_found"),
        "comment_inside_target_method": diagnostic.get("comment_inside_target_method"),
        "callers_found": diagnostic.get("callers_found", 0),
        "callees_found": diagnostic.get("callees_found", 0),
        "related_tests_found": diagnostic.get("related_tests_found", 0),
        "related_satd_found": diagnostic.get("related_satd_found", 0),
        "caller_examples": diagnostic.get("caller_examples", [])[:5],
        "related_test_examples": diagnostic.get("related_test_examples", [])[:5],
    }
    return (
        "\n\nAdditional SATD diagnostic evidence collected before repair:\n"
        "```json\n"
        + json.dumps(fields, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def diagnostic_trajectory_step(diagnostic, step, attempt=0):
    return {
        "step": step,
        "event_index": step,
        "timestamp": diagnostic.get("timestamp") or _now(),
        "rand_index": diagnostic.get("rand_index"),
        "attempt": attempt,
        "event": "satd_diagnostic",
        "response": "",
        "thought": "",
        "action": "collect_satd_diagnostics",
        "observation": json.dumps(diagnostic, ensure_ascii=False),
        "state": json.dumps({
            "target_file": diagnostic.get("target_file", ""),
            "target_method": diagnostic.get("target_method", ""),
        }, ensure_ascii=False),
        "query": [],
        "query_exact": True,
    }


def _tool_events(trajectory_steps):
    for step in trajectory_steps or []:
        if step.get("event") == "tool_use":
            yield step


def analyze_evidence_use(row, worktree, trajectory_steps, agent, diagnostic=None):
    """Summarize what external evidence the agent appeared to use."""
    target_file = str(row.get("url_file_path", "")).replace("\\", "/")
    method_name = str(row.get("method_name", ""))
    tool_steps = list(_tool_events(trajectory_steps))
    action_text = "\n".join(str(step.get("action", "")) for step in tool_steps)
    action_lower = action_text.lower().replace("\\", "/")
    tool_names = {str(step.get("tool", "")).lower() for step in tool_steps}

    read_target_file = target_file.lower() in action_lower and any(
        tool in tool_names
        for tool in {
            "read", "grep", "glob", "bash",
            "read_file", "read_target_method", "inspect_satd_context", "search_text",
        }
    )
    searched_repo = bool({"grep", "glob", "search_text"} & tool_names) or "rg " in action_lower
    ran_tests = bool(re.search(r"\b(pytest|unittest|mvn test|gradle test|npm test|tox)\b", action_lower))
    edited_target_file = target_file.lower() in action_lower and any(
        tool in tool_names
        for tool in {"edit", "multiedit", "write", "replace_in_file", "write_file"}
    )
    inspected_callers = searched_repo and bool(method_name and method_name.lower() in action_lower)

    changed_files = git_changed_files(worktree)
    analysis = {
        "event": "evidence_use_analysis",
        "timestamp": _now(),
        "rand_index": int(row["rand_index"]),
        "read_target_file": read_target_file,
        "searched_repo": searched_repo,
        "inspected_callers_or_references": inspected_callers,
        "ran_tests_or_checks": ran_tests,
        "edited_target_file_via_tool": edited_target_file,
        "changed_files_count": len(changed_files),
        "changed_files": [item.get("path", "") for item in changed_files],
        "edited_target_method": bool(agent.get("original_method_updated", False)),
        "comment_only_target_edit": bool(agent.get("comment_only_target_edit", False)),
        "repayment_status": agent.get("repayment_status", ""),
        "diagnostic_debt_type_guess": (diagnostic or {}).get("debt_type_guess", ""),
        "diagnostic_repair_scope_guess": (diagnostic or {}).get("repair_scope_guess", ""),
        "diagnostic_related_tests_found": (diagnostic or {}).get("related_tests_found", 0),
        "ignored_available_tests": bool((diagnostic or {}).get("related_tests_found", 0) and not ran_tests),
        "git_status_after": git_status_porcelain(worktree),
        "git_diff_after": _compact(git_diff(worktree), limit=40000),
    }
    return analysis


def evidence_use_trajectory_step(analysis, step, attempt):
    return {
        "step": step,
        "event_index": step,
        "timestamp": analysis.get("timestamp") or _now(),
        "rand_index": analysis.get("rand_index"),
        "attempt": attempt,
        "event": "evidence_use_analysis",
        "response": "",
        "thought": "",
        "action": "analyze_observable_evidence_use",
        "observation": json.dumps(analysis, ensure_ascii=False),
        "state": json.dumps({
            "repayment_status": analysis.get("repayment_status", ""),
            "changed_files": analysis.get("changed_files", []),
        }, ensure_ascii=False),
        "query": [],
        "query_exact": True,
    }
