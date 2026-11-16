"""Helpers for repository setup, repair extraction, evaluation, and execution logging."""

import ast
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import difflib
from pathlib import Path

import pandas as pd
from openai import OpenAI

try:
    from tree_sitter_language_pack import get_parser
except Exception:  # Use regex extraction if tree-sitter is unavailable.
    get_parser = None

BASELINE_DIR = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
EXCEL_CELL_LIMIT = 32767
EXCEL_DISPLAY_LIMIT = 32000
EXCEL_ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

sys.path.insert(0, str(BASELINE_DIR))
import fix_satd_eval  


def load_env_file(path):
    """Load simple KEY=value entries from an .env file without overwriting env."""
    path = Path(path)
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    return True


def load_project_env(path=None):
    """Load an optional environment file from the replication package."""
    env_path = Path(path) if path else BASELINE_DIR.parent / "conf" / ".env"
    return load_env_file(env_path)


def configure_openrouter_for_claude_agent():
    """Route the Claude Agent SDK through OpenRouter when a key is present."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return False
    os.environ["ANTHROPIC_BASE_URL"] = os.getenv(
        "OPENROUTER_ANTHROPIC_BASE_URL", "https://openrouter.ai/api"
    ).strip()
    os.environ["ANTHROPIC_AUTH_TOKEN"] = key
    os.environ["ANTHROPIC_API_KEY"] = ""
    return True


def openrouter_client(base_url=None, api_key=None):
    return OpenAI(
        base_url=base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
    )


def call_llm(client, model, prompt, max_tokens, system_prompt="You are a helpful assistant."):
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def load_prompt_yaml(name):
    path = AGENT_DIR / "prompts" / name
    data, key, block = {}, None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if key is not None:
            if raw.startswith("  ") or not raw.strip():
                block.append(raw[2:] if raw.startswith("  ") else "")
                continue
            data[key], key, block = "\n".join(block).rstrip() + "\n", None, []
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        if v.strip() == "|":
            key, block = k.strip(), []
        else:
            data[k.strip()] = v.strip().strip('"')
    if key is not None:
        data[key] = "\n".join(block).rstrip() + "\n"
    return data


def render_prompt_template(prompt_file, template_key, **values):
    return load_prompt_yaml(prompt_file)[template_key].format(**values)


def language_from_path(path):
    path = str(path).lower()
    if path.endswith(".py"):
        return "Python"
    if path.endswith(".java"):
        return "Java"
    return "Unsupported"


def derive_method_name(method_body):
    text = str(method_body or "")
    m = re.search(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", text, re.MULTILINE)
    if m:
        return m.group(1)
    m = re.search(
        r"^\s*(?:public|protected|private|static|final|synchronized|abstract|\s)+"
        r"[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\(",
        text,
        re.MULTILINE,
    )
    return m.group(1) if m else ""


def normalize_dataset_for_runner(df):
    df = df.copy()
    if "repo_slug" not in df.columns and {"user", "project"}.issubset(df.columns):
        df["repo_slug"] = df["user"].astype(str) + "/" + df["project"].astype(str)
    if "fix_commit" not in df.columns and "deleted_in_commit" in df.columns:
        df["fix_commit"] = df["deleted_in_commit"]
    if "parent_commit" not in df.columns:
        # Start from the parent of the developer fix.
        df["parent_commit"] = df["fix_commit"].astype(str) + "^"
    if "created_in_file" in df.columns:
        df["url_file_path"] = df["created_in_file"]
    if "method_name" not in df.columns:
        df["method_name"] = df["containing_method_before_repayment"].map(derive_method_name)
    else:
        df["method_name"] = df.apply(
            lambda r: derive_method_name(r.get("containing_method_before_repayment", ""))
            if pd.isna(r.get("method_name")) or not str(r.get("method_name", "")).strip()
            else r.get("method_name"),
            axis=1,
        )
    if "language" not in df.columns:
        df["language"] = df["url_file_path"].map(language_from_path)
    if "SATD_comment" not in df.columns and "last_content" in df.columns:
        df["SATD_comment"] = df["last_content"]
    return df


def validate_runner_dataset(df):
    required = [
        "rand_index", "repo_slug", "fix_commit", "parent_commit", "url_file_path",
        "language", "SATD_comment",
        "containing_method_before_repayment", "containing_method_after_repayment",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Dataset missing required columns: " + ", ".join(missing))


def run_git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        stderr=subprocess.STDOUT,
    ).strip()


def try_run_git(repo, *args):
    try:
        return run_git(repo, *args)
    except subprocess.CalledProcessError:
        return ""


def repo_dir(repos_root, repo_slug):
    repos_root.mkdir(parents=True, exist_ok=True)
    slug = str(repo_slug).replace("/", "__")
    for candidate in (repos_root / slug, repos_root / str(repo_slug)):
        if (candidate / ".git").exists():
            run_git(candidate, "config", "core.longpaths", "true")
            return candidate
    target = repos_root / slug
    subprocess.check_output(
        ["git", "clone", f"https://github.com/{repo_slug}.git", str(target)],
        text=True, stderr=subprocess.STDOUT,
    )
    run_git(target, "config", "core.longpaths", "true")
    return target


def prepare_worktree(repo, row, worktrees_root):
    """Check out a detached worktree at the commit before the developer fix."""
    fix_commit = str(row["fix_commit"]).strip()
    parent = str(row.get("parent_commit", "")).strip()
    baseline = parent if parent and parent.lower() != "nan" else f"{fix_commit}^"
    expected = run_git(repo, "rev-parse", baseline)
    target = worktrees_root / f"{repo.name}_{expected[:12]}"

    if target.exists() and try_run_git(target, "rev-parse", "HEAD") != expected:
        shutil.rmtree(target)
        run_git(repo, "worktree", "prune")
    if target.exists():
        clean_worktree(target)
    else:
        run_git(repo, "worktree", "prune")
        try:
            run_git(repo, "worktree", "add", "--detach", str(target), expected)
        except subprocess.CalledProcessError:
            # Prune stale worktree registrations before retrying.
            run_git(repo, "worktree", "prune")
            run_git(repo, "worktree", "add", "--detach", str(target), expected)
        run_git(target, "config", "core.longpaths", "true")

    visible_date = run_git(target, "show", "-s", "--format=%cI", "HEAD")
    fix_date = run_git(repo, "show", "-s", "--format=%cI", fix_commit)
    return target, expected, visible_date, fix_date


def clean_worktree(worktree):
    run_git(worktree, "reset", "--hard", "HEAD")
    run_git(worktree, "clean", "-fd")


def read_worktree_file(worktree, relative_path):
    target = Path(worktree) / relative_path
    if not target.is_file():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def git_status_porcelain(worktree):
    try:
        return subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            text=True, stderr=subprocess.STDOUT,
        ).rstrip("\r\n")
    except subprocess.CalledProcessError:
        return ""


def git_changed_files(worktree):
    """Changed-file rows (path, change_type, patch), including new files."""
    rows = []
    for line in git_status_porcelain(worktree).splitlines():
        if not line:
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if status == "??":
            content = read_worktree_file(worktree, path)
            patch = "\n".join(
                [f"diff --git a/{path} b/{path}", "new file mode 100644",
                 "--- /dev/null", f"+++ b/{path}", "@@"]
                + ["+" + l for l in content.splitlines()]
            )
            rows.append({"path": path, "change_type": "add", "patch": patch})
        else:
            change = "add" if status.startswith("A") else "delete" if status.startswith("D") else "modify"
            rows.append({"path": path, "change_type": change,
                         "patch": try_run_git(worktree, "diff", "--", path)})
    return rows


def git_diff(worktree):
    tracked = try_run_git(worktree, "diff", "--binary")
    extra = [r["patch"] for r in git_changed_files(worktree) if r["change_type"] == "add" and r["patch"]]
    return "\n".join(p for p in [tracked, *extra] if p)


def normalize_code_for_compare(code):
    return re.sub(r"\s+", "", str(code or ""))


def extracted_method_is_valid(method_code, language):
    """Valid = non-empty; for Python, additionally parseable after dedent."""
    text = str(method_code or "").strip()
    if not text:
        return False
    if str(language) == "Python":
        try:
            ast.parse(textwrap.dedent(text))
            return True
        except SyntaxError:
            return False
    return True


def tree_sitter_extract_method(code, method_name, file_path):
    """Extract a method by name with tree-sitter when available, else ''. """
    if get_parser is None:
        return ""
    lang = {"py": "python", "java": "java"}.get(Path(str(file_path)).suffix.lstrip(".").lower(), "")
    if not lang:
        return ""
    try:
        tree = get_parser(lang).parse(str(code or "").encode("utf-8"))
    except Exception:
        return ""
    source = str(code or "").encode("utf-8")
    method_types = {"function_definition", "method_declaration", "constructor_declaration"}
    result = []

    def visit(node):
        if node.type in method_types:
            name_node = node.child_by_field_name("name")
            if name_node is not None and source[name_node.start_byte:name_node.end_byte].decode() == method_name:
                result.append(source[node.start_byte:node.end_byte].decode("utf-8", "replace"))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return result[0] if result else ""


def extract_method_from_code(code, method_name, file_path, language):
    """Tree-sitter first, then the shared regex extractors."""
    method = tree_sitter_extract_method(code, method_name, file_path)
    if method:
        return method
    if language == "Python":
        return fix_satd_eval._extract_python_method_block(code, method_name)
    if language == "Java":
        return fix_satd_eval._extract_java_method_block(code, method_name)
    return ""


def extract_updated_method_from_worktree(worktree, row):
    code = read_worktree_file(worktree, str(row.get("url_file_path", "")))
    if not code:
        return ""
    return extract_method_from_code(
        code, str(row.get("method_name", "")).strip(),
        row.get("url_file_path", ""), str(row.get("language", "")),
    )


def target_method_matches_dataset_before(worktree, row):
    expected = normalize_code_for_compare(row.get("containing_method_before_repayment", ""))
    actual = normalize_code_for_compare(extract_updated_method_from_worktree(worktree, row))
    return bool(expected) and expected == actual


COMMENT_PREFIXES = ("#", "//", "*", "/*", "*/", "<!--", '"""', "'''")
SATD_KEYWORDS = ("xxx", "todo", "fixme", "hack", "workaround", "temporary", "temp", "kludge", "debt")


def is_comment_line(line):
    stripped = str(line or "").strip()
    return not stripped or any(stripped.startswith(p) for p in COMMENT_PREFIXES)


def debt_text_remains(method_after, satd_comment=""):
    """True when the SATD comment (or any debt-keyword comment) is still present."""
    text = str(method_after or "")
    for line in text.splitlines():
        s = line.strip()
        if is_comment_line(s) and any(k in s.lower() for k in SATD_KEYWORDS):
            return True
    fragment = str(satd_comment or "").strip()[:40].lower()
    return bool(fragment and fragment in text.lower())


def code_without_comments(code, language):
    text = str(code or "")
    if language == "Python":
        lines = [re.sub(r"\s+#.*$", "", l) for l in text.splitlines() if not l.lstrip().startswith("#")]
        return normalize_code_for_compare("\n".join(lines))
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    text = re.sub(r"(?m)//.*$", "", text)
    return normalize_code_for_compare(text)


def is_comment_only_method_change(before, after, language):
    if normalize_code_for_compare(before) == normalize_code_for_compare(after):
        return False
    return code_without_comments(before, language) == code_without_comments(after, language)


FENCE_RE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def answer_status(text):
    """Read the 'Status:' line -> 'changed' | 'no_safe_change' | 'unknown'."""
    m = re.search(r"(?im)^\s*(?:\*\*)?status(?:\*\*)?\s*:\s*(.+)$", str(text or ""))
    if not m:
        return "unknown"
    value = m.group(1).strip().lower()
    if "no safe change" in value or "no change" in value:
        return "no_safe_change"
    if "changed" in value:
        return "changed"
    return "unknown"


def diff_to_after_code(diff_text):
    """Convert a unified diff into the proposed after-code text."""
    after = []
    for line in str(diff_text or "").splitlines():
        if line.startswith(("+++", "---", "@@", "diff --git", "index ")):
            continue
        if line.startswith("+"):
            after.append(line[1:])
        elif line.startswith("-"):
            continue
        elif line.startswith(" "):
            after.append(line[1:])
        else:
            after.append(line)
    return "\n".join(after).strip()


def looks_like_diff(text):
    return bool(re.search(r"(?m)^(?:@@|diff --git|\+\+\+ |--- )", str(text or "")))


def extract_replacement_from_answer(text, row):
    """Extract the target method from replacement fences, code fences, or transcript text."""
    method_name = str(row.get("method_name", "")).strip()
    file_path = str(row.get("url_file_path", ""))
    language = str(row.get("language", ""))
    before = str(row.get("containing_method_before_repayment", ""))
    if not method_name:
        return "", False

    replacement_blocks, other_blocks = [], []
    for lang, code in FENCE_RE.findall(str(text or "")):
        code = code.strip()
        if not code or lang.lower() == "json":
            continue
        if lang.lower() == "replacement":
            replacement_blocks.append(code)
        elif lang.lower() == "diff" or looks_like_diff(code):
            other_blocks.append(diff_to_after_code(code))
        else:
            other_blocks.append(code)

    best_valid, best_invalid = "", ""
    for candidate in replacement_blocks + other_blocks:
        method = extract_method_from_code(candidate, method_name, file_path, language)
        if not method and method_name in candidate:
            method = candidate  # The fence contains only a method body.
        if not method:
            continue
        changed = normalize_code_for_compare(method) != normalize_code_for_compare(before)
        if extracted_method_is_valid(method, language):
            if changed:
                return method, True
            best_valid = best_valid or method
        else:
            best_invalid = best_invalid or method
    if best_valid:
        return best_valid, True
    if best_invalid:
        return best_invalid, False
    return "", False


def agent_from_git(row, answer, transcript, worktree):
    """Build the agent result: Git evidence first, replacement fence fallback."""
    before = str(row.get("containing_method_before_repayment", ""))
    language = str(row.get("language", ""))
    satd_comment = str(row.get("SATD_comment", ""))

    changed_files = git_changed_files(worktree)
    diff_text = git_diff(worktree)
    method_after = extract_updated_method_from_worktree(worktree, row)
    target_edited = (
        bool(diff_text.strip())
        and extracted_method_is_valid(method_after, language)
        and normalize_code_for_compare(method_after) != normalize_code_for_compare(before)
    )
    comment_only = target_edited and is_comment_only_method_change(before, method_after, language)

    updated_method, extraction_source, status = "", "", ""
    if target_edited:
        updated_method = method_after
        extraction_source = "git_target_method"
        status = "comment_only_change" if comment_only else "generated_fix"
    else:
        # Prefer the replacement fence, then inspect the session transcript.
        proposed, valid = extract_replacement_from_answer(answer, row)
        source_label = "answer_replacement"
        if not proposed and transcript:
            proposed, valid = extract_replacement_from_answer(transcript, row)
            source_label = "transcript_replacement"
        if proposed:
            updated_method = proposed
            extraction_source = source_label if valid else source_label + "_malformed"
            if diff_text.strip():
                status = "non_target_change_with_proposal"
            else:
                status = "proposed_only" if valid else "proposed_only_malformed"
        elif diff_text.strip():
            status = "non_target_change"
        else:
            status = "no_change"

    return {
        "repayment_status": status,
        "target_file": str(row.get("url_file_path", "")),
        "target_method": str(row.get("method_name", "")),
        "method_level_only": len(changed_files) <= 1,
        "repayment_scope": "multi_file" if len(changed_files) > 1 else ("file" if changed_files else "none"),
        "files_touched": [r["path"] for r in changed_files],
        "update_files": sorted({r["path"] for r in changed_files if r["change_type"] != "delete"}),
        "original_method_updated": target_edited,
        "comment_only_target_edit": comment_only,
        "updated_method": updated_method,
        "updated_target_method": method_after,
        "extraction_source": extraction_source,
        "proposed_files": changed_files,
        "debt_text_remains": debt_text_remains(updated_method, satd_comment),
        "summary": answer,
        "git_diff": diff_text,
        "git_status": git_status_porcelain(worktree),
    }


def parse_json_object(text):
    clean = re.sub(r"```(?:json)?|```", "", str(text or "")).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            raise ValueError(f"Judge returned non-JSON: {text}")
        return json.loads(m.group())


async def judge_fix_with_llm(row, agent, judge_model, judge_base_url=None, judge_api_key="", timeout=120):
    user_prompt = render_prompt_template(
        "judge_prompt.yaml", "user_template",
        satd_comment=str(row.get("SATD_comment", "")),
        method_before=str(row.get("containing_method_before_repayment", ""))[:3000],
        method_after=str(agent.get("updated_method", ""))[:3000],
        diff=str(agent.get("git_diff", ""))[:3000],
    )
    system_prompt = render_prompt_template("judge_prompt.yaml", "system_template")
    client = openrouter_client(judge_base_url, judge_api_key)
    content = await asyncio.wait_for(
        asyncio.to_thread(call_llm, client, judge_model, user_prompt, 1500, system_prompt),
        timeout=timeout,
    )
    return parse_json_object(content)


def judge_decision(agent, judge_result=None):
    """(decision, reason). Hard structural checks first, then the LLM verdict."""
    status = agent.get("repayment_status", "")
    if status not in {"generated_fix", "proposed_only", "non_target_change_with_proposal"}:
        return "retry", f"No strong fix extracted (status={status or 'unknown'})."
    if status == "generated_fix" and agent.get("comment_only_target_edit"):
        return "retry", "Comment-only edit: SATD not repaid."
    if agent.get("debt_text_remains"):
        return "retry", "SATD/debt comment still present in the updated method."
    if judge_result is None:
        return "accept", "Structural checks passed (no LLM judge)."
    verdict = str(judge_result.get("verdict", "")).lower()
    reason = str(judge_result.get("reason", ""))
    if verdict == "accept":
        return "accept", f"Judge accepted: {reason}"
    return "retry", f"Judge verdict={verdict or 'unknown'}: {reason}"


def message_text(message):
    """Assistant text from an SDK message (dict or typed)."""
    if type(message).__name__ == "SessionMessage":
        message = getattr(message, "message", message)
    if isinstance(message, dict):
        if message.get("role") not in (None, "assistant"):
            return ""
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        return "\n".join(
            b.get("text", "") for b in content or []
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if type(message).__name__ != "AssistantMessage":
        return ""
    return "\n".join(
        getattr(b, "text", "") for b in getattr(message, "content", []) or [] if getattr(b, "text", None)
    )


def compact_for_activity(value, limit=1200):
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is None:
        return text
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


def claude_activity_events(message, text_limit=1200, result_limit=2000):
    """Extract visible text, tool calls, and tool results from an SDK message."""
    if type(message).__name__ == "SessionMessage":
        message = getattr(message, "message", message)
    events = []
    blocks = message.get("content", []) if isinstance(message, dict) else getattr(message, "content", []) or []
    for block in blocks or []:
        get = block.get if isinstance(block, dict) else lambda k, d=None: getattr(block, k, d)
        block_type = get("type", type(block).__name__)
        if get("text"):
            events.append({"event": "text", "text": compact_for_activity(get("text"), limit=text_limit)})
        elif get("name"):
            events.append({"event": "tool_use", "tool": get("name"),
                           "input": compact_for_activity(get("input", ""), limit=text_limit)})
        elif block_type in ("tool_result", "ToolResultBlock") or get("tool_use_id"):
            events.append({"event": "tool_result",
                           "text": compact_for_activity(get("content", ""), limit=result_limit)})
    result = getattr(message, "result", None)
    if result:
        events.append({"event": "result", "text": compact_for_activity(result, limit=text_limit)})
    return events


def json_safe(value, depth=0, max_depth=8):
    """Convert SDK message objects to JSON-safe values for lossless-ish traces."""
    if depth > max_depth:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
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
            **{str(k): json_safe(v, depth + 1, max_depth)
               for k, v in vars(value).items()
               if not str(k).startswith("_")},
        }
    return repr(value)


def raw_message_payload(message):
    return {
        "message_type": type(message).__name__,
        "message": json_safe(message),
    }


def claude_trajectory_steps(message, rand_index, attempt, start_step, prompt, cwd):
    """Convert SDK messages into trajectory rows with the task prompt and raw payload."""
    raw = raw_message_payload(message)
    events = claude_activity_events(message, text_limit=None, result_limit=None)
    steps = []
    for offset, event in enumerate(events):
        step_no = start_step + offset
        row = {
            "step": step_no,
            "event_index": step_no,
            "timestamp": event.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rand_index": rand_index,
            "attempt": attempt,
            "event": event.get("event", ""),
            "response": "",
            "thought": "",
            "action": "",
            "observation": "",
            "state": json.dumps({"working_dir": str(cwd)}, ensure_ascii=False),
            "query": [{"role": "user", "content": prompt}] if step_no == 0 else [],
            "query_exact": False,
            "query_note": "Claude Agent SDK did not expose SWE-agent-style exact per-step LM query.",
            "raw_message": raw,
        }
        if event.get("event") == "text":
            row["response"] = event.get("text", "")
            row["thought"] = event.get("text", "")
        elif event.get("event") == "tool_use":
            tool_input = event.get("input", "")
            row["action"] = f"{event.get('tool', '')} {tool_input}".strip()
            row["tool"] = event.get("tool", "")
            row["tool_input"] = tool_input
        elif event.get("event") == "tool_result":
            row["observation"] = event.get("text", "")
        elif event.get("event") == "result":
            row["response"] = event.get("text", "")
        steps.append(row)
    return steps


def timestamped_trajectory_events(events, rand_index=None):
    """Add timestamps and event indices while preserving the original event payload."""
    normalized = []
    fallback_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for index, event in enumerate(events or []):
        if isinstance(event, dict):
            item = dict(event)
        else:
            item = {"event": "raw_event", "payload": event}
        item.setdefault("event_index", index)
        item.setdefault("timestamp", item.get("time") or item.get("created_at") or fallback_timestamp)
        if rand_index is not None:
            item.setdefault("rand_index", rand_index)
        normalized.append(item)
    return normalized


def log_jsonl(trace_handle, event):
    if trace_handle is not None:
        json.dump(event, trace_handle, ensure_ascii=False, default=str)
        trace_handle.write("\n")
        trace_handle.flush()


def log_claude_activity(logger, trace_handle, rand_index, event, verbose=False):
    """Always write the JSONL trace (paper evidence); console output is optional."""
    event = {"rand_index": rand_index, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    if verbose:
        logger.info("CLAUDE_ACTIVITY rand_index=%s %s %s", rand_index, event.get("event"),
                    event.get("tool", "") or event.get("text", ""))
    if trace_handle is not None:
        json.dump(event, trace_handle, ensure_ascii=False, default=str)
        trace_handle.write("\n")
        trace_handle.flush()


def clean_agent_error(text):
    if isinstance(text, asyncio.TimeoutError):
        return "Timed out while waiting for Claude response"
    value = str(text)
    if value.startswith("CLAUDE_AGENT_ERROR:"):
        return value[len("CLAUDE_AGENT_ERROR:"):].strip()
    return value


def method_diff(before, after):
    return "\n".join(difflib.unified_diff(
        str(before).splitlines(), str(after).splitlines(),
        fromfile="before_method", tofile="generated_method", lineterm="",
    ))


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def excel_safe_value(value):
    if isinstance(value, str):
        value = EXCEL_ILLEGAL_CHAR_RE.sub("", value)
        if len(value) > EXCEL_CELL_LIMIT:
            return value[:EXCEL_DISPLAY_LIMIT] + f"\n...[truncated for Excel: {len(value)} chars total]"
    return value


def excel_safe_df(df):
    return df.apply(lambda col: col.map(excel_safe_value))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, default=str)
            handle.write("\n")


def build_satd_metadata(row, args, baseline=None, run_config=None, agent=None, paths=None, extra=None):
    """Link one SATD task to its model, commit, and recorded artifacts."""
    baseline = baseline or {}
    run_config = run_config or {}
    agent = agent or {}
    paths = paths or {}
    extra = extra or {}
    rand_index = int(row["rand_index"])
    metadata = {
        "event": "satd_instance_metadata",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_config.get("run_id", ""),
        "rand_index": rand_index,
        "agent": run_config.get("agent", getattr(args, "agent", "")),
        "model": run_config.get("model", getattr(args, "model", "")),
        "template": run_config.get("template", ""),
        "tool_backend": run_config.get("tool_backend", run_config.get("backend", "")),
        "dataset": run_config.get("dataset", getattr(args, "dataset", "")),
        "dataset_slug": run_config.get("dataset_slug", ""),
        "sheet": run_config.get("sheet", getattr(args, "sheet", "")),
        "repo_slug": str(row.get("repo_slug", "")),
        "language": str(row.get("language", "")),
        "target_file": str(row.get("url_file_path", "")),
        "target_method": str(row.get("method_name", "")),
        "satd_comment": str(row.get("SATD_comment", "")),
        "agent_visible_commit": baseline.get("agent_visible_commit", ""),
        "agent_visible_commit_date": baseline.get("agent_visible_commit_date", ""),
        "fix_commit": str(row.get("fix_commit", "")),
        "parent_commit": str(row.get("parent_commit", "")),
        "fix_commit_date": baseline.get("fix_commit_date", ""),
        "paths": {
            "task_path": paths.get("task_path", agent.get("task_path", "")),
            "activity_path": paths.get(
                "activity_path",
                agent.get("openai_agents_activity_path", "")
                or agent.get("claude_activity_path", "")
                or agent.get("openhands_stdout_path", "")
                or agent.get("swe_agent_stdout_path", "")
                or agent.get("copilot_activity_path", ""),
            ),
            "stderr_path": paths.get(
                "stderr_path",
                agent.get("openhands_stderr_path", "")
                or agent.get("swe_agent_stderr_path", ""),
            ),
            "trajectory_path": paths.get("trajectory_path", agent.get("trajectory_path", "")),
            "diagnostic_path": paths.get("diagnostic_path", agent.get("satd_diagnostic_path", "")),
            "evidence_use_path": paths.get("evidence_use_path", agent.get("satd_evidence_use_path", "")),
            "global_activity_trace_path": paths.get("global_activity_trace_path", run_config.get("activity_trace_path", "")),
        },
        "result": {
            "repayment_status": agent.get("repayment_status", ""),
            "extraction_source": agent.get("extraction_source", ""),
            "original_method_updated": agent.get("original_method_updated", ""),
            "comment_only_target_edit": agent.get("comment_only_target_edit", ""),
            "files_touched": agent.get("files_touched", []),
            "summary": agent.get("summary", ""),
        },
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def write_satd_metadata(metadata_dir, row, args, baseline=None, run_config=None, agent=None, paths=None, extra=None):
    """Write `<rand_index>.metadata.json` and return its path as a string."""
    if metadata_dir is None:
        return ""
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / f"{int(row['rand_index'])}.metadata.json"
    payload = build_satd_metadata(row, args, baseline, run_config, agent, paths, extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    return str(path)


def setup_execution_logger():
    logger = logging.getLogger("claude_satd_fix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger
