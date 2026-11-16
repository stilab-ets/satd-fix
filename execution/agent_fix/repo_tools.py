"""Repository traversal, reading, searching, editing, and Git inspection tools."""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

try:
    import pathspec
except Exception:  # pragma: no cover - optional dependency
    pathspec = None

try:
    import lizard
except Exception:  # pragma: no cover - optional dependency
    lizard = None

try:
    from git import Repo
except Exception:  # pragma: no cover - optional dependency
    Repo = None


DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".venv", "venv", "env", "node_modules", "dist", "build",
}


def clean_text(text):
    return str(text or "").encode("utf-8", errors="replace").decode("utf-8")


def compact_text(text, limit=20000):
    text = clean_text(text)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated: {len(text)} chars total]"


def json_text(payload, limit=20000):
    return compact_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), limit=limit)


class RepoToolkit:
    def __init__(self, worktree, command_timeout=120):
        self.root = Path(worktree).resolve()
        self.command_timeout = int(command_timeout or 120)
        self.ignore_spec = self._load_ignore_spec()

    def safe_join(self, relative_path):
        target = (self.root / str(relative_path or ".").replace("\\", "/")).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError(f"Path escapes worktree: {relative_path}")
        return target

    def _load_ignore_spec(self):
        if pathspec is None:
            return None
        gitignore = self.root / ".gitignore"
        if not gitignore.is_file():
            return None
        try:
            return pathspec.PathSpec.from_lines(
                "gitwildmatch",
                gitignore.read_text(encoding="utf-8", errors="replace").splitlines(),
            )
        except Exception:
            return None

    def _rel(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def _is_ignored(self, path):
        try:
            rel = self._rel(path)
        except Exception:
            return True
        if any(part in DEFAULT_EXCLUDE_DIRS for part in Path(rel).parts):
            return True
        return bool(self.ignore_spec and self.ignore_spec.match_file(rel))

    def list_directory(self, path="."):
        target = self.safe_join(path)
        if not target.exists():
            return f"Path does not exist: {path}"
        if target.is_file():
            return self._rel(target)
        rows = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if self._is_ignored(child):
                continue
            suffix = "/" if child.is_dir() else ""
            rows.append(self._rel(child) + suffix)
        return "\n".join(rows)

    def explore_worktree(self, path=".", max_depth=2, max_entries=200):
        start = self.safe_join(path)
        max_depth = max(0, int(max_depth or 0))
        max_entries = max(1, int(max_entries or 200))
        entries = []
        truncated = False

        def visit(directory, depth):
            nonlocal truncated
            if truncated or depth > max_depth:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except Exception:
                return
            for child in children:
                if self._is_ignored(child):
                    continue
                if len(entries) >= max_entries:
                    truncated = True
                    return
                entries.append({
                    "path": self._rel(child),
                    "type": "directory" if child.is_dir() else "file",
                })
                if child.is_dir():
                    visit(child, depth + 1)

        if start.is_file():
            entries.append({"path": self._rel(start), "type": "file"})
        elif start.exists():
            visit(start, 0)
        else:
            return f"Path does not exist: {path}"

        return json_text({
            "root": self._rel(start) if start != self.root else ".",
            "max_depth": max_depth,
            "entry_count": len(entries),
            "truncated": truncated,
            "entries": entries,
            "libraries": {
                "pathspec": pathspec is not None,
                "gitpython": Repo is not None,
                "lizard": lizard is not None,
            },
        })

    def read_file(self, path, start_line=1, max_lines=300):
        target = self.safe_join(path)
        if not target.is_file():
            return f"File does not exist: {path}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line or 1))
        end = min(len(lines), start + max(1, int(max_lines or 300)) - 1)
        numbered = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
        header = f"{self._rel(target)} lines {start}-{end} of {len(lines)}"
        return compact_text(header + "\n" + "\n".join(numbered), limit=30000)

    def write_file(self, path, content):
        target = self.safe_join(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return json_text({"status": "written", "path": self._rel(target), "chars": len(str(content))})

    def replace_in_file(self, path, old, new, count=1):
        target = self.safe_join(path)
        if not target.is_file():
            return f"File does not exist: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        limit = 0 if int(count or 1) <= 0 else int(count or 1)
        occurrences = text.count(old)
        if occurrences == 0:
            return json_text({"status": "not_found", "path": self._rel(target), "old_text_chars": len(str(old))})
        target.write_text(text.replace(old, new, limit), encoding="utf-8")
        changed = occurrences if limit == 0 else min(occurrences, limit)
        return json_text({"status": "replaced", "path": self._rel(target), "occurrences_changed": changed})

    def search_text(self, pattern, path_glob="", max_results=50):
        max_results = max(1, int(max_results or 50))
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "-n", "--hidden", "--glob", "!.git", "--", str(pattern)]
            if path_glob:
                cmd[2:2] = ["--glob", str(path_glob)]
            completed = subprocess.run(
                cmd,
                cwd=str(self.root),
                text=True,
                capture_output=True,
                timeout=30,
            )
            output = completed.stdout if completed.returncode in (0, 1) else completed.stderr
            lines = output.splitlines()[:max_results]
            return "\n".join(lines) or "No matches"

        regex = re.compile(str(pattern))
        rows = []
        for file_path in self.root.rglob("*"):
            if len(rows) >= max_results:
                break
            if self._is_ignored(file_path) or not file_path.is_file():
                continue
            try:
                for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if regex.search(line):
                        rows.append(f"{self._rel(file_path)}:{line_no}:{line}")
                        if len(rows) >= max_results:
                            break
            except Exception:
                continue
        return "\n".join(rows) or "No matches"

    def _lizard_functions(self, file_path):
        if lizard is None:
            return []
        try:
            analysis = lizard.analyze_file(str(file_path))
        except Exception:
            return []
        functions = []
        for fn in getattr(analysis, "function_list", []) or []:
            functions.append({
                "name": getattr(fn, "name", ""),
                "long_name": getattr(fn, "long_name", getattr(fn, "name", "")),
                "start_line": int(getattr(fn, "start_line", 0) or 0),
                "end_line": int(getattr(fn, "end_line", 0) or 0),
                "cyclomatic_complexity": getattr(fn, "cyclomatic_complexity", None),
                "nloc": getattr(fn, "nloc", None),
                "token_count": getattr(fn, "token_count", None),
                "parameter_count": len(getattr(fn, "parameters", []) or []),
            })
        return functions

    def _ast_functions(self, file_path):
        if file_path.suffix.lower() != ".py":
            return []
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return []
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = int(getattr(node, "lineno", 0) or 0)
                end = int(getattr(node, "end_lineno", start) or start)
                functions.append({
                    "name": node.name,
                    "long_name": node.name,
                    "start_line": start,
                    "end_line": end,
                    "source": "ast",
                })
        return functions

    def _functions(self, file_path):
        functions = self._lizard_functions(file_path)
        if functions:
            for fn in functions:
                fn["source"] = "lizard"
            return functions
        return self._ast_functions(file_path)

    def read_target_method(self, path, method_name, max_lines=260):
        target = self.safe_join(path)
        if not target.is_file():
            return f"File does not exist: {path}"
        functions = self._functions(target)
        method_name = str(method_name or "").strip()
        matches = [
            fn for fn in functions
            if fn.get("name") == method_name
            or fn.get("long_name") == method_name
            or str(fn.get("long_name", "")).endswith("." + method_name)
        ]
        if not matches:
            return json_text({
                "status": "not_found",
                "path": self._rel(target),
                "method_name": method_name,
                "available_functions_sample": functions[:20],
                "library": "lizard" if lizard is not None else "ast",
            }, limit=30000)
        fn = sorted(matches, key=lambda item: (item.get("end_line", 0) - item.get("start_line", 0)))[0]
        content = self.read_file(target.relative_to(self.root), fn["start_line"], min(max_lines, fn["end_line"] - fn["start_line"] + 1))
        return json_text({"status": "found", "path": self._rel(target), "method": fn}, limit=8000) + "\n" + content

    def inspect_satd_context(self, path, satd_comment, context_lines=25):
        target = self.safe_join(path)
        if not target.is_file():
            return f"File does not exist: {path}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        needle = str(satd_comment or "").strip()
        hit_line = None
        if needle:
            for index, line in enumerate(lines, 1):
                if needle in line or needle.lstrip("#/ ").strip() in line:
                    hit_line = index
                    break
        if hit_line is None:
            return json_text({
                "status": "satd_comment_not_found",
                "path": self._rel(target),
                "satd_comment": compact_text(needle, 500),
            })

        functions = self._functions(target)
        enclosing = [
            fn for fn in functions
            if int(fn.get("start_line", 0) or 0) <= hit_line <= int(fn.get("end_line", 0) or 0)
        ]
        enclosing_fn = sorted(enclosing, key=lambda fn: fn.get("end_line", 0) - fn.get("start_line", 0))[0] if enclosing else None
        start = max(1, hit_line - int(context_lines or 25))
        end = min(len(lines), hit_line + int(context_lines or 25))
        context = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
        return json_text({
            "status": "found",
            "path": self._rel(target),
            "satd_line": hit_line,
            "enclosing_function": enclosing_fn,
            "analysis_library": "lizard" if lizard is not None else "ast",
            "context_range": [start, end],
        }, limit=8000) + "\n" + "\n".join(context)

    def git_status(self):
        if Repo is not None:
            try:
                repo = Repo(str(self.root), search_parent_directories=False)
                return "\n".join(repo.git.status("--porcelain").splitlines())
            except Exception:
                pass
        completed = subprocess.run(
            ["git", "-C", str(self.root), "status", "--porcelain"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=20,
        )
        return (completed.stdout or "").rstrip("\r\n") or "clean"

    def git_diff(self, max_chars=40000):
        if Repo is not None:
            try:
                repo = Repo(str(self.root), search_parent_directories=False)
                return compact_text(repo.git.diff("--binary"), limit=max_chars)
            except Exception:
                pass
        completed = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--binary"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
        )
        return compact_text(completed.stdout or "", limit=max_chars) or "No diff"

    def run_command(self, command, timeout_seconds=60):
        blocked = [
            "git reset", "git checkout", "git clean", "remove-item", " del ",
            " rmdir ", "format ", "shutdown", "reg delete",
        ]
        lowered = f" {str(command).lower()} "
        if any(term in lowered for term in blocked):
            return json_text({"status": "blocked", "command": command, "reason": "potentially destructive"})
        timeout = max(1, min(int(timeout_seconds or self.command_timeout), self.command_timeout))
        completed = subprocess.run(
            str(command),
            cwd=str(self.root),
            shell=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        return compact_text(
            f"exit_code={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
            limit=30000,
        )
