"""Create an independent pre-fix repository containing only the selected commit's history."""

import os
from pathlib import Path
import re
import subprocess
import tempfile


def _environment():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_TERMINAL_PROMPT="0")
    return env


def _git(repo, *args, executable="git", check=True):
    return subprocess.run(
        [executable, "-C", str(repo), *args], env=_environment(),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=check, timeout=300,
    )


def _commit(repo, ref, executable):
    ref = str(ref).strip()
    if not ref or ref.startswith("-") or "\n" in ref:
        raise ValueError("Invalid commit reference for history isolation.")
    return _git(repo, "rev-parse", "--verify", "--end-of-options",
                f"{ref}^{{commit}}", executable=executable).stdout.strip()


def _transfer_ancestors(source, target, commit, executable):
    """Stream an ancestor-only pack into the independent object database."""
    producer = consumer = None
    with tempfile.TemporaryFile() as errors:
        try:
            producer = subprocess.Popen(
                [executable, "-C", str(source), "pack-objects", "--revs", "--stdout"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                env=_environment(),
            )
            consumer = subprocess.Popen(
                [executable, "-C", str(target), "index-pack", "--stdin"],
                stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_environment(),
            )
            producer.stdout.close()
            producer.stdin.write((commit + "\n").encode("ascii"))
            producer.stdin.close()
            _, consumer_error = consumer.communicate(timeout=300)
            producer.wait(timeout=300)
            if producer.returncode or consumer.returncode:
                errors.seek(0)
                detail = (errors.read() + consumer_error).decode("utf-8", errors="replace")
                raise RuntimeError("Pre-fix history transfer failed: " + detail[-2000:])
        finally:
            for process in (consumer, producer):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()


def verify_history_isolation(target, baseline, withheld_fix, *, git_executable="git"):
    """Abort if the repository exposes a remote, shared objects, or future commits."""
    target = Path(target).resolve()
    git = lambda *args, **kwargs: _git(target, *args, executable=git_executable, **kwargs)
    actual_dir = Path(git("rev-parse", "--absolute-git-dir").stdout.strip()).resolve()
    if actual_dir != target / ".git":
        raise RuntimeError("Isolation requires an independent .git directory, not a linked worktree.")
    common = Path(git("rev-parse", "--git-common-dir").stdout.strip())
    if (common if common.is_absolute() else target / common).resolve() != actual_dir:
        raise RuntimeError("The repository shares a Git object database.")
    alternate = actual_dir / "objects" / "info" / "alternates"
    if alternate.exists() and alternate.read_text(encoding="utf-8").strip():
        raise RuntimeError("Git object alternates would expose the source repository.")
    if git("remote").stdout.strip() or git("for-each-ref", "--format=%(refname)").stdout.strip():
        raise RuntimeError("An isolated detached repository must not expose remotes or refs.")
    if git("rev-parse", "HEAD").stdout.strip() != baseline:
        raise RuntimeError("The isolated repository is not at the selected pre-fix commit.")
    probe = git("cat-file", "-e", f"{withheld_fix}^{{commit}}", check=False)
    if probe.returncode == 0:
        raise RuntimeError("Isolation failed: the developer fixing commit is accessible.")
    allowed = set(git("rev-list", baseline).stdout.splitlines())
    objects = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)").stdout
    stored_commits = {line.split()[0] for line in objects.splitlines() if line.endswith(" commit")}
    if stored_commits != allowed:
        raise RuntimeError("The object database contains commits outside the pre-fix ancestry.")
    git("fsck", "--connectivity-only", "--no-reflogs")
    return {"verified": True, "policy": "pre_fix_ancestors_only",
            "history_commit_count": len(allowed), "fix_commit_accessible": False,
            "independent_object_database": True, "remotes": []}


def prepare_isolated_worktree(source, row, worktrees_root, *, git_executable="git"):
    """Create and verify a pre-fix checkout; return its path and commit metadata."""
    # Reject inherited Git path overrides before repository setup.
    forbidden = {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_COUNT",
                 "GIT_CONFIG_PARAMETERS", "GIT_REPLACE_REF_BASE"}
    if any(key.upper() in forbidden and value for key, value in os.environ.items()):
        raise RuntimeError("Remove inherited Git path/config overrides before starting the agent.")
    source = Path(source).resolve()
    root = Path(worktrees_root).resolve()
    fix = _commit(source, row["fix_commit"], git_executable)
    parent = str(row.get("parent_commit", "")).strip()
    before = _commit(source, parent if parent and parent.lower() != "nan" else fix + "^", git_executable)
    if before == fix or _git(source, "merge-base", "--is-ancestor", before, fix,
                            executable=git_executable, check=False).returncode != 0:
        raise RuntimeError("Selected baseline is not an ancestor preceding the fixing commit.")
    visible_date = _git(source, "show", "-s", "--format=%cI", before, executable=git_executable).stdout.strip()
    fix_date = _git(source, "show", "-s", "--format=%cI", fix, executable=git_executable).stdout.strip()
    root.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", source.name)[:80] + f"_{before[:12]}_isolated_"
    target = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    _git(target, "init", "--template=", executable=git_executable)
    _git(target, "config", "core.longpaths", "true", executable=git_executable)
    _git(target, "config", "core.hooksPath", os.devnull, executable=git_executable)
    _git(target, "config", "protocol.allow", "never", executable=git_executable)
    _transfer_ancestors(source, target, before, git_executable)
    _git(target, "checkout", "--detach", before, executable=git_executable)
    # Prune recovery pointers only in the new private repository.
    _git(target, "reflog", "expire", "--expire=now", "--all", executable=git_executable)
    report = verify_history_isolation(target, before, fix, git_executable=git_executable)
    return target, before, visible_date, fix_date, report
