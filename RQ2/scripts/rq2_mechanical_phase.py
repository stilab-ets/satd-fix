# -*- coding: utf-8 -*-
"""Core rules for label-free RQ2 failure-phase attribution.

Each failed trajectory's phase is decided only from objective,
runner-recorded evidence: the tool-call sequence in the .traj log and
the run's recorded status, independently of manually assigned
failure-category labels. Rules, applied in this order (first match):

  M1 Context Acquisition: an edit is invoked before any completed, successful
                    retrieval of nonempty repository content by read_file,
                    read_target_method, inspect_satd_context, search_text,
                    or an identifiable equivalent read/search command.
                    Directory listings, empty searches, failed reads and
                    LLM calls do not qualify. This is a proxy for editing
                    before additional context retrieval, not proof that
                    necessary information was absent or misunderstood:
                    the prompt already contains the SATD and method.
                    Uninterpretable pre-edit retrieval evidence is flagged,
                    not silently treated as an unsuccessful retrieval.
  M2 Localization   the run's recorded status is a non-target change:
                    an edit was applied, but only outside the provided
                    target method/file.
  M3 Orchestration  destructive edit followed by state confusion: a
                    replace_in_file fails (status not_found) after an
                    earlier write_file on the worktree -- the agent
                    destroyed content and kept acting on a stale model
                    of the file, unrecovered. Empirically this rule
                    attributes no case in this dataset: the only two
                    trajectories with the signature (122, 1000013) are
                    non-target edits already claimed by M2, i.e., the
                    destructive confusion follows an earlier
                    wrong-target commitment.
  M4 Orchestration  no file-modifying tool call in the whole
                    trajectory, or the turn budget expired
                    (turn-budget termination): termination without (completed)
                    action.
  M5 Editing        residual class: an edit was invoked and none of
                    the preceding rules applies.

The RQ2 results driver imports this module to assign phases; it writes the
result workbook.
"""
import json
import re
import shlex
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / 'RQ3' / 'scripts'))
from trajectory_evidence import recorded_evidence

CONTENT_TOOLS = {"read_file", "read_target_method", "inspect_satd_context", "search_text"}
INSPECT_TOOLS = CONTENT_TOOLS | {"explore_worktree", "list_directory"}
EDIT_TOOLS = {"write_file", "replace_in_file"}
NOT_FOUND = re.compile(r"not[ _]found|no[ _]match|not[ _]present|nothing (?:was )?replaced|0 replacements", re.I)
PHASES = ["Context Acquisition", "Localization", "Editing", "Orchestration", "Unresolved"]


def tool_records(evidence):
    """Pair raw events by call ID; positions are one-based event positions.

    Content must have completed BEFORE the first edit request, including when
    calls run concurrently. Preserve the existing completion sequence for M2-M5.
    """
    raw = Path(evidence['evidence_path']).read_text(encoding='utf-8-sig')
    events = (json.loads(raw)['trajectory'] if evidence['evidence_kind'] == 'trajectory_export'
              else [json.loads(line) for line in raw.splitlines() if line.strip()])
    starts, ends = {}, {}
    for pos, event in enumerate(events, 1):
        try:
            inner = json.loads(event.get('observation', ''))
            if isinstance(inner, dict) and inner.get('event') == event.get('event'):
                event = inner
        except (ValueError, TypeError):
            pass
        if event.get('event') not in ('tool_start', 'tool_end'):
            continue
        call_id = event.get('tool_call_id')
        if not call_id or not event.get('tool'):
            raise ValueError('Missing tool identity or call ID')
        dest = starts if event['event'] == 'tool_start' else ends
        if call_id in dest:
            raise ValueError('Duplicate tool-call event')
        dest[call_id] = (pos, event)
    if set(starts) != set(ends):
        raise ValueError('Incomplete tool-call pairs')
    records = []
    for key, (start, event) in starts.items():
        end, completed = ends[key]
        if end <= start or event['tool'] != completed['tool']:
            raise ValueError('Invalid tool-call pairing/order')
        arguments = event.get('arguments', {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or 'result' not in completed:
            raise ValueError('Missing tool arguments or result')
        records.append({'tool': event['tool'], 'call_id': key,
                        'start_position': start, 'end_position': end,
                        'arguments': arguments, 'result': str(completed['result'])})
    if len(records) != evidence['n_tool_calls']:
        raise ValueError('Tool count differs from recorded evidence')
    return sorted(records, key=lambda r: r['start_position'])


def command_kind(command):
    """Conservative recognition of single read/search commands; never execute."""
    command = str(command).strip()
    if re.search(r'[|;&<>\r\n]', command):
        return 'ambiguous'
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return 'ambiguous'
    if not tokens:
        return 'other'
    name = tokens[0].replace('\\', '/').split('/')[-1].lower().removesuffix('.exe')
    args = [s.lower() for s in tokens[1:]]
    if name in {'cat', 'head', 'tail', 'type', 'get-content'}:
        return 'content' if args else 'other'
    if name in {'rg', 'grep', 'findstr'}:
        # Filename-only, count-only and quiet searches do not return content.
        if any(s in {'--files', '--files-with-matches', '--files-without-match',
                     '--count', '--count-matches', '--quiet'} for s in args):
            return 'other'
        if any(s.startswith('-') and not s.startswith('--') and any(c in s[1:] for c in 'lcq') for s in args):
            return 'other'
        return 'content'
    if name == 'sed':
        return 'other' if any(s.startswith('-i') for s in args) else 'ambiguous'
    if name == 'git':
        return 'content' if args and args[0] in {'show', 'blame'} else 'other'
    if name in {'pytest', 'pip', 'mvn', 'ant', 'gradlew', 'gradlew.bat', 'find', 'ls', 'dir'}:
        return 'other'
    return 'ambiguous'


def acquisition(record):
    """Return (qualifies, reason). None means unsupported/ambiguous evidence."""
    tool, text = record['tool'], record['result']
    if tool in {'list_directory', 'explore_worktree'}:
        return False, 'structure_only'
    if tool == 'run_command':
        kind = command_kind(record['arguments'].get('command', ''))
        if kind == 'other': return False, 'not_content_retrieval'
        match = re.search(r'^exit_code=(-?\d+)\s*\nSTDOUT:\n(.*?)\nSTDERR:\n', text, re.S)
        if not match:
            if text.lstrip().startswith('{'):
                try:
                    if json.loads(text).get('status') in {'blocked', 'error'}:
                        return False, 'command_failed'
                except ValueError: pass
            return None, 'unrecognized_command_result'
        if int(match[1]) != 0: return False, 'command_failed'
        if kind == 'ambiguous': return None, 'unrecognized_command'
        return (bool(match[2].strip()), 'content_returned' if match[2].strip() else 'empty_content')
    if tool not in CONTENT_TOOLS:
        return False, 'not_content_retrieval'
    if not text.strip(): return False, 'empty_content'
    # Match the toolkit's numbered source lines, not success metadata alone.
    if tool == 'search_text':
        if re.search(r'^.+?:\d+:[ \t]*\S', text, re.M): return True, 'matching_content_returned'
        if text.strip() == 'No matches': return False, 'no_matches'
    elif re.search(r'^\d+:[ \t]*\S', text, re.M):
        return True, 'source_content_returned'
    if re.match(r'(File does not exist:|Error\b|Traceback\b)', text.lstrip(), re.I):
        return False, 'retrieval_failed'
    if text.lstrip().startswith('{'):
        try:
            status = json.JSONDecoder().raw_decode(text.lstrip())[0].get('status')
            if status in {'not_found', 'satd_comment_not_found', 'error', 'blocked'}:
                return False, 'retrieval_failed'
        except (ValueError, AttributeError): pass
    if re.search(r' lines \d+-\d+ of \d+', text) and not re.search(r'^\d+:[ \t]*\S', text, re.M):
        return False, 'empty_content'
    return None, 'unrecognized_retrieval_result'


def context_audit(records):
    first_edit = next((r for r in records if r['tool'] in EDIT_TOOLS), None)
    before = [r for r in records if first_edit and r['end_position'] < first_edit['start_position']]
    checks = []
    for r in before:
        if r['tool'] not in INSPECT_TOOLS | {'run_command'}: continue
        success, reason = acquisition(r)
        checks.append({**r, 'qualifies': success, 'reason': reason})
    successful = sorted([r for r in checks if r['qualifies'] is True],
                        key=lambda r: r['end_position'])
    unknown = [r for r in checks if r['qualifies'] is None]
    if successful:
        first = successful[0]
        lines = first['result'].splitlines()
        source = next((line for line in lines if re.search(r'^\d+:[ \t]*\S|^.+?:\d+:[ \t]*\S', line)), '')
        if not source and first['tool'] == 'run_command':
            source = first['result'].split('STDOUT:\n', 1)[-1].split('\nSTDERR:', 1)[0].strip().splitlines()[0]
        audit_summary = (f"{first['tool']}; call {first['call_id']}; result at event {first['end_position']}. "
                         f"Content: {source[:160]}")
    else:
        audit_summary = 'No qualifying content returned before the edit.' if first_edit else 'No edit invoked.'
    details = {
        'first_edit_position': first_edit['start_position'] if first_edit else None,
        'pre_edit_content_calls': len(successful),
        'first_content_tool': successful[0]['tool'] if successful else '',
        'first_content_position': successful[0]['end_position'] if successful else None,
        # Full arguments/results remain at the recorded source path and hash.
        'pre_edit_context_evidence': audit_summary,
    }
    if first_edit and not successful and unknown:
        raise ValueError('Cannot establish M1: ambiguous pre-edit retrieval evidence')
    return bool(first_edit and not successful), details


def assign(steps, repayment_status, result_subtype, records):
    first_edit = next((i for i, (t, _) in enumerate(steps) if t in EDIT_TOOLS), None)
    # M1: completed successful content retrieval, not merely an inspection call.
    no_context, _ = context_audit(records)
    if no_context:
        return "Context Acquisition", "M1_edit_before_content_retrieval"

    # M2: applied change landed only outside the target
    if str(repayment_status).startswith("non_target"):
        return "Localization", "M2_non_target_change"

    # M3: destructive write followed by a failed replace (stale file model)
    seen_write = False
    for t, obs in steps:
        if t == "write_file":
            seen_write = True
        elif t == "replace_in_file" and seen_write and NOT_FOUND.search(obs):
            return "Orchestration", "M3_destructive_confusion"

    # M4: never edited, or budget expired
    if first_edit is None or str(result_subtype) in {"returncode_2", "turn_budget_exhausted"}:
        return "Orchestration", "M4_no_edit_or_budget"

    # M5: edited the target but the change is wrong / comment-only
    return "Editing", "M5_wrong_target_edit"
