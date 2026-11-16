"""Read recorded-run evidence, recovering missing exports from same-run JSONL."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def _same_path(left, right):
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(os.path.abspath(str(right)))


def _same_recorded_location(recorded, current):
    """Allow an intact same-run artifact to move with the replication package.

    Metadata retains its original absolute paths and hash. The run layout,
    model, case ID, and event identities must still match below.
    """
    if _same_path(recorded,current):
        return True
    old=Path(str(recorded)).parts
    new=Path(str(current)).parts
    return len(old)>=4 and len(new)>=4 and tuple(os.path.normcase(x) for x in old[-4:])==tuple(os.path.normcase(x) for x in new[-4:])


def _same_run_activity(path):
    if path.parent.parent.parent.name != 'trajectories':
        raise FileNotFoundError(f'Cannot locate same-run activity for: {path}')
    run_id, model, case_id = path.parent.name, path.parent.parent.name, int(path.stem)
    root = path.parent.parent.parent.parent
    source = root / 'agent_activity' / 'stdout' / run_id / f'{case_id}.jsonl'
    metadata_path = root / 'metadata' / model / run_id / f'{case_id}.metadata.json'
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw.decode('utf-8-sig'))
    if (metadata.get('rand_index') != case_id or metadata.get('run_id') != run_id
            or metadata.get('model') != model
            or not _same_recorded_location(metadata.get('paths', {}).get('trajectory_path', ''), path)
            or not _same_recorded_location(metadata.get('paths', {}).get('activity_path', ''), source)):
        raise ValueError(f'Same-run metadata mismatch for {path}')
    raw = source.read_bytes()
    events = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if not events or any(e.get('rand_index') != case_id for e in events):
        raise ValueError(f'Activity case identity mismatch for {source}')
    if not any(e.get('event') in ('run_result', 'max_turns_exceeded') for e in events):
        raise ValueError(f'No recorded termination in {source}')
    started = [e for e in events if e.get('event') == 'tool_start']
    ended = [e for e in events if e.get('event') == 'tool_end']
    start_ids = [e.get('tool_call_id') for e in started]
    end_ids = [e.get('tool_call_id') for e in ended]
    if (None in start_ids or None in end_ids or len(set(start_ids)) != len(start_ids)
            or len(set(end_ids)) != len(end_ids) or set(start_ids) != set(end_ids)):
        raise ValueError(f'Incomplete or duplicate tool-call pairs in {source}')
    start_tools = {e['tool_call_id']: e.get('tool') for e in started}
    if any(start_tools[e['tool_call_id']] != e.get('tool') for e in ended):
        raise ValueError(f'Tool identity mismatch in {source}')
    return events, raw, source, {
        'metadata_path': str(metadata_path),
        'metadata_sha256': hashlib.sha256(metadata_raw).hexdigest(),
        'recorded_repayment_status': metadata.get('result', {}).get('repayment_status'),
    }


def recorded_evidence(traj_path):
    path = Path(str(traj_path))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if path.is_file():
        file_raw = path.read_bytes()
        export = json.loads(file_raw.decode('utf-8-sig'))
        reconstruction = export.get('reconstruction')
        if reconstruction is not None:
            if reconstruction.get('kind') != 'same_run_activity_jsonl':
                raise ValueError(f'Unsupported trajectory reconstruction marker in {path}')
            trajectory, raw, source, extra = _same_run_activity(path)
            package_root = Path(__file__).resolve().parents[2]
            metadata_path = Path(extra['metadata_path'])
            if (export.get('trajectory') != trajectory
                    or reconstruction.get('source_path') != source.relative_to(package_root).as_posix()
                    or reconstruction.get('metadata_path') != metadata_path.relative_to(package_root).as_posix()
                    or reconstruction.get('source_sha256') != hashlib.sha256(raw).hexdigest()
                    or reconstruction.get('metadata_sha256') != extra['metadata_sha256']):
                raise ValueError(f'Reconstructed trajectory differs from its recorded source: {path}')
            kind = 'reconstructed_activity_jsonl'
        else:
            raw = file_raw
            trajectory = export['trajectory']
            source, kind, extra = path, 'trajectory_export', {}
    else:
        trajectory, raw, source, extra = _same_run_activity(path)
        kind = 'same_run_activity_jsonl'
    steps = []
    for event in trajectory:
        if event.get('event') != 'tool_end':
            continue
        tool = event.get('tool')
        if not tool:
            action = event.get('action', '')
            try:
                tool = json.loads(action).get('tool', '')
            except (TypeError, ValueError):
                tool = str(action)
        if not tool:
            raise ValueError(f'Tool name missing in {path}')
        observation = event.get('observation')
        if observation is None:
            observation = json.dumps(event, ensure_ascii=False)
        steps.append((tool, str(observation)))
    return {'steps': steps, 'counts': Counter(t for t, _ in steps),
            'n_tool_calls': len(steps),
            'n_llm_turns': sum(e.get('event') == 'llm_end' for e in trajectory),
            'trajectory_path': str(path),
            'trajectory_sha256': hashlib.sha256(raw).hexdigest() if kind == 'trajectory_export' else '',
            'evidence_kind': kind, 'evidence_path': str(source),
            'evidence_sha256': hashlib.sha256(raw).hexdigest(), **extra}
