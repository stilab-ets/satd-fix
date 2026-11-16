# -*- coding: utf-8 -*-
"""RQ3 additional analyses over the 297 failed Agentic-AI cases.

All category analyses are MULTI-LABEL: a failed case can carry several
failure categories (failure_categories, pipe-separated), so occurrence
percentages can sum to more than 100%.

Outputs RQ3/results/RQ3_analysis.xlsx with:
  category_occurrence      multi-label counts/% (297 failed)
  cooccurrence_matrix      9x9 symmetric co-occurrence counts
  subcategory_counts       category -> subcategory counts (taxonomy tree data)
  behavior_signatures      median tool-use profile per category (multi-label)
  phase_x_category_mechanical phase counts for each multi-label category
"""
import itertools
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / 'RQ3' / 'scripts'))
from final_labels import read_final_failed
from trajectory_evidence import recorded_evidence

OUT_DIR = PACKAGE_ROOT / 'RQ3' / 'results'
RQ2_WORKBOOK = PACKAGE_ROOT / 'RQ2/results/RQ2_phase_attribution.xlsx'

CATS = ["Debt-Retention Failure", "Patch Application Failure", "Semantic Repair Failure",
        "Trajectory-Control Failure", "Context Acquisition Failure", "Turn Budget Exhaustion",
        "Localization Failure", "API hallucination", "Misleading demonstrative reference"]

TOOL_GROUPS = {
    "read": ["read_file", "read_target_method", "inspect_satd_context", "list_directory", "explore_worktree"],
    "search": ["search_text"],
    "edit": ["write_file", "replace_in_file"],
    "command": ["run_command"],
    "git_inspect": ["get_git_status", "get_git_diff"],
}


def labels_of(s):
    return [c.strip() for c in str(s or "").split("|") if c.strip()]


def tool_profile(tc):
    m = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", str(tc or ""))}
    return {g: sum(m.get(t, 0) for t in tools) for g, tools in TOOL_GROUPS.items()}


def build_frames(phases=None):
    fc = read_final_failed()
    n = len(fc)
    fc["labels"] = fc["failure_categories"].map(labels_of)

    # ---- A. multi-label occurrence
    occ = pd.Series([l for ls in fc["labels"] for l in ls]).value_counts().reindex(CATS).fillna(0).astype(int)
    occ_df = pd.DataFrame({"category": occ.index, "count": occ.values})
    occ_df["percent_of_failed"] = (occ_df["count"] / n * 100).round(1)

    # ---- B. co-occurrence matrix
    co = pd.DataFrame(0, index=CATS, columns=CATS)
    for ls in fc["labels"]:
        for a in ls:
            co.loc[a, a] += 1
        for a, b in itertools.combinations(sorted(set(ls)), 2):
            co.loc[a, b] += 1
            co.loc[b, a] += 1

    # ---- D. subcategory counts (taxonomy tree data)
    # Debt-Retention is partitioned exhaustively into three subcategories:
    # Superficial Repayment (manual label) plus two derived mechanically
    # from the recorded run status (how the debt survived).
    def dr_sub(r):
        return r['debt_retention_subcategory_derived'] if 'Debt-Retention Failure' in r['labels'] else ''
    sub_rows = []
    drv = fc.apply(dr_sub, axis=1)
    for v, c in drv[drv.ne("")].value_counts().items():
        sub_rows.append({"category": "Debt-Retention Failure", "subcategory": v, "count": int(c)})
    for col, cat in [("context_failure_subcategories", "Context Acquisition Failure"),
                     ("localization_failure_subcategories", "Localization Failure")]:
        selected = fc[fc['labels'].map(lambda ls: cat in ls)]
        vals = [v.strip() for s in selected[col].dropna().astype(str) for v in s.split("|")
                if v.strip() and v.strip().lower() != "nan"]
        for v, c in pd.Series(vals).value_counts().items():
            sub_rows.append({"category": cat, "subcategory": v, "count": int(c)})
    sem = fc["semantic_failure_subcategory"].dropna().astype(str).str.strip()
    for v, c in sem[sem.ne("") & sem.ne("nan")].value_counts().items():
        sub_rows.append({"category": "Semantic Repair Failure", "subcategory": v, "count": int(c)})
    sub_df = pd.DataFrame(sub_rows)

    # ---- E. behavioral signatures (median tool profile of cases carrying label)
    profile_rows = []
    for _, r in fc.iterrows():
        try:
            ev = recorded_evidence(r['trajectory_path'])
            if (ev.get('recorded_repayment_status') is not None
                    and ev['recorded_repayment_status'] != r['repayment_status']):
                raise ValueError('Recovered activity status differs from failed-case metadata')
            profile = {g: sum(ev['counts'].get(t, 0) for t in tools) for g, tools in TOOL_GROUPS.items()}
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Missing recorded tool evidence for case {r['rand_index']}") from exc
        profile_rows.append(profile)
    prof = pd.DataFrame(profile_rows, index=fc.index)
    fcp = pd.concat([fc[["labels"]], prof], axis=1)
    sig_rows = []
    for c in CATS:
        sel = fcp[fcp["labels"].map(lambda ls: c in ls)]
        verified = sel.dropna(subset=list(TOOL_GROUPS))
        if len(verified) != len(sel):
            raise ValueError(f'Missing recorded tool evidence for category: {c}')
        med = verified[list(TOOL_GROUPS)].median().round(1)
        sig_rows.append({'category': c, 'n_cases': len(verified), **med.to_dict()})
    sig_df = pd.DataFrame(sig_rows)

    # Keep phase assignment independent of the final multi-label occurrence.
    if phases is None:
        phases = pd.read_excel(RQ2_WORKBOOK, sheet_name='per_case_attribution', usecols=['rand_index', 'phase'])
    if set(phases['rand_index']) != set(fc['rand_index']):
        raise ValueError('Phase metadata must cover the same final-label cases')
    joined = fc[['rand_index', 'labels']].merge(phases[['rand_index', 'phase']], on='rand_index', validate='one_to_one')
    phase_order = ['Context Acquisition', 'Localization', 'Editing', 'Orchestration', 'Unresolved']
    assigned = joined.explode('labels', ignore_index=True)
    phase_counts = pd.crosstab(assigned.phase, assigned.labels).reindex(
        index=phase_order, columns=CATS, fill_value=0)
    phase_counts.index.name = 'phase'

    frames = {
        'category_occurrence': (occ_df, False),
        'cooccurrence_matrix': (co, True),
        'subcategory_counts': (sub_df, False),
        'behavior_signatures': (sig_df, False),
        'phase_x_category_mechanical': (phase_counts, True),
    }
    return frames


def main():
    sys.path.insert(0,str(PACKAGE_ROOT/'scripts'))
    from workbook_io import matrix,write_tables
    out=OUT_DIR/'RQ3_analysis.xlsx'
    write_tables(out,{name:matrix(frame,index) for name,(frame,index) in build_frames().items()})
    print('Wrote',out)


if __name__ == '__main__': main()
