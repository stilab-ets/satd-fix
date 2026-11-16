# -*- coding: utf-8 -*-
"""RQ4: What is the price of autonomy?

Conditions the failure analyses on the RQ1 outcome quadrants, in both
directions (all category analyses are MULTI-LABEL):

  * Agent failure-category prevalence
    for the 117 agent-only failures (LLM succeeded) vs the 180
    both-failed instances.
  * What agency fixes: LLM failure-category mix (stored final
    LLM labels) for the 129 agent-rescued instances vs the 180
    both-failed instances.

Outputs RQ4/results/RQ4_analysis.xlsx.
"""
from pathlib import Path
import sys

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / 'RQ3' / 'scripts'))
from final_labels import read_final_failed
sys.path.insert(0,str(PACKAGE_ROOT/'RQ1/scripts'))
from prepare_final_excel import load_records

LLM_FAILED = PACKAGE_ROOT / 'RQ3' / 'data' / 'llm_failure_labels.xlsx'
OUT_DIR = PACKAGE_ROOT / 'RQ4' / 'results'

CATS = ["Debt-Retention Failure", "Patch Application Failure", "Semantic Repair Failure",
        "Trajectory-Control Failure", "Context Acquisition Failure", "Turn Budget Exhaustion",
        "Localization Failure", "API hallucination", "Misleading demonstrative reference"]


def labels_of(s):
    return ['Context Acquisition Failure' if c.strip() == 'Missing Context' else c.strip()
            for c in str(s or "").split("|") if c.strip()]


def mix(df, label_col, cats, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        cnt = pd.Series([l for ls in sub[label_col] for l in ls]).value_counts()
        for c in cats:
            rows.append({"group": g, "category": c, "count": int(cnt.get(c, 0)),
                         "percent_of_group": round(cnt.get(c, 0) / len(sub) * 100, 1),
                         "group_size": len(sub)})
    return pd.DataFrame(rows)


def context_standardization(failures, outcomes):
    dimensions = ['difficulty_label', 'score_context_dependence']
    case_scores = pd.DataFrame(outcomes)[['rand_index', *dimensions]]
    cases = failures[['rand_index', 'group', 'labels']].merge(case_scores, on='rand_index', validate='one_to_one')
    if len(cases) != len(failures) or cases[dimensions].isna().any().any():
        raise ValueError('Every baseline failure must have difficulty and context-dependence scores')
    cases['has_context_failure'] = cases.labels.map(lambda labels: 'Context Acquisition Failure' in labels)
    observed = cases.groupby('group').has_context_failure.mean()
    by_stratum = cases.groupby([*dimensions, 'group']).has_context_failure.agg(['size', 'mean']).unstack()
    if by_stratum['size'].isna().any().any():
        raise ValueError('Both outcome groups must have cases in every observed stratum')
    weights = by_stratum['size'].sum(axis=1) / len(cases)
    standardized = by_stratum['mean'].mul(weights, axis=0).sum()
    rows = []
    for method, rates in [('Observed', observed), ('Standardized', standardized)]:
        rescued, both_failed = 100 * rates['agent_rescued'], 100 * rates['both_fail']
        rows.append({'method': method, 'agent_rescued_percent': round(rescued, 1),
                     'both_failed_percent': round(both_failed, 1), 'gap_pp': round(rescued - both_failed, 1)})
    return pd.DataFrame(rows)


def build_frames():
    outcomes=load_records()
    agent_only={c['rand_index'] for c in outcomes if c['quadrant']=='B'}
    rescued={c['rand_index'] for c in outcomes if c['quadrant']=='C'}

    # ---- forward direction: what agency breaks (agent failure labels)
    fc = read_final_failed()
    fc["labels"] = fc["failure_categories"].map(labels_of)
    fc["group"] = fc["rand_index"].map(lambda r: "agent_only_fail" if r in agent_only else "both_fail")
    quad_df = mix(fc, "labels", CATS, "group")

    # ---- inverse direction: what agency fixes (LLM failure labels)
    lf = pd.read_excel(LLM_FAILED, sheet_name="failed_cases")
    expected_failed = rescued | set(fc.loc[fc["group"] == "both_fail", "rand_index"])
    if lf["rand_index"].duplicated().any() or set(lf["rand_index"]) != expected_failed:
        raise ValueError("Final baseline failure labels must match the corrected outcome groups")
    lf["labels"] = lf["final_failure_categories"].map(labels_of)
    lf["group"] = lf["rand_index"].map(lambda r: "agent_rescued" if r in rescued else "both_fail")
    llm_cats = sorted({l for ls in lf["labels"] for l in ls})
    inv_df = mix(lf, "labels", llm_cats, "group")

    # ---- diverging deltas: percentage-point shift of each category
    # relative to the both-failed baseline (the "price of autonomy" figure)
    def deltas(df_mix, group_name):
        # Preserve full precision until the final percentage-point difference.
        prevalence = df_mix.assign(prevalence=100 * df_mix["count"] / df_mix["group_size"])
        piv = prevalence.pivot_table(index="category", columns="group", values="prevalence")
        d = (piv[group_name] - piv["both_fail"]).round(1)
        return d.sort_values(ascending=False)
    d1 = deltas(quad_df, "agent_only_fail")
    d2 = deltas(inv_df, "agent_rescued")
    delta_df = pd.concat([
        pd.DataFrame({"direction": "agent_fail_llm_success (what agency breaks)",
                      "category": d1.index, "delta_pp": d1.values}),
        pd.DataFrame({"direction": "llm_fail_agent_success (what agency fixes)",
                      "category": d2.index, "delta_pp": d2.values}),
    ], ignore_index=True)

    return {'agent_fail_llm_success':(quad_df,False),
            'llm_fail_agent_success':(inv_df,False),
            'delta_price_of_autonomy':(delta_df,False),
            'context_standardization':(context_standardization(lf, outcomes), False)}


def main():
    sys.path.insert(0,str(PACKAGE_ROOT/'scripts'))
    from workbook_io import matrix,write_tables
    out=OUT_DIR/'RQ4_analysis.xlsx'
    write_tables(out,{name:matrix(frame,index) for name,(frame,index) in build_frames().items()})
    print('Wrote',out)


if __name__ == '__main__': main()
