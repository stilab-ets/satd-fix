"""Read the authoritative Excel labels and join fixed recorded-run evidence.

The failed_cases sheet supplies all current primary/secondary and subcategory
assignments. Author sheets are never rewritten by analysis commands.
"""
from pathlib import Path, PureWindowsPath
import json
import re
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FINAL_WORKBOOK = PACKAGE_ROOT/'RQ3/data/agent_failed_cases_final_labels.xlsx'
EVIDENCE = PACKAGE_ROOT/'RQ3/data/agent_failure_evidence.json'
OUTCOMES_WORKBOOK = PACKAGE_ROOT/'RQ1/results/rq1_agentic_ai_vs_llm_metrics.xlsx'
CATEGORIES = ('Debt-Retention Failure','Patch Application Failure','Semantic Repair Failure',
              'Trajectory-Control Failure','Context Acquisition Failure','Turn Budget Exhaustion',
              'Localization Failure','API hallucination','Misleading demonstrative reference')
ALIASES={c.casefold():c for c in CATEGORIES}
ALIASES.update({'missing context / context acquisition failure':'Context Acquisition Failure',
                'missing context':'Context Acquisition Failure','debt retention':'Debt-Retention Failure'})


def final_set(value):
    if pd.isna(value): return set()
    parts=[p.strip() for p in re.split(r'[;|]',str(value)) if p.strip()]
    try: return {ALIASES[p.casefold()] for p in parts}
    except KeyError as exc: raise ValueError(f'Unknown failure category: {exc.args[0]}') from exc


def read_final_failed():
    labels=pd.read_excel(FINAL_WORKBOOK,sheet_name='failed_cases')
    if labels.rand_index.isna().any() or labels.rand_index.duplicated().any():
        raise ValueError('Missing or duplicate final-label IDs')
    cases=json.loads(EVIDENCE.read_text(encoding='utf-8'))['cases']
    evidence=pd.DataFrame(cases)[['rand_index', 'tool_counts']]
    if evidence.rand_index.duplicated().any() or set(evidence.rand_index)!=set(labels.rand_index):
        raise ValueError('Recorded evidence must cover the same final-label cases')
    frame=labels.merge(evidence,on='rand_index',validate='one_to_one')
    frame['failure_category_primary']=frame.failure_category_primary.map(lambda s:next(iter(final_set(s))) if len(final_set(s))==1 else '')
    sets=frame.failure_categories.map(final_set)
    for i,r in frame.iterrows():
        if r.failure_category_primary not in sets[i]: raise ValueError(f'Primary category is missing from label set: {r.rand_index}')
    frame['failure_categories']=sets.map(lambda ls:' | '.join(c for c in CATEGORIES if c in ls))
    for col in ['debt_retention_subcategory_derived','context_failure_subcategories',
                'semantic_failure_subcategory','localization_failure_subcategories']:
        if col not in frame: raise ValueError(f'Missing annotation column in final workbook: {col}')
        frame[col]=frame[col].fillna('')
    frame['trajectory_path']=frame.trajectory_path.map(lambda value:str(PACKAGE_ROOT.joinpath('data/trajectories',*PureWindowsPath(str(value)).parts[-3:])))
    # Validate scope against final outcomes, not a historical 233-case snapshot.
    outcomes = pd.read_excel(OUTCOMES_WORKBOOK, sheet_name='final_outcomes',
                             usecols=['rand_index', 'agent_outcome'])
    failed = set(outcomes.loc[outcomes.agent_outcome.eq('failure'), 'rand_index'])
    if set(frame.rand_index)!=failed: raise ValueError('Final labels must cover every final agent failure')
    return frame.sort_values('rand_index').reset_index(drop=True)
