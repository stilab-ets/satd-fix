"""Pooled multi-label agreement for the current agent and baseline failure sets."""
from pathlib import Path
import re
import pandas as pd
from calculate_llm_agreement import kappa

PACKAGE_ROOT=Path(__file__).resolve().parents[2]
AGENT_CATEGORIES=(
    "Patch Application Failure", "Turn Budget Exhaustion", "Context Acquisition Failure",
    "Localization Failure", "Debt-Retention Failure", "Semantic Repair Failure",
    "Trajectory-Control Failure", "API Hallucination", "Misleading Demonstrative Reference",
)
BASELINE_CATEGORIES=tuple(c for c in AGENT_CATEGORIES if c not in {
    "Turn Budget Exhaustion","Trajectory-Control Failure","Misleading Demonstrative Reference"})
ALIASES={c.casefold():c for c in AGENT_CATEGORIES}
ALIASES.update({
    "missing context":"Context Acquisition Failure",
    "missing context / context acquisition failure":"Context Acquisition Failure",
    "debt retention":"Debt-Retention Failure",
    "data alignment artifact":"Data Alignment Artifact",
})

def labels(*values):
    assigned=set()
    for value in values:
        if pd.isna(value):
            continue
        for part in re.split(r"[;|]",str(value)):
            key=" ".join(part.split()).casefold()
            if key:
                if key not in ALIASES:
                    raise ValueError(f"Unknown label: {part}")
                assigned.add(ALIASES[key])
    if not assigned:
        raise ValueError("Missing author label set")
    return assigned

def main():
    label_data=PACKAGE_ROOT/"RQ3"/"data"
    author_file=label_data/'agent_failed_cases_final_labels.xlsx'
    az=pd.read_excel(author_file,sheet_name='zaki_labeling')
    aj=pd.read_excel(author_file,sheet_name='Jassem_labeling')
    if az.rand_index.duplicated().any() or aj.rand_index.duplicated().any() or set(az.rand_index)!=set(aj.rand_index):
        raise ValueError('Agent author sheets must contain the same unique case IDs')
    paired=az.merge(aj,on='rand_index',suffixes=('_zaki','_jassem'),validate='one_to_one')
    ready=(paired.labeling_status_zaki.ne('pending') & paired.labeling_status_jassem.ne('pending')
           & paired.zaki_primary.notna() & paired.jasem_primary.notna())
    agent=paired[ready]
    print(f'Agent author pairs complete: {len(agent)}; pending: {len(paired)-len(agent)}')
    file=label_data/"llm_failure_labels.xlsx"
    final=pd.read_excel(file,sheet_name="failed_cases")
    zaki=pd.read_excel(file,sheet_name="zaki_labeling")
    jassem=pd.read_excel(file,sheet_name="Jassem_labeling")
    for frame in (agent,final,zaki,jassem):
        if frame.rand_index.isna().any() or frame.rand_index.duplicated().any():
            raise ValueError("Missing or duplicate case identifier")
    if set(zaki.rand_index)!=set(jassem.rand_index):
        raise ValueError("Author snapshots cover different cases")
    baseline=zaki.merge(jassem,on="rand_index",validate="one_to_one")
    baseline=baseline[baseline.rand_index.isin(final.rand_index)]
    if set(baseline.rand_index)!=set(final.rand_index):
        raise ValueError("Author snapshots do not cover all current baseline failures")
    records=[(labels(r.zaki_primary,r.zaki_secondary),labels(r.jasem_primary,r.jasem_secondary),AGENT_CATEGORIES)
             for r in agent.itertuples()]
    records.extend((labels(r.llm_failure_categories),labels(r.failure_categories),BASELINE_CATEGORIES)
                   for r in baseline.itertuples())
    for left,right,categories in records:
        if (left|right)-set(categories):
            raise ValueError(f"Active failure has an unsupported category: {(left|right)-set(categories)}")
    left=[category in a for a,b,categories in records for category in categories]
    right=[category in b for a,b,categories in records for category in categories]
    agreement,coefficient=kappa(left,right)
    print(f"Active failures: {len(agent)} agent + {len(baseline)} LLM = {len(records)}")
    print(f"Binary decisions: {len(left)}; agreements: {sum(a==b for a,b in zip(left,right))}")
    print(f"Agreement: {agreement:.6%}; Cohen's kappa: {coefficient:.9f}")
    print(f"Identical complete sets: {sum(a==b for a,b,_ in records)}/{len(records)}")

if __name__=="__main__":
    main()
