"""Prepare RQ3/RQ4 analysis tables from the current final-label workbooks.

Export final analysis workbooks without rewriting author annotations.
"""
from pathlib import Path
import sys
import json
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'RQ3/scripts'))
sys.path.insert(0,str(PACKAGE/'RQ4/scripts'))
import rq3_analysis
import rq4_analysis


def matrix(frame,index=False):
    if index: frame=frame.reset_index()
    frame=frame.astype(object).where(pd.notna(frame),None)
    return [list(frame.columns)]+[list(row) for row in frame.itertuples(index=False,name=None)]


def build_payload(rq2_outputs=None):
    if rq2_outputs is not None:
        rows = next(out['sheets']['per_case_attribution'] for out in rq2_outputs
                    if 'per_case_attribution' in out['sheets'])
        phases = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        phases = pd.read_excel(rq3_analysis.RQ2_WORKBOOK, sheet_name='per_case_attribution',
                              usecols=['rand_index', 'phase'])
    rq3=rq3_analysis.build_frames(phases)
    rq4=rq4_analysis.build_frames()
    outputs=[]
    for rq,frames in [('RQ3',rq3),('RQ4',rq4)]:
        outputs.append({'path':str(PACKAGE/rq/'results'/f'{rq}_analysis.xlsx'),
                        'sheets':{name:matrix(frame,index) for name,(frame,index) in frames.items()}})
    summary={'agent_failures':len(phases),
             'occurrence':rq3['category_occurrence'][0].to_dict('records'),
             'subcategories':rq3['subcategory_counts'][0].to_dict('records'),
             'behavior_signatures':rq3['behavior_signatures'][0].to_dict('records'),
             'deltas':rq4['delta_price_of_autonomy'][0].to_dict('records')}
    return {'outputs':outputs},summary


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-excel',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    payload,summary=build_payload()
    if not args.no_excel:
        sys.path.insert(0,str(PACKAGE/'scripts'))
        from workbook_io import export_outputs
        export_outputs(payload['outputs'],args.output_dir)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
