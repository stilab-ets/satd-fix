# -*- coding: utf-8 -*-
"""Plot failure-category assignments by difficulty and heuristic phase.

Join RQ2 phase metadata to the final RQ3 label sets by case ID. Each
trajectory contributes one count per assigned category to both parts
of the Sankey.
"""
from pathlib import Path
import difflib
import sys
import os

import pandas as pd

PACKAGE_ROOT=Path(__file__).resolve().parents[2]
PAPER_DIR=Path(os.environ.get('SATD_PAPER_DIR',PACKAGE_ROOT/'paper'))
sys.path.insert(0,str(PACKAGE_ROOT/'RQ3/scripts'))
from final_labels import read_final_failed, final_set

IN_PATH = PACKAGE_ROOT/'RQ2/results/RQ2_phase_attribution.xlsx'
OUT_PATH = PAPER_DIR/'Figures/rq3_difficulty_phase_category_sankey.tex'

diff_nodes = ["easy", "moderate", "hard"]
diff_label = {"easy": "Easy", "moderate": "Moderate", "hard": "Hard"}
phase_nodes = ["Context Acquisition", "Comprehension", "Localization", "Editing", "Orchestration", "Unresolved"]
cat_nodes = [
    "Context Acquisition Failure", "Misleading demonstrative reference",
    "Localization Failure", "API hallucination", "Debt-Retention Failure",
    "Semantic Repair Failure", "Patch Application Failure",
    "Trajectory-Control Failure", "Turn Budget Exhaustion",
]
cat_label = {
    "Context Acquisition Failure": "Context Acquisition Failure",
    "Misleading demonstrative reference": "Misleading Reference",
    "Localization Failure": "Localization Failure",
    "API hallucination": "API Hallucination",
    "Debt-Retention Failure": "Debt-Retention",
    "Semantic Repair Failure": "Semantic Repair",
    "Patch Application Failure": "Patch Application",
    "Trajectory-Control Failure": "Trajectory-Control",
    "Turn Budget Exhaustion": "Turn Budget Exhaustion",
}

# phase colors reuse the paper's established convention (rq2_phase_difficulty.tex)
phase_color = {"Context Acquisition": "cAgent", "Comprehension": "cAgent", "Localization": "cBench", "Editing": "cTraj", "Orchestration": "cFail", "Unresolved": "cGrey"}
diff_color = {"easy": "green", "moderate": "cAgent", "hard": "red"}
X_D, X_P, X_C = 0.0, 6.6, 13.4
NW = 0.28            # node rect width
FLOW_HEIGHT = 7.45   # total height of the stacked nodes before gaps
GAP = {"diff": 0.55, "phase": 0.55, "cat": 0.22}
CANVAS_H = 11.0


def load_links():
    pc = pd.read_excel(IN_PATH, sheet_name="per_case_attribution",
                       usecols=['rand_index', 'difficulty_label', 'phase'])
    final = read_final_failed()
    if set(pc['rand_index']) != set(final['rand_index']):
        raise ValueError('Phase metadata and final labels must cover the same cases')
    pc = pc.merge(final[['rand_index', 'failure_categories']], on='rand_index', validate='one_to_one')
    pc['category'] = pc['failure_categories'].map(final_set)
    assignments = pc.explode('category', ignore_index=True)
    if assignments[['difficulty_label', 'phase', 'category']].isna().any().any():
        raise ValueError('Every failed case needs difficulty, phase, and at least one category')
    d2p = (assignments.groupby(["difficulty_label", "phase"]).size()
             .reset_index(name="n"))
    links_d2p = [(r.difficulty_label, r.phase, int(r.n)) for r in d2p.itertuples()]
    p2c = (assignments.groupby(["phase", "category"]).size()
             .reset_index(name="n"))
    links_p2c = [(r.phase, r.category, int(r.n)) for r in p2c.itertuples()]
    return links_d2p, links_p2c, len(pc), len(assignments)


def layout_column(order, totals, gap, scale):
    heights = {n: totals[n] * scale for n in order}
    col_h = sum(heights.values()) + gap * (len(order) - 1)
    top = (CANVAS_H - col_h) / 2
    y = {}
    cur = CANVAS_H - top
    for n in order:
        y1 = cur
        y0 = cur - heights[n]
        y[n] = (y0, y1)
        cur = y0 - gap
    return y


def totals_from(links, key):
    t = {}
    for a, b, v in links:
        k = a if key == 0 else b
        t[k] = t.get(k, 0) + v
    return t


def stack_links(links, node_order_src, node_order_tgt, y_src, y_tgt, scale):
    """Sub-segments stacked top-to-bottom within each node, ordered by
    the *other* column's node order (reduces ribbon crossings)."""
    out_by_src = {n: [] for n in node_order_src}
    in_by_tgt = {n: [] for n in node_order_tgt}
    for a, b, v in links:
        out_by_src[a].append((b, v))
        in_by_tgt[b].append((a, v))
    for n in out_by_src:
        out_by_src[n].sort(key=lambda t: node_order_tgt.index(t[0]))
    for n in in_by_tgt:
        in_by_tgt[n].sort(key=lambda t: node_order_src.index(t[0]))

    src_cursor = {n: y_src[n][1] for n in node_order_src}
    tgt_cursor = {n: y_tgt[n][1] for n in node_order_tgt}
    src_seg, tgt_seg = {}, {}
    for n in node_order_src:
        for b, v in out_by_src[n]:
            h = v * scale
            top = src_cursor[n]
            src_seg[(n, b)] = (top - h, top)
            src_cursor[n] = top - h
    for n in node_order_tgt:
        for a, v in in_by_tgt[n]:
            h = v * scale
            top = tgt_cursor[n]
            tgt_seg[(a, n)] = (top - h, top)
            tgt_cursor[n] = top - h

    return [(a, b, v, src_seg[(a, b)], tgt_seg[(a, b)]) for a, b, v in links]


def ribbon(x1, x2, seg, color):
    (sy0, sy1), (ty0, ty1) = seg[3], seg[4]
    xm = (x1 + x2) / 2
    return (
        f"\\draw[fill={color}!32,draw=none,fill opacity=0.62] "
        f"({x1+NW:.3f},{sy1:.3f}) .. controls ({xm:.3f},{sy1:.3f}) and ({xm:.3f},{ty1:.3f}) .. ({x2:.3f},{ty1:.3f}) -- "
        f"({x2:.3f},{ty0:.3f}) .. controls ({xm:.3f},{ty0:.3f}) and ({xm:.3f},{sy0:.3f}) .. ({x1+NW:.3f},{sy0:.3f}) -- cycle;"
    )


def main():
    links_d2p, links_p2c, n_cases, n_assignments = load_links()

    diff_totals = totals_from(links_d2p, 0)
    phase_totals = totals_from(links_d2p, 1)
    cat_totals = totals_from(links_p2c, 1)
    if (sum(diff_totals.values()) != n_assignments or sum(cat_totals.values()) != n_assignments
            or phase_totals != totals_from(links_p2c, 0)):
        raise ValueError('The Sankey must conserve all label assignments at each phase')
    diff_order = [n for n in diff_nodes if n in diff_totals]
    phase_order = [n for n in phase_nodes if n in phase_totals]
    cat_order = [n for n in cat_nodes if n in cat_totals]
    scale = FLOW_HEIGHT / n_assignments

    y_diff = layout_column(diff_order, diff_totals, GAP["diff"], scale)
    y_phase = layout_column(phase_order, phase_totals, GAP["phase"], scale)
    y_cat = layout_column(cat_order, cat_totals, GAP["cat"], scale)

    seg_d2p = stack_links(links_d2p, diff_order, phase_order, y_diff, y_phase, scale)
    seg_p2c = stack_links(links_p2c, phase_order, cat_order, y_phase, y_cat, scale)
    strongest_phase = {category: phase for phase, category, _ in sorted(links_p2c, key=lambda link: link[2])}

    lines = [
        f"% RQ3: {n_assignments} failure-category assignments from {n_cases} failed trajectories.",
        "% Phase/difficulty metadata: RQ2/RQ2_phase_attribution.xlsx.",
        "% Final categories: agent_failed_cases_final_labels.xlsx, failure_categories.",
        "% Generated by RQ3/scripts/rq3_sankey_figure.py -- do not hand-edit.",
        "\\begin{tikzpicture}[font=\\scriptsize]",
    ]

    for seg in seg_d2p:
        lines.append(ribbon(X_D, X_P, seg, diff_color[seg[0]]))
    for seg in seg_p2c:
        lines.append(ribbon(X_P, X_C, seg, phase_color[seg[0]]))

    for n in diff_order:
        y0, y1 = y_diff[n]
        lines.append(f"\\draw[fill={diff_color[n]}!70,draw={diff_color[n]}!85!black,line width=0.5pt] "
                     f"({X_D:.3f},{y0:.3f}) rectangle ({X_D+NW:.3f},{y1:.3f});")
        lines.append(f"\\node[anchor=east,align=right] at ({X_D-0.12:.3f},{(y0+y1)/2:.3f}) "
                     f"{{{diff_label[n]} ({diff_totals[n]})}};")

    for n in phase_order:
        y0, y1 = y_phase[n]
        lines.append(f"\\draw[fill={phase_color[n]}!70,draw={phase_color[n]}!85!black,line width=0.5pt] "
                     f"({X_P:.3f},{y0:.3f}) rectangle ({X_P+NW:.3f},{y1:.3f});")
        lines.append(f"\\node[anchor=south,align=center] at ({X_P+NW/2:.3f},{y1+0.14:.3f}) "
                     f"{{{n} ({phase_totals[n]})}};")

    for n in cat_order:
        y0, y1 = y_cat[n]
        color = phase_color[strongest_phase[n]]
        lines.append(f"\\draw[fill={color}!70,draw={color}!85!black,line width=0.5pt] "
                     f"({X_C:.3f},{y0:.3f}) rectangle ({X_C+NW:.3f},{y1:.3f});")
        lines.append(f"\\node[anchor=west,align=left] at ({X_C+NW+0.12:.3f},{(y0+y1)/2:.3f}) "
                     f"{{{cat_label[n]} ({cat_totals[n]})}};")

    lines.append(f"\\node[anchor=south,font=\\scriptsize\\bfseries] at ({X_D+NW/2:.3f},{CANVAS_H+0.35:.3f}) {{Difficulty}};")
    lines.append(f"\\node[anchor=south,font=\\scriptsize\\bfseries] at ({X_P+NW/2:.3f},{CANVAS_H+0.35:.3f}) {{Phase (\\RQ{{2}})}};")
    lines.append(f"\\node[anchor=south,font=\\scriptsize\\bfseries] at ({X_C+NW/2:.3f},{CANVAS_H+0.35:.3f}) {{Failure category (\\RQ{{3}})}};")
    lines.append(f"\\node[anchor=north] at ({X_P+NW/2:.3f},0) "
                 f"{{{n_assignments} label assignments from {n_cases} failed trajectories}};")
    lines.append("\\end{tikzpicture}")

    content = '\n'.join(lines) + '\n'
    if '--emit-patch' in sys.argv:
        diff = list(difflib.unified_diff(OUT_PATH.read_text(encoding='utf-8').splitlines(), content.splitlines(), n=3))
        if diff:
            print('*** Begin Patch\n*** Update File: ' + OUT_PATH.as_posix())
            for line in diff[2:]:
                print('@@' if line.startswith('@@') else line)
            print('*** End Patch')
        return
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    OUT_PATH.write_text(content, encoding="utf-8")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
