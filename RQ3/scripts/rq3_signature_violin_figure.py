# -*- coding: utf-8 -*-
"""RQ3 violin figure: per-category distribution of Search and Edit tool
calls (multi-label failed cases), replacing/complementing the median
table (tab:rq3_behavior_signatures) with the full distribution.

Reads the failed_cases sheet (failure_categories, tool_counts) and
groups tool calls the same way rq3_analysis.py does (TOOL_GROUPS).
Categories with n<5 get a min-max range + individual dots instead of a
KDE violin (too few points for a meaningful density estimate).
Writes a native TikZ figure; reuses cTraj (Search) / cFail (Edit).
"""
import re
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT=Path(__file__).resolve().parents[2]
PAPER_DIR=Path(os.environ.get('SATD_PAPER_DIR',PACKAGE_ROOT/'paper'))
sys.path.insert(0,str(PACKAGE_ROOT/'RQ3/scripts'))
from final_labels import read_final_failed
from trajectory_evidence import recorded_evidence

OUT_PATH = PAPER_DIR/'Figures/rq3_signature_violin.tex'

TOOL_GROUPS = {
    "search": ["search_text"],
    "edit": ["write_file", "replace_in_file"],
}
CATS = ["Debt-Retention Failure", "Patch Application Failure", "Semantic Repair Failure",
        "Trajectory-Control Failure", "Context Acquisition Failure", "Turn Budget Exhaustion",
        "Localization Failure", "API hallucination", "Misleading demonstrative reference"]
SHORT = {
    "Debt-Retention Failure": "Debt-Ret.", "Patch Application Failure": "Patch-App.",
    "Semantic Repair Failure": "Sem.Repair", "Trajectory-Control Failure": "Traj-Ctrl",
    "Context Acquisition Failure": "Ctx.Acq.", "Turn Budget Exhaustion": "Turn-Budg.",
    "Localization Failure": "Local.", "API hallucination": "API-Hall.",
    "Misleading demonstrative reference": "Misl.Ref.",
}

X_GAP = 1.55
HALF_W = 0.62
Y_SCALE = 0.11
Y_MAX_VAL = 25
PANEL_GAP = 1.4


def tool_profile(tc):
    m = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", str(tc or ""))}
    return {g: sum(m.get(t, 0) for t in tools) for g, tools in TOOL_GROUPS.items()}


def kde(values, grid, bw):
    diffs = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * diffs ** 2).sum(axis=1) / (values.size * bw * np.sqrt(2 * np.pi))


def silverman_bw(values):
    std = values.std(ddof=1) if values.size > 1 else 1.0
    return max(0.9 * std * values.size ** (-1 / 5), 0.6)


def violin_or_strip(vals, x0, y0, color):
    vals = np.asarray(vals, dtype=float)
    lines = []
    if vals.size >= 5:
        grid = np.linspace(0, Y_MAX_VAL, 160)
        dens = kde(vals, grid, silverman_bw(vals))
        dens = dens / dens.max() * HALF_W
        right = [(x0 + w, y0 + y * Y_SCALE) for w, y in zip(dens, grid)]
        left = [(x0 - w, y0 + y * Y_SCALE) for w, y in zip(dens, grid)][::-1]
        path = " -- ".join(f"({x:.3f},{y:.3f})" for x, y in right + left) + " -- cycle"
        lines.append(f"\\draw[fill={color}!32,draw={color}!75!black,line width=0.5pt] {path};")
    else:
        lo, hi = vals.min(), vals.max()
        lines.append(f"\\draw[{color}!75!black,line width=0.7pt] ({x0:.3f},{y0+lo*Y_SCALE:.3f}) -- ({x0:.3f},{y0+hi*Y_SCALE:.3f});")
        for v in vals:
            lines.append(f"\\fill[{color}!75!black] ({x0:.3f},{y0+v*Y_SCALE:.3f}) circle (0.9pt);")
    med = float(np.median(vals))
    lines.append(f"\\draw[{color}!85!black,line width=0.9pt] ({x0-HALF_W*0.5:.3f},{y0+med*Y_SCALE:.3f}) -- ({x0+HALF_W*0.5:.3f},{y0+med*Y_SCALE:.3f});")
    lines.append(f"\\node[anchor=west,font=\\tiny,text={color}!85!black] at ({x0+HALF_W*0.55:.3f},{y0+med*Y_SCALE:.3f}) {{{med:.0f}}};")
    return lines


def panel(prof, labels, metric, y0, color, panel_label):
    out = [f"\\node[anchor=south west,font=\\scriptsize\\bfseries] at ({-1.3:.3f},{y0+Y_MAX_VAL*Y_SCALE+0.18:.3f}) {{{panel_label}}};"]
    for yv in (0, 10, 20):
        y = y0 + yv * Y_SCALE
        out.append(f"\\draw[cGrey!18,line width=0.4pt] (-1.0,{y:.3f}) -- ({(len(CATS)-1)*X_GAP+1.0:.3f},{y:.3f});")
        out.append(f"\\node[anchor=east,font=\\tiny,text=cGrey] at (-1.1,{y:.3f}) {{{yv}}};")
    for i, c in enumerate(CATS):
        x0 = i * X_GAP
        sel = prof[labels.map(lambda ls: c in ls)][metric].values
        out += violin_or_strip(sel, x0, y0, color)
    return out


def main():
    fc = read_final_failed()
    profiles=[]
    for path in fc['trajectory_path']:
        counts=recorded_evidence(path)['counts']
        profiles.append({g:sum(counts.get(tool,0) for tool in tools) for g,tools in TOOL_GROUPS.items()})
    prof = pd.DataFrame(profiles,index=fc.index)
    labels = fc["failure_categories"].map(lambda s: [c.strip() for c in str(s or "").split("|") if c.strip()])

    lines = [
        "% RQ3: violin/strip plot of per-case Search and Edit tool-call counts",
        "% by failure category (multi-label), replacing tab:rq3_behavior_signatures'",
        "% median-only columns for these two tools with the full distribution.",
        "% Categories with n<5 (Misleading demonstrative reference)",
        "% show individual points + range instead of a KDE (too few for a density).",
        "% Data: failed_cases sheet, tool_counts. Generated by",
        "% RQ3/scripts/rq3_signature_violin_figure.py.",
        "\\begin{tikzpicture}[font=\\scriptsize]",
    ]
    top_y0 = Y_MAX_VAL * Y_SCALE + PANEL_GAP
    lines += panel(prof, labels, "search", top_y0, "cTraj", "Search calls")
    lines += panel(prof, labels, "edit", 0, "cFail", "Edit calls")
    for i, c in enumerate(CATS):
        x0 = i * X_GAP
        n = int(labels.map(lambda ls: c in ls).sum())
        lines.append(f"\\node[anchor=north,rotate=35,font=\\tiny,align=right] at ({x0:.3f},{-0.15:.3f}) {{{SHORT[c]} ({n})}};")
    lines.append("\\end{tikzpicture}")

    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
