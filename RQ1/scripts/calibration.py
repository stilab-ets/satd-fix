"""Recompute the SATD success cutoff from the manual calibration labels only."""

from pathlib import Path

import pandas as pd


RQ1_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_INPUT = RQ1_DIR / "data" / "for_calibration_labels.xlsx"
MANUAL_LABEL_SHEET = "Manual_label"
LABEL_COLUMN = "Label"
BLEU_COLUMN = "bleu_diff"
SYSTEM_COLUMN = "system"

MIN_PRECISION = 0.90
MIN_ITEMS_ABOVE_THRESHOLD = 3
THRESHOLD_STEP = 0.05


def load_manual_labels():
    """Load only the designated sheet and human correctness labels."""
    df = pd.read_excel(CALIBRATION_INPUT, sheet_name=MANUAL_LABEL_SHEET)
    required = {LABEL_COLUMN, BLEU_COLUMN, SYSTEM_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{MANUAL_LABEL_SHEET} is missing columns: {sorted(missing)}")

    df = df[[SYSTEM_COLUMN, BLEU_COLUMN, LABEL_COLUMN]].copy()
    df[BLEU_COLUMN] = pd.to_numeric(df[BLEU_COLUMN], errors="coerce")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
    invalid = df[df[BLEU_COLUMN].isna() | ~df[LABEL_COLUMN].isin({"correct", "wrong"})]
    if not invalid.empty:
        raise ValueError(
            f"{MANUAL_LABEL_SHEET} has {len(invalid)} rows without a valid BLEU score "
            f"or {LABEL_COLUMN} value. Resolve them before calibrating."
        )
    return df


def threshold_results(df):
    """Return pooled and per-system precision across the existing 0.05 grid."""
    groups = [("combined", df)]
    groups.extend((str(system), part) for system, part in df.groupby(SYSTEM_COLUMN))
    results = []

    thresholds = [round(i * THRESHOLD_STEP, 10) for i in range(round(1 / THRESHOLD_STEP) + 1)]
    for system, part in groups:
        total_correct = int(part[LABEL_COLUMN].eq("correct").sum())
        for threshold in thresholds:
            accepted = part[part[BLEU_COLUMN] >= threshold]
            correct = int(accepted[LABEL_COLUMN].eq("correct").sum())
            wrong = int(accepted[LABEL_COLUMN].eq("wrong").sum())
            covered = len(accepted)
            results.append({
                "system": system,
                "threshold": threshold,
                "accepted": covered,
                "correct": correct,
                "wrong": wrong,
                "precision": correct / covered if covered else None,
                "recall": correct / total_correct if total_correct else None,
            })
    return results


def choose_thresholds(results):
    chosen = []
    for system in dict.fromkeys(row["system"] for row in results):
        qualifying = [
            row for row in results
            if row["system"] == system
            and row["accepted"] >= MIN_ITEMS_ABOVE_THRESHOLD
            and row["precision"] is not None
            and row["precision"] >= MIN_PRECISION
        ]
        chosen.append({"system": system, "result": min(qualifying, key=lambda row: row["threshold"])} if qualifying else {"system": system, "result": None})
    return chosen


def main():
    manual = load_manual_labels()
    results = threshold_results(manual)
    print(f"Source: {CALIBRATION_INPUT} [{MANUAL_LABEL_SHEET}]")
    print(f"Manual labels: {len(manual)}; target precision: {MIN_PRECISION:.0%}; grid: {THRESHOLD_STEP:.2f}")
    print("system\tthreshold\taccepted\tcorrect\twrong\tprecision\trecall")
    for selection in choose_thresholds(results):
        row = selection["result"]
        if row is None:
            print(f"{selection['system']}\tno qualifying threshold")
            continue
        print(
            f"{row['system']}\t{row['threshold']:.2f}\t{row['accepted']}\t"
            f"{row['correct']}\t{row['wrong']}\t{row['precision']:.1%}\t{row['recall']:.1%}"
        )


if __name__ == "__main__":
    main()
