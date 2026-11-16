"""Compare baseline author-label sheets using pooled presence/absence agreement.

Reference: calculate_agreement.py in this directory.
Reads the current author sheets; does not modify labels or the workbook.
"""
from collections import Counter
from pathlib import Path
import re
import openpyxl

WORKBOOK = Path(__file__).resolve().parents[1] / "data" / "llm_failure_labels.xlsx"
CATEGORIES = (
    "Patch Application Failure",
    "Context Acquisition Failure",
    "Localization Failure",
    "Debt-Retention Failure",
    "Semantic Repair Failure",
    "API Hallucination",
)
ARTIFACT = "Data Alignment Artifact"
ALIASES = {name.casefold(): name for name in (*CATEGORIES, ARTIFACT)}
ALIASES.update({
    "missing context": "Context Acquisition Failure",
    "missing context / context acquisition failure": "Context Acquisition Failure",
    "debt retention": "Debt-Retention Failure",
})


def kappa(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Paired, nonempty decisions are required")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    counts_left, counts_right = Counter(left), Counter(right)
    expected = sum(
        counts_left[label] * counts_right[label] for label in counts_left.keys() | counts_right.keys()
    ) / len(left) ** 2
    return observed, (observed - expected) / (1 - expected)


def labels(value):
    result = set()
    for part in re.split(r"[;|]", str(value or "")):
        if not part.strip():
            continue
        key = " ".join(part.split()).casefold()
        if key not in ALIASES:
            raise ValueError(f"Unrecognized category: {part!r}")
        result.add(ALIASES[key])
    if not result:
        raise ValueError("Missing label set")
    return result


def rows(book, sheet, label_column):
    records = list(book[sheet].values)
    headers = records[0]
    id_col, label_col = headers.index("rand_index"), headers.index(label_column)
    out = {}
    for row in records[1:]:
        identifier = row[id_col]
        if identifier is None:
            if any(v is not None for v in row):
                raise ValueError("Nonempty row without an identifier")
            continue
        if identifier in out:
            raise ValueError(f"Duplicate identifier: {identifier}")
        out[identifier] = labels(row[label_col])
    return out


def report(name, identifiers, zaki, jassem, categories):
    left = [c in zaki[i] for i in identifiers for c in categories]
    right = [c in jassem[i] for i in identifiers for c in categories]
    observed, coefficient = kappa(left, right)
    agree = sum(a == b for a, b in zip(left, right))
    exact = sum(zaki[i] == jassem[i] for i in identifiers)
    print(name)
    print(f"Cases: {len(identifiers)}; category dimensions: {len(categories)}")
    print(f"Pooled agreement: {agree}/{len(left)} = {observed:.6%}")
    print(f"Category-decision disagreements: {len(left)-agree}")
    print(f"Cohen's kappa: {coefficient:.9f}")
    print(f"Identical complete label sets: {exact}/{len(identifiers)} = {exact/len(identifiers):.6%}")
    print(f"Cases with any label-set disagreement: {len(identifiers)-exact}")
    return observed, coefficient


def main():
    book = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    zaki = rows(book, "zaki_labeling", "llm_failure_categories")
    jassem = rows(book, "Jassem_labeling", "failure_categories")
    active_rows = list(book["failed_cases"].values)
    active_column = active_rows[0].index("rand_index")
    active_ids = [r[active_column] for r in active_rows[1:] if r[active_column] is not None]
    book.close()
    if len(active_ids) != len(set(active_ids)) or not active_ids:
        raise ValueError("Expected distinct active failure identifiers")
    if zaki.keys() != jassem.keys() or not set(active_ids).issubset(zaki):
        raise ValueError("Author snapshots must cover the same cases and all active failures")
    identifiers = sorted(active_ids)
    for i in identifiers:
        if ARTIFACT in zaki[i] or ARTIFACT in jassem[i]:
            raise ValueError(f"Active failure still carries an administrative label: {i}")
    print(f"Source: {WORKBOOK}")
    print("Compared columns: zaki_labeling.llm_failure_categories; Jassem_labeling.failure_categories")
    print(f"Author records: {len(zaki)}; active failures: {len(identifiers)}")
    report("Active baseline failures: six-category codebook",
           identifiers, zaki, jassem, CATEGORIES)


if __name__ == "__main__":
    main()
