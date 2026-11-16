# -*- coding: utf-8 -*-
"""Deterministic difficulty scoring for the 500 evaluated SATD instances.

Code identifiers are extracted with Pygments; natural-language SATD
mentions and comment clarity use fixed lexical rules. Pass
identifier_method='regex' to score_row for the previous extractor.
Numeric cutoffs are the linearly interpolated 1/3 and 2/3 quantiles of
the 500 evaluated instances, fitted independently of repayment outcomes.
No LLM clarity scores are used by the default configuration.
"""
import difflib
import keyword
import re
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PATH = PACKAGE_ROOT / "data" / "satd_dataset.xlsx"
OUTPUT_PATH = PATH
# Output of clarity_gemini.py: checkpointed continuously during the run
# (so usable even from a partial run), and the source of truth for the
# LLM clarity scores -- see clarity_gemini.py's OUTPUT_DIR/CACHE_PATH.
LLM_CLARITY_CACHE = PACKAGE_ROOT / "data" / "clarity_llm_cache.csv"

EVALUATED_RESULTS = PACKAGE_ROOT / "data" / "agent_metrics.xlsx"

USE_LLM_CLARITY = False

ACTION_VERBS = re.compile(
    r"\b(use|replace|remove|delete|add|move|rename|fix|implement|support|"
    r"handle|switch|update|refactor|check|avoid|make|convert|extract|"
    r"merge|split|cache|optimi[sz]e|simplify|clean|enable|disable|"
    r"return|raise|throw|catch|validate|test|document|rewrite|uncomment)\b",
    re.I,
)
VAGUE_MARKERS = re.compile(
    r"(\bugly\b|\bhack\b|\bwhy\b|\?|better way|revisit|temporary|for now|"
    r"not sure|really\b|\bweird\b|\bwtf\b|figure out|work out|some ?day|"
    r"\bmess\b|is this|should this)",
    re.I,
)
CODE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9]*(_[A-Za-z0-9_]+|\(\)|\.[A-Za-z_])|[A-Z][a-z]+[A-Z]")

# Non-informative-token stoplist for identifiers() / D4 (context
# dependence). Built from:
#
#   1. Python reserved words: keyword.kwlist (stdlib, always in sync
#      with the running interpreter) + keyword.softkwlist when present
#      (Python >=3.9, e.g. "match", "case", "type").
#   2. Java reserved words: the fixed, official list of 53 keywords
#      plus the boolean/null literals (Java Language Specification
#      SS3.9); Java has no stdlib equivalent in Python, so this is a
#      static table, unlikely to change.
#   3. English stop words: the standard NLTK English stop-word list
#      (Bird, Klein & Loper, "Natural Language Processing with
#      Python", O'Reilly, 2009), hardcoded here to avoid adding an
#      nltk runtime dependency for one fixed list.
#
# The comment/method text mixes Python, Java, and English (docstrings,
# SATD comments), so all three are unioned into one stoplist -- the
# extraction in identifiers() does not know the source language of the
# snippet it is filtering.
PYTHON_KEYWORDS = {kw.lower() for kw in keyword.kwlist}
PYTHON_KEYWORDS |= {kw.lower() for kw in getattr(keyword, "softkwlist", [])}
PYTHON_KEYWORDS |= {"self", "cls"}  # not reserved words, but ubiquitous by convention

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
    "true", "false", "null",
    # contextual keywords (Java 10+)
    "var", "yield", "record", "sealed", "permits", "non-sealed",
}

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "may", "me", "might", "more", "most",
    "must", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}

# Common non-Java, non-Python short identifier conventions that carry
# no project-specific meaning (kept from the original hand-picked
# list; not part of any language's reserved-word table).
GENERIC_SHORT_NAMES = {"len", "str", "val"}

PY_JAVA_KEYWORDS = (PYTHON_KEYWORDS | JAVA_KEYWORDS | ENGLISH_STOPWORDS
                    | GENERIC_SHORT_NAMES)


def code_lines(text):
    return [ln for ln in str(text or "").splitlines() if ln.strip()]


def diff_stats(before, after):
    b, a = code_lines(before), code_lines(after)
    sm = difflib.SequenceMatcher(None, b, a)
    deleted = inserted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            deleted += i2 - i1
        if tag in ("replace", "insert"):
            inserted += j2 - j1
    return deleted, inserted, len(b)


def identifiers(text):
    toks = set(t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(text or "")))
    return toks - PY_JAVA_KEYWORDS


def identifiers_pygments(text, source_language):
    """Extract code names with a language lexer, excluding comments/strings.

    Keep the original minimum length, case normalization, and stoplist.
    Pygments is imported when this extractor is selected.
    """
    from pygments import lex
    from pygments.lexers import JavaLexer, PythonLexer
    from pygments.token import Name

    language = str(source_language).strip().lower()
    lexers = {"java": JavaLexer, "python": PythonLexer}
    if language not in lexers:
        raise ValueError(f"Unsupported source language: {source_language!r}")
    names = {
        part.lower()
        for token_type, value in lex(str(text or ""), lexers[language]())
        if token_type in Name
        for part in value.lstrip("@").split(".")
        if len(part) >= 3 and part.isidentifier()
    }
    return names - PY_JAVA_KEYWORDS


def context_identifiers(before, after, comment, source_language, method="pygments"):
    """Return repair names absent from the original method and SATD text.

    The SATD is natural language, so both alternatives use text matching
    for identifier mentions in that comment.
    """
    if method == "regex":
        before_ids, after_ids = identifiers(before), identifiers(after)
    elif method == "pygments":
        before_ids = identifiers_pygments(before, source_language)
        after_ids = identifiers_pygments(after, source_language)
    else:
        raise ValueError(f"Unknown identifier extraction method: {method!r}")
    return after_ids - before_ids - identifiers(comment)


def heuristic_clarity(comment):
    words = len(re.findall(r"\w+", comment))
    actionable = bool(ACTION_VERBS.search(comment))
    codeish = bool(CODE_TOKEN.search(comment))
    vague = bool(VAGUE_MARKERS.search(comment))
    if actionable and (codeish or words >= 8) and not vague:
        clar_s = 0
    elif vague and not actionable:
        clar_s = 2
    else:
        clar_s = 1
    return clar_s, words, actionable, codeish, vague


def measure_row(row, llm_clarity, identifier_method="pygments"):
    """Extract measurements before fitting dataset-based numeric cutoffs.

    Pass an empty llm_clarity mapping to use deterministic clarity.
    """
    # SATD_comment_full includes continuation comment lines that
    # SATD_comment truncates (~21% of instances); fall back to
    # SATD_comment if extract_full_satd_comment.py has not been run.
    comment = str(row.get("SATD_comment_full") or row.get("SATD_comment") or "")
    before = str(row.get("containing_method_before_repayment") or "")
    after = str(row.get("containing_method_after_repayment") or "")

    deleted, inserted, method_len = diff_stats(before, after)
    changed = deleted + inserted

    # Keep full precision for quantile estimation and score assignment.
    frac = changed / max(method_len, 1)

    # D3 comment clarity: LLM judgment (Gemini 2.5 Flash) when available,
    # else the deterministic heuristic; the heuristic is always computed
    # and kept for comparison.
    heur_s, words, actionable, codeish, vague = heuristic_clarity(comment)
    rand_index = row.get("rand_index")
    llm_s = llm_clarity.get(rand_index)
    clar_s = llm_s if llm_s is not None else heur_s

    # D4 context dependence: new identifiers in the fix that appear neither
    # in the pre-fix method nor in the comment (proxy for knowledge the
    # agent must obtain from the wider project or external APIs)
    new_ids = context_identifiers(
        before, after, comment, row.get("source_language"), identifier_method
    )
    return {
        "changed_lines": changed,
        "deleted_lines": deleted,
        "inserted_lines": inserted,
        "method_lines_before": method_len,
        "changed_fraction": frac,
        "comment_words": words,
        "comment_actionable_verb": actionable,
        "comment_code_reference": codeish,
        "comment_vague_marker": vague,
        "new_identifiers_in_fix": len(new_ids),
        "score_comment_clarity": clar_s,
        "score_comment_clarity_heuristic": heur_s,
        "score_comment_clarity_llm": llm_s,
    }


NUMERIC_DIMENSIONS = {
    "changed_lines": "score_change_size",
    "changed_fraction": "score_locality",
    "new_identifiers_in_fix": "score_context_dependence",
}


def fit_tertile_cutoffs(measurements):
    """Fit both boundaries on all 500 instances without using outcomes.

    pandas linear interpolation uses rank (n - 1) * q. Values tied at a
    boundary stay together in the lower score, so counts need not be equal.
    """
    frame = pd.DataFrame(measurements)
    if len(frame) != 500:
        raise ValueError("Fit tertile cutoffs on the 500 evaluated instances")
    return {
        name: tuple(float(value) for value in frame[name].quantile(
            [1 / 3, 2 / 3], interpolation="linear"))
        for name in NUMERIC_DIMENSIONS
    }


def apply_tertile_scores(measurement, cutoffs):
    result = dict(measurement)
    for name, score_name in NUMERIC_DIMENSIONS.items():
        lower, upper = cutoffs[name]
        result[score_name] = (0 if result[name] <= lower else
                              1 if result[name] <= upper else 2)
    result["difficulty_score"] = sum(result[name] for name in (
        "score_change_size", "score_locality", "score_comment_clarity",
        "score_context_dependence"))
    total = result["difficulty_score"]
    result["difficulty_label"] = "easy" if total <= 2 else "moderate" if total <= 5 else "hard"
    return result


def score_row(row, llm_clarity, identifier_method="pygments", *, cutoffs):
    """Score one instance using boundaries fitted on the complete dataset."""
    return apply_tertile_scores(measure_row(row, llm_clarity, identifier_method), cutoffs)


def score_dataset(rows, llm_clarity, identifier_method="pygments"):
    measurements = [measure_row(row, llm_clarity, identifier_method) for row in rows]
    cutoffs = fit_tertile_cutoffs(measurements)
    return [apply_tertile_scores(row, cutoffs) for row in measurements], cutoffs


def rubric_rows(cutoffs):
    def boundaries(name):
        low, high = cutoffs[name]
        return [f"<={low:.17g}", f"({low:.17g},{high:.17g}]", f">{high:.17g}"]
    return [
        ["Dimension", "Proxy", "Score 0 (low)", "Score 1", "Score 2 (high)"],
        ["Change size", "Inserted plus deleted nonblank method lines (difflib.SequenceMatcher)", *boundaries("changed_lines")],
        ["Locality", "Changed lines / max(nonblank original method lines, 1); full precision", *boundaries("changed_fraction")],
        ["SATD clarity", "Deterministic lexical rules; full stored SATD comment preferred", "Action verb AND (code-like token OR >=8 words) AND no vague marker", "Otherwise", "Vague marker AND no action verb"],
        ["Context dependence", "Distinct filtered Pygments identifiers in developer repair absent from original method and SATD text", *boundaries("new_identifiers_in_fix")],
        ["Overall", "Sum of four scores (0-8)", "Easy: 0-2", "Moderate: 3-5", "Hard: 6-8"],
        ["Numeric boundaries", "1/3 and 2/3 quantiles of all 500 evaluated instances; pandas linear interpolation, rank (n-1)*q; independent of outcomes; ties stay in the lower score; no rounding before scoring", "", "", ""],
        ["Grouping precedent", "Baysal et al. (EMSE 2016), https://doi.org/10.1007/s10664-015-9366-8; their quartile grouping is adapted here to three levels", "", "", ""],
    ]


def load_llm_clarity():
    """rand_index -> LLM clarity score (0/1/2), produced by
    clarity_gemini.py. Prefers the CSV checkpoint (LLM_CLARITY_CACHE,
    updated continuously during the run, so usable even from a partial
    run); falls back to the 'clarity_llm_gemini' sheet in the dataset
    workbook (written only once the full run completes) for backward
    compatibility. Returns {} if neither source exists yet."""
    if not USE_LLM_CLARITY:
        return {}
    if LLM_CLARITY_CACHE.exists():
        llm = pd.read_csv(LLM_CLARITY_CACHE)
        n = llm["score_comment_clarity_llm"].notna().sum()
        print(f"Loaded {n} LLM clarity scores from {LLM_CLARITY_CACHE}.")
        return dict(zip(llm["rand_index"], llm["score_comment_clarity_llm"]))
    try:
        llm = pd.read_excel(PATH, sheet_name="clarity_llm_gemini")
        return dict(zip(llm["rand_index"], llm["score_comment_clarity_llm"]))
    except (FileNotFoundError, ValueError):
        print("No LLM clarity scores found (checked "
              f"{LLM_CLARITY_CACHE} and the 'clarity_llm_gemini' sheet) "
              "-- run clarity_gemini.py first. Falling back to the "
              "deterministic heuristic for score_comment_clarity.")
        return {}


def main():
    df = pd.read_excel(PATH, sheet_name="samples")
    evaluated = pd.read_excel(EVALUATED_RESULTS, sheet_name="repayment_results")
    ids = set(evaluated["rand_index"])
    df = df[df["rand_index"].isin(ids)].copy()
    if len(ids) != 500 or len(df) != 500 or df["rand_index"].nunique() != 500:
        raise ValueError("Difficulty scoring requires the 500 unique evaluated instances")
    llm_clarity = load_llm_clarity()
    keep = ["rand_index", "project", "source_language", "SATD_comment", "SATD_comment_full"]
    out = df[keep].copy()
    records, cutoffs = score_dataset(df.to_dict("records"), llm_clarity)
    scores = pd.DataFrame(records, index=df.index)
    out = pd.concat([out, scores.drop(columns=[
        "score_comment_clarity_heuristic", "score_comment_clarity_llm"
    ])], axis=1)

    rubric = pd.DataFrame(rubric_rows(cutoffs))

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        out.to_excel(w, sheet_name="defficulty_SATD", index=False)
        rubric.to_excel(w, sheet_name="defficulty_rubric", index=False, header=False)

    if llm_clarity:
        agree = (scores["score_comment_clarity_llm"] == scores["score_comment_clarity_heuristic"]).mean()
        print(f"LLM vs. heuristic clarity exact agreement: {agree*100:.1f}%")
    print(out["difficulty_label"].value_counts())
    print(out[["score_change_size", "score_locality", "score_comment_clarity",
               "score_context_dependence", "difficulty_score"]].describe().round(2))


if __name__ == "__main__":
    main()
