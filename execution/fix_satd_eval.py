import ast
import re
import textwrap
import time
from collections import Counter

import pandas as pd
from nltk.util import ngrams

try:
    from codebleu import calc_codebleu
except Exception:
    calc_codebleu = None

from SATD_tool import (
    extract_code,
    get_BLEU,
    get_crystalBLEU,
    get_deleted_and_inserted_lines,
    get_exact_matches,
    get_itmes_having_specific_number_of_inserted_lines,
    get_linePRF,
    get_updated_or_new_lines,
    load_from_json,
    remove_comments_and_javadoc_from_java,
    remove_docstrings_and_comments_by_ast_from_python,
    remove_docstrings_and_comments_by_regex_from_python,
    remove_imports,
)

PROMPT_TEMPLATES = ["NoExplain", "CoT1"]
PROMPT_VERSION = "v2"


def _indentation_width(line):
    expanded = str(line or "").expandtabs(4)
    return len(expanded) - len(expanded.lstrip())


def get_language(row):
    path = str(
        row.get("url_file_path")
        or row.get("last_appeared_in_file")
        or row.get("created_in_file")
        or ""
    ).lower()
    if path.endswith(".py"):
        return "Python"
    if path.endswith(".java"):
        return "Java"
    return "Unsupported"


def load_dataset(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)

    if "rand_index" not in df.columns:
        df["rand_index"] = list(range(len(df)))

    if "SATD_comment" not in df.columns:
        if "last_tracked_comment" in df.columns:
            df["SATD_comment"] = df["last_tracked_comment"]
        elif "content" in df.columns:
            df["SATD_comment"] = df["content"]
        else:
            raise ValueError("Dataset is missing SATD_comment, last_tracked_comment, and content columns.")

    df["language"] = df.apply(get_language, axis=1)
    unsupported = df[df["language"] == "Unsupported"]
    if len(unsupported) > 0:
        raise ValueError("Unsupported files found. Only .py and .java are supported by SATD_tool.py.")

    df["rand_index"] = df["rand_index"].astype(int)
    df[["number_of_deleted_lines", "number_of_inserted_lines"]] = df.apply(
        lambda row: pd.Series(
            get_deleted_and_inserted_lines(
                str(row["containing_method_before_repayment"]),
                str(row["containing_method_after_repayment"]),
            )
        ).map(len),
        axis=1,
    )
    return df


def answer_path(output_dir, model_name, prompt_template):
    return output_dir / "generated_outputs" / model_name / f"answers_{prompt_template}_{PROMPT_VERSION}.json"


def _extract_python_method_block(code, method_name):
    source = str(code or "")
    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
        dedented_lines = dedented.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == str(method_name):
                end_lineno = getattr(node, "end_lineno", None)
                if end_lineno:
                    return "\n".join(dedented_lines[node.lineno - 1:end_lineno]).strip()
    except SyntaxError:
        pass

    lines = source.splitlines()
    start = None
    indent = None
    for i, line in enumerate(lines):
        if re.match(rf"(?:async\s+)?def\s+{re.escape(str(method_name))}\s*\(", line.lstrip()):
            start = i
            indent = _indentation_width(line)
            break
    if start is None:
        return ""
    end = len(lines)
    signature_done = lines[start].rstrip().endswith(":")
    for j in range(start + 1, len(lines)):
        line = lines[j]
        stripped = line.strip()
        if stripped == "":
            continue
        if not signature_done:
            if stripped.endswith(":"):
                signature_done = True
            continue
        curr_indent = _indentation_width(line)
        if curr_indent <= indent:
            end = j
            break
    has_body = any(
        line.strip() and _indentation_width(line) > indent
        for line in lines[start + 1:end]
    )
    if not has_body:
        return ""
    return "\n".join(lines[start:end]).strip()


def _extract_java_method_block(code, method_name):
    lines = code.splitlines()
    start = None
    start_brace_seen = False
    brace_count = 0
    method_pat = f"{method_name}("
    for i, line in enumerate(lines):
        if method_pat in line and any(k in line for k in ("public", "private", "protected", "static", "final", "synchronized")):
            start = i
            break
    if start is None:
        for i, line in enumerate(lines):
            if method_pat in line:
                start = i
                break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        line = lines[j]
        opens = line.count("{")
        closes = line.count("}")
        if opens > 0:
            start_brace_seen = True
        brace_count += opens - closes
        if start_brace_seen and brace_count <= 0:
            end = j + 1
            break
    return "\n".join(lines[start:end]).strip()

def average(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def codebleu_language(language):
    mapping = {
        "Python": "python",
        "Java": "java",
    }
    return mapping.get(str(language), "")


def get_codebleu_scores(references, candidates, language):
    """Return per-item CodeBLEU scores, falling back to 0 when unavailable."""
    scores = {}
    lang = codebleu_language(language)
    for rand_index, reference in references.items():
        candidate = candidates.get(rand_index, "")
        if not calc_codebleu or not lang:
            scores[rand_index] = 0.0
            continue
        try:
            result = calc_codebleu([str(reference)], [str(candidate)], lang=lang)
            scores[rand_index] = float(result.get("codebleu", 0.0))
        except Exception:
            scores[rand_index] = 0.0
    return scores


def remove_icd(code, language):
    if language == "Python":
        try:
            return remove_imports(remove_docstrings_and_comments_by_ast_from_python(code))
        except Exception:
            return remove_imports(remove_docstrings_and_comments_by_regex_from_python(code))
    if language == "Java":
        return remove_imports(remove_comments_and_javadoc_from_java(code))
    raise ValueError("Language not supported")


def build_trivial_ngrams(df, k=500):
    tokenized_corpus = [line.split() for line in df["containing_method_after_repayment"].astype(str).tolist()]
    tokenized_corpus = [item for sublist in tokenized_corpus for item in sublist]
    all_ngrams = []
    for n in range(1, 5):
        all_ngrams.extend(list(ngrams(tokenized_corpus, n)))
    frequencies = Counter(all_ngrams)
    return dict(frequencies.most_common(k))


def extract_answer_code(answer, language):
    return extract_code(str(answer), language)


def extract_answer_code_for_row(answer, row):
    language = row["language"]
    method_name = str(row.get("method_name", "")).strip()
    raw = str(answer)
    candidate = extract_code(raw, language)
    if method_name:
        if language == "Python":
            method = _extract_python_method_block(candidate, method_name) or _extract_python_method_block(raw, method_name)
            if method:
                return method
        elif language == "Java":
            method = _extract_java_method_block(candidate, method_name) or _extract_java_method_block(raw, method_name)
            if method:
                return method
    return candidate


def load_answers_for_model(output_dir, model_name, prompt_template, df):
    path = answer_path(output_dir, model_name, prompt_template)
    if not path.exists():
        return {}
    answers = {int(k): v for k, v in load_from_json(path).items()}
    valid_ids = set(df["rand_index"].astype(int))
    return {k: v for k, v in answers.items() if k in valid_ids}


def evaluate_exact_match(df, output_dir, models):
    rows = []
    model_template_em = {}

    for model_name, _ in models:
        model_template_em[model_name] = {}
        for prompt_template in PROMPT_TEMPLATES:
            answers = load_answers_for_model(output_dir, model_name, prompt_template, df)
            if not answers:
                continue

            em_all = set()
            em_no_change_all = set()
            em_ignore_ws_all = set()
            distances = {}
            black_failed_count = 0
            black_failed_em_count = 0

            for language in sorted(df["language"].unique()):
                lang_df = df[df["language"] == language]
                lang_ids = set(lang_df["rand_index"].astype(int))
                lang_answers = {k: v for k, v in answers.items() if k in lang_ids}
                if not lang_answers:
                    continue

                no_change_distance, em_no_change, _, _ = get_exact_matches(
                    lang_df,
                    lang_answers,
                    language,
                    "MixedSATD",
                    True,
                    ignore_whitespace=False,
                    format_code=False,
                )
                ignore_ws_distance, em_ignore_ws, _, _ = get_exact_matches(
                    lang_df,
                    lang_answers,
                    language,
                    "MixedSATD",
                    True,
                    ignore_whitespace=True,
                )
                rand_distance, em, black_failed, black_failed_em = get_exact_matches(
                    lang_df,
                    lang_answers,
                    language,
                    "MixedSATD",
                    True,
                    ignore_docstrings_and_comments=True,
                )

                distances.update(rand_distance)
                em_no_change_all.update(em_no_change.keys())
                em_ignore_ws_all.update(em_ignore_ws.keys())
                em_all.update(em.keys())
                black_failed_count += len(black_failed)
                black_failed_em_count += len(black_failed_em)

            model_template_em[model_name][prompt_template] = em_all
            items_easy = get_itmes_having_specific_number_of_inserted_lines(df, em_all, 0, 2)
            items_hard = get_itmes_having_specific_number_of_inserted_lines(df, em_all, 3, 1000)

            deletes = []
            inserts = []
            for rand_index, answer in answers.items():
                row = df[df["rand_index"] == rand_index].iloc[0]
                answer_code = extract_answer_code_for_row(answer, row)
                input_code = row["containing_method_before_repayment"]
                deleted, inserted = get_deleted_and_inserted_lines(answer_code, input_code)
                deletes.append(len(deleted))
                inserts.append(len(inserted))

            rows.append(
                {
                    "Model": model_name,
                    "Template": prompt_template,
                    "Items": len(answers),
                    "Avg deletes": average(deletes),
                    "Avg inserts": average(inserts),
                    "EM": len(em_all) / len(answers),
                    "EM on Easy": len(items_easy) / len(answers),
                    "EM on Hard": len(items_hard) / len(answers),
                    "Avg Levenshtein Distance": average(distances.values()),
                    "Exact Match without any changes": len(em_no_change_all),
                    "Exact Match by ignoring whitespaces": len(em_ignore_ws_all),
                    "Exact Match ignoring imports/comments/docstrings": len(em_all),
                    "Failed by black": black_failed_count,
                    "Failed by black but EM ignoring whitespace": black_failed_em_count,
                }
            )

    oracle_rows = []
    for model_name, _ in models:
        if model_name not in model_template_em:
            continue
        oracle = set()
        for prompt_template in PROMPT_TEMPLATES:
            oracle.update(model_template_em[model_name].get(prompt_template, set()))
        if oracle:
            oracle_rows.append({"Model": model_name, "Oracle EM": len(oracle) / len(df), "Oracle Count": len(oracle)})

    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


def metric_dicts_for_whole_code(df, answers):
    trivial_ngrams = build_trivial_ngrams(df)
    bleu = {}
    crystal = {}
    codebleu = {}
    line_p = {}
    line_r = {}
    line_f = {}
    per_item = []

    for language in sorted(df["language"].unique()):
        lang_df = df[df["language"] == language]
        lang_ids = set(lang_df["rand_index"].astype(int))
        lang_answers = {k: v for k, v in answers.items() if k in lang_ids}
        if not lang_answers:
            continue

        references = {
            int(row["rand_index"]): str(row["containing_method_after_repayment"])
            for _, row in lang_df.iterrows()
            if int(row["rand_index"]) in lang_answers
        }
        candidates = {}
        for i, answer in lang_answers.items():
            row = lang_df[lang_df["rand_index"] == i].iloc[0]
            candidates[i] = extract_answer_code_for_row(answer, row)

        lang_bleu = get_BLEU(references, candidates, language, True)
        lang_crystal = get_crystalBLEU(references, candidates, language, True, trivial_ngrams)
        lang_codebleu = get_codebleu_scores(references, candidates, language)
        lang_p, lang_r, lang_f = get_linePRF(references, candidates, language, True)

        bleu.update(lang_bleu)
        crystal.update(lang_crystal)
        codebleu.update(lang_codebleu)
        line_p.update(lang_p)
        line_r.update(lang_r)
        line_f.update(lang_f)

        for rand_index in candidates:
            per_item.append(
                {
                    "rand_index": rand_index,
                    "language": language,
                    "BLEU": lang_bleu[rand_index],
                    "CrystalBLEU": lang_crystal[rand_index],
                    "CodeBLEU": lang_codebleu[rand_index],
                    "LineP": lang_p[rand_index],
                    "lineR": lang_r[rand_index],
                    "lineF": lang_f[rand_index],
                }
            )

    return bleu, crystal, codebleu, line_p, line_r, line_f, per_item


def metric_dicts_for_diff(df, answers):
    trivial_ngrams = build_trivial_ngrams(df)
    bleu = {}
    crystal = {}
    codebleu = {}
    line_p = {}
    line_r = {}
    line_f = {}
    per_item = []

    for language in sorted(df["language"].unique()):
        lang_df = df[df["language"] == language]
        lang_ids = set(lang_df["rand_index"].astype(int))
        lang_answers = {k: v for k, v in answers.items() if k in lang_ids}
        if not lang_answers:
            continue

        references = {}
        candidates = {}
        for _, row in lang_df.iterrows():
            rand_index = int(row["rand_index"])
            if rand_index not in lang_answers:
                continue

            before_code = str(row["containing_method_before_repayment"])
            after_code = str(row["containing_method_after_repayment"])
            answer_code = extract_answer_code_for_row(lang_answers[rand_index], row)

            before_code_clean = remove_icd(before_code, language)
            after_code_clean = remove_icd(after_code, language)
            answer_code_clean = remove_icd(answer_code, language)

            references[rand_index] = "\n".join(
                get_updated_or_new_lines(before_code_clean, after_code_clean, "MixedSATD")
            )
            candidates[rand_index] = "\n".join(
                get_updated_or_new_lines(before_code_clean, answer_code_clean, "MixedSATD")
            )

        lang_bleu = get_BLEU(references, candidates, language, True)
        lang_crystal = get_crystalBLEU(references, candidates, language, True, trivial_ngrams)
        lang_codebleu = get_codebleu_scores(references, candidates, language)
        lang_p, lang_r, lang_f = get_linePRF(references, candidates, language, True)

        bleu.update(lang_bleu)
        crystal.update(lang_crystal)
        codebleu.update(lang_codebleu)
        line_p.update(lang_p)
        line_r.update(lang_r)
        line_f.update(lang_f)

        for rand_index in candidates:
            per_item.append(
                {
                    "rand_index": rand_index,
                    "language": language,
                    "BLEU": lang_bleu[rand_index],
                    "CrystalBLEU": lang_crystal[rand_index],
                    "CodeBLEU": lang_codebleu[rand_index],
                    "lineP": lang_p[rand_index],
                    "lineR": lang_r[rand_index],
                    "lineF": lang_f[rand_index],
                }
            )

    return bleu, crystal, codebleu, line_p, line_r, line_f, per_item


def evaluate_bleu_tables(df, output_dir, models):
    whole_rows = []
    diff_rows = []
    per_item_whole = []
    per_item_diff = []

    for model_name, _ in models:
        for prompt_template in PROMPT_TEMPLATES:
            answers = load_answers_for_model(output_dir, model_name, prompt_template, df)
            if not answers:
                continue

            bleu, crystal, codebleu, line_p, line_r, line_f, item_rows = metric_dicts_for_whole_code(df, answers)
            whole_rows.append(
                {
                    "Model": model_name,
                    "Template": prompt_template,
                    "BLEU": average(bleu.values()),
                    "CrystalBLEU": average(crystal.values()),
                    "CodeBLEU": average(codebleu.values()),
                    "LineP": average(line_p.values()),
                    "lineR": average(line_r.values()),
                    "lineF": average(line_f.values()),
                }
            )
            for row in item_rows:
                row.update({"Model": model_name, "Template": prompt_template})
            per_item_whole.extend(item_rows)

            bleu, crystal, codebleu, line_p, line_r, line_f, item_rows = metric_dicts_for_diff(df, answers)
            diff_rows.append(
                {
                    "Model": model_name,
                    "Template": prompt_template,
                    "Group": "all",
                    "BLEU": average(bleu.values()),
                    "CrystalBLEU": average(crystal.values()),
                    "CodeBLEU": average(codebleu.values()),
                    "lineP": average(line_p.values()),
                    "lineR": average(line_r.values()),
                    "lineF": average(line_f.values()),
                }
            )
            for row in item_rows:
                row.update({"Model": model_name, "Template": prompt_template, "Group": "all"})
            per_item_diff.extend(item_rows)

    return (
        pd.DataFrame(whole_rows),
        pd.DataFrame(diff_rows),
        pd.DataFrame(per_item_whole),
        pd.DataFrame(per_item_diff),
    )


def evaluate_baseline_input(df):
    answers = {
        int(row["rand_index"]): str(row["containing_method_before_repayment"])
        for _, row in df.iterrows()
    }
    bleu, crystal, codebleu, line_p, line_r, line_f, _ = metric_dicts_for_whole_code(df, answers)
    return pd.DataFrame(
        [
            {
                "Candidate": "containing_method_before_repayment",
                "BLEU": average(bleu.values()),
                "CrystalBLEU": average(crystal.values()),
                "CodeBLEU": average(codebleu.values()),
                "LineP": average(line_p.values()),
                "lineR": average(line_r.values()),
                "lineF": average(line_f.values()),
            }
        ]
    )


def collect_answers_table(df, output_dir, models):
    rows = []
    for model_name, _ in models:
        for prompt_template in PROMPT_TEMPLATES:
            answers = load_answers_for_model(output_dir, model_name, prompt_template, df)
            for rand_index, answer in answers.items():
                row = df[df["rand_index"] == rand_index].iloc[0]
                rows.append(
                    {
                        "Model": model_name,
                        "Template": prompt_template,
                        "rand_index": rand_index,
                        "language": row["language"],
                        "SATD_comment": row["SATD_comment"],
                        "answer": answer,
                        "extracted_code": extract_answer_code_for_row(answer, row),
                    }
                )
    return pd.DataFrame(rows)


def write_excel(df, output_dir, models):
    exact_df, oracle_df = evaluate_exact_match(df, output_dir, models)
    whole_df, diff_df, per_item_whole_df, per_item_diff_df = evaluate_bleu_tables(df, output_dir, models)
    baseline_df = evaluate_baseline_input(df)
    answers_df = collect_answers_table(df, output_dir, models)

    summary_df = pd.DataFrame(
        [
            {"Metric": "Rows", "Value": len(df)},
            {"Metric": "Python rows", "Value": len(df[df["language"] == "Python"])},
            {"Metric": "Java rows", "Value": len(df[df["language"] == "Java"])},
            {"Metric": "Easy rows (<=2 inserted lines)", "Value": len(df[df["number_of_inserted_lines"] <= 2])},
            {"Metric": "Hard rows (>=3 inserted lines)", "Value": len(df[df["number_of_inserted_lines"] >= 3])},
            {"Metric": "Prompts", "Value": ", ".join(PROMPT_TEMPLATES)},
            {"Metric": "Models", "Value": ", ".join(name for name, _ in models)},
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f"satd_llm_fix_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="dataset_summary", index=False)
        answers_df.to_excel(writer, sheet_name="llm_answers", index=False)
        exact_df.to_excel(writer, sheet_name="exact_match", index=False)
        oracle_df.to_excel(writer, sheet_name="oracle_prompt", index=False)
        whole_df.to_excel(writer, sheet_name="bleu_whole_code", index=False)
        baseline_df.to_excel(writer, sheet_name="baseline_input", index=False)
        diff_df.to_excel(writer, sheet_name="bleu_diff", index=False)
        per_item_whole_df.to_excel(writer, sheet_name="per_item_whole", index=False)
        per_item_diff_df.to_excel(writer, sheet_name="per_item_diff", index=False)
    return excel_path
