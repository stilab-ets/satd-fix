"""Generate and evaluate method-level LLM repairs for the selected SATD instances."""

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI


BASELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASELINE_DIR.parent
LLM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(LLM_DIR))

from SATD_tool import load_from_json, save_to_json
import fix_satd_eval
from env_loader import load_project_env


DATASET_PATH = PROJECT_DIR / "data" / "satd_dataset.xlsx"
DATASET_SHEET = "samples"
OUTPUT_DIR = PROJECT_DIR / "execution_runs" / "llm"
PROMPT_TEMPLATES = ["NoExplain", "CoT1"] 
PROMPT_VERSION = "v2"
DEFAULT_MODEL = "gpt-4o-2024-08-06"

DEFAULT_MODELS = [
    (DEFAULT_MODEL, DEFAULT_MODEL),
    ("QwenCoder30B", "qwen/qwen3-coder-30b-a3b-instruct"),
    ("gpt-4o-mini", "openai/gpt-4o-mini-2024-07-18"),
    ("Gemma4Free", "google/gemma-4-31b-it:free"),
    ("claude-sonnet", "anthropic/claude-sonnet-4.5"),
    ("glm-4.7-flash", "z-ai/glm-4.7-flash")
]


def get_prompt(row, prompt_template):
    """Build the baseline prompt from the original method and SATD comment."""
    if prompt_template == "NoExplain":
        return f"""How to update the following code to resolve the SATD? No need to explain. Just provide the updated code.

### Code:
{row['containing_method_before_repayment']}

### SATD comment:
{row['SATD_comment']}

### Updated code after SATD repayment:"""

    if prompt_template == "CoT1":
        return f"""How to update the following code to resolve the SATD?

### Code:
{row['containing_method_before_repayment']}

### SATD comment:
{row['SATD_comment']}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code."""

    raise ValueError(f"Unknown prompt template: {prompt_template}")


def call_llm(client, full_model_name, prompt, max_tokens):
    """Request a candidate repair from the configured model."""
    extra_headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_NAME", "SATD LLM Fix Evaluation"),
    }

    response = client.chat.completions.create(
        model=full_model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        extra_headers=extra_headers,
    )
    return response.choices[0].message.content or ""


def answer_path(output_dir, model_name, prompt_template):
    return output_dir / "generated_outputs" / model_name / f"answers_{prompt_template}_{PROMPT_VERSION}.json"

def generate_answers(df, output_dir, models, base_url, api_key, max_tokens, save_every):
    """Generate missing repairs and periodically save progress for resuming a run."""
    client = OpenAI(base_url=base_url, api_key=api_key)

    for model_name, full_model_name in models:
        for prompt_template in PROMPT_TEMPLATES:
            path = answer_path(output_dir, model_name, prompt_template)
            path.parent.mkdir(parents=True, exist_ok=True)
            answers = {}
            if path.exists():
                answers = {int(k): v for k, v in load_from_json(path).items()}

            unsaved = {}
            for _, row in df.iterrows():
                rand_index = int(row["rand_index"])
                if rand_index in answers:
                    continue

                prompt = get_prompt(row, prompt_template)
                print(f"Getting answer: model={model_name}, prompt={prompt_template}, rand_index={rand_index}")
                try:
                    answers[rand_index] = call_llm(
                        client,
                        full_model_name,
                        prompt,
                        max_tokens,
                    )
                except Exception as exc:
                    answers[rand_index] = f"LLM_ERROR: {exc}"
                unsaved[rand_index] = answers[rand_index]

                if len(unsaved) >= save_every:
                    save_to_json(answers, str(path))
                    unsaved = {}

            save_to_json(answers, str(path))


def select_models(selection):
    if selection.strip().lower() == "all":
        return list(DEFAULT_MODELS)
    catalog = dict(DEFAULT_MODELS)
    selected = []
    for item in selection.split(","):
        name = item.strip()
        if not name:
            continue
        normalized = name.removeprefix("openai/")
        if name in catalog:
            model = (name, catalog[name])
        elif normalized == DEFAULT_MODEL or name == "gpt-4o":
            model = (DEFAULT_MODEL, name if normalized == DEFAULT_MODEL else DEFAULT_MODEL)
        else:
            model = next((entry for entry in DEFAULT_MODELS if entry[1] == name), None)
            if model is None:
                raise ValueError(f"Unknown model: {name}. Available: {', '.join(catalog)}")
        if model not in selected:
            selected.append(model)
    if not selected:
        raise ValueError("No models selected.")
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--sheet", default=DATASET_SHEET)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--models", default=os.getenv(
        "OPENROUTER_MODEL", os.getenv("OPENAI_AGENTS_MODEL", DEFAULT_MODEL)
    ), help="Comma-separated model names/IDs, or 'all'; defaults to the project .env setting or gpt-4o-2024-08-06.")
    parser.add_argument("--templates", default="NoExplain",
                        help="Comma-separated prompt templates, or 'all'.")
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main():
    global PROMPT_TEMPLATES
    load_project_env()
    args = parse_args()

    output_dir = Path(args.output_dir)
    models = select_models(args.models)

    if args.templates.strip().lower() != "all":
        requested_templates = [item.strip() for item in args.templates.split(",") if item.strip()]
        unknown = [item for item in requested_templates if item not in PROMPT_TEMPLATES]
        if unknown:
            raise ValueError(f"Unknown prompt templates: {', '.join(unknown)}")
        PROMPT_TEMPLATES = requested_templates

    df = fix_satd_eval.load_dataset(args.dataset, sheet_name=args.sheet)

    if not args.evaluate_only:
        if not args.api_key:
            raise ValueError("API key is missing. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or pass --api-key.")
        generate_answers(
            df,
            output_dir,
            models,
            args.base_url,
            args.api_key,
            args.max_tokens,
            args.save_every,
        )

    fix_satd_eval.PROMPT_TEMPLATES = PROMPT_TEMPLATES
    fix_satd_eval.PROMPT_VERSION = PROMPT_VERSION
    excel_path = fix_satd_eval.write_excel(df, output_dir, models)
    print(f"Excel results saved to: {excel_path}")


if __name__ == "__main__":
    main()
