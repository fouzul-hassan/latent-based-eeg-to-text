"""
Cross-LLM Semantic Evaluation.

Addresses reviewer concerns (EcLa W2, oD5V W5):
"The same Llama 3.1-70B model is used to generate paraphrases AND evaluate
semantic consistency. No independent LLM judge or human evaluation is provided."

This script evaluates GRAPE-generated text using a DIFFERENT LLM family as judge,
removing the circular-evaluation concern.

Supported judge backends (set --judge):
  - gpt4o_mini  : OpenAI GPT-4o-mini (requires OPENAI_API_KEY)
  - claude       : Anthropic Claude (requires ANTHROPIC_API_KEY)
  - llama_local  : Local Llama model via HuggingFace (different variant / size)

Tasks evaluated:
  1. Corpus-level semantic classification (what is the topic?)
  2. Relation-type classification (9 classes)
  3. Sentiment classification (positive / neutral / negative)

Usage:
    # GPT-4o-mini judge:
    export OPENAI_API_KEY=sk-...
    python experiments/crossllm_eval.py \
        --predictions evaluations/v2/relation_results.txt \
        --judge gpt4o_mini --task relation \
        --out results/crossllm_relation_v2.json

    # Claude judge:
    export ANTHROPIC_API_KEY=sk-ant-...
    python experiments/crossllm_eval.py \
        --predictions evaluations/v2/corpus_results.txt \
        --judge claude --task corpus \
        --out results/crossllm_corpus_v2.json

    # Compare V1 and V2 with GPT-4o-mini:
    python experiments/crossllm_eval.py \
        --predictions evaluations/v1/relation_results.txt \
                      evaluations/v2/relation_results.txt \
        --names V1 V2 \
        --judge gpt4o_mini --task relation \
        --out results/crossllm_comparison.json
"""
import os
import re
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Literal


# ── Judge backends ────────────────────────────────────────────────────────────

def judge_gpt4o_mini(prompts: list[str], model: str = 'gpt-4o-mini') -> list[str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    responses = []
    for prompt in prompts:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                responses.append(resp.choices[0].message.content.strip())
                break
            except Exception as e:
                if attempt == 2:
                    responses.append("ERROR")
                time.sleep(2 ** attempt)
    return responses


def judge_claude(prompts: list[str], model: str = 'claude-haiku-4-5-20251001') -> list[str]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    responses = []
    for prompt in prompts:
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                responses.append(msg.content[0].text.strip())
                break
            except Exception as e:
                if attempt == 2:
                    responses.append("ERROR")
                time.sleep(2 ** attempt)
    return responses


def judge_llama_local(prompts: list[str], model_id: str = 'meta-llama/Llama-3.1-8B-Instruct',
                      bsz: int = 8) -> list[str]:
    """Use a smaller Llama variant locally as a different-scale cross-check."""
    import torch
    import transformers
    from torch.utils.data import DataLoader
    pipe = transformers.pipeline(
        'text-generation', model_id,
        model_kwargs={'torch_dtype': torch.float16},
        device_map='auto',
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    responses = []
    loader = DataLoader(prompts, batch_size=bsz, shuffle=False, drop_last=False)
    for batch in loader:
        out = pipe(list(batch), max_new_tokens=10, do_sample=False,
                   pad_token_id=pipe.tokenizer.eos_token_id)
        for p, o in zip(batch, out):
            text = o[0]['generated_text'][len(p):].strip()
            responses.append(text)
    return responses


JUDGE_FNS = {
    'gpt4o_mini':   judge_gpt4o_mini,
    'claude':       judge_claude,
    'llama_local':  judge_llama_local,
}


# ── Task-specific prompt builders ─────────────────────────────────────────────

RELATION_CLASSES = [
    'awarding', 'education', 'employment', 'foundation', 'job title',
    'marriage', 'nationality', 'political affiliation', 'visit',
]
SENTIMENT_CLASSES = ['positive', 'neutral', 'negative']
CORPUS_CLASSES = ['movie review', 'biography', 'news article', 'other']


def build_relation_prompt(generated_text: str) -> str:
    classes_str = ', '.join(RELATION_CLASSES)
    return (
        f"Classify the semantic relation expressed in this text into exactly one of:\n"
        f"{classes_str}\n\n"
        f"Text: \"{generated_text}\"\n\n"
        f"Reply with ONLY the class name, nothing else."
    )


def build_sentiment_prompt(generated_text: str) -> str:
    return (
        f"Classify the sentiment of this text as exactly one of: positive, neutral, negative\n\n"
        f"Text: \"{generated_text}\"\n\n"
        f"Reply with ONLY one word: positive, neutral, or negative."
    )


def build_corpus_prompt(generated_text: str) -> str:
    return (
        f"What type of text is this? Choose one of: movie review, biography, news article, other\n\n"
        f"Text: \"{generated_text}\"\n\n"
        f"Reply with ONLY the category name."
    )


PROMPT_BUILDERS = {
    'relation': build_relation_prompt,
    'sentiment': build_sentiment_prompt,
    'corpus': build_corpus_prompt,
}

CLASS_SETS = {
    'relation': RELATION_CLASSES,
    'sentiment': SENTIMENT_CLASSES,
    'corpus': CORPUS_CLASSES,
}


# ── Prediction file parser ────────────────────────────────────────────────────

def parse_predictions(path: str) -> list[dict]:
    """
    Parse a predictions .txt file written by predict.py.
    Extracts (label, raw_text, gen_text) for each sample.
    """
    samples = []
    with open(path) as f:
        content = f.read()

    # Match blocks like: [ 1] -  Label: X  Pred: Y\n   Raw: ...\n   Gen: ...
    pattern = re.compile(
        r'\[\s*\d+\]\s+-\s+Label:\s*(.*?)\s+Pred:.*?\n'
        r'\s+Raw:\s+(.*?)\n'
        r'\s+Gen:\s+(.*?)(?=\n\[|\Z)',
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        label = m.group(1).strip()
        raw   = m.group(2).strip()
        gen   = m.group(3).strip()
        if label.lower() not in ('nan', 'none', ''):
            samples.append({'label': label, 'raw': raw, 'gen': gen})
    return samples


# ── Evaluation ────────────────────────────────────────────────────────────────

def normalise(text: str, valid_classes: list[str]) -> str:
    t = text.lower().strip().rstrip('.').strip('"\'')
    for cls in valid_classes:
        if cls in t:
            return cls
    return t


def evaluate_predictions(samples: list[dict], task: str, judge_fn,
                          judge_name: str, verbose: bool = False) -> dict:
    print(f"  Building {len(samples)} prompts (task={task}, judge={judge_name}) ...")
    valid = CLASS_SETS[task]
    builder = PROMPT_BUILDERS[task]

    prompts_gen = [builder(s['gen']) for s in samples]
    prompts_raw = [builder(s['raw']) for s in samples]

    print(f"  Querying judge on generated text ...")
    preds_gen = judge_fn(prompts_gen)
    print(f"  Querying judge on raw input text ...")
    preds_raw = judge_fn(prompts_raw)

    labels = [s['label'].lower().strip() for s in samples]

    gen_preds_norm = [normalise(p, valid) for p in preds_gen]
    raw_preds_norm = [normalise(p, valid) for p in preds_raw]

    def acc(preds, labs):
        correct = sum(p == l for p, l in zip(preds, labs) if l in valid)
        total   = sum(1 for l in labs if l in valid)
        return correct / total if total else 0.0

    gen_acc = acc(gen_preds_norm, labels)
    raw_acc = acc(raw_preds_norm, labels)

    result = {
        'judge': judge_name,
        'task': task,
        'n_samples': len(samples),
        'gen_acc': gen_acc,
        'raw_acc': raw_acc,
        'delta_gen_vs_raw': gen_acc - raw_acc,
        'per_sample': [
            {'label': s['label'], 'pred_gen': pg, 'pred_raw': pr}
            for s, pg, pr in zip(samples, gen_preds_norm, raw_preds_norm)
        ] if verbose else [],
    }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', nargs='+', required=True,
                   help='Prediction .txt files from predict.py')
    p.add_argument('--names', nargs='*', default=None,
                   help='Short names for each prediction file (optional)')
    p.add_argument('--judge', required=True, choices=list(JUDGE_FNS.keys()),
                   help='Which LLM judge to use')
    p.add_argument('--task', required=True, choices=list(PROMPT_BUILDERS.keys()))
    p.add_argument('--judge_model', default=None,
                   help='Override default model for the judge (optional)')
    p.add_argument('--out', required=True)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    names = args.names or [Path(f).parent.name for f in args.predictions]
    assert len(names) == len(args.predictions), "Number of names must match number of files"

    judge_fn = JUDGE_FNS[args.judge]
    if args.judge_model:
        import functools
        judge_fn = functools.partial(judge_fn, model=args.judge_model)

    all_results = []
    for pred_file, name in zip(args.predictions, names):
        print(f"\n{'─'*60}")
        print(f"  Model: {name}  |  File: {pred_file}")
        samples = parse_predictions(pred_file)
        print(f"  Parsed {len(samples)} labelled samples")
        res = evaluate_predictions(samples, args.task, judge_fn,
                                   judge_name=args.judge, verbose=args.verbose)
        res['model'] = name
        res['file'] = pred_file
        all_results.append(res)
        print(f"  Gen-text acc:  {res['gen_acc']:.4f}")
        print(f"  Raw-text acc:  {res['raw_acc']:.4f}")
        print(f"  Δ (gen-raw):   {res['delta_gen_vs_raw']:+.4f}")

    # Comparison table
    print(f"\n{'='*60}")
    print(f"  CROSS-LLM EVALUATION SUMMARY  (judge={args.judge}, task={args.task})")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Gen Acc':>10} {'Raw Acc':>10} {'Δ':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8}")
    for r in all_results:
        print(f"  {r['model']:<20} {r['gen_acc']:>10.4f} {r['raw_acc']:>10.4f}"
              f" {r['delta_gen_vs_raw']:>+8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
