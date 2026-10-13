"""
Live inference helpers for the GLIM demo.

- Auto-discover checkpoints from ./runs/v1/ and ./runs/v2/
- Load model (singleton per version, cached in memory)
- Run inference on one demo sample: generate text + all metrics
- Disk-based JSON cache so repeated clicks are instant
"""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Callable, Optional
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate
from transformers.modeling_outputs import BaseModelOutput
from model.energy import ETESEvaluator

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

VARIANT_KEYS = [
    'lexical simplification (v0)', 'lexical simplification (v1)',
    'semantic clarity (v0)',        'semantic clarity (v1)',
    'syntax simplification (v0)',   'syntax simplification (v1)',
    'naive rewritten',              'naive simplified',
]

SENTIMENT_LABELS  = ['negative', 'neutral', 'positive']
RELATION_LABELS   = ['awarding', 'education', 'employment', 'foundation',
                     'job title', 'nationality', 'political affiliation', 'visit', 'marriage']
CORPUS_LABELS     = ["The topic is about: movie, good or bad",
                     "The topic is about: life experiences, relationship"]
CORPUS_DISPLAY    = ['movie review', 'life / relationship']
PARADIGM_LABELS   = ["Normal reading: the participant reads sentences for general comprehension",
                     "Task-specific reading: the participant reads sentences to extract semantic relations"]
PARADIGM_DISPLAY  = ['NR', 'TSR']


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_all_checkpoints() -> dict[str, str]:
    """
    Return {version_label: ckpt_path} for all auto-discovered checkpoints.
    Scans ./runs/v1/, ./runs/v2/, ./runs/v3/, ... up to v9.
    Picks the most recently modified .ckpt per version directory.
    """
    base = os.path.join(os.path.dirname(__file__), '..', 'runs')
    found = {}
    for version_dir in sorted(glob.glob(os.path.join(base, 'v*'))):
        version = os.path.basename(version_dir)
        ckpts   = sorted(glob.glob(os.path.join(version_dir, '*.ckpt')),
                         key=os.path.getmtime)
        if ckpts:
            found[version] = ckpts[-1]   # most recently modified = best epoch
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Model singleton cache (stays loaded between Gradio calls)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}

def _patched_setup(model, target_device: torch.device):
    """
    Memory-safe replacement for GLIM.setup() in inference mode.

    On CUDA: loads T5 directly onto the GPU via device_map (no CPU staging),
    which avoids the Windows paging-file exhaustion (OSError 1455).
    On CPU: loads with low_cpu_mem_usage=True to minimize RAM pressure.
    """
    import os as _os
    from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration

    _os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.tokenizer = AutoTokenizer.from_pretrained(model.text_model_id)

    if target_device.type == 'cuda':
        # Stream weights directly to GPU, never staging in CPU RAM.
        # This bypasses the Windows page file completely.
        load_kwargs = dict(
            device_map=str(target_device),   # e.g. "cuda:0"
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        )
    else:
        load_kwargs = dict(
            torch_dtype=torch.float32,
            tie_word_embeddings=False,
            low_cpu_mem_usage=True,
        )

    if 'bart' in model.text_model_id.lower():
        text_model = BartForConditionalGeneration.from_pretrained(
            model.text_model_id, **load_kwargs)
    else:
        text_model = T5ForConditionalGeneration.from_pretrained(
            model.text_model_id, **load_kwargs)

    model.text_model = text_model.requires_grad_(False)
    if target_device.type != 'cuda':
        model.text_model = model.text_model.to(target_device)

    # Fix: when tie_word_embeddings=False, T5/BART pretrained checkpoints only
    # save `shared.weight` — encoder.embed_tokens & decoder.embed_tokens are
    # treated as independent and initialized to zeros (MISSING in safetensors).
    # We must copy shared.weight into both so decoding is not garbage.
    with torch.no_grad():
        shared_w = model.text_model.shared.weight.data
        if hasattr(model.text_model, 'encoder') and hasattr(model.text_model.encoder, 'embed_tokens'):
            model.text_model.encoder.embed_tokens.weight.data.copy_(shared_w)
        if hasattr(model.text_model, 'decoder') and hasattr(model.text_model.decoder, 'embed_tokens'):
            model.text_model.decoder.embed_tokens.weight.data.copy_(shared_w)

    model.energy_generator = None
    model.etes_evaluator   = None


def load_model(ckpt_path: str, device: torch.device):
    """Load GLIM model; cache in memory so it's only loaded once per checkpoint."""
    if ckpt_path in _model_cache:
        return _model_cache[ckpt_path]

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from model.glim import GLIM
    print(f"  Loading model from {ckpt_path} ...")
    model = GLIM.load_from_checkpoint(
        ckpt_path, map_location='cpu', strict=False, weights_only=False)
    # Use our memory-safe setup instead of model.setup() to avoid Windows
    # paging-file exhaustion (OSError 1455).
    # If another model with the same text_model_id is already cached, share its
    # T5 weights to avoid loading a second copy of ~800MB from disk.
    shared_text_model = None
    for cached_model in _model_cache.values():
        if (hasattr(cached_model, 'text_model_id') and
                cached_model.text_model_id == model.text_model_id and
                hasattr(cached_model, 'text_model')):
            shared_text_model = cached_model.text_model
            shared_tokenizer  = cached_model.tokenizer
            break

    if shared_text_model is not None:
        import os as _os
        _os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model.tokenizer   = shared_tokenizer
        model.text_model  = shared_text_model
        model.energy_generator = None
        model.etes_evaluator   = None
        print("  Shared T5 weights from cached model (memory-efficient)")
    else:
        _patched_setup(model, device)

    if getattr(model, 'etes_evaluator', None) is None:
        model.etes_evaluator = ETESEvaluator(
            aligner=model.aligner,
            text_encoder=model.text_model.get_encoder(),
            tokenizer=model.tokenizer,
            include_fluency=False,
        )
    model.eval().to(device)
    _model_cache[ckpt_path] = model
    print("  Model loaded OK")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bleu(gen, refs, n):
    try:    return bleu_score([gen], [refs], n_gram=n).item()
    except: return 0.0

def _rouge(gen, refs):
    try:
        d = rouge_score([gen], [refs], rouge_keys='rouge1')
        return {k: v.item() for k, v in d.items()}
    except:
        return {'rouge1_fmeasure': 0.0, 'rouge1_precision': 0.0, 'rouge1_recall': 0.0}

def _wer(gen, ref):
    try:    return word_error_rate([gen], [ref]).item()
    except: return 1.0


def _ensure_etes_evaluator(model) -> ETESEvaluator:
    """Return a ready-to-use ETES evaluator for the loaded GLIM model."""
    evaluator = getattr(model, 'etes_evaluator', None)
    if evaluator is None:
        evaluator = ETESEvaluator(
            aligner=model.aligner,
            text_encoder=model.text_model.get_encoder(),
            tokenizer=model.tokenizer,
            include_fluency=False,
        )
        model.etes_evaluator = evaluator
    return evaluator


def _summarize_attention_weights(attn_weights: dict) -> dict:
    """Collapse per-layer attention tensors into a single heatmap-friendly summary."""
    matrices = []
    for _, weights in sorted(attn_weights.items()):
        if weights is None:
            continue
        tensor = weights.detach()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 3:
            tensor = tensor[0]
        if tensor.dim() != 2:
            continue
        matrices.append(tensor.float())

    if not matrices:
        return {}

    stacked = torch.stack(matrices, dim=0)
    mean_matrix = stacked.mean(dim=0)
    return {
        'attention_matrix': mean_matrix.cpu().tolist(),
        'attention_profile': mean_matrix.mean(dim=0).cpu().tolist(),
        'attention_layers': len(matrices),
    }


def _progress_report(progress: Optional[Callable[[float], None]], value: float, desc: str) -> None:
    """Send a progress update if the caller provided a progress hook."""
    if progress is not None:
        progress(value, desc=desc)


def compute_etes_metrics(model, eeg_emb_vectors: torch.Tensor,
                         generated_text: str, raw_input: str,
                         progress: Optional[Callable[[float], None]] = None) -> dict:
    """Compute ETES metrics for a single demo sample."""
    evaluator = _ensure_etes_evaluator(model)
    _progress_report(progress, 0.9, "Computing ETES alignment")
    etes_results = evaluator.evaluate(
        eeg_emb_vectors=eeg_emb_vectors,
        generated_texts=[generated_text],
        reference_texts=[raw_input],
    )
    return {
        'etes_alignment': etes_results.get('etes_alignment', 0.0),
        'etes_total': etes_results.get('etes_total', 0.0),
        'etes_reference': etes_results.get('etes_reference', 0.0),
        'etes_gap': etes_results.get('etes_gap', 0.0),
    }


def compute_all_text_metrics(gen: str, raw_input: str, variants: list[str]) -> dict:
    valid = [v for v in variants if isinstance(v, str) and v.strip()]
    m = {}
    for n in [1, 2, 3, 4]:
        m[f'bleu{n}_mtv'] = _bleu(gen, valid or [raw_input], n)
        m[f'bleu{n}_raw'] = _bleu(gen, [raw_input], n)
    for suffix, refs in [('mtv', valid or [raw_input]), ('raw', [raw_input])]:
        r = _rouge(gen, refs)
        for stat in ['fmeasure', 'precision', 'recall']:
            m[f'rouge1_{stat}_{suffix}'] = r[f'rouge1_{stat}']
    m['wer'] = _wer(gen, raw_input)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(sample_idx: int, row, ckpt_path: str,
                  version_label: str, device: torch.device,
                  progress: Optional[Callable[[float], None]] = None) -> dict:
    """
    Run full inference on one demo sample.
    Results are disk-cached; subsequent calls are instant.

    Returns a rich dict with generated text + all metrics.
    """
    cache_key  = f"{version_label}_sample_{sample_idx}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    raw_input  = str(row.get('input text', ''))
    _progress_report(progress, 0.05, f"Checking cache for {version_label}")

    # ── Cache hit ──
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if 'etes_metrics' not in cached:
            eeg_emb_vector = cached.get('eeg_emb_vector')
            gen_text = cached.get('gen_text')
            if eeg_emb_vector is not None and gen_text:
                try:
                    model = load_model(ckpt_path, device)
                    eeg_emb = torch.tensor(eeg_emb_vector, dtype=torch.float32, device=device).unsqueeze(0)
                    cached['etes_metrics'] = compute_etes_metrics(
                        model, eeg_emb, gen_text, raw_input, progress=progress)
                    with open(cache_path, 'w') as f:
                        json.dump(cached, f, indent=2)
                    _progress_report(progress, 1.0, "Done")
                    cached['_from_cache'] = True
                    return cached
                except Exception as exc:
                    print(f"  [WARN] Could not backfill ETES cache ({type(exc).__name__}: {exc})")
            else:
                print("  [INFO] Refreshing stale cache so ETES can be computed.")
        else:
            cached['_from_cache'] = True
            _progress_report(progress, 1.0, "Done")
            return cached

    # ── Cache miss: run model ──
    _progress_report(progress, 0.15, f"Loading model for {version_label}")
    model = load_model(ckpt_path, device)

    eeg_np  = row['eeg']   if isinstance(row['eeg'],  np.ndarray) else np.array(row['eeg'],  dtype=np.float32)
    mask_np = row['mask']  if isinstance(row['mask'], np.ndarray) else np.array(row['mask'], dtype=np.int8)
    if eeg_np.shape[0] == 128:   # (C,T) → (T,C)
        eeg_np = eeg_np.T

    _progress_report(progress, 0.3, "Preparing EEG tensors")
    eeg_t  = torch.from_numpy(eeg_np).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

    task    = str(row.get('task',    'task1'))
    subj    = str(row.get('subject', '<UNK>'))
    dataset = str(row.get('dataset', '<UNK>'))
    t_token = '<NR>' if task != 'task3' else '<TSR>'

    use_amp = device.type == 'cuda'
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        # ── Encode EEG with task-aware prompt ──
        _progress_report(progress, 0.45, "Encoding EEG")
        prompts      = ([t_token], [dataset], [subj])
        p_ids        = model.p_embedder.encode(prompts, device=device)
        p_embed      = model.p_embedder(p_ids, model.eval_pembed)
        eeg_hiddens, attn_weights = model.eeg_encoder(eeg_t, mask_t, p_embed, need_weights=True)
        eeg_embeds, eeg_emb = model.aligner.embed_eeg(eeg_hiddens)
        if eeg_emb.dim() == 1: eeg_emb = eeg_emb.unsqueeze(0)

        # ── Encode EEG with <UNK> task prompt (for reading paradigm) ──
        unk_prompts      = (['<UNK>'], [dataset], [subj])
        unk_pids         = model.p_embedder.encode(unk_prompts, device=device)
        unk_pembed       = model.p_embedder(unk_pids, model.eval_pembed)
        unk_hid, _       = model.eeg_encoder(eeg_t, mask_t, unk_pembed)
        _, unk_emb       = model.aligner.embed_eeg(unk_hid)
        if unk_emb.dim() == 1: unk_emb = unk_emb.unsqueeze(0)

        # ── Classification helper ──
        def cls_probs(cand_emb):
            e = eeg_emb / eeg_emb.norm(dim=1, keepdim=True)
            c = cand_emb / cand_emb.norm(dim=1, keepdim=True)
            return (e @ c.T).softmax(dim=-1).squeeze(0).cpu().tolist()

        def paradigm_probs_fn(cand_emb):
            e = unk_emb / unk_emb.norm(dim=1, keepdim=True)
            c = cand_emb / cand_emb.norm(dim=1, keepdim=True)
            return (e @ c.T).softmax(dim=-1).squeeze(0).cpu().tolist()

        # Pre-compute label embeddings
        se_embs = model.cal_label_embs(SENTIMENT_LABELS, "Sentiment classification: It is <MASK>.")
        re_embs = model.cal_label_embs(RELATION_LABELS,  "Relation classification: It is about <MASK>.")
        co_embs = model.cal_label_embs(CORPUS_LABELS)
        pa_embs = model.cal_label_embs(PARADIGM_LABELS)

        # ── Text generation (beam search) ──
        _progress_report(progress, 0.7, "Generating text")
        gen_ids = model.text_model.generate(
            encoder_outputs=BaseModelOutput(eeg_embeds),
            num_beams=4,
            min_new_tokens=5,          # prevent trivially empty outputs
            max_length=model.tgt_text_len,
            early_stopping=True,
            no_repeat_ngram_size=3,    # prevent repetitive garbage like '... ... ...'
        )
        gen_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # ── Text metrics ──
    _progress_report(progress, 0.82, "Computing text metrics")
    variants  = [str(row.get(k, '')) for k in VARIANT_KEYS]
    text_metrics = compute_all_text_metrics(gen_text, raw_input, variants)
    etes_metrics = compute_etes_metrics(model, eeg_emb.detach(), gen_text, raw_input, progress=progress)
    attention_summary = _summarize_attention_weights(attn_weights)

    # ── Build result dict ──
    result = {
        'gen_text':         gen_text,
        'text_metrics':     text_metrics,
        'etes_metrics':     etes_metrics,
        'eeg_emb_vector':   eeg_emb.detach().float().cpu().tolist(),
        **attention_summary,

        'sentiment_probs':  cls_probs(se_embs),
        'sentiment_labels': SENTIMENT_LABELS,

        'relation_probs':   cls_probs(re_embs),
        'relation_labels':  RELATION_LABELS,

        'corpus_probs':     cls_probs(co_embs),
        'corpus_labels':    CORPUS_DISPLAY,

        'paradigm_probs':   paradigm_probs_fn(pa_embs),
        'paradigm_labels':  PARADIGM_DISPLAY,

        '_from_cache': False,
        '_version':    version_label,
        '_ckpt':       ckpt_path,
    }

    # ── Save to disk cache ──
    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)
    _progress_report(progress, 1.0, "Done")
    print(f"  Cached -> {cache_path}")

    return result
