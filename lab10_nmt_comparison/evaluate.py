"""
Lab 10 (A1): Evaluation - BLEU scoring + qualitative comparison
==================================================================
Computes BLEU scores on the test set and generates side-by-side sample
translations for all 4 models. Run this AFTER training each model and
saving its checkpoint (see train.py).

Usage:
    python evaluate.py
"""

import pickle
import csv
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu

from utils import DEVICE, greedy_decode  # shared helpers (utils.py)

smoothie = SmoothingFunction().method4


# ---------------------------------------------------------------------------
# Load processed data (vocabs + test set) produced by preprocess.py
# ---------------------------------------------------------------------------

def load_processed_data(path='processed_data.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# ---------------------------------------------------------------------------
# BLEU evaluation
# ---------------------------------------------------------------------------

def evaluate_bleu(model, test_pairs, src_vocab, tgt_vocab, max_len=20, n=None):
    """
    Runs the model on every (or first `n`) test pair, computes corpus-level
    BLEU. Returns (bleu_score, list_of_(src, ref, hyp)_tuples) so the same
    call can feed both the quantitative table and qualitative samples.
    """
    model.eval()
    pairs = test_pairs if n is None else test_pairs[:n]

    references = []   # list of list-of-tokens (nltk expects list of references per sentence)
    hypotheses = []    # list of tokens
    triples = []        # (source, reference, hypothesis) strings, for display

    with torch.no_grad():
        for src_sent, tgt_sent in pairs:
            hyp = greedy_decode(model, src_sent, src_vocab, tgt_vocab, max_len=max_len)
            ref_tokens = tgt_sent.split()
            hyp_tokens = hyp.split()

            references.append([ref_tokens])
            hypotheses.append(hyp_tokens)
            triples.append((src_sent, tgt_sent, hyp))

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu, triples


# ---------------------------------------------------------------------------
# Qualitative comparison across all 4 models
# ---------------------------------------------------------------------------

def qualitative_comparison(models_dict, test_pairs, src_vocab, tgt_vocab,
                             n_samples=10, max_len=20, seed=0):
    """
    models_dict: {'RNN': model1, 'EncDec': model2, 'Additive': model3, 'Multiplicative': model4}
    Picks n_samples test sentences and shows each model's translation side by side.
    """
    import random
    rng = random.Random(seed)
    samples = rng.sample(test_pairs, min(n_samples, len(test_pairs)))

    rows = []
    for src_sent, tgt_sent in samples:
        row = {'source': src_sent, 'reference': tgt_sent}
        for name, model in models_dict.items():
            model.eval()
            with torch.no_grad():
                row[name] = greedy_decode(model, src_sent, src_vocab, tgt_vocab, max_len=max_len)
        rows.append(row)
    return rows


def print_qualitative_table(rows, model_names):
    for i, row in enumerate(rows, 1):
        print(f"\n--- Example {i} ---")
        print(f"Source:    {row['source']}")
        print(f"Reference: {row['reference']}")
        for name in model_names:
            print(f"{name:>15}: {row[name]}")


def save_qualitative_csv(rows, model_names, path='results/sample_translations.csv'):
    fieldnames = ['source', 'reference'] + model_names
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved qualitative samples to {path}")


def save_bleu_csv(bleu_scores, path='results/bleu_scores.csv'):
    """bleu_scores: dict like {'RNN': 0.12, 'EncDec': 0.18, ...}"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'bleu_score'])
        for name, score in bleu_scores.items():
            writer.writerow([name, round(score, 4)])
    print(f"Saved BLEU scores to {path}")


# ---------------------------------------------------------------------------
# Main - assumes each model checkpoint has already been trained & saved
# by train.py (e.g. checkpoints/model1_rnn.pt, checkpoints/model2_*.pt, ...)
# ---------------------------------------------------------------------------

def main():
    data = load_processed_data()
    test_pairs = data['test']
    src_vocab = data['src_vocab']
    tgt_vocab = data['tgt_vocab']

    # --- Load trained models ---
    # Each model module should provide a `load_model(checkpoint_path, src_vocab, tgt_vocab)`
    # function returning a ready-to-eval model. Import lazily so this script
    # doesn't fail if you haven't built all 4 models yet.
    models_dict = {}

    try:
        from models.model1_rnn import load_model as load_rnn
        models_dict['RNN'] = load_rnn('checkpoints/model1_rnn.pt', src_vocab, tgt_vocab).to(DEVICE)
    except Exception as e:
        print(f"[skip] Model 1 (RNN) not available: {e}")

    try:
        from models.model2_encoder_decoder import load_model as load_encdec
        models_dict['EncDec'] = load_encdec('checkpoints/model2_encdec.pt', src_vocab, tgt_vocab).to(DEVICE)
    except Exception as e:
        print(f"[skip] Model 2 (EncDec) not available: {e}")

    try:
        from models.model3_additive_attention import load_model as load_additive
        models_dict['Additive'] = load_additive('checkpoints/model3_additive.pt', src_vocab, tgt_vocab).to(DEVICE)
    except Exception as e:
        print(f"[skip] Model 3 (Additive Attention) not available: {e}")

    try:
        from models.model4_multiplicative_attention import load_model as load_mult
        models_dict['Multiplicative'] = load_mult('checkpoints/model4_multiplicative.pt', src_vocab, tgt_vocab).to(DEVICE)
    except Exception as e:
        print(f"[skip] Model 4 (Multiplicative Attention) not available: {e}")

    if not models_dict:
        print("No trained models found yet. Train models first (see train.py), then rerun evaluate.py.")
        return

    # --- BLEU scores ---
    bleu_scores = {}
    for name, model in models_dict.items():
        score, _ = evaluate_bleu(model, test_pairs, src_vocab, tgt_vocab)
        bleu_scores[name] = score
        print(f"{name}: BLEU = {score:.4f}")

    save_bleu_csv(bleu_scores)

    # --- Qualitative comparison ---
    rows = qualitative_comparison(models_dict, test_pairs, src_vocab, tgt_vocab, n_samples=10)
    print_qualitative_table(rows, list(models_dict.keys()))
    save_qualitative_csv(rows, list(models_dict.keys()))


if __name__ == '__main__':
    main()