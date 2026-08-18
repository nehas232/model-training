"""
Lab 10 (A1): Preprocessing pipeline for Comparative Analysis of NMT Models
============================================================================
Expects a tab-separated file downloaded from https://www.manythings.org/anki/
Format per line: <english sentence>\t<target language sentence>\t<attribution>

Usage:
    python preprocess.py --file data\spa_sample.txt --roll_number <your_roll_number>
"""

import re
import random
import argparse
import unicodedata
import pickle

from vocab import Vocab  # shared class -- see vocab.py (fixes pickle cross-script loading)

# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------

def unicode_to_ascii(s):
    """Strip accents (optional - only use if you want to DROP accents,
    which is usually NOT ideal for Spanish/French/etc. since accents carry
    meaning). Kept here for reference / off by default."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def clean_sentence(s, strip_accents=False):
    s = s.lower().strip()
    if strip_accents:
        s = unicode_to_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZÀ-ÿ.!?']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# 2. Load + filter
# ---------------------------------------------------------------------------

def load_pairs(path, max_len=15, min_len=1):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            eng, tgt = parts[0], parts[1]
            eng_c = clean_sentence(eng)
            tgt_c = clean_sentence(tgt)
            if not eng_c or not tgt_c:
                continue
            if min_len <= len(eng_c.split()) <= max_len and \
               min_len <= len(tgt_c.split()) <= max_len:
                pairs.append((eng_c, tgt_c))
    return pairs


# ---------------------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------------------

def train_val_test_split(pairs, random_state, train_frac=0.8, val_frac=0.1):
    rng = random.Random(random_state)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='path to e.g. data\\spa_sample.txt')
    ap.add_argument('--roll_number', required=True, type=int,
                     help='used as random_state, per assignment instructions')
    ap.add_argument('--max_len', type=int, default=15)
    ap.add_argument('--min_freq', type=int, default=2)
    args = ap.parse_args()

    print(f"Loading pairs from {args.file} ...")
    pairs = load_pairs(args.file, max_len=args.max_len)
    print(f"Loaded {len(pairs)} sentence pairs after cleaning/filtering")
    print("Example pair:", pairs[0])

    train, val, test = train_val_test_split(pairs, random_state=args.roll_number)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    src_vocab = Vocab([p[0] for p in train], min_freq=args.min_freq)
    tgt_vocab = Vocab([p[1] for p in train], min_freq=args.min_freq)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({
            'train': train, 'val': val, 'test': test,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab,
            'random_state': args.roll_number,
        }, f)
    print("Saved processed_data.pkl")


if __name__ == '__main__':
    main()