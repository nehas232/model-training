"""
Lab 10 (A1): Shared utilities used by train.py and evaluate.py
==================================================================
Batching, padding, training/eval loops, and greedy decoding for inference.
All 4 models share these so behavior is consistent across the comparison.
"""

import random
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Must match the special token order used in preprocess.py's Vocab class:
# itos = [PAD, SOS, EOS, UNK, ...]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def pad_sequence(seqs, pad_idx=PAD_IDX):
    """seqs: list of list[int]. Returns a [batch, max_len] LongTensor, padded."""
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return padded


def batchify(pairs, src_vocab, tgt_vocab, batch_size, shuffle=True):
    """
    Encodes (src_sentence, tgt_sentence) string pairs into padded tensor
    batches. Returns a list of (src_batch, tgt_batch, src_lengths) tuples,
    each on CPU -- move to DEVICE inside the train/eval loop.
    """
    data = pairs[:]
    if shuffle:
        random.shuffle(data)

    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        # sort by source length (descending) -- helps if you use packed sequences;
        # harmless if you don't.
        chunk.sort(key=lambda p: len(p[0].split()), reverse=True)

        src_ids = [src_vocab.encode(s) for s, t in chunk]
        tgt_ids = [tgt_vocab.encode(t) for s, t in chunk]
        src_lengths = torch.tensor([len(s) for s in src_ids], dtype=torch.long)

        src_batch = pad_sequence(src_ids)
        tgt_batch = pad_sequence(tgt_ids)

        batches.append((src_batch, tgt_batch, src_lengths))
    return batches


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, batches, optimizer, criterion, clip_grad=1.0, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0

    for src_batch, tgt_batch, src_lengths in batches:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)
        src_lengths = src_lengths.to(DEVICE)

        optimizer.zero_grad()

        # Model forward signature (implemented identically by all 4 models):
        #   logits = model(src_batch, src_lengths, tgt_batch, teacher_forcing_ratio)
        # logits shape: [batch, tgt_len-1, tgt_vocab_size]  (predicts tokens 1..T given 0..T-1)
        logits = model(src_batch, src_lengths, tgt_batch, teacher_forcing_ratio)

        # Targets are tgt_batch shifted by one (predict next token)
        targets = tgt_batch[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(batches)


@torch.no_grad()
def evaluate_loss(model, batches, criterion):
    model.eval()
    total_loss = 0.0

    for src_batch, tgt_batch, src_lengths in batches:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)
        src_lengths = src_lengths.to(DEVICE)

        # teacher_forcing_ratio=0 at eval time -> model uses its own predictions
        logits = model(src_batch, src_lengths, tgt_batch, teacher_forcing_ratio=0.0)
        targets = tgt_batch[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        total_loss += loss.item()

    return total_loss / len(batches)


# ---------------------------------------------------------------------------
# Greedy decoding for inference (used by evaluate.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(model, sentence, src_vocab, tgt_vocab, max_len=20):
    """
    Translates a single raw (already-cleaned) source sentence string.
    Assumes each model exposes `model.encode(src_tensor, src_lengths)` and
    `model.decode_step(prev_token, hidden, encoder_outputs)` -- see the
    per-model docstrings for exact signatures.
    """
    model.eval()

    src_ids = src_vocab.encode(sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)  # [1, src_len]
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=DEVICE)

    encoder_outputs, hidden = model.encode(src_tensor, src_lengths)

    prev_token = torch.tensor([SOS_IDX], dtype=torch.long, device=DEVICE)
    output_tokens = []

    for _ in range(max_len):
        logits, hidden = model.decode_step(prev_token, hidden, encoder_outputs)
        next_token = logits.argmax(dim=-1)  # [1]
        token_id = next_token.item()

        if token_id == EOS_IDX:
            break
        output_tokens.append(token_id)
        prev_token = next_token

    words = [tgt_vocab.itos[t] for t in output_tokens]
    return ' '.join(words)