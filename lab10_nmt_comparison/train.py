"""
Lab 10 (A1): Shared training loop for all 4 NMT models
=========================================================
Same hyperparameters, same data, same training procedure across all 4
models -- so that differences in results are attributable to architecture,
not to inconsistent training setup.

Usage:
    python train.py --model rnn            --epochs 15
    python train.py --model encdec         --epochs 15
    python train.py --model additive       --epochs 15
    python train.py --model multiplicative --epochs 15
"""

import argparse
import pickle
import time
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import DEVICE, PAD_IDX, batchify, train_one_epoch, evaluate_loss


MODEL_REGISTRY = {
    'rnn':            ('models.model1_rnn', 'checkpoints/model1_rnn.pt'),
    'encdec':         ('models.model2_encoder_decoder', 'checkpoints/model2_encdec.pt'),
    'additive':       ('models.model3_additive_attention', 'checkpoints/model3_additive.pt'),
    'multiplicative': ('models.model4_multiplicative_attention', 'checkpoints/model4_multiplicative.pt'),
}

# Shared hyperparameters - kept IDENTICAL across all 4 models for a fair comparison
HPARAMS = {
    'embed_size': 256,
    'hidden_size': 512,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'clip_grad': 1.0,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_data(path='processed_data.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_model(model_key, src_vocab, tgt_vocab):
    module_path, ckpt_path = MODEL_REGISTRY[model_key]
    module = __import__(module_path, fromlist=['build_model'])
    model = module.build_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=HPARAMS['embed_size'],
        hidden_size=HPARAMS['hidden_size'],
    )
    return model.to(DEVICE), ckpt_path


def plot_losses(train_losses, val_losses, model_key):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss curve - {model_key}')
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = f'results/loss_curve_{model_key}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data', default='processed_data.pkl')
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"Loading processed data from {args.data} ...")
    data = load_processed_data(args.data)
    train_pairs = data['train']
    val_pairs = data['val']
    src_vocab = data['src_vocab']
    tgt_vocab = data['tgt_vocab']
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    print(f"Src vocab: {len(src_vocab)} | Tgt vocab: {len(tgt_vocab)}")

    print(f"Building model: {args.model}")
    model, ckpt_path = build_model(args.model, src_vocab, tgt_vocab)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS['learning_rate'])

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_batches = batchify(train_pairs, src_vocab, tgt_vocab, HPARAMS['batch_size'], shuffle=True)
        val_batches = batchify(val_pairs, src_vocab, tgt_vocab, HPARAMS['batch_size'], shuffle=False)

        train_loss = train_one_epoch(model, train_batches, optimizer, criterion, HPARAMS['clip_grad'])
        val_loss = evaluate_loss(model, val_batches, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'hparams': HPARAMS,
                'src_vocab_itos': src_vocab.itos,
                'tgt_vocab_itos': tgt_vocab.itos,
            }, ckpt_path)
            print(f"  -> saved best checkpoint ({ckpt_path})")

    plot_losses(train_losses, val_losses, args.model)
    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoint at {ckpt_path}")


if __name__ == '__main__':
    main()