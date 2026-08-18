"""
Lab 10 (A1) - Model 2: Encoder-Decoder (no attention)
=========================================================
A proper encoder-decoder split: the encoder is a multi-layer GRU that
processes the full source sentence and produces a final hidden state
(the "context vector"). This context vector is used to INITIALIZE the
decoder's hidden state, and the decoder generates the target sentence
token by token, conditioned only on that fixed-size context (and its
own previous outputs) -- there is still no attention, so the entire
source sentence must be compressed into one fixed-size vector. This is
the classic Sutskever et al. (2014) architecture, and the well-known
"fixed-length bottleneck" problem it has (especially for long sentences)
is exactly what attention (Models 3 & 4) is designed to solve.
"""

import random
import torch
import torch.nn as nn

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src_batch, src_lengths):
        emb = self.embedding(src_batch)                     # [B, S, E]
        outputs, hidden = self.rnn(emb)                       # outputs: [B, S, H], hidden: [L, B, H]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden):
        # input_token: [B, 1]
        emb = self.embedding(input_token)                     # [B, 1, E]
        out, hidden = self.rnn(emb, hidden)                     # out: [B, 1, H]
        logits = self.fc_out(out.squeeze(1))                    # [B, V]
        return logits, hidden


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    # -----------------------------------------------------------------
    # Training forward pass (teacher forcing)
    # -----------------------------------------------------------------
    def forward(self, src_batch, src_lengths, tgt_batch, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt_batch.shape
        device = src_batch.device

        _, hidden = self.encoder(src_batch, src_lengths)          # hidden: [L, B, H] -- the context

        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=device)
        input_token = tgt_batch[:, 0].unsqueeze(1)                 # <sos>

        for t in range(tgt_len - 1):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=-1, keepdim=True)
            input_token = tgt_batch[:, t + 1].unsqueeze(1) if teacher_force else top1

        return outputs

    # -----------------------------------------------------------------
    # Inference helpers (used by utils.greedy_decode)
    # -----------------------------------------------------------------
    def encode(self, src_tensor, src_lengths):
        encoder_outputs, hidden = self.encoder(src_tensor, src_lengths)
        return encoder_outputs, hidden

    def decode_step(self, prev_token, hidden, encoder_outputs):
        # prev_token: [B] -> reshape to [B, 1] to match decoder's expected input
        input_token = prev_token.unsqueeze(1)
        logits, hidden = self.decoder(input_token, hidden)
        return logits, hidden


def build_model(src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512):
    return EncoderDecoder(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)


def load_model(checkpoint_path, src_vocab, tgt_vocab):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['hparams']
    model = build_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=hparams['embed_size'],
        hidden_size=hparams['hidden_size'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model