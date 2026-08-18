"""
Lab 10 (A1) - Model 1: Plain RNN (baseline)
==============================================
A single RNN reads the source sentence, its final hidden state seeds a
decoder RNN that generates the target sentence one token at a time.
This is the simplest architecture and expected to be the weakest --
it has to compress the ENTIRE source sentence into one fixed-size hidden
state, which becomes a bottleneck for longer sentences. No attention.
"""

import random
import torch
import torch.nn as nn

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class RNNSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=PAD_IDX)

        # Single-layer GRU for both "encoder" and "decoder" roles
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

        self.fc_out = nn.Linear(hidden_size, tgt_vocab_size)

    # -----------------------------------------------------------------
    # Training forward pass (teacher forcing)
    # -----------------------------------------------------------------
    def forward(self, src_batch, src_lengths, tgt_batch, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt_batch.shape
        device = src_batch.device

        # --- "Encode": run RNN over source, take final hidden state ---
        src_emb = self.src_embedding(src_batch)                     # [B, S, E]
        _, hidden = self.encoder_rnn(src_emb)                        # hidden: [1, B, H]

        # --- Decode step by step ---
        outputs = torch.zeros(batch_size, tgt_len - 1, self.fc_out.out_features, device=device)
        input_token = tgt_batch[:, 0].unsqueeze(1)                   # <sos>, [B, 1]

        for t in range(tgt_len - 1):
            emb = self.tgt_embedding(input_token)                    # [B, 1, E]
            out, hidden = self.decoder_rnn(emb, hidden)               # out: [B, 1, H]
            logits = self.fc_out(out.squeeze(1))                      # [B, V]
            outputs[:, t, :] = logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=-1, keepdim=True)                # [B, 1]
            input_token = tgt_batch[:, t + 1].unsqueeze(1) if teacher_force else top1

        return outputs

    # -----------------------------------------------------------------
    # Inference helpers (used by utils.greedy_decode)
    # -----------------------------------------------------------------
    def encode(self, src_tensor, src_lengths):
        src_emb = self.src_embedding(src_tensor)
        encoder_outputs, hidden = self.encoder_rnn(src_emb)
        # No attention in this model, but we still return encoder_outputs
        # for interface consistency with the attention-based models.
        return encoder_outputs, hidden

    def decode_step(self, prev_token, hidden, encoder_outputs):
        emb = self.tgt_embedding(prev_token).unsqueeze(1)             # [B, 1, E]
        out, hidden = self.decoder_rnn(emb, hidden)
        logits = self.fc_out(out.squeeze(1))                          # [B, V]
        return logits, hidden


def build_model(src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512):
    return RNNSeq2Seq(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)


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