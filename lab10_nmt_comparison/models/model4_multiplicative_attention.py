"""
Lab 10 (A1) - Model 4: Encoder-Decoder + Multiplicative (Luong) Attention
=============================================================================
Same idea as Model 3 (decoder attends over all encoder states instead of
relying on one fixed context vector), but the alignment SCORE is computed
differently. Instead of a small feedforward network (additive/Bahdanau),
Luong's "general" multiplicative attention scores alignment via a single
learned bilinear form:

    score(s_t, h_i) = s_t^T * W * h_i

This is cheaper to compute than additive attention (one matrix multiply
vs. a feedforward network with a nonlinearity), and in practice often
performs comparably or better. Also, unlike Model 3, Luong attention is
applied AFTER the decoder RNN step (not before), and Luong's original
paper uses a unidirectional encoder -- both differences are kept here to
reflect the actual architectural distinction described in the literature,
which is worth discussing in your report.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        # Unidirectional encoder, as in Luong et al. (2015)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src_batch, src_lengths):
        emb = self.embedding(src_batch)                       # [B, S, E]
        outputs, hidden = self.rnn(emb)                          # outputs: [B, S, H], hidden: [1, B, H]
        return outputs, hidden


class MultiplicativeAttention(nn.Module):
    """Luong-style 'general' multiplicative attention: score = s^T W h"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: [1, B, H] -> [B, H]
        dec_h = decoder_hidden[-1]                                  # [B, H]
        # Project encoder outputs: [B, S, H] -> [B, S, H]
        proj = self.W(encoder_outputs)
        # Bilinear score: [B, S, H] x [B, H, 1] -> [B, S, 1] -> [B, S]
        scores = torch.bmm(proj, dec_h.unsqueeze(2)).squeeze(2)     # [B, S]

        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)                     # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, H]
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.attention = MultiplicativeAttention(hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        # Luong-style: combine RNN output + context AFTER the RNN step,
        # via a "concat" layer, then predict from that.
        self.fc_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, src_mask=None):
        emb = self.dropout(self.embedding(input_token))              # [B, 1, E]
        rnn_out, hidden = self.rnn(emb, hidden)                        # rnn_out: [B, 1, H]

        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)  # [B,1,H]

        concat = torch.cat([rnn_out, context], dim=2).squeeze(1)        # [B, 2H]
        attended_hidden = torch.tanh(self.fc_concat(concat))             # [B, H]
        logits = self.fc_out(attended_hidden)                             # [B, V]

        return logits, hidden, attn_weights


class Seq2SeqMultiplicativeAttention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, dropout=dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def make_src_mask(self, src_batch):
        return (src_batch != PAD_IDX).float()  # [B, S]

    # -----------------------------------------------------------------
    # Training forward pass (teacher forcing)
    # -----------------------------------------------------------------
    def forward(self, src_batch, src_lengths, tgt_batch, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt_batch.shape
        device = src_batch.device

        encoder_outputs, hidden = self.encoder(src_batch, src_lengths)
        src_mask = self.make_src_mask(src_batch)

        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=device)
        input_token = tgt_batch[:, 0].unsqueeze(1)                       # <sos>

        for t in range(tgt_len - 1):
            logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
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
        input_token = prev_token.unsqueeze(1)                             # [B, 1]
        logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask=None)
        return logits, hidden


def build_model(src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512):
    return Seq2SeqMultiplicativeAttention(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)


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