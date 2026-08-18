"""
Lab 10 (A1) - Model 3: Encoder-Decoder + Additive (Bahdanau) Attention
===========================================================================
Same encoder-decoder skeleton as Model 2, but now the decoder does NOT
rely on a single fixed context vector. At each decoding step, it looks
back at ALL encoder hidden states and computes a weighted combination of
them (the "context"), where the weights are learned via a small
feedforward network that scores how well each encoder position aligns
with the decoder's current state. This is the "additive" or "Bahdanau"
attention mechanism (Bahdanau et al., 2015):

    score(s_t, h_i) = v^T * tanh(W1*s_t + W2*h_i)

This directly solves the fixed-length bottleneck that Models 1 & 2 have:
long sentences no longer need to be squeezed into one vector, since the
decoder can "attend" to different parts of the source at each step.
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
        # Bidirectional encoder -> richer representations for attention to draw from
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        # Project concatenated bidirectional hidden state down to hidden_size
        # so it can be used to initialize the (unidirectional) decoder.
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src_batch, src_lengths):
        emb = self.embedding(src_batch)                          # [B, S, E]
        outputs, hidden = self.rnn(emb)                           # outputs: [B, S, 2H], hidden: [2*L, B, H]

        # Combine forward+backward final hidden states -> [B, H], then
        # give it a layer dim of 1 to seed the decoder's GRU.
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)   # [B, 2H]
        hidden_proj = torch.tanh(self.fc_hidden(hidden_cat))       # [B, H]
        hidden_proj = hidden_proj.unsqueeze(0)                     # [1, B, H]

        return outputs, hidden_proj                                 # outputs: [B, S, 2H]


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention: score = v^T tanh(W1*s + W2*h)"""
    def __init__(self, hidden_size, encoder_output_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)          # for decoder state
        self.W2 = nn.Linear(encoder_output_size, hidden_size)   # for encoder outputs
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: [1, B, H] -> [B, H] -> [B, 1, H]
        dec_h = decoder_hidden[-1].unsqueeze(1)                   # [B, 1, H]
        # encoder_outputs: [B, S, 2H]
        scores = self.v(torch.tanh(self.W1(dec_h) + self.W2(encoder_outputs)))  # [B, S, 1]
        scores = scores.squeeze(-1)                                 # [B, S]

        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)                    # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, 2H]
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_output_size, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.attention = AdditiveAttention(hidden_size, encoder_output_size)
        # GRU input = embedding + attended context
        self.rnn = nn.GRU(embed_size + encoder_output_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size + encoder_output_size + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs, src_mask=None):
        emb = self.dropout(self.embedding(input_token))            # [B, 1, E]

        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)  # [B,1,2H]

        rnn_input = torch.cat([emb, context], dim=2)                 # [B, 1, E+2H]
        out, hidden = self.rnn(rnn_input, hidden)                     # out: [B, 1, H]

        logits = self.fc_out(torch.cat([out, context, emb], dim=2).squeeze(1))  # [B, V]
        return logits, hidden, attn_weights


class Seq2SeqAdditiveAttention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, dropout=0.2):
        super().__init__()
        encoder_output_size = hidden_size * 2  # bidirectional encoder
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, encoder_output_size, dropout=dropout)
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
        input_token = tgt_batch[:, 0].unsqueeze(1)                    # <sos>

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
        input_token = prev_token.unsqueeze(1)                          # [B, 1]
        # src_mask is None at inference time (single sentence, no padding assumed)
        logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask=None)
        return logits, hidden


def build_model(src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512):
    return Seq2SeqAdditiveAttention(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)


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