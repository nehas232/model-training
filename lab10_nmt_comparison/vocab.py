"""
Lab 10 (A1): Shared Vocab class
==================================
Moved out of preprocess.py into its own module so that pickle can
correctly resolve `Vocab` when processed_data.pkl is loaded from a
DIFFERENT script (train.py, evaluate.py) than the one that created it
(preprocess.py). Pickle ties a class to the module it was defined in --
keeping it in one shared file avoids "module has no attribute Vocab" errors.
"""

from collections import Counter

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = '<pad>', '<sos>', '<eos>', '<unk>'


class Vocab:
    def __init__(self, sentences, min_freq=2):
        counter = Counter()
        for s in sentences:
            counter.update(s.split())

        self.itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.itos += [w for w, c in counter.items() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, add_sos_eos=True):
        ids = [self.stoi.get(w, self.stoi[UNK_TOKEN]) for w in sentence.split()]
        if add_sos_eos:
            ids = [self.stoi[SOS_TOKEN]] + ids + [self.stoi[EOS_TOKEN]]
        return ids

    def decode(self, ids):
        words = [self.itos[i] for i in ids
                 if self.itos[i] not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)]
        return ' '.join(words)