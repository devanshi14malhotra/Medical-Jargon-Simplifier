"""
dataset.py
----------
Shared data loading, preprocessing, vocabulary building, and dataset classes
used across all 3 training notebooks.

Usage:
    from src.dataset import load_data, Vocabulary, MedicalDataset, collate_fn
"""

import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────────────────────
# SPECIAL TOKENS
# ─────────────────────────────────────────────────────────────────────────────

PAD_TOKEN = "<pad>"   # padding
SOS_TOKEN = "<sos>"   # start of sequence
EOS_TOKEN = "<eos>"   # end of sequence
UNK_TOKEN = "<unk>"   # unknown word

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic text cleaning pipeline:
    - lowercase
    - remove special characters except basic punctuation
    - collapse multiple spaces
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\.\,\?\!\'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer. Returns list of tokens."""
    return clean_text(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Word-level vocabulary built from a list of sentences.

    Attributes:
        word2idx : dict mapping word → integer index
        idx2word : dict mapping integer index → word
        n_words   : total vocabulary size
    """

    def __init__(self):
        self.word2idx = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.n_words = 4

    def build(self, sentences: list, min_freq: int = 1):
        """
        Build vocab from a list of raw sentences.

        Args:
            sentences : list of strings
            min_freq  : minimum word frequency to include (default 1)
        """
        from collections import Counter
        freq = Counter()
        for sentence in sentences:
            freq.update(tokenize(sentence))

        for word, count in freq.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

        print(f"Vocabulary built: {self.n_words} words")
        return self

    def encode(self, sentence: str) -> list:
        """Convert sentence string → list of token indices. Adds EOS at end."""
        tokens = tokenize(sentence)
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        return indices + [EOS_IDX]

    def decode(self, indices: list, skip_special: bool = True) -> str:
        """Convert list of token indices → sentence string."""
        words = []
        for idx in indices:
            token = self.idx2word.get(idx, UNK_TOKEN)
            if skip_special and token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                continue
            if token == EOS_TOKEN:
                break
            words.append(token)
        return " ".join(words)

    def __len__(self):
        return self.n_words


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    train_path: str,
    val_path: str,
    test_path: str,
    src_col: str = "Expert",
    tgt_col: str = "Simple",
    max_len: int = 100,
    min_freq: int = 1,
):
    """
    Load CSVs, clean text, build shared vocabulary, return splits.

    Args:
        train_path : path to train.csv
        val_path   : path to validation.csv
        test_path  : path to test.csv
        src_col    : column name for complex medical text
        tgt_col    : column name for simplified text
        max_len    : max token length per sentence (longer pairs dropped)
        min_freq   : minimum word frequency for vocab inclusion

    Returns:
        train_pairs, val_pairs, test_pairs : list of (src_str, tgt_str) tuples
        vocab : shared Vocabulary object (built on train only)
    """
    train_df = pd.read_csv(train_path)[[src_col, tgt_col]].dropna()
    val_df   = pd.read_csv(val_path)[[src_col, tgt_col]].dropna()
    test_df  = pd.read_csv(test_path)[[src_col, tgt_col]].dropna()

    def filter_length(df):
        mask = (
            df[src_col].apply(lambda x: len(tokenize(x)) <= max_len) &
            df[tgt_col].apply(lambda x: len(tokenize(x)) <= max_len)
        )
        return df[mask]

    train_df = filter_length(train_df)
    val_df   = filter_length(val_df)
    test_df  = filter_length(test_df)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Build vocab on train only — prevents data leakage
    all_train_sentences = (
        train_df[src_col].tolist() + train_df[tgt_col].tolist()
    )
    vocab = Vocabulary().build(all_train_sentences, min_freq=min_freq)

    def to_pairs(df):
        return list(zip(df[src_col].tolist(), df[tgt_col].tolist()))

    return to_pairs(train_df), to_pairs(val_df), to_pairs(test_df), vocab


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MedicalDataset(Dataset):
    """
    PyTorch Dataset for medical jargon simplification.

    Each item returns:
        src : tensor of token indices for the complex sentence
        tgt : tensor of token indices for the simple sentence (with SOS prepended)
    """

    def __init__(self, pairs: list, vocab: Vocabulary):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.pairs[idx]

        src = torch.tensor(self.vocab.encode(src_str), dtype=torch.long)

        # Target: SOS + tokens + EOS
        tgt_indices = [SOS_IDX] + self.vocab.encode(tgt_str)
        tgt = torch.tensor(tgt_indices, dtype=torch.long)

        return src, tgt


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION (for DataLoader batching)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads sequences in a batch to the same length.

    Args:
        batch : list of (src_tensor, tgt_tensor) tuples

    Returns:
        src_padded : [batch_size, src_max_len]
        tgt_padded : [batch_size, tgt_max_len]
        src_lengths: list of original src lengths (useful for packed sequences)
    """
    src_batch, tgt_batch = zip(*batch)

    src_lengths = [len(s) for s in src_batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return src_padded, tgt_padded, src_lengths


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    train_pairs, val_pairs, test_pairs,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Build train/val/test DataLoaders.

    Args:
        train_pairs, val_pairs, test_pairs : list of (src, tgt) string tuples
        vocab       : shared Vocabulary
        batch_size  : number of samples per batch
        num_workers : parallel data loading workers (0 = main process)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = MedicalDataset(train_pairs, vocab)
    val_ds   = MedicalDataset(val_pairs,   vocab)
    test_ds  = MedicalDataset(test_pairs,  vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_pairs, val_pairs, test_pairs, vocab = load_data(
        train_path="data/train.csv",
        val_path="data/validation.csv",
        test_path="data/test.csv",
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_pairs, val_pairs, test_pairs, vocab, batch_size=32
    )

    src_batch, tgt_batch, src_lengths = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  src shape : {src_batch.shape}")
    print(f"  tgt shape : {tgt_batch.shape}")
    print(f"  src lengths: {src_lengths[:5]}")

    # Decode first sample to verify
    print(f"\nFirst sample decoded:")
    print(f"  SRC: {vocab.decode(src_batch[0].tolist())}")
    print(f"  TGT: {vocab.decode(tgt_batch[0].tolist())}")