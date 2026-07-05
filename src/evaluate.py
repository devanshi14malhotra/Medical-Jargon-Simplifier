"""
evaluate.py
-----------
Shared evaluation functions used across all notebooks and app.py.

Metrics:
    - BLEU score  (nltk implementation, corpus-level and sentence-level)
    - Perplexity  (derived from cross-entropy loss)
    - Inference   (greedy decoding for a single input sentence)
"""

import math
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from src.dataset import Vocabulary, SOS_IDX, EOS_IDX, PAD_IDX, tokenize


# ─────────────────────────────────────────────────────────────────────────────
# PERPLEXITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(loss: float) -> float:
    """
    Perplexity = exp(cross_entropy_loss).
    Lower is better. Perfect model = 1.0, random = vocab_size.

    Args:
        loss : average cross-entropy loss over a dataset split

    Returns:
        perplexity as float
    """
    return math.exp(min(loss, 100))  # cap to avoid overflow


# ─────────────────────────────────────────────────────────────────────────────
# BLEU SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(references: list, hypotheses: list) -> float:
    """
    Corpus-level BLEU score.

    Args:
        references  : list of reference strings (ground truth)
        hypotheses  : list of hypothesis strings (model output)

    Returns:
        BLEU score as float between 0 and 1
    """
    refs = [[tokenize(ref)] for ref in references]
    hyps = [tokenize(hyp) for hyp in hypotheses]

    smoothie = SmoothingFunction().method1
    score = corpus_bleu(refs, hyps, smoothing_function=smoothie)
    return round(score, 4)


def compute_sentence_bleu(reference: str, hypothesis: str) -> float:
    """
    Sentence-level BLEU score for a single pair.
    Used in the Streamlit app for per-prediction scoring.
    """
    ref  = [tokenize(reference)]
    hyp  = tokenize(hypothesis)
    smoothie = SmoothingFunction().method1
    return round(sentence_bleu(ref, hyp, smoothing_function=smoothie), 4)


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(model, src_tensor: torch.Tensor, vocab: Vocabulary,
                  device: torch.device, max_len: int = 100,
                  model_type: str = "rnn") -> tuple:
    """
    Run greedy decoding for a single source sentence.

    Args:
        model      : trained model (RNN/LSTM/GRU, Attention, or Transformer)
        src_tensor : [src_len] token index tensor
        vocab      : shared Vocabulary
        device     : torch device
        max_len    : maximum output length
        model_type : one of "rnn", "attention", "transformer"

    Returns:
        translation : decoded output string
        attn_weights: attention weight matrix (None if model_type != "attention")
    """
    model.eval()
    attn_weights = None

    with torch.no_grad():
        src = src_tensor.unsqueeze(0).to(device)   # [1, src_len]

        if model_type in ("rnn", "lstm", "gru"):
            hidden, cell = model.encoder(src)
            dec_input = torch.tensor([[SOS_IDX]], device=device)
            tokens = []

            for _ in range(max_len):
                output, hidden, cell = model.decoder(dec_input, hidden, cell)
                top1 = output.argmax(2)
                token_idx = top1.item()

                if token_idx == EOS_IDX:
                    break

                tokens.append(token_idx)
                dec_input = top1

            translation = vocab.decode(tokens)

        elif model_type == "attention":
            enc_outputs, hidden, cell = model.encoder(src)
            dec_input = torch.tensor([[SOS_IDX]], device=device)
            tokens = []
            attn_weights = []
            prev_token = None
            repeat_count = 0

            for _ in range(max_len):
                output, hidden, cell, weights = model.decoder(
                    dec_input, hidden, cell, enc_outputs
                )
                token_idx = output.argmax(2).item()

                if token_idx == EOS_IDX:
                    break

                # Stop if same token repeats 3 times in a row
                if token_idx == prev_token:
                    repeat_count += 1
                    if repeat_count >= 3:
                        break
                else:
                    repeat_count = 0

                prev_token = token_idx
                tokens.append(token_idx)
                attn_weights.append(weights.squeeze().cpu().tolist())
                dec_input = torch.tensor([[token_idx]], device=device)

            translation = vocab.decode(tokens)

        elif model_type == "transformer":
            src_key_padding_mask = (src == PAD_IDX)
            memory = model.encode(src, src_key_padding_mask)

            tgt = torch.tensor([[SOS_IDX]], device=device)
            tokens = []

            for _ in range(max_len):
                tgt_mask = model.generate_square_subsequent_mask(
                    tgt.size(1)
                ).to(device)

                output = model.decode(tgt, memory, tgt_mask, src_key_padding_mask)
                logits = model.fc_out(output[:, -1, :])
                top1 = logits.argmax(1).unsqueeze(0)
                token_idx = top1.item()

                if token_idx == EOS_IDX:
                    break

                tokens.append(token_idx)
                tgt = torch.cat([tgt, top1], dim=1)

            translation = vocab.decode(tokens)

    return translation, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ON FULL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, criterion, device: torch.device, model_type: str="rnn") -> tuple:
    """
    Compute average loss and perplexity over a DataLoader split.

    Args:
        model     : trained model
        loader    : DataLoader (val or test)
        criterion : loss function (CrossEntropyLoss)
        device    : torch device

    Returns:
        avg_loss    : float
        perplexity  : float
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt, _ in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # tgt input = all tokens except last
            # tgt target = all tokens except first (SOS)
            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            if model_type == "attention":
                output, _ = model(src,tgt)
            elif model_type == "transformer":
                output = model(src, tgt_input)
            else:
                output = model(src, tgt)
            # output: [batch, tgt_len, vocab_size]

            vocab_size = output.shape[-1]
            output_flat = output.reshape(-1, vocab_size)
            target_flat = tgt_target.reshape(-1)

            loss = criterion(output_flat, target_flat)

            # Count non-padding tokens
            non_pad = (target_flat != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE TRANSLATION TABLE (for NB4 comparison)
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_table(models: dict, test_pairs: list,
                               vocab: Vocabulary, device: torch.device,
                               n_samples: int = 20) -> list:
    """
    Run all models on the same n_samples test sentences.
    Returns a list of dicts for easy DataFrame construction in NB4.

    Args:
        models     : dict of {model_name: (model, model_type)}
        test_pairs : list of (src_str, tgt_str) tuples
        vocab      : shared Vocabulary
        device     : torch device
        n_samples  : number of test sentences to evaluate

    Returns:
        list of dicts with keys:
            source, reference, and one key per model name
    """
    import random
    samples = random.sample(test_pairs, min(n_samples, len(test_pairs)))
    results = []

    for src_str, tgt_str in samples:
        row = {"source": src_str, "reference": tgt_str}

        src_tensor = torch.tensor(vocab.encode(src_str), dtype=torch.long)

        for model_name, (model, model_type) in models.items():
            translation, _ = greedy_decode(
                model, src_tensor, vocab, device, model_type=model_type
            )
            row[model_name] = translation

        results.append(row)

    return results