"""
train.py
--------
Shared training loop used by all 3 notebooks.

Functions:
    train_epoch  — one pass over the training DataLoader
    train        — full training loop with val evaluation + model saving
"""

import time
import torch
import torch.nn as nn

from src.evaluate import evaluate_model, compute_perplexity


def train_epoch(model, loader, optimizer, criterion, device,
                teacher_forcing_ratio=0.5, model_type="rnn",
                clip=1.0):
    """
    One full pass over the training DataLoader.

    Args:
        model                : Seq2SeqRNN, Seq2SeqAttn, or Seq2SeqTransformer
        loader               : training DataLoader
        optimizer            : torch optimizer
        criterion            : CrossEntropyLoss (ignore_index=PAD)
        device               : torch device
        teacher_forcing_ratio: float, passed to model.forward
        model_type           : "rnn"/"lstm"/"gru", "attention", or "transformer"
        clip                 : gradient clipping max norm

    Returns:
        avg_loss : float
    """
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    for src, tgt, _ in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # tgt input  = all tokens except last  [SOS ... last_word]
        # tgt target = all tokens except first [first_word ... EOS]
        tgt_input  = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        optimizer.zero_grad()

        if model_type == "attention":
            output, _ = model(src, tgt, teacher_forcing_ratio)
        elif model_type == "transformer":
            output = model(src, tgt_input)
        else:
            output = model(src, tgt, teacher_forcing_ratio)
        # output: [batch, tgt_len-1, vocab_size]

        vocab_size  = output.shape[-1]
        output_flat = output.reshape(-1, vocab_size)
        target_flat = tgt_target.reshape(-1)

        loss = criterion(output_flat, target_flat)
        loss.backward()

        # Gradient clipping — critical for RNNs, good practice for all
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()

        non_pad      = (target_flat != criterion.ignore_index).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train(
    model,
    train_loader,
    val_loader,
    model_type: str,
    save_path: str,
    n_epochs: int = 20,
    lr: float = 3e-4,
    clip: float = 1.0,
    teacher_forcing_start: float = 0.5,
    teacher_forcing_end: float = 0.2,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
):
    """
    Full training loop with:
    - Linear teacher forcing decay
    - Validation after every epoch
    - Early stopping
    - Best model saving

    Args:
        model                 : model instance
        train_loader          : training DataLoader
        val_loader            : validation DataLoader
        model_type            : "rnn"/"lstm"/"gru", "attention", "transformer"
        save_path             : path to save best model weights (.pt file)
        n_epochs              : maximum number of epochs
        lr                    : learning rate
        clip                  : gradient clip max norm
        teacher_forcing_start : initial teacher forcing ratio
        teacher_forcing_end   : final teacher forcing ratio (linearly decayed)
        patience              : early stopping patience (epochs without improvement)
        device                : torch device

    Returns:
        history : dict with keys "train_loss", "val_loss", "val_ppl"
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore PAD token

    # Learning rate scheduler — reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  Training {model_type.upper()} | {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Epochs: {n_epochs} | LR: {lr} | Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, n_epochs + 1):
        start = time.time()

        # Linear teacher forcing decay
        tf_ratio = teacher_forcing_start - (
            (teacher_forcing_start - teacher_forcing_end) * (epoch / n_epochs)
        )
        # Transformer doesn't use teacher forcing (it's parallel by design)
        if model_type == "transformer":
            tf_ratio = 0.0

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, tf_ratio, model_type, clip
        )
        val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device, model_type)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"TF: {tf_ratio:.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved → {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    return history