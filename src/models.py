"""
models.py
---------
All three architecture classes for the Medical Jargon Simplifier.

Models:
    1. Seq2SeqRNN   — vanilla encoder-decoder, supports RNN / LSTM / GRU cell
    2. Seq2SeqAttn  — BiLSTM encoder + Bahdanau attention + LSTM decoder
    3. Seq2SeqTransformer — standard Transformer encoder-decoder

All models share the same interface:
    forward(src, tgt) → logits [batch, tgt_len, vocab_size]

This makes the training loop in train.py identical for all three.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. VANILLA SEQ2SEQ  (RNN / LSTM / GRU)
# ─────────────────────────────────────────────────────────────────────────────

class RNNEncoder(nn.Module):
    """
    Encoder using a vanilla RNN, LSTM, or GRU cell.

    Args:
        vocab_size  : source vocabulary size
        embed_size  : embedding dimension
        hidden_size : hidden state dimension
        cell_type   : one of "rnn", "lstm", "gru"
        n_layers    : number of stacked RNN layers
        dropout     : dropout probability
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 cell_type="lstm", n_layers=2, dropout=0.3):
        super().__init__()
        self.cell_type   = cell_type
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            embed_size, hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

    def forward(self, src):
        """
        Args:
            src : [batch, src_len]
        Returns:
            hidden : final hidden state  [n_layers, batch, hidden]
            cell   : final cell state    [n_layers, batch, hidden]  (None for RNN/GRU)
        """
        embedded = self.dropout(self.embedding(src))   # [batch, src_len, embed]
        outputs, hidden = self.rnn(embedded)

        if self.cell_type == "lstm":
            hidden, cell = hidden
            return hidden, cell
        else:
            return hidden, None


class RNNDecoder(nn.Module):
    """
    Decoder using the same cell type as the encoder.
    Takes previous token + previous hidden state → next token logits.

    Args:
        vocab_size  : target vocabulary size
        embed_size  : embedding dimension
        hidden_size : hidden state dimension
        cell_type   : one of "rnn", "lstm", "gru"
        n_layers    : number of stacked RNN layers
        dropout     : dropout probability
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 cell_type="lstm", n_layers=2, dropout=0.3):
        super().__init__()
        self.cell_type = cell_type

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            embed_size, hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_token, hidden, cell=None):
        """
        Args:
            tgt_token : [batch, 1]   current input token
            hidden    : [n_layers, batch, hidden]
            cell      : [n_layers, batch, hidden]  (None for RNN/GRU)
        Returns:
            logits  : [batch, 1, vocab_size]
            hidden  : updated hidden state
            cell    : updated cell state (None for RNN/GRU)
        """
        embedded = self.dropout(self.embedding(tgt_token))  # [batch, 1, embed]

        if self.cell_type == "lstm":
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            output, hidden = self.rnn(embedded, hidden)
            cell = None

        logits = self.fc_out(output)   # [batch, 1, vocab_size]
        return logits, hidden, cell


class Seq2SeqRNN(nn.Module):
    """
    Full vanilla Seq2Seq model with RNN / LSTM / GRU.

    Supports teacher forcing during training.
    cell_type controls which recurrent cell is used in both encoder and decoder.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_size=256, hidden_size=512,
                 cell_type="lstm", n_layers=2, dropout=0.3,
                 sos_idx=1, eos_idx=2):
        super().__init__()
        self.sos_idx   = sos_idx
        self.eos_idx   = eos_idx
        self.cell_type = cell_type

        self.encoder = RNNEncoder(
            src_vocab_size, embed_size, hidden_size, cell_type, n_layers, dropout
        )
        self.decoder = RNNDecoder(
            tgt_vocab_size, embed_size, hidden_size, cell_type, n_layers, dropout
        )

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src                  : [batch, src_len]
            tgt                  : [batch, tgt_len]  includes SOS at position 0
            teacher_forcing_ratio: probability of using ground truth token

        Returns:
            outputs : [batch, tgt_len-1, vocab_size]
        """
        import random

        batch_size  = src.shape[0]
        tgt_len     = tgt.shape[1]
        vocab_size  = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        hidden, cell = self.encoder(src)

        # First decoder input = SOS token
        dec_input = tgt[:, 0].unsqueeze(1)   # [batch, 1]

        for t in range(tgt_len - 1):
            logits, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t, :] = logits.squeeze(1)

            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher:
                dec_input = tgt[:, t + 1].unsqueeze(1)
            else:
                dec_input = logits.argmax(2)   # [batch, 1]

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEQ2SEQ + BAHDANAU ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.

    At each decoder step t:
        energy_i = v · tanh(W1·h_i + W2·s_{t-1})
        alpha    = softmax(energy)
        context  = Σ alpha_i · h_i

    h_i   : encoder hidden state at position i
    s_t-1 : previous decoder hidden state
    """

    def __init__(self, enc_hidden_size, dec_hidden_size, attn_size=256):
        super().__init__()
        self.W1 = nn.Linear(enc_hidden_size * 2, attn_size, bias=False)
        self.W2 = nn.Linear(dec_hidden_size,     attn_size, bias=False)
        self.v  = nn.Linear(attn_size, 1,         bias=False)

    def forward(self, enc_outputs, dec_hidden):
        """
        Args:
            enc_outputs : [batch, src_len, 2*enc_hidden]
            dec_hidden  : [batch, dec_hidden]   (single layer)

        Returns:
            context     : [batch, 2*enc_hidden]
            weights     : [batch, src_len]
        """
        src_len = enc_outputs.shape[1]

        # Expand decoder hidden to match encoder sequence length
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # dec_hidden: [batch, src_len, dec_hidden]

        energy = torch.tanh(self.W1(enc_outputs) + self.W2(dec_hidden))
        # energy: [batch, src_len, attn_size]

        scores  = self.v(energy).squeeze(2)         # [batch, src_len]
        weights = torch.softmax(scores, dim=1)       # [batch, src_len]

        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        # context: [batch, 2*enc_hidden]

        return context, weights


class AttnEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for the attention model.
    Final hidden/cell states are projected to decoder hidden size.
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding  = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout    = nn.Dropout(dropout)
        self.lstm       = nn.LSTM(
            embed_size, hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        # Project bidirectional → single direction for decoder init
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell   = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        """
        Args:
            src : [batch, src_len]
        Returns:
            enc_outputs : [batch, src_len, 2*hidden]
            hidden      : [batch, hidden]   projected
            cell        : [batch, hidden]   projected
        """
        embedded = self.dropout(self.embedding(src))      # [batch, src_len, embed]
        enc_outputs, (hidden, cell) = self.lstm(embedded)
        # enc_outputs: [batch, src_len, 2*hidden]
        # hidden:      [2*n_layers, batch, hidden]

        # Take last layer forward + backward states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, 2*hidden]
        cell   = torch.cat([cell[-2],   cell[-1]],   dim=1)

        hidden = torch.tanh(self.fc_hidden(hidden))          # [batch, hidden]
        cell   = torch.tanh(self.fc_cell(cell))

        return enc_outputs, hidden, cell


class AttnDecoder(nn.Module):
    """
    LSTM decoder with Bahdanau attention.

    At each step:
        1. Embed previous output token
        2. Compute attention → context vector
        3. Concat [embed, context] → LSTM input
        4. Project LSTM output → vocabulary logits
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 enc_hidden_size, attn_size=256, dropout=0.3):
        super().__init__()
        self.attention = BahdanauAttention(enc_hidden_size, hidden_size, attn_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.lstm      = nn.LSTM(
            embed_size + enc_hidden_size * 2,
            hidden_size,
            batch_first=True,
        )
        self.fc_out = nn.Linear(
            hidden_size + enc_hidden_size * 2 + embed_size,
            vocab_size
        )

    def forward(self, tgt_token, hidden, cell, enc_outputs):
        """
        Args:
            tgt_token   : [batch, 1]
            hidden      : [batch, hidden]
            cell        : [batch, hidden]
            enc_outputs : [batch, src_len, 2*enc_hidden]

        Returns:
            logits   : [batch, 1, vocab_size]
            hidden   : [batch, hidden]
            cell     : [batch, hidden]
            weights  : [batch, src_len]  attention weights
        """
        embedded = self.dropout(self.embedding(tgt_token))  # [batch, 1, embed]

        context, weights = self.attention(enc_outputs, hidden)
        # context: [batch, 2*enc_hidden]

        lstm_input = torch.cat(
            [embedded, context.unsqueeze(1)], dim=2
        )   # [batch, 1, embed + 2*enc_hidden]

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden.unsqueeze(0), cell.unsqueeze(0))
        )
        # output: [batch, 1, hidden]

        output   = output.squeeze(1)    # [batch, hidden]
        embedded = embedded.squeeze(1)  # [batch, embed]

        prediction = self.fc_out(
            torch.cat([output, context, embedded], dim=1)
        )   # [batch, vocab_size]

        hidden = hidden.squeeze(0)  # [batch, hidden]
        cell   = cell.squeeze(0)

        return prediction.unsqueeze(1), hidden, cell, weights


class Seq2SeqAttn(nn.Module):
    """
    Full Seq2Seq model with Bahdanau attention.
    Encoder: BiLSTM
    Decoder: LSTM + additive attention
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_size=256, hidden_size=512,
                 attn_size=256, n_layers=1, dropout=0.3,
                 sos_idx=1, eos_idx=2):
        super().__init__()
        self.sos_idx = sos_idx

        self.encoder = AttnEncoder(
            src_vocab_size, embed_size, hidden_size, n_layers, dropout
        )
        self.decoder = AttnDecoder(
            tgt_vocab_size, embed_size, hidden_size,
            hidden_size, attn_size, dropout
        )

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src : [batch, src_len]
            tgt : [batch, tgt_len]

        Returns:
            outputs      : [batch, tgt_len-1, vocab_size]
            attn_weights : [batch, tgt_len-1, src_len]
        """
        import random

        batch_size = src.shape[0]
        tgt_len    = tgt.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        src_len    = src.shape[1]

        outputs      = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)
        attn_weights = torch.zeros(batch_size, tgt_len - 1, src_len,    device=src.device)

        enc_outputs, hidden, cell = self.encoder(src)
        dec_input = tgt[:, 0].unsqueeze(1)   # SOS

        for t in range(tgt_len - 1):
            logits, hidden, cell, weights = self.decoder(
                dec_input, hidden, cell, enc_outputs
            )
            outputs[:, t, :]      = logits.squeeze(1)
            attn_weights[:, t, :] = weights

            use_teacher = random.random() < teacher_forcing_ratio
            dec_input   = tgt[:, t + 1].unsqueeze(1) if use_teacher else logits.argmax(2)

        return outputs, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSFORMER ENCODER-DECODER
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — injects position information
    into token embeddings since Transformer has no recurrence.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices

        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            x + positional encoding, with dropout
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    Standard Transformer encoder-decoder for sequence-to-sequence tasks.

    Architecture follows "Attention Is All You Need" (Vaswani et al., 2017)
    — simplified to basic encoder-decoder with multi-head self-attention
    and cross-attention.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        d_model        : embedding / model dimension (must be divisible by n_heads)
        n_heads        : number of attention heads
        n_encoder_layers: number of Transformer encoder layers
        n_decoder_layers: number of Transformer decoder layers
        d_ff           : feed-forward inner dimension
        dropout        : dropout probability
        max_len        : maximum sequence length for positional encoding
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, n_heads=8,
                 n_encoder_layers=3, n_decoder_layers=3,
                 d_ff=512, dropout=0.1, max_len=512,
                 sos_idx=1, pad_idx=0):
        super().__init__()
        self.d_model  = d_model
        self.sos_idx  = sos_idx
        self.pad_idx  = pad_idx

        # Embeddings + positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding  = PositionalEncoding(d_model, dropout, max_len)

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying — share embedding weights with output projection
        # (standard practice, improves performance + reduces parameters)
        self.fc_out.weight = self.tgt_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Causal mask for decoder — prevents attending to future tokens.
        Upper triangle = -inf, diagonal and below = 0.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def encode(self, src, src_key_padding_mask=None):
        """Run encoder only — used during inference."""
        src_emb = self.pos_encoding(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        return self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, tgt, memory, tgt_mask=None, src_key_padding_mask=None):
        """Run decoder only — used during inference."""
        tgt_emb = self.pos_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )
        return self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

    def forward(self, src, tgt):
        """
        Args:
            src : [batch, src_len]
            tgt : [batch, tgt_len]   includes SOS, excludes last EOS

        Returns:
            logits : [batch, tgt_len, vocab_size]
        """
        src_key_padding_mask = (src == self.pad_idx)   # [batch, src_len]
        tgt_key_padding_mask = (tgt == self.pad_idx)   # [batch, tgt_len]
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(src.device)

        src_emb = self.pos_encoding(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.pos_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )

        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        # output: [batch, tgt_len, d_model]

        logits = self.fc_out(output)   # [batch, tgt_len, vocab_size]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, vocab_size: int, **kwargs):
    """
    Convenience factory function.

    Args:
        model_type : one of "rnn", "lstm", "gru", "attention", "transformer"
        vocab_size : shared vocab size (src and tgt use same vocab here)
        **kwargs   : hyperparameters passed to the model constructor

    Returns:
        model instance
    """
    if model_type in ("rnn", "lstm", "gru"):
        return Seq2SeqRNN(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            cell_type=model_type,
            **kwargs
        )
    elif model_type == "attention":
        return Seq2SeqAttn(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            **kwargs
        )
    elif model_type == "transformer":
        return Seq2SeqTransformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose from: rnn, lstm, gru, attention, transformer")