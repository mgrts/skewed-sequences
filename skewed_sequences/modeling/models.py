import math

import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10_000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerWithPE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        max_len: int = 10_000,
    ) -> None:
        super().__init__()

        self.encoder_embed = nn.Linear(in_dim, embed_dim)
        self.decoder_embed = nn.Linear(out_dim, embed_dim)

        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Linear(embed_dim, out_dim)
        # Non-persistent: the cache is a derived lookup table, not a learned
        # parameter. Persisting it would add/remove a state_dict key depending on
        # whether forward() has run, breaking load_state_dict into a fresh model.
        self.register_buffer("cached_mask", None, persistent=False)

    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        if self.cached_mask is None or self.cached_mask.size(0) < sz:
            mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
            self.cached_mask = mask
        return self.cached_mask[:sz, :sz]

    def _run_decoder(self, src: Tensor, decoder_in: Tensor) -> Tensor:
        """Encode ``src`` and decode an already-shifted ``decoder_in``.

        ``decoder_in`` position i is the token used to predict output position i;
        no internal shift happens here, so this op is shared verbatim by the
        teacher-forced ``forward`` and the autoregressive ``infer``.
        """
        src = self.pos_encoding(self.encoder_embed(src))
        decoder_in = self.pos_encoding(self.decoder_embed(decoder_in))

        tgt_mask = self._generate_square_subsequent_mask(decoder_in.size(1)).to(decoder_in.device)

        output = self.transformer(src, decoder_in, tgt_mask=tgt_mask)
        return self.output_layer(output)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """Teacher-forced forward with a one-step-shifted decoder input.

        The decoder is seeded with the last observed context value and fed the
        *previous* ground-truth targets, so output position i predicts ``tgt[i]``
        from ``tgt[<i]`` (and ``src``) only — never from ``tgt[i]`` itself. This
        matches the generation process in ``infer`` and removes the otherwise
        trivial copy of the decoder input (which, at OUTPUT_LENGTH=1, would leak
        the single value being predicted).

        Args:
            src: (B, src_len, in_dim)
            tgt: (B, tgt_len, out_dim)
        Returns:
            Tensor: (B, tgt_len, out_dim)
        """
        out_dim = self.output_layer.out_features
        seed = src[:, -1:, :out_dim]  # last observed value, (B, 1, out_dim)
        decoder_in = torch.cat([seed, tgt[:, :-1, :]], dim=1)  # shift right by one
        return self._run_decoder(src, decoder_in)

    def infer(self, src: Tensor, tgt_len: int) -> Tensor:
        """
        Args:
            src: (B, src_len, in_dim)
            tgt_len: int
        Returns:
            Tensor: (B, tgt_len, out_dim)
        """
        out_dim = self.output_layer.out_features

        # Seed from the last observed value — the same seed the shifted forward
        # uses for output position 0.
        decoder_input = src[:, -1:, :out_dim]  # (B, 1, out_dim)

        outputs = []
        for _ in range(tgt_len):
            next_token = self._run_decoder(src, decoder_input)[:, -1:]  # (B, 1, out_dim)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            outputs.append(next_token)

        return torch.cat(outputs, dim=1)  # (B, tgt_len, out_dim)


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Autoregressive LSTM decoder, mimicking Transformer-style causal decoding.

        Uses one-step-shifted teacher forcing: the decoder is seeded with the
        last observed context step and then fed the *previous* ground-truth
        target, so step t predicts ``tgt[:,t]`` from ``tgt[:,t-1]`` (or the seed)
        — never from ``tgt[:,t]`` itself. This matches ``infer`` and avoids the
        identity leak of feeding ``tgt[:,t]`` to predict ``tgt[:,t]``.

        Args:
            src: (B, src_len, input_dim)
            tgt: (B, tgt_len, output_dim) — ground-truth targets for teacher forcing

        Returns:
            Tensor: (B, tgt_len, output_dim)
        """
        B, tgt_len, _ = tgt.size()
        device = src.device

        # Encode the src sequence
        h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        _, (h, c) = self.lstm(src, (h0, c0))  # Only get final hidden and cell states

        # Seed the decoder with the last observed context step (same as infer()).
        outputs = []
        input_t = src[:, -1:, :]  # (B, 1, input_dim)

        for t in range(tgt_len):
            out, (h, c) = self.lstm(input_t, (h, c))  # Step input
            pred = self.fc(out)  # (B, 1, output_dim)
            outputs.append(pred)

            # Teacher forcing — next input is the ground-truth target at step t.
            if t + 1 < tgt_len:
                input_t = tgt[:, t].unsqueeze(1)

        return torch.cat(outputs, dim=1)  # (B, tgt_len, output_dim)

    def infer(self, src: Tensor, tgt_len: int) -> Tensor:
        """
        Autoregressive inference: predict ``tgt_len`` steps into the future
        based on the source sequence.

        Args:
            src: Tensor of shape (B, src_len, input_dim)
            tgt_len: number of future steps to predict

        Returns:
            Tensor of shape (B, tgt_len, output_dim)
        """
        B = src.size(0)
        device = src.device

        # Encode the source sequence to get the initial hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        _, (h, c) = self.lstm(src, (h0, c0))

        # Use the last time step from src as the first decoder input
        input_t = src[:, -1:].detach()  # (B, 1, input_dim)

        outputs = []

        for _ in range(tgt_len):
            out, (h, c) = self.lstm(input_t, (h, c))
            pred = self.fc(out)  # (B, 1, output_dim)
            outputs.append(pred)
            input_t = pred  # use model prediction as next input

        return torch.cat(outputs, dim=1)  # (B, sequence_length, output_dim)
