"""
Step 6: Full transformer model.

This assembles everything we've built:

    Token IDs
        ↓
    [Embedding + Positional Encoding]  (Step 2)
        ↓
    [Transformer Block] × N            (Step 5 = Step 3 + Step 4)
        ↓
    [Layer Norm]
        ↓
    [Linear → vocab_size]              (output head: predicts next token)
        ↓
    Logits (raw scores for each possible next character)

The output head is a linear layer that converts from d_model dimensions
to vocab_size dimensions. Each output value is a score ("logit") for how
likely that character is to come next. During training, we compare these
logits to the actual next character using cross-entropy loss.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import TransformerInput
from block import TransformerBlock


class TinyMoreh(nn.Module):
    """
    A small GPT-style language model trained on Maimonides.

    Architecture:
        - Token + positional embeddings
        - N transformer blocks (attention + feed-forward)
        - Final layer norm
        - Linear output head → vocab_size logits

    Args:
        vocab_size: number of unique characters in the corpus
        d_model: embedding dimension
        num_heads: attention heads per block
        num_layers: how many transformer blocks to stack
        block_size: max sequence length (context window)
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        block_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        # Input: token IDs → embedded vectors with position info
        self.input_layer = TransformerInput(vocab_size, d_model, block_size, dropout)

        # Stack of transformer blocks — this is the "deep" in deep learning
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm (stabilizes the output)
        self.ln_f = nn.LayerNorm(d_model)

        # Output head: project from d_model → vocab_size
        # Each output is a score for "how likely is this character next?"
        self.output_head = nn.Linear(d_model, vocab_size)

        # Weight tying: share weights between token embedding and output head.
        # Intuition: the embedding maps tokens→vectors, the output head maps
        # vectors→tokens. Using the same weights for both improves performance
        # and reduces parameter count.
        self.output_head.weight = self.input_layer.token_emb.embedding.weight

        # Initialize weights (important for stable training)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier/Glorot initialization — standard for transformers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass.

        Args:
            x: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs (for computing loss)

        Returns:
            logits: (batch, seq_len, vocab_size) — raw prediction scores
            loss: scalar loss value (only if targets provided)
        """
        # Embed tokens + add position info
        h = self.input_layer(x)            # (batch, seq_len, d_model)

        # Pass through all transformer blocks
        h = self.blocks(h)                  # (batch, seq_len, d_model)

        # Final norm + project to vocabulary
        h = self.ln_f(h)                    # (batch, seq_len, d_model)
        logits = self.output_head(h)        # (batch, seq_len, vocab_size)

        # Compute loss if we have targets
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: it expects (N, C) and (N,)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text autoregressively.

        Starting from idx (a sequence of token IDs), predict one token at a time
        and append it. Repeat until we've generated max_new_tokens.

        Temperature controls randomness:
            - temp=1.0: normal sampling
            - temp<1.0: more conservative (picks likely tokens)
            - temp>1.0: more creative (more random)

        Args:
            idx: (batch, seq_len) starting token IDs
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature

        Returns:
            (batch, seq_len + max_new_tokens) — original + generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is too long
            # (model can only handle block_size tokens at once)
            idx_cond = idx[:, -self.block_size:]

            # Forward pass — get logits for next token
            logits, _ = self(idx_cond)

            # Take only the last position's logits (that's our prediction)
            logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    vocab_size = 80
    d_model = 128
    num_heads = 4
    num_layers = 4
    block_size = 128
    batch_size = 4

    model = TinyMoreh(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Test forward pass with loss
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))

    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Expected initial loss: -ln(1/vocab_size) = ln(80) ≈ 4.38
    # (random model assigns equal probability to each character)
    import math
    expected = math.log(vocab_size)
    print(f"Expected initial loss (random): {expected:.4f}")

    # Test generation
    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=20)
    print(f"Generated token IDs: {generated[0].tolist()}")

    print("\nStep 6 OK — full model working")