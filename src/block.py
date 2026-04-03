"""
Step 5: Transformer block.

This is the repeatable building block of a transformer. Stack 4 of these
and you have a 4-layer transformer. Stack 96 and you have GPT-3.

WHAT'S INSIDE ONE BLOCK:
    input
      │
      ├──→ Multi-Head Attention ──→ Add & Layer Norm ──→ output₁
      │         ↑                        ↑
      └─────────┘ (residual)  ───────────┘
      │
      ├──→ Feed-Forward Network ──→ Add & Layer Norm ──→ output₂
      │         ↑                         ↑
      └─────────┘ (residual)   ───────────┘

TWO NEW CONCEPTS:

1. RESIDUAL CONNECTIONS (the "Add" part):
   Instead of: output = layer(input)
   We do:      output = input + layer(input)

   This lets gradients flow directly backward through the + operation,
   which prevents the vanishing gradient problem in deep networks.
   Think of it as a "skip connection" — the original signal always gets through.

2. LAYER NORMALIZATION (the "Norm" part):
   Normalizes values so they have mean=0 and std=1 across the embedding dimension.
   This keeps numbers from getting too big or too small as they flow through layers.
   Without it, training becomes unstable.

3. FEED-FORWARD NETWORK:
   After attention gathers context from other tokens, this processes each token
   independently. It's two linear layers with a ReLU (or GELU) activation in between.

   The hidden dimension is typically 4x the model dimension. This gives the network
   more capacity to process the information that attention collected.

   Think of attention as "gathering information" and feed-forward as "thinking about it."
"""

import torch
import torch.nn as nn
from multihead import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied to each token independently (same weights for every position).
    Expands to 4x dimension, applies GELU activation, projects back down.

    Why GELU instead of ReLU?
    GELU (Gaussian Error Linear Unit) is smoother than ReLU and has been
    shown to work better in transformers. GPT-2 and BERT both use it.
    ReLU would also work — it's not a huge difference for a tiny model.

    Args:
        d_model: embedding dimension
        dropout: dropout rate
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # expand to 4x
            nn.GELU(),                          # non-linear activation
            nn.Linear(4 * d_model, d_model),    # project back down
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block: attention + feed-forward, with residuals and norms.

    We use "pre-norm" architecture (norm BEFORE attention/ff, not after).
    This is what GPT-2 does. It's more stable to train than "post-norm"
    (the original transformer paper's approach).

    Args:
        d_model: embedding dimension
        num_heads: number of attention heads
        block_size: max sequence length
        dropout: dropout rate
    """

    def __init__(
        self, d_model: int, num_heads: int, block_size: int, dropout: float = 0.1
    ):
        super().__init__()

        # Layer norms — one before attention, one before feed-forward
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # The two sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq_len, d_model)
        output:  (batch, seq_len, d_model)
        """
        # Attention with residual connection
        # x + attention(norm(x))  — the x+ is the residual/skip connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward with residual connection
        x = x + self.ff(self.ln2(x))

        return x


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    d_model = 64
    num_heads = 4
    block_size = 32
    batch_size = 4

    block = TransformerBlock(d_model, num_heads, block_size)

    x = torch.randn(batch_size, block_size, d_model)
    print(f"Input shape:  {x.shape}")

    out = block(x)
    print(f"Output shape: {out.shape}")

    assert out.shape == (batch_size, block_size, d_model)

    # Count parameters
    num_params = sum(p.numel() for p in block.parameters())
    print(f"Parameters in one block: {num_params:,}")

    # Test stacking — this is how you build a deeper model
    blocks = nn.Sequential(*[
        TransformerBlock(d_model, num_heads, block_size)
        for _ in range(4)
    ])
    out_deep = blocks(x)
    print(f"\n4 stacked blocks:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_deep.shape}")

    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"  Total params: {total_params:,}")

    print("\nStep 5 OK — transformer block working")