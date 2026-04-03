"""
Step 4: Multi-head attention.

WHY MULTIPLE HEADS?
A single attention head can only learn one "type" of relationship at a time.
Multi-head attention runs several heads in parallel, each with its own Q/K/V
weights, so they can specialize:
    - Head 1 might learn: "which noun does this verb refer to?"
    - Head 2 might learn: "which adjective modifies this noun?"
    - Head 3 might learn: "what's the topic of this sentence?"

HOW IT WORKS:
1. Split d_model into num_heads smaller pieces
   (e.g., d_model=64 with 4 heads -> each head works with d_k=16)
2. Each head runs self-attention independently on its piece
3. Concatenate all head outputs back together -> d_model dimensions
4. One final linear layer mixes information across heads

This is more efficient than running separate full-sized attention heads
because the total computation stays the same — we just divide it up.
"""

import torch
import torch.nn as nn
from attention import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.

    Args:
        d_model: embedding dimension
        num_heads: how many parallel attention heads
        block_size: max sequence length
        dropout: dropout rate
    """

    def __init__(
        self, d_model: int, num_heads: int, block_size: int, dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Each head gets a fraction of the total dimensions
        d_k = d_model // num_heads

        # Create all heads as a ModuleList so PyTorch tracks their parameters
        self.heads = nn.ModuleList(
            [SingleHeadAttention(d_model, d_k, block_size) for _ in range(num_heads)]
        )

        # After concatenating all heads, this linear layer lets them
        # exchange information. Without it, heads can't "talk" to each other.
        self.output_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq_len, d_model)
        output:  (batch, seq_len, d_model)

        Each head outputs (batch, seq_len, d_k).
        We concatenate along the last dimension: num_heads * d_k = d_model.
        Then project back to d_model.
        """
        # Run all heads in parallel and concatenate
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, d_model)

        # Final projection + dropout
        out = self.dropout(self.output_proj(concat))
        return out


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    d_model = 64
    num_heads = 4  # 4 heads, each with d_k = 64/4 = 16
    block_size = 32
    batch_size = 4

    mha = MultiHeadAttention(d_model, num_heads, block_size)

    x = torch.randn(batch_size, block_size, d_model)
    print(f"Input shape:  {x.shape}")

    out = mha(x)
    print(f"Output shape: {out.shape}")

    assert out.shape == (batch_size, block_size, d_model)

    # Count parameters
    num_params = sum(p.numel() for p in mha.parameters())
    print(f"Parameters:   {num_params:,}")
    print(f"Heads: {num_heads}, each with d_k = {d_model // num_heads}")

    print("\nStep 4 OK — multi-head attention working")