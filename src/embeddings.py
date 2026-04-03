"""
Step 2: Token embeddings + positional encoding.

TWO PROBLEMS TO SOLVE:
1. The model receives token IDs (integers like 42, 7, 13). It needs to convert
   these into vectors (lists of numbers) that it can do math on. That's what
   token embeddings do.

2. "The dog bit the man" and "The man bit the dog" have the same tokens but
   different meanings. The model needs to know word ORDER. That's what
   positional encoding does — it adds a unique signal to each position.

HOW THEY CONNECT:
    token IDs  ->  [token embedding] + [position embedding]  ->  vectors
    (integers)     (learned vectors)   (learned vectors)        (ready for attention)

We use LEARNED positional embeddings (not sinusoidal). GPT-2 did it this way.
It means the model learns the best position representations during training,
just like it learns token representations.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors.

    nn.Embedding is basically a lookup table:
    - Row 0 = vector for token 0
    - Row 1 = vector for token 1
    - etc.

    These vectors start random and get updated during training.
    The model learns that similar tokens should have similar vectors.

    Args:
        vocab_size: number of unique tokens (characters in our case)
        d_model: dimensionality of each embedding vector
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len) — integer token IDs
        output:  (batch_size, seq_len, d_model) — embedding vectors
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Adds position information to token embeddings.

    Without this, the model can't tell the difference between
    "dog bites man" and "man bites dog" — same tokens, different order.

    We learn a separate embedding vector for each position (0, 1, 2, ...).
    Position 0 always gets the same vector, position 1 always gets the same
    vector, etc. These get ADDED to the token embeddings.

    Args:
        block_size: maximum sequence length (max number of positions)
        d_model: dimensionality (must match token embedding size)
    """

    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        # One learned vector per position
        self.embedding = nn.Embedding(block_size, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns position embeddings for positions 0..seq_len-1.
        output shape: (seq_len, d_model)
        """
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)


class TransformerInput(nn.Module):
    """
    Combines token embeddings + positional embeddings.

    This is the very first thing that happens in the model:
        raw token IDs -> meaningful vectors with position info

    Args:
        vocab_size: number of unique tokens
        d_model: embedding dimension
        block_size: max sequence length
        dropout: dropout rate (randomly zeros some values during training
                 to prevent overfitting)
    """

    def __init__(
        self, vocab_size: int, d_model: int, block_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len) — token IDs
        output:  (batch_size, seq_len, d_model) — embedded + positioned vectors

        The addition works because both embeddings have shape (..., d_model).
        Position embeddings broadcast across the batch dimension.
        """
        seq_len = x.shape[1]
        tok = self.token_emb(x)           # (batch, seq_len, d_model)
        pos = self.pos_emb(seq_len)        # (seq_len, d_model)
        return self.dropout(tok + pos)      # (batch, seq_len, d_model)


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters for our tiny model
    vocab_size = 80   # ~80 unique characters in the corpus
    d_model = 64      # embedding dimension (small for testing)
    block_size = 32   # context window: 32 characters
    batch_size = 4

    # Create the input layer
    input_layer = TransformerInput(vocab_size, d_model, block_size)

    # Fake batch of token IDs
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    print(f"Input shape:  {x.shape}")   # (4, 32)

    # Forward pass
    out = input_layer(x)
    print(f"Output shape: {out.shape}")  # (4, 32, 64)

    # Verify the shapes make sense
    assert out.shape == (batch_size, block_size, d_model)
    print("Step 2 OK — embeddings + positional encoding working")