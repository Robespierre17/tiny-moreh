"""
Step 3: Single-head self-attention.

THE INTUITION:
Imagine reading "The prophet spoke and the people listened."
When processing "listened", the model needs to figure out WHO listened
and WHO spoke. Self-attention lets "listened" look back at every previous
token and decide: "people" is very relevant, "prophet" is somewhat relevant,
"The" is not very relevant.

HOW IT WORKS (the Q/K/V mechanism):
For each token, we compute three vectors:
    Q (Query)  = "What am I looking for?"
    K (Key)    = "What do I contain?"
    V (Value)  = "What information do I give if selected?"

Attention score = how well a Query matches a Key.
High score = "this token is relevant to me" = take more of its Value.

THE MATH:
    1. Compute Q, K, V by multiplying input by learned weight matrices
    2. Score = Q @ K^T  (dot product: how similar is my query to each key?)
    3. Scale by sqrt(d_k) (keeps gradients stable)
    4. Mask future tokens (can't peek ahead when generating text!)
    5. Softmax (turn scores into probabilities that sum to 1)
    6. Output = probabilities @ V (weighted sum of values)

WHY MASKING?
During training, we process whole sequences at once for efficiency.
But at generation time, token 5 can't see tokens 6, 7, 8...
So we mask (set to -infinity) all positions ahead of the current token.
This is called "causal" attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """
    One head of self-attention.

    Args:
        d_model: input embedding dimension
        d_k: dimension of Q, K, V vectors (usually d_model // num_heads,
             but here we just use one head so d_k can equal d_model)
        block_size: max sequence length (needed for the causal mask)
    """

    def __init__(self, d_model: int, d_k: int, block_size: int):
        super().__init__()

        # These three linear layers ARE the Q, K, V weight matrices.
        # They project the input into query/key/value spaces.
        # bias=False is a common choice — original transformer used it.
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

        self.d_k = d_k

        # Causal mask: a lower-triangular matrix of 1s.
        # Position (i, j) = 1 means "token i CAN attend to token j"
        # Position (i, j) = 0 means "token i CANNOT attend to token j" (it's in the future)
        #
        # Example for seq_len=4:
        #   [[1, 0, 0, 0],    token 0 can only see itself
        #    [1, 1, 0, 0],    token 1 can see tokens 0-1
        #    [1, 1, 1, 0],    token 2 can see tokens 0-2
        #    [1, 1, 1, 1]]    token 3 can see tokens 0-3
        #
        # register_buffer: saves it with the model but doesn't train it
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq_len, d_model)
        output:  (batch, seq_len, d_k)
        """
        B, T, C = x.shape  # batch, seq_len (Time), channels (d_model)

        # Step 1: Project input into Q, K, V
        q = self.W_q(x)  # (B, T, d_k)
        k = self.W_k(x)  # (B, T, d_k)
        v = self.W_v(x)  # (B, T, d_k)

        # Step 2: Compute attention scores
        # q @ k^T gives a (T, T) matrix of how much each token attends to each other
        scores = q @ k.transpose(-2, -1)  # (B, T, T)

        # Step 3: Scale by sqrt(d_k)
        # Without this, dot products get very large for high dimensions,
        # causing softmax to produce near-0 and near-1 values (vanishing gradients)
        scores = scores / math.sqrt(self.d_k)

        # Step 4: Apply causal mask
        # Where mask is 0 (future tokens), set score to -infinity
        # After softmax, -infinity becomes 0 probability
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # Step 5: Softmax — turn scores into probabilities
        weights = F.softmax(scores, dim=-1)  # (B, T, T), rows sum to 1

        # Step 6: Weighted sum of values
        out = weights @ v  # (B, T, d_k)

        return out


# -------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------
if __name__ == "__main__":
    d_model = 64
    d_k = 64
    block_size = 32
    batch_size = 4

    head = SingleHeadAttention(d_model, d_k, block_size)

    # Fake input (as if coming from the embedding layer)
    x = torch.randn(batch_size, block_size, d_model)
    print(f"Input shape:  {x.shape}")

    out = head(x)
    print(f"Output shape: {out.shape}")

    assert out.shape == (batch_size, block_size, d_k)

    # Verify causal masking works: token 0 should only attend to itself
    # Run a single example to check
    test_x = torch.randn(1, 4, d_model)
    test_head = SingleHeadAttention(d_model, d_k, block_size)
    
    # Hook into the weights to verify
    with torch.no_grad():
        q = test_head.W_q(test_x)
        k = test_head.W_k(test_x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        scores = scores.masked_fill(test_head.mask[:4, :4] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        print(f"\nAttention weights for 4-token sequence:")
        print(f"(each row shows how much that token attends to each position)")
        print(weights[0].detach())

    print("\nStep 3 OK — self-attention working")