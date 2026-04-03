"""
Step 1: Character-level tokenizer and dataset.

WHY CHARACTER-LEVEL?
A "real" tokenizer like BPE (used by GPT) groups common character sequences
into tokens. That's better for performance but adds complexity. For a tiny
model trained on a small corpus, character-level works fine and lets us focus
on the transformer itself.

HOW IT WORKS:
1. Read the corpus
2. Build a vocabulary: every unique character gets an integer ID
3. Encode the entire text as a sequence of integers
4. Split into training (90%) and validation (10%) sets
5. Create a function that pulls random chunks ("batches") for training
"""

import torch
import os


class CharTokenizer:
    """
    Maps characters <-> integers.

    stoi = string-to-integer dict  (e.g. {'a': 0, 'b': 1, ...})
    itos = integer-to-string dict  (e.g. {0: 'a', 1: 'b', ...})
    """

    def __init__(self, text: str):
        # sorted() so the mapping is deterministic
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        """Text -> list of integers."""
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """List of integers -> text."""
        return "".join(self.itos[t] for t in tokens)


def load_corpus(data_dir: str = "data") -> str:
    """Load the Maimonides corpus. Run download_data.py first."""
    path = os.path.join(data_dir, "maimonides_corpus.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run: python data/download_data.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_data(
    text: str, train_split: float = 0.9
) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """
    Tokenize text and split into train/val tensors.

    Returns:
        tokenizer: CharTokenizer instance
        train_data: 1D tensor of token IDs (90% of corpus)
        val_data: 1D tensor of token IDs (10% of corpus)
    """
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split point
    n = int(len(data) * train_split)
    train_data = data[:n]
    val_data = data[n:]

    return tokenizer, train_data, val_data


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull a random batch of training examples.

    Each example is a chunk of `block_size` tokens (input) and the same chunk
    shifted right by 1 (target). This is how next-token prediction works:

        Input:  [T, h, e, ' ', s]   (predict each next character)
        Target: [h, e, ' ', s, o]

    Args:
        data: 1D tensor of token IDs
        block_size: how many tokens per training example (context window)
        batch_size: how many examples per batch
        device: 'cpu' or 'mps' or 'cuda'

    Returns:
        x: (batch_size, block_size) input tensor
        y: (batch_size, block_size) target tensor (shifted by 1)
    """
    # Pick random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack into batches
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


# -------------------------------------------------------------------
# Quick test: run this file directly to verify everything works
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small test corpus
    test_text = "Thus spoke Maimonides: the truth is not found in the opinions of the many."

    tokenizer, train, val = prepare_data(test_text, train_split=0.9)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Characters: {''.join(tokenizer.itos[i] for i in range(tokenizer.vocab_size))}")
    print(f"Train tokens: {len(train)}")
    print(f"Val tokens:   {len(val)}")

    # Test encoding/decoding roundtrip
    encoded = tokenizer.encode("truth")
    decoded = tokenizer.decode(encoded)
    print(f"\n'truth' -> {encoded} -> '{decoded}'")

    # Test get_batch
    x, y = get_batch(train, block_size=8, batch_size=4)
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    print(f"First example input:  '{tokenizer.decode(x[0].tolist())}'")
    print(f"First example target: '{tokenizer.decode(y[0].tolist())}'")