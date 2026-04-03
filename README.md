# TinyMoreh

A small GPT-style transformer built from scratch in PyTorch, trained on the writings of Maimonides.

The model learns to generate text in the style of the *Guide for the Perplexed* and *Mishneh Torah* — not coherent philosophy, but text that captures the vocabulary, rhythm, and structure of medieval Jewish philosophical writing.

**This is a learning project.** Every component — tokenizer, embeddings, self-attention, multi-head attention, transformer blocks, training loop — is written from scratch to understand how transformer language models actually work under the hood.

## Sample Output

Prompt: *"It is known that"*

> It is known that the Universe is governed after the reverse, as is distinctly stated. For it is the case with the words, For the Lord thy God which was the mighty people (Josh. v. 13). In fact, however, they must suffice to destroy the word of God, and to sacrifice of the stars in his seven species similar to that

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only transformer (GPT-style) |
| Parameters | ~4.8M (GPU) / ~827K (CPU) |
| Embedding dim | 256 (GPU) / 128 (CPU) |
| Attention heads | 8 (GPU) / 4 (CPU) |
| Layers | 6 (GPU) / 4 (CPU) |
| Context window | 256 (GPU) / 128 (CPU) characters |
| Tokenizer | Character-level |
| Training corpus | ~1.7M characters of Maimonides |

## How It Works

The model is a stack of transformer blocks, each containing:

1. **Multi-head self-attention** — each token attends to all previous tokens using Query/Key/Value projections with causal masking to prevent looking ahead
2. **Feed-forward network** — two linear layers with GELU activation that process each token's representation after attention gathers context
3. **Residual connections + layer normalization** — keep gradients flowing and training stable

The full pipeline: character → integer token → embedding vector + positional encoding → N transformer blocks → linear projection → next-character probabilities.

## Project Structure

```
tiny-moreh/
├── data/
│   └── download_data.py     # Downloads Maimonides texts
├── src/
│   ├── data.py              # Character tokenizer + batching
│   ├── embeddings.py        # Token + positional embeddings
│   ├── attention.py         # Single-head self-attention (Q/K/V)
│   ├── multihead.py         # Multi-head attention
│   ├── block.py             # Transformer block (attn + FF + residuals)
│   ├── model.py             # Full TinyMoreh model
│   ├── train.py             # Training loop
│   └── generate.py          # Text generation script
├── notebooks/
│   └── train_colab.ipynb    # Colab notebook for GPU training
├── checkpoints/             # Saved model weights (not in git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Robespierre17/tiny-moreh.git
cd tiny-moreh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download training data
python data/download_data.py

# Train (CPU — ~6 min for 5K steps)
python src/train.py

# Train (Apple Silicon GPU)
python src/train.py --device mps

# Generate text
python src/generate.py --prompt "The nature of God"
python src/generate.py --prompt "The prophet spoke" --temperature 0.5
python src/generate.py --num_samples 3 --tokens 500
```

## Training

The model was trained on ~1.7 million characters from two public domain sources:
- *Guide for the Perplexed* (Friedländer translation, Project Gutenberg)
- *Mishneh Torah* selections (Touger translation, Sefaria, Creative Commons)

Local training (5K steps, CPU): loss drops from ~4.98 to ~1.22 in about 6 minutes. The model learns Maimonides-specific vocabulary and sentence patterns but produces semi-coherent output.

GPU training (10K steps, Colab T4): loss drops from ~4.41 to ~0.64. The larger model learns to cite biblical verses, reference Aristotle and the Mutakallemim, and construct multi-clause philosophical sentences in Maimonides' style. Use the Colab notebook (`notebooks/train_colab.ipynb`) to reproduce.

## Data Sources

- [Guide for the Perplexed](https://www.gutenberg.org/ebooks/73584) — Project Gutenberg (public domain)
- [Mishneh Torah](https://www.sefaria.org/texts/Halakhah/Mishneh%20Torah) — Sefaria (Creative Commons)