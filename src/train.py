"""
Step 6b: Training loop.

THE TRAINING CYCLE (every single step):
    1. Grab a random batch of text chunks from the corpus
    2. Feed them through the model → get predictions (logits)
    3. Compare predictions to actual next characters → compute loss
    4. Backpropagation: compute how each weight contributed to the error
    5. Gradient descent: nudge each weight slightly to reduce the error
    6. Repeat thousands of times

This script trains locally on CPU/MPS. For the final training run,
we'll use Google Colab with a GPU.

USAGE:
    python src/train.py                    # train on CPU
    python src/train.py --device mps       # train on Apple GPU
    python src/train.py --device cuda      # train on NVIDIA GPU (Colab)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import torch
from data import load_corpus, prepare_data, get_batch
from model import TinyMoreh


# ----- Hyperparameters -----
# These control how the model trains. We'll use small values for local
# testing and scale up for the real Colab run.

CONFIG = {
    # Model architecture
    "d_model": 128,       # embedding dimension
    "num_heads": 4,       # attention heads
    "num_layers": 4,      # transformer blocks
    "block_size": 128,    # context window (characters)
    "dropout": 0.1,

    # Training
    "batch_size": 32,     # examples per gradient update
    "learning_rate": 3e-4,  # how big each weight update is (AdamW default)
    "max_steps": 5000,    # total training steps
    "eval_interval": 250, # evaluate every N steps
    "eval_steps": 20,     # how many batches to average for eval loss

    # Generation
    "generate_every": 1000,  # generate sample text every N steps
    "generate_tokens": 200,  # how many characters to generate
}


def estimate_loss(
    model: TinyMoreh,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: dict,
    device: str,
) -> dict:
    """
    Estimate train and val loss by averaging over several batches.

    We do this because any single batch's loss is noisy. Averaging
    gives a more reliable picture of how the model is doing.
    """
    model.eval()  # turn off dropout (don't want randomness during evaluation)
    losses = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        for _ in range(config["eval_steps"]):
            x, y = get_batch(data, config["block_size"], config["batch_size"], device)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[split_name] = total_loss / config["eval_steps"]

    model.train()  # turn dropout back on
    return losses


def generate_sample(model: TinyMoreh, tokenizer, device: str, num_tokens: int = 200):
    """Generate a text sample from the model."""
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    tokens = model.generate(start, max_new_tokens=num_tokens, temperature=0.8)
    text = tokenizer.decode(tokens[0].tolist())
    model.train()
    return text


def train(device: str = "cpu"):
    print(f"Device: {device}")
    print(f"Loading corpus...")

    # Load and prepare data
    text = load_corpus(data_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
    tokenizer, train_data, val_data = prepare_data(text)

    print(f"Corpus: {len(text):,} characters")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Create model
    model = TinyMoreh(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        block_size=CONFIG["block_size"],
        dropout=CONFIG["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer: AdamW is the standard for transformers.
    # It's Adam (adaptive learning rates per parameter) + weight decay
    # (gently pushes weights toward zero to prevent overfitting).
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Training loop ---
    print(f"\nTraining for {CONFIG['max_steps']} steps...")
    print("-" * 60)

    start_time = time.time()

    for step in range(CONFIG["max_steps"]):
        # Evaluate periodically
        if step % CONFIG["eval_interval"] == 0 or step == CONFIG["max_steps"] - 1:
            losses = estimate_loss(model, train_data, val_data, CONFIG, device)
            elapsed = time.time() - start_time
            print(
                f"Step {step:>5d} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"time: {elapsed:.0f}s"
            )

        # Generate sample text periodically
        if step > 0 and step % CONFIG["generate_every"] == 0:
            sample = generate_sample(model, tokenizer, device, CONFIG["generate_tokens"])
            print(f"\n--- Sample at step {step} ---")
            print(sample)
            print("--- End sample ---\n")

        # 1. Get a batch
        x, y = get_batch(
            train_data, CONFIG["block_size"], CONFIG["batch_size"], device
        )

        # 2. Forward pass
        logits, loss = model(x, y)

        # 3. Backward pass (compute gradients)
        optimizer.zero_grad(set_to_none=True)  # clear old gradients
        loss.backward()                         # compute new gradients

        # 4. Update weights
        optimizer.step()

    # --- Done training ---
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")

    # Final generation
    print("\n" + "=" * 60)
    print("FINAL GENERATED TEXT:")
    print("=" * 60)
    sample = generate_sample(model, tokenizer, device, 500)
    print(sample)

    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tiny_moreh.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
        "vocab_size": tokenizer.vocab_size,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on: cpu, mps (Apple), or cuda (NVIDIA)",
    )
    args = parser.parse_args()

    # Validate device
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    train(device=device)