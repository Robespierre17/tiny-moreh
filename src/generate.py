"""
Generate text from a trained TinyRambam model.

USAGE:
    python src/generate.py                              # random start
    python src/generate.py --prompt "The truth of God"  # custom prompt
    python src/generate.py --temperature 0.5            # more conservative
    python src/generate.py --temperature 1.2            # more creative
    python src/generate.py --tokens 500                 # longer output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from model import TinyRambam


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    vocab_size = checkpoint["vocab_size"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = TinyRambam(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=0.0,  # no dropout during generation
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos, config


def generate(
    model: TinyRambam,
    stoi: dict,
    itos: dict,
    prompt: str = "",
    max_tokens: int = 300,
    temperature: float = 0.8,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt."""
    if prompt:
        # Encode prompt, skipping any characters not in vocabulary
        tokens = [stoi[ch] for ch in prompt if ch in stoi]
        if not tokens:
            print("Warning: no valid characters in prompt, starting from zero")
            tokens = [0]
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    return "".join(itos[t] for t in output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from TinyRambam")
    parser.add_argument("--prompt", type=str, default="", help="Starting text")
    parser.add_argument("--tokens", type=int, default=300, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1=conservative, 1.5=creative)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, mps, or cuda")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tiny_rambam.pt")

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        print("Train a model first: python src/train.py")
        sys.exit(1)

    # Load model
    print(f"Loading model from {ckpt_path}")
    model, stoi, itos, config = load_model(ckpt_path, args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")
    print(f"Temperature: {args.temperature}")
    if args.prompt:
        print(f"Prompt: \"{args.prompt}\"")
    print("-" * 60)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Sample {i + 1} ---")
        text = generate(
            model, stoi, itos,
            prompt=args.prompt,
            max_tokens=args.tokens,
            temperature=args.temperature,
            device=args.device,
        )
        print(text)

    print("-" * 60)


if __name__ == "__main__":
    main()