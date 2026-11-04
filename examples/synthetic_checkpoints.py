"""Generate a tiny, deterministic example at runtime; no checkpoints are bundled."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/example")
    parser.add_argument("--versions", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(-0.2, 0.4, 4097)
    for version in range(args.versions):
        master = x + version * 1e-5
        state = {
            "step": 100 * (version + 1),
            "master": {"weight": master},
            "model": {"weight": master.to(torch.bfloat16)},
            "optimizer": {"m": master.sin(), "v": master.square()},
            "frozen": torch.ones(16),
            "config": {"learning_rate": 0.001},
        }
        path = root / f"checkpoint-{version:02d}.pt"
        torch.save(state, path)
        print(path)


if __name__ == "__main__":
    main()
