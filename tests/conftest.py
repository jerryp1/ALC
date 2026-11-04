import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch


def states(count=5, elements=513):
    result = []
    for version in range(count):
        master = torch.nn.Parameter(
            torch.linspace(-0.2, 0.4, elements) + version * 1e-5
        )
        model = master.detach().to(torch.bfloat16)
        bits = np.array(
            [0x7FC1, 0xFFFF, 0x0001, 0x8000, 0x8001, 0x3F80 + version], np.uint16
        )
        result.append(
            {
                "step": version + 1,
                "master": {"weight": master},
                "model": {
                    "weight": model,
                    "unmatched": torch.from_numpy(bits.view(np.int16).copy()).view(
                        torch.bfloat16
                    ),
                },
                "tied": model,
                "optimizer": {
                    0: {
                        "m": master.detach().sin() * (-1) ** version,
                        "v": master.detach().square(),
                    }
                },
                "frozen": torch.arange(17, dtype=torch.float32),
                "counter": torch.tensor(version, dtype=torch.int64),
                "empty": torch.empty((0, 3)),
                "metadata": (None, [True, 0.001, torch.Size([2, 3])]),
            }
        )
    return result


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, type(left))
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.requires_grad == right.requires_grad
        a = left.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        b = right.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert a == b
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert type(left) is type(right) and left == right
