"""Seeding, and the one flag that turned out to matter more than the seed.

``cudnn.benchmark`` times the available convolution algorithms against the free
workspace and keeps the fastest. Measured on this project's card, the same
seeded sliding-window inference run with 3 GB of VRAM already occupied selected
a different algorithm and produced different numbers - softmax probabilities
agreeing to only about four decimal places, and one predicted component
crossing the 50-voxel floor that lesion-wise Dice scores on. With
``benchmark=False`` the same experiment was bit-identical.

So ``mri3d.predict`` seeds with ``benchmark=False``. These tests pin that,
because the failure it prevents is invisible: nothing errors, the numbers just
quietly stop matching the ones in ``reports/``.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from mri3d.seeding import set_seed


def test_set_seed_makes_the_random_streams_repeat():
    set_seed(1234)
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_seed(1234)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b


def test_different_seeds_give_different_streams():
    set_seed(0)
    a = torch.rand(5).tolist()
    set_seed(1)
    assert torch.rand(5).tolist() != a


def test_benchmark_flag_is_honoured():
    """MONAI's set_determinism forces benchmark off; set_seed must restore it."""
    set_seed(0, benchmark=True)
    assert torch.backends.cudnn.benchmark is True
    set_seed(0, benchmark=False)
    assert torch.backends.cudnn.benchmark is False


def test_prediction_seeds_with_autotuning_off():
    """The inference entry point must not leave cuDNN free to autotune.

    Asserted against the source rather than by running inference, which needs a
    GPU and a checkpoint. Crude, but it fails if someone drops the argument.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "mri3d" / "predict.py"
           ).read_text(encoding="utf-8")
    assert "set_seed(args.seed, benchmark=False)" in src
