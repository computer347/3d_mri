"""One place to make a run repeatable.

Nothing in this project was seeded until now, which is an awkward gap for a
codebase whose argument is measurement discipline: every "before -> after"
number in ``reports/results.md`` implicitly claims that re-running would
produce the same thing, and until this module existed that claim was untested.

Three sources of randomness matter here, and MONAI's ``set_determinism`` alone
does not cover all of them:

* ``random`` / ``numpy`` - split sampling, and MONAI's ``Rand*d`` transforms,
  which draw from numpy's global stream unless given their own ``set_random_state``.
* ``torch`` (CPU and CUDA) - dropout, weight init, and the MC-dropout passes in
  ``mri3d.predict``, which are *deliberately* stochastic and therefore need a
  seed to be reproducible rather than a switch to turn them off.
* cuDNN algorithm selection - benchmark mode picks whichever convolution
  algorithm is fastest on the day, and those algorithms are not all
  bit-identical.

``set_determinism`` is called with ``use_deterministic_algorithms=False``.
Turning it on would raise on the 3D convolution backward kernels this model
uses (no deterministic implementation exists for several of them), so the
honest statement is: seeded and reproducible run-to-run on the same machine and
library versions, not bit-reproducible across hardware.

**The seed alone is not enough for inference**, and getting this wrong here
briefly made things worse rather than better. ``cudnn.benchmark`` times the
available convolution algorithms and keeps the fastest - timing them against
whatever workspace happens to be free. An early version of this module turned
it on (PyTorch's own default is off, and ``set_determinism`` also turns it off,
so "restoring throughput" looked like the considerate thing to do). Measured on
this card, the same seeded sliding-window inference run with 3 GB of VRAM
already occupied then selected a different algorithm and produced different
numbers: softmax probabilities agreeing to only about four decimal places, one
SNFH component in ~7100 changing size by 8 voxels, volumetric Dice moving by up
to 0.0095 on a case, and two of the per-class false-positive counts in
``reports/`` shifting by one. Two runs under equal memory conditions agreed
exactly, which is why it was not obvious.

With ``benchmark=False`` the same experiment is bit-identical at 0 GB and 3 GB
occupied, and it reproduces the original pre-seeding predictions exactly.
``mri3d.predict`` therefore seeds with ``benchmark=False``: the predictions are
the evidence in ``reports/``, and a prediction that changes depending on what
else is on the GPU is not evidence. Training keeps ``benchmark=True``, where
autotuning is worth real throughput and the output is a checkpoint rather than
a number that gets reported.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from monai.utils import set_determinism


def set_seed(seed: int = 0, benchmark: bool = True) -> int:
    """Seed python, numpy, torch and MONAI. Returns the seed, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed, use_deterministic_algorithms=False)
    # set_determinism forces cudnn.benchmark off; training throughput on this
    # card depends on it, so restore it unless the caller wants it off. It
    # affects algorithm choice, not the seeded random streams.
    torch.backends.cudnn.benchmark = benchmark
    return seed
