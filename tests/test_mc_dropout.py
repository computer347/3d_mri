"""MC dropout at inference: stochastic on purpose, and only in one place.

The trick is to leave the model in ``eval()`` and re-enable *only* the dropout
modules. Putting the whole model into ``train()`` would also un-freeze the
batch-norm statistics, so the "uncertainty" measured would be dominated by
batch-norm recomputing its statistics from a single window - noise that has
nothing to do with whether the lesion is real.

The restore afterwards matters as much: ``Module.train(mode)`` is recursive, so
a careless restore loop can leave dropout active for every subsequent case,
quietly making the saved masks stochastic. Nothing would error; the predictions
would just stop being reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from mri3d.predict import mc_dropout_variance


class Tiny(torch.nn.Module):
    """Conv - batch-norm - dropout, the same ordering SegResNet uses."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 3, 3, padding=1)
        self.norm = torch.nn.BatchNorm3d(3)
        self.drop = torch.nn.Dropout3d(0.5)

    def forward(self, x):
        return self.drop(self.norm(self.conv(x)))


def _model():
    m = Tiny()
    m.eval()
    return m


def test_variance_is_nonzero_and_shaped_like_the_probabilities():
    torch.manual_seed(0)
    model, image = _model(), torch.randn(1, 1, 16, 16, 16)
    var = mc_dropout_variance(model, image, (16, 16, 16), 8, torch.device("cpu"))
    assert var.shape == (3, 16, 16, 16)
    assert np.isfinite(var).all()
    assert var.max() > 0, "8 dropout passes must disagree somewhere"


def test_every_module_is_left_exactly_as_it_was_found():
    model, image = _model(), torch.randn(1, 1, 16, 16, 16)
    before = {name: m.training for name, m in model.named_modules()}
    mc_dropout_variance(model, image, (16, 16, 16), 3, torch.device("cpu"))
    assert {name: m.training for name, m in model.named_modules()} == before
    assert not any(before.values()), "the fixture should start fully in eval()"


def test_batch_norm_stays_frozen_during_the_passes():
    """Running statistics must not move, or the variance measures the wrong thing."""
    model, image = _model(), torch.randn(1, 1, 16, 16, 16)
    mean = model.norm.running_mean.clone()
    var = model.norm.running_var.clone()
    mc_dropout_variance(model, image, (16, 16, 16), 4, torch.device("cpu"))
    assert torch.equal(model.norm.running_mean, mean)
    assert torch.equal(model.norm.running_var, var)


def test_it_is_seeded_and_therefore_repeatable():
    model, image = _model(), torch.randn(1, 1, 16, 16, 16)
    torch.manual_seed(7)
    a = mc_dropout_variance(model, image, (16, 16, 16), 4, torch.device("cpu"))
    torch.manual_seed(7)
    b = mc_dropout_variance(model, image, (16, 16, 16), 4, torch.device("cpu"))
    assert np.array_equal(a, b)


def test_welford_matches_the_naive_variance():
    """The streaming estimate is an optimisation, not an approximation.

    Holding T volumes to call np.var would be 1.2 GB at T=8 on a real case;
    Welford holds two arrays. It must give the same answer.
    """
    torch.manual_seed(3)
    model, image = _model(), torch.randn(1, 1, 8, 8, 8)
    torch.manual_seed(11)
    streamed = mc_dropout_variance(model, image, (8, 8, 8), 6, torch.device("cpu"))

    torch.manual_seed(11)
    passes = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout3d):
            m.train()
    for _ in range(6):
        with torch.no_grad():
            from monai.inferers import sliding_window_inference
            logits = sliding_window_inference(image, (8, 8, 8), 1, model,
                                              overlap=0.25, mode="gaussian")
        passes.append(torch.softmax(logits.float(), dim=1)[0].numpy())
    model.eval()
    naive = np.var(np.stack(passes), axis=0, ddof=1)
    assert np.allclose(streamed, naive, atol=1e-6)
