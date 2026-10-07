"""Flip test-time augmentation: every flip must be undone before averaging.

The failure this guards against is silent. Forget to flip a prediction back and
the averaged mask is a blend of the tumour and its mirror image - still a
plausible-looking blob, and still scoring well on any case where the tumour is
near the midline.
"""

from __future__ import annotations

import torch

from mri3d.predict import FLIP_DIMS, _infer


class PerVoxel(torch.nn.Module):
    """A 1x1x1 convolution: commutes with any flip, so TTA must change nothing."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 3, 1)

    def forward(self, x):
        return self.conv(x)


class Positional(torch.nn.Module):
    """Adds a fixed, asymmetric pattern: NOT flip-equivariant, so TTA must change it."""

    def __init__(self, shape):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-3, 3, shape[0]).view(1, 1, -1, 1, 1)
                                       .expand(1, 3, *shape).clone())

    def forward(self, x):
        return x.repeat(1, 3, 1, 1, 1) + self.bias


def test_all_eight_flips_are_distinct_and_include_identity():
    assert len(FLIP_DIMS) == 8
    assert len(set(FLIP_DIMS)) == 8
    assert () in FLIP_DIMS


def test_flip_equivariant_model_is_unchanged():
    torch.manual_seed(0)
    model, image = PerVoxel().eval(), torch.randn(1, 1, 12, 10, 8)
    with torch.no_grad():
        plain = torch.softmax(_infer(model, image, (12, 10, 8)), dim=1)
        tta = torch.softmax(_infer(model, image, (12, 10, 8), tta=True), dim=1)
    assert torch.allclose(plain, tta, atol=1e-5)


def test_output_is_a_probability_after_softmax_and_averages_the_flips():
    torch.manual_seed(1)
    shape = (12, 10, 8)
    model, image = Positional(shape).eval(), torch.randn(1, 1, *shape)
    with torch.no_grad():
        tta = torch.softmax(_infer(model, image, shape, tta=True), dim=1)
        manual = sum(
            torch.flip(torch.softmax(model(torch.flip(image, d) if d else image), dim=1), d)
            if d else torch.softmax(model(image), dim=1)
            for d in FLIP_DIMS) / 8
    assert torch.allclose(tta.sum(dim=1), torch.ones(1, *shape), atol=1e-5)
    assert torch.allclose(tta, manual, atol=1e-5)
