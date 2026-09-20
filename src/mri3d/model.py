"""Model and loss.

``SegResNet`` is a residual 3D encoder-decoder from the BraTS lineage - the
architecture family that has won or placed in these challenges repeatedly, and
which still outperforms transformer variants at this data scale. At 8 GB it
also leaves room for a 128^3 patch, which SwinUNETR does not.

The loss is Dice + cross entropy, together. Cross entropy alone optimises
happily by predicting background everywhere: tumour is ~1% of voxels, so that
strategy is 99% "accurate". Dice is a per-class overlap ratio, so a small
tumour counts as much as the huge background, while the cross-entropy term
keeps gradients well behaved early on when Dice is near zero for every class.
"""

from __future__ import annotations

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet


def build_model(n_classes: int = 5, in_channels: int = 4, width: int = 16) -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        init_filters=width,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )


def build_loss() -> DiceCELoss:
    return DiceCELoss(
        include_background=False,  # background Dice is ~1.0 and would mask real movement
        to_onehot_y=True,          # labels arrive as integers (0-4), not one-hot
        softmax=True,              # the 2024 sub-regions are mutually exclusive
        squared_pred=True,
        batch=True,                # pool the Dice denominator across the batch, so a
                                   # patch that happens to lack a class does not emit a
                                   # degenerate 0/0 gradient for it
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


def describe(model: torch.nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{type(model).__name__}, {n/1e6:.1f}M parameters"
