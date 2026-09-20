"""Datasets and transforms.

Training happens on random patches; inference always runs on whole volumes via
sliding windows (see ``mri3d.predict``).

Measured on the 8 GB RTX 3070 Ti this project targets (7.45 GB actually free,
the desktop holds the rest), one training step of forward + backward at
bfloat16:

    patch 128^3, width 16      2.76 GB
    patch 160^3, width 16      5.37 GB
    full volume, width 8       5.39 GB   <- fits
    full volume, width 16      9.90 GB   <- exceeds VRAM; see below

So "a full volume does not fit" is only true at full width. Note that the
width-16 full-volume case does not raise OutOfMemoryError on Windows: the
driver silently spills to system RAM over PCIe, which is why peak allocation
can exceed the card's 8.6 GB at all. That path is not a free win - see
``docs/patch-vs-full-volume.md`` for what it costs and which configuration this
project ended up using.

The sampling detail that matters most is ``RandCropByLabelClassesd``. Uniformly
random patches would be almost all background - tumour is ~1% of voxels, and
NETC appears in only 42% of cases - so the rare classes would barely appear in
a batch. Ratios below oversample patches centred on each class instead.
"""

from __future__ import annotations

from monai import transforms as T
from monai.data import CacheDataset, DataLoader, Dataset

from .index import load_cases
from .splits import load_split

PATCH = (128, 128, 128)
# One slot per label value (0=background, then the four tumour classes). The
# background share is deliberately small; every class is otherwise starved.
CLASS_RATIOS = [1, 2, 2, 3, 2]


def case_dicts(dataset: str = "gli", split: str = "train") -> list[dict]:
    """MONAI-style records: ``{"image": [4 paths], "label": path, "case_id": str}``."""
    wanted = set(load_split(dataset, split))
    which = "gli_train" if dataset == "gli" else "men_train"
    out = []
    for c in load_cases(which):
        if c.case_id not in wanted:
            continue
        out.append({
            "image": [c.images[m].as_posix() for m in sorted(c.images)],
            "label": c.label.as_posix(),
            "case_id": c.case_id,
        })
    if not out:
        raise RuntimeError(f"no cases for {dataset}/{split}")
    return out


def base_transforms(n_classes: int) -> list:
    """Load, orient and normalise - shared by training and validation."""
    return [
        T.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        # Stack the modalities into channels; single-modality datasets pass through.
        T.ConcatItemsd(keys="image", name="image", dim=0),
        T.EnsureTyped(keys=["image", "label"]),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Per channel, per case: BraTS intensities have no absolute meaning, and
        # nonzero-only statistics stop the black air around the head from
        # dominating the mean.
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]


def train_transforms(n_classes: int = 5, patch=PATCH) -> T.Compose:
    return T.Compose(base_transforms(n_classes) + [
        T.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        T.SpatialPadd(keys=["image", "label"], spatial_size=patch),
        T.RandCropByLabelClassesd(
            keys=["image", "label"], label_key="label", spatial_size=patch,
            ratios=CLASS_RATIOS[:n_classes], num_classes=n_classes, num_samples=2,
        ),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])


def full_volume_transforms(n_classes: int = 5) -> T.Compose:
    """Train on whole heads instead of patches.

    Viable only at width 8 on this card. The model sees every case in full
    context and needs no sliding window at inference, but it loses the
    class-balanced sampling that ``RandCropByLabelClassesd`` provides, so rare
    classes are represented only as often as they naturally occur.
    """
    return T.Compose(base_transforms(n_classes) + [
        T.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        T.DivisiblePadd(keys=["image", "label"], k=8),  # SegResNet downsamples 3x
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])


def val_transforms(n_classes: int = 5) -> T.Compose:
    # No cropping or augmentation: validation scores whole volumes, the way
    # inference will run.
    return T.Compose(base_transforms(n_classes))


def loaders(dataset: str = "gli", batch_size: int = 1, workers: int = 4,
            cache_rate: float = 0.0, limit: int = 0, patch=PATCH,
            mode: str = "patch", val_limit: int = 0):
    n_classes = 5 if dataset == "gli" else 2
    train_files = case_dicts(dataset, "train")
    val_files = case_dicts(dataset, "val")
    if limit:  # debug mode: overfit a handful of cases
        train_files, val_files = train_files[:limit], train_files[:limit]
    else:
        if val_limit:
            val_files = val_files[:val_limit]

    make = (lambda f, t: CacheDataset(f, t, cache_rate=cache_rate, num_workers=workers)) \
        if cache_rate > 0 else (lambda f, t: Dataset(f, t))

    tf = full_volume_transforms(n_classes) if mode == "full" else train_transforms(n_classes, patch)
    train_ds = make(train_files, tf)
    val_ds = make(val_files, val_transforms(n_classes))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                   pin_memory=True, persistent_workers=workers > 0, drop_last=True),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, workers // 2),
                   pin_memory=True, persistent_workers=workers > 0),
        n_classes,
    )
