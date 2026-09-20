# Patches or whole volumes? Measuring instead of assuming

The usual advice for 3D medical segmentation on a consumer GPU is "train on
patches, because the volume won't fit". On an 8 GB RTX 3070 Ti with the BraTS
2024 glioma data (182×218×182 × 4 modalities), that advice turns out to be
half wrong, and the half that's wrong is the interesting one.

## What actually fits

One training step, forward + backward, bfloat16 autocast, batch size 1,
SegResNet. 7.45 GB of the card's 8.59 GB was free (the Windows desktop holds
the rest).

| Configuration | Peak allocated | Step time | Throughput |
|---|---|---|---|
| patch 128³, width 16 | 2.80 GB | 0.14 s | 15.2M voxels/s |
| patch 160³, width 16 | 5.41 GB | 0.27 s | 15.3M voxels/s |
| **full volume, width 8** | **5.40 GB** | **0.36 s** | **21.3M voxels/s** |
| full volume, width 16 | 9.94 GB | 15.20 s | 0.5M voxels/s |

Two things stand out.

**A whole volume fits at width 8.** 5.40 GB, comfortably inside 7.45 GB, and it
is the *fastest* configuration measured per voxel — a single large tensor keeps
the GPU busier than a small one does, and a whole case is covered in one step
instead of several overlapping patches.

**Width 16 on a full volume does not raise `OutOfMemoryError`.** It reports a
peak of 9.94 GB, which is more memory than the card physically has. On Windows
the driver silently spills to system RAM over PCIe rather than failing, so the
step still completes — in 15.2 seconds instead of 0.36. That is a **42×**
slowdown, which works out to roughly four hours per epoch over 944 training
cases. It "works" in the sense that it never crashes, and that is exactly what
makes it dangerous: a run left going overnight looks healthy and simply never
finishes.

So the honest constraint is not "a full volume doesn't fit". It is: **a full
volume fits only if you halve the network width, and exceeding VRAM degrades
silently rather than loudly.**

## Why this is a real trade-off

Neither option dominates, which is why it needed an experiment rather than an
opinion.

**Patches, width 16** — twice the feature width at every level, and
`RandCropByLabelClassesd` can oversample the rare classes. Tumour is ~1% of
voxels and NETC appears in only 42% of cases, so class-balanced sampling puts
far more rare-class voxels into each gradient step than nature does. The cost:
the model never sees a whole head during training, so it has no way to learn
that (say) a resection cavity is a single object rather than a texture.

**Full volumes, width 8** — every step sees the entire head in context, and
inference needs no sliding window, so there are no window-boundary artefacts to
blend away. The cost: half the width, and no class balancing at all — each case
contributes its natural class mix, so the rare classes are rare in training too.

## Result

See `reports/ab_patch_vs_full.md` for the measured comparison — both
configurations trained on the same cases, for the same number of epochs, scored
on the same validation split.
