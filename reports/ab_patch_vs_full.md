# A/B: patch training vs whole-volume training

Whole volumes turned out to fit on an 8 GB card at half width
(`docs/patch-vs-full-volume.md`), so the question became empirical: does
training on whole heads beat training on class-balanced patches?

## Setup

Identical for all three arms — same 100 training cases, same 30 validation
cases, 35 epochs, AdamW at 1e-3 with cosine decay, Dice + cross-entropy,
bfloat16. Validation is whole-volume in every arm, so the arms are scored the
same way regardless of how they were trained.

Three arms rather than two, because patch-vs-full is confounded with width: a
full volume only fits at width 8, so a straight A/B would not say whether any
difference came from the sampling or from the smaller network. Arm C breaks
that tie.

| Arm | Sampling | Width | Params | Peak VRAM | Epoch time | **Best val Dice** |
|---|---|---|---|---|---|---|
| **A** | patches 128³ | 16 | 4.7M | 6.0 GB | 31 s | **0.561** |
| C | patches 128³ | 8 | 1.2M | 3.1 GB | 24 s | 0.479 |
| B | whole volumes | 8 | 1.2M | 3.0 GB | 23 s | 0.387 |

Per-class Dice at the best epoch:

| Arm | NETC | SNFH | ET | RC |
|---|---|---|---|---|
| **A** patches, w16 | **0.494** | 0.793 | **0.631** | **0.325** |
| C patches, w8 | 0.331 | **0.810** | 0.621 | 0.154 |
| B whole, w8 | 0.185 | 0.645 | 0.495 | 0.152 |

## Reading it

**Patching is worth more than width.** C vs B holds width fixed and changes
only the sampling: +0.09 mean Dice. A vs C holds sampling fixed and doubles
the width: +0.08. Both help; sampling helps slightly more, and they stack.

**The gap is concentrated in the rare classes, exactly where the theory says it
should be.** NETC — present in only 42% of cases and 0.047% of voxels — scores
0.494 with class-balanced patches and 0.185 on whole volumes. SNFH, which is
present in every single case, barely moves (0.793 vs 0.645). Whole-volume
training feeds the network each case's natural class mix, so a class that is
rare in the data stays rare in the gradient. `RandCropByLabelClassesd`
deliberately breaks that proportionality.

**Arm B was also the only arm that stopped improving.** It peaked at epoch 7
(0.387) and drifted down to 0.369 by epoch 35, while A and C rose steadily
throughout. It was fitting the easy, common tissue while losing the rare
classes.

**A fourth arm was not needed.** Whole volumes at width 16 would fit after
foreground cropping (arm B peaked at 3.0 GB, not the 5.4 GB the uncropped
measurement suggested). But width is worth about +0.08 and whole-volume
sampling costs about −0.09, so that arm projects to ~0.45 — still below arm A,
and it would still lack class balancing.

## Decision

**Train on class-balanced patches at width 16** (arm A). Whole-volume training
was measured, and it lost.

Absolute numbers here are low because each arm saw 100 cases for 35 epochs;
they are for ranking the configurations, not for reporting as results. The
production run uses all 944 training cases.
