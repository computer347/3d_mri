# Results

One SegResNet (4.7M parameters), trained on one RTX 3070 Ti for 8.3 hours. No
ensembling, no test-time augmentation. Scored once on 203 held-out patients
that nothing in training, checkpoint selection, or post-processing tuning ever
touched.

## Glioma (BraTS 2024 post-treatment), test split

Lesion-wise Dice is the metric the challenge ranks on. Volumetric Dice is the
familiar overlap ratio, shown because the difference between the two columns is
the most interesting thing here.

| Region | Lesion-wise | + post-proc | Volumetric | Cases | BraTS best | BraTS median |
|---|---|---|---|---|---|---|
| NETC | 0.478 | 0.478 | 0.482 | 96 | 0.789 | 0.730 |
| SNFH | 0.533 | **0.575** | 0.851 | 203 | 0.876 | 0.840 |
| ET | 0.625 | 0.614 | 0.766 | 154 | 0.763 | 0.712 |
| RC | 0.655 | **0.675** | 0.725 | 169 | 0.715 | 0.672 |
| TC | 0.605 | 0.593 | 0.739 | 158 | 0.750 | 0.692 |
| WT | 0.561 | **0.597** | 0.861 | 203 | 0.879 | 0.846 |
| **mean** | **0.576** | **0.589** | | | | |

Challenge figures are from the BraTS 2024 validation-phase leaderboard (188
scored submissions, Synapse table `syn61779417`). They are not strictly
comparable: those are validation-phase scores that teams could resubmit against
repeatedly, from ensembles with far more compute. The column is there for scale,
not for a claim of parity.

Validation Dice peaked at 0.7026 (epoch 80 of 100).

### The gap between the two metrics is the finding

SNFH scores 0.851 volumetric and 0.533 lesion-wise. WT scores 0.861 and 0.561.
A 0.30 spread between two metrics on the same masks is not noise - it is a
specific, diagnosable failure.

The cause is false-positive *lesions*: 192 spurious SNFH components and 162
spurious WT components across 203 cases, roughly one per case. The model finds
the main tumour well and then marks additional tissue elsewhere. Volumetric
Dice hardly registers this, because the extra components are small relative to
the true mass. Lesion-wise Dice scores each connected component separately and
charges a full zero for every invented one.

Reporting only volumetric Dice would have hidden this completely. That is the
argument for the challenge's choice of metric, visible in our own numbers.

## Post-processing: a hypothesis that mostly failed

Predicting that removing small connected components would be the single biggest
available win, I swept a minimum-lesion-size threshold. It was not.

**A single global threshold does nothing.** Best on validation was +0.009; on
test it was **-0.0006**. The invented lesions are not specks - the scoring
already ignores components under 50 voxels - so any threshold large enough to
remove them also deletes true lesions. At 100 voxels, false positives fell
796 -> 477 while missed lesions rose 140 -> 340.

**Per-class thresholds help, modestly.** Selected on validation
(NETC 0, SNFH 400, ET 100, RC 400 voxels), they gained +0.039 on validation and
**+0.0126 on test** - about a third of what validation promised. SNFH (+0.042)
and WT (+0.036) improved; ET (-0.011) and TC (-0.012) got slightly worse,
because the ET threshold that helped on validation did not transfer.

The threshold was chosen on validation and applied to test once. Choosing it on
test would have turned a held-out score into a fitted one, and - given the
validation gain overstated the test gain threefold - would have produced a
number that looked better and meant less.

## Meningioma (BraTS 2024 MEN-RT)

Validation GTV Dice 0.456 after 40 epochs, still climbing steeply when the
schedule ended. Retrained for 120 epochs; see the training-curve report for the
converged figure.

Preprocessing to 1 mm isotropic (`mri3d.preprocess`) halved epoch time from 317s
to 166s and removed the per-epoch resampling that had exhausted system memory.

## What would improve these numbers

In rough order of expected value:

1. **More training.** Glioma converged at epoch 80, but the meningioma curve
   shows how much is lost to an under-length schedule.
2. **Attack the false positives at the source** rather than by filtering.
   Probability thresholding, deep supervision, or a loss term that penalises
   spurious components would address the cause instead of the symptom.
3. **Test-time augmentation** (flip-averaging) typically adds 0.01-0.02 and
   costs only inference time.
4. **Ensembling** across folds - the standard reason challenge entries score
   where they do.
5. **NETC is the weakest class** (0.478) and the rarest (42% of cases, 0.047%
   of voxels). Class-balanced sampling already rescued it once, from 0.487 to
   0.675 on validation during training; more aggressive oversampling may help
   further.
