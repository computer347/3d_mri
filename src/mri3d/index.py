"""Case discovery.

A *case* is one imaging session: a set of co-registered modalities plus (in the
training sets) a label map. Several cases can belong to the same *patient* -
BraTS-GLI case IDs are ``BraTS-GLI-<patient>-<timepoint>``, and 1350 glioma
cases come from only 613 patients. Every split in this project is made over
``patient``, never over ``case_id``; splitting on cases would put two scans of
the same brain on both sides of the split and inflate the scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

GLI_MODALITIES = ("t1c", "t1n", "t2f", "t2w")

_GLI_RE = re.compile(r"BraTS-GLI-(\d+)-(\d+)$")
_MEN_RE = re.compile(r"BraTS-MEN-RT-(\d+)-(\d+)$")


@dataclass(frozen=True)
class Case:
    dataset: str  # "gli" | "men_rt"
    subset: str  # "train" | "val"
    case_id: str
    patient: str  # dataset-prefixed, so IDs never collide across datasets
    timepoint: str
    images: dict[str, Path] = field(compare=False)
    label: Path | None = field(default=None, compare=False)

    @property
    def has_label(self) -> bool:
        return self.label is not None


def _gli_cases(root: Path, subset: str) -> list[Case]:
    cases = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = _GLI_RE.match(d.name)
        if not m:
            continue
        images = {mod: d / f"{d.name}-{mod}.nii.gz" for mod in GLI_MODALITIES}
        missing = [mod for mod, p in images.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{d.name}: missing modalities {missing}")
        seg = d / f"{d.name}-seg.nii.gz"
        cases.append(
            Case("gli", subset, d.name, f"gli-{m.group(1)}", m.group(2),
                 images, seg if seg.exists() else None)
        )
    return cases


def _men_cases(root: Path, subset: str) -> list[Case]:
    cases = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = _MEN_RE.match(d.name)
        if not m:
            continue
        t1c = d / f"{d.name}_t1c.nii.gz"
        gtv = d / f"{d.name}_gtv.nii.gz"
        if not t1c.exists():
            raise FileNotFoundError(f"{d.name}: missing t1c")
        cases.append(
            Case("men_rt", subset, d.name, f"men-{m.group(1)}", m.group(2),
                 {"t1c": t1c}, gtv if gtv.exists() else None)
        )
    return cases


def load_cases(which: str = "all") -> list[Case]:
    """Enumerate cases. ``which`` is one of gli_train, gli_val, men_train, all."""
    sources = {
        "gli_train": lambda: _gli_cases(paths.GLI_TRAIN, "train"),
        "gli_val": lambda: _gli_cases(paths.GLI_VAL, "val"),
        "men_train": lambda: _men_cases(paths.MEN_TRAIN, "train"),
    }
    if which == "all":
        return [c for fn in sources.values() for c in fn()]
    if which not in sources:
        raise ValueError(f"unknown selection {which!r}; pick from {list(sources)} or 'all'")
    return sources[which]()


def patients(cases: list[Case]) -> dict[str, list[Case]]:
    out: dict[str, list[Case]] = {}
    for c in cases:
        out.setdefault(c.patient, []).append(c)
    return out
