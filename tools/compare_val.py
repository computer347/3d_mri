"""Gate for a pre-registered decision: does run A beat run B on validation?

Compares the mean lesion-wise Dice over all regions (mean over the cases where
each region is present), from reports/summary_{run}_val.json. Exit code 0 when
A is strictly better, 1 otherwise - so a queue script can stop before the test
split is touched.

    python tools/compare_val.py gli_main_rl gli_main
"""

import json
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def mean_lesion(run: str) -> float:
    s = json.loads((REPORTS / f"summary_{run}_val.json").read_text(encoding="utf-8"))
    vals = [v["mean_where_present"] for k, v in s.items()
            if k.startswith("lesion_dice_") and v["mean_where_present"] is not None]
    return sum(vals) / len(vals)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    ma, mb = mean_lesion(a), mean_lesion(b)
    better = ma > mb
    print(f"validation mean lesion-wise: {a} {ma:.4f} vs {b} {mb:.4f} -> "
          f"{'PROCEED' if better else 'STOP'}")
    sys.exit(0 if better else 1)
