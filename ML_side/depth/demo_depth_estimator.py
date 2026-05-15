"""Simple demo for the extracted depth proxy module."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ML_SIDE_ROOT = Path(__file__).resolve().parents[1]
if str(ML_SIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_SIDE_ROOT))

from depth.depth_estimator import estimate_depth


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sample_image = repo_root / "ML_side" / "testing_pipeline" / "test_assets" / "test.png"

    result = estimate_depth(
        sample_image,
        bounding_boxes=[
            {"x_min": 40, "y_min": 50, "x_max": 180, "y_max": 310},
            {"x_min": 250, "y_min": 110, "x_max": 370, "y_max": 240},
        ],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
