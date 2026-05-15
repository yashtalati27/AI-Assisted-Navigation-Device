"""Notebook-free depth proxy utilities for ML-side integration.

This module extracts the relative depth proxy logic from
`ML_side/notebooks/cohort-2/04_training_and_depth_estimation.ipynb` into a
plain Python module that can be imported by scripts or backend code.

Public API:
    estimate_depth(image, bounding_boxes=None, alpha=0.7, beta=0.3)

Args:
    image:
        Either a NumPy array with shape `(H, W)` or `(H, W, C)`, or a file path
        to a PNG or JPEG image.
    bounding_boxes:
        Optional iterable of pixel-coordinate boxes. Each box may be either a
        mapping with keys `x_min`, `y_min`, `x_max`, `y_max`, or a 4-item
        sequence in that order.
    alpha:
        Weight for box-height contribution in the improved score.
    beta:
        Weight for vertical-position contribution in the improved score.

Returns:
    A dictionary with image dimensions, the relative unit name, and one depth
    proxy record per bounding box. Each box record contains the original bbox,
    `baseline_depth_score`, and `improved_depth_score`.

Notes:
    - Scores are relative proxy values, not metric distances in metres.
    - This module does not produce a full depth map.
    - No model weights are required for this v1 implementation.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BoundingBoxInput = Mapping[str, Any] | Sequence[float]


def baseline_depth_score(box_height_px: float, img_height_px: int) -> float:
    """Baseline depth proxy: larger boxes are assumed closer."""
    _validate_positive_image_height(img_height_px)
    return float(box_height_px) / float(img_height_px)


def improved_depth_score(
    box_height_px: float,
    box_center_y_px: float,
    img_height_px: int,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """Improved proxy: combine size and vertical position cues."""
    _validate_positive_image_height(img_height_px)
    size_score = float(box_height_px) / float(img_height_px)
    position_score = float(box_center_y_px) / float(img_height_px)
    return alpha * size_score + beta * position_score


def estimate_depth(
    image: np.ndarray | str | Path,
    bounding_boxes: Iterable[BoundingBoxInput] | None = None,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> dict[str, Any]:
    """Estimate relative depth proxy scores for bounding boxes in an image.

    The function accepts an image array or image file path and returns relative
    depth proxy scores for each supplied box. If no boxes are supplied, the
    function still returns image metadata with an empty `boxes` list.
    """
    image_height, image_width = _image_dimensions(image)
    results = []

    for index, raw_box in enumerate(bounding_boxes or ()):
        box = _normalize_bounding_box(raw_box)
        box_height_px = box["y_max"] - box["y_min"]
        box_center_y_px = (box["y_min"] + box["y_max"]) / 2.0
        results.append(
            {
                "index": index,
                "bbox": box,
                "baseline_depth_score": baseline_depth_score(
                    box_height_px,
                    image_height,
                ),
                "improved_depth_score": improved_depth_score(
                    box_height_px,
                    box_center_y_px,
                    image_height,
                    alpha=alpha,
                    beta=beta,
                ),
            }
        )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "unit": "relative_depth_score",
        "depth_map": None,
        "boxes": results,
    }


def _validate_positive_image_height(img_height_px: int) -> None:
    if img_height_px <= 0:
        raise ValueError("img_height_px must be greater than zero")


def _image_dimensions(image: np.ndarray | str | Path) -> tuple[int, int]:
    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise ValueError("image array must have at least 2 dimensions")
        return int(image.shape[0]), int(image.shape[1])

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_type = _detect_image_type(image_path)
    if image_type == "png":
        return _png_dimensions(image_path)
    if image_type == "jpeg":
        return _jpeg_dimensions(image_path)

    raise ValueError(
        f"Unsupported image format for {image_path}. Only PNG and JPEG are supported."
    )


def _detect_image_type(image_path: Path) -> str | None:
    with image_path.open("rb") as file_obj:
        header = file_obj.read(16)

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    return None


def _png_dimensions(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as file_obj:
        signature = file_obj.read(24)
    if len(signature) < 24 or signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG file: {image_path}")
    width, height = struct.unpack(">II", signature[16:24])
    return int(height), int(width)


def _jpeg_dimensions(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as file_obj:
        if file_obj.read(2) != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG file: {image_path}")

        while True:
            marker_prefix = file_obj.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue

            marker_type = file_obj.read(1)
            while marker_type == b"\xff":
                marker_type = file_obj.read(1)
            if not marker_type:
                break

            if marker_type in {b"\xd8", b"\xd9"}:
                continue

            segment_length_bytes = file_obj.read(2)
            if len(segment_length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"Corrupt JPEG segment in {image_path}")

            if marker_type in {
                b"\xc0",
                b"\xc1",
                b"\xc2",
                b"\xc3",
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                data = file_obj.read(5)
                if len(data) != 5:
                    break
                _, height, width = struct.unpack(">BHH", data)
                return int(height), int(width)

            file_obj.seek(segment_length - 2, 1)

    raise ValueError(f"Could not determine JPEG dimensions for {image_path}")


def _normalize_bounding_box(raw_box: BoundingBoxInput) -> dict[str, int]:
    if isinstance(raw_box, Mapping):
        try:
            x_min = raw_box["x_min"]
            y_min = raw_box["y_min"]
            x_max = raw_box["x_max"]
            y_max = raw_box["y_max"]
        except KeyError as exc:
            raise ValueError(
                "Bounding box mapping must contain x_min, y_min, x_max, y_max"
            ) from exc
    else:
        if len(raw_box) != 4:
            raise ValueError("Bounding box sequence must contain four values")
        x_min, y_min, x_max, y_max = raw_box

    box = {
        "x_min": int(round(float(x_min))),
        "y_min": int(round(float(y_min))),
        "x_max": int(round(float(x_max))),
        "y_max": int(round(float(y_max))),
    }

    if box["x_max"] <= box["x_min"] or box["y_max"] <= box["y_min"]:
        raise ValueError("Bounding box must have positive width and height")

    return box
