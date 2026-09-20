"""
Buhler drawing -> camera validation

Dynamic annotation-driven validator.

Rules:
- The JSON file is the ONLY source of feature/annotation geometry.
- Every annotation is processed. Nothing is filtered by tag_id/tag_name.
- The processing method is selected ONLY from annotation["shape"].
- circle   -> bbox top/bottom/left/right -> circle ROI -> image circle fit
- rect/square -> bbox top/bottom/left/right -> rectangle measurement/drawing
- polygon  -> annotation points -> polygon measurement/drawing
- A polygon with 6 points is reported as a 6-vertex polygon (hexagon).
- No CVAT ZIP/XML is read.
- No feature coordinates or nominal values are hard-coded in this script.

Inputs are supplied at runtime:
    --json
    --image
    --calibration
    --distance-mm

Outputs:
    overlay image
    JSON report
"""

from __future__ import annotations

import argparse
import base64
import csv
from fileinput import filename
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def shape_name(annotation: Dict[str, Any]) -> str:
    """The annotation shape is the ONLY dispatch key."""
    return str(annotation.get("shape", "")).strip().lower()


def annotation_bbox(annotation: Dict[str, Any]) -> Tuple[float, float, float, float]:
    left = number(annotation.get("left"), 0.0)
    top = number(annotation.get("top"), 0.0)
    width = number(annotation.get("width"), 0.0)
    height = number(annotation.get("height"), 0.0)
    assert left is not None and top is not None and width is not None and height is not None
    return left, top, width, height


def bbox_edges(annotation: Dict[str, Any]) -> Dict[str, float]:
    left, top, width, height = annotation_bbox(annotation)
    return {
        "left": left,
        "top": top,
        "right": left + width,
        "bottom": top + height,
        "width": width,
        "height": height,
        "center_x": left + width / 2.0,
        "center_y": top + height / 2.0,
    }


def clip_bbox(
    annotation: Dict[str, Any], image_shape: Sequence[int]
) -> Tuple[int, int, int, int]:
    """Return x1,y1,x2,y2 clipped to the camera image."""
    h, w = image_shape[:2]
    b = bbox_edges(annotation)
    x1 = max(0, min(w - 1, int(math.floor(b["left"]))))
    y1 = max(0, min(h - 1, int(math.floor(b["top"]))))
    x2 = max(x1 + 1, min(w, int(math.ceil(b["right"]))))
    y2 = max(y1 + 1, min(h, int(math.ceil(b["bottom"]))))
    return x1, y1, x2, y2


def points_from_annotation(annotation: Dict[str, Any]) -> List[Tuple[float, float]]:
    points = []
    for p in annotation.get("points", []) or []:
        x = number(p.get("x")) if isinstance(p, dict) else None
        y = number(p.get("y")) if isinstance(p, dict) else None
        if x is not None and y is not None:
            points.append((x, y))
    return points


def mm_per_pixel(distance_mm: float, fx: float, fy: float) -> Tuple[float, float]:
    return distance_mm / fx, distance_mm / fy


def px_distance(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def convert_dxdy_to_mm(dx_px: float, dy_px: float, x_scale: float, y_scale: float) -> float:
    return math.hypot(dx_px * x_scale, dy_px * y_scale)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def load_calibration(path: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Load camera calibration from a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Camera matrix
    if "camera_matrix" not in data:
        raise KeyError("Could not find 'camera_matrix' in calibration JSON.")

    # Distortion coefficients
    if "distortion_coefficients" not in data:
        raise KeyError(
            "Could not find 'distortion_coefficients' in calibration JSON."
        )

    camera_matrix = np.asarray(
        data["camera_matrix"],
        dtype=np.float64
    )

    distortion = np.asarray(
        data["distortion_coefficients"],
        dtype=np.float64
    )

    # Validate camera matrix
    if camera_matrix.shape != (3, 3):
        raise ValueError(
            f"Invalid camera matrix shape {camera_matrix.shape}; "
            "expected (3, 3)."
        )

    # Get focal lengths directly from camera matrix
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])

    if fx <= 0 or fy <= 0:
        raise ValueError(
            "Calibration contains invalid fx/fy values."
        )

    print(f"Calibration camera matrix key : camera_matrix")
    print(f"Calibration distortion key    : distortion_coefficients")
    print(f"Calibration JSON camera       : {data.get('camera', 'N/A')}")
    print(f"Calibration JSON lens         : {data.get('lens', 'N/A')}")
    print(f"Calibration fx                : {fx}")
    print(f"Calibration fy                : {fy}")
    print(f"Calibration reprojection err  : "
          f"{data.get('reprojection_error_pixels', 'N/A')} px")

    return camera_matrix, distortion, fx, fy
# -----------------------------------------------------------------------------
# Circle method
# -----------------------------------------------------------------------------

def fit_circle_least_squares(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Algebraic least-squares circle fit: x^2+y^2+Ax+By+C=0."""
    if points is None or len(points) < 3:
        return None

    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x * x + y * y)

    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, bb, c = coef
    cx = -a / 2.0
    cy = -bb / 2.0
    r2 = cx * cx + cy * cy - c
    if r2 <= 0 or not np.isfinite(r2):
        return None
    r = math.sqrt(r2)
    return float(cx), float(cy), float(r)


def circle_edge_support(
    edge_image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    band_px: float = 1.0,
) -> float:
    """Fraction of sampled circumference points supported by image edges."""
    if radius <= 0:
        return 0.0

    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    support = 0

    h, w = edge_image.shape[:2]
    for c, s in zip(cos_a, sin_a):
        x = int(round(cx + radius * c))
        y = int(round(cy + radius * s))
        found = False
        rr = max(1, int(math.ceil(band_px)))
        for yy in range(max(0, y - rr), min(h, y + rr + 1)):
            for xx in range(max(0, x - rr), min(w, x + rr + 1)):
                if edge_image[yy, xx] != 0:
                    found = True
                    break
            if found:
                break
        support += int(found)

    return support / len(angles)


def refine_circle_from_edges(
    edge_image: np.ndarray,
    initial: Tuple[float, float, float],
    max_distance_px: float = 3.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Refine a Hough circle using edge points near its circumference.
    Returns cx,cy,radius,edge_support.
    """
    cx0, cy0, r0 = initial
    ys, xs = np.nonzero(edge_image)
    if len(xs) < 3:
        return None

    d = np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2)
    keep = np.abs(d - r0) <= max_distance_px
    pts = np.column_stack((xs[keep], ys[keep]))
    if len(pts) < 3:
        return None

    # Reject extreme point counts caused by unrelated edges.
    if len(pts) > 12000:
        step = int(math.ceil(len(pts) / 12000))
        pts = pts[::step]

    fitted = fit_circle_least_squares(pts)
    if fitted is None:
        return None

    cx, cy, radius = fitted
    support = circle_edge_support(edge_image, cx, cy, radius)
    return cx, cy, radius, support


def hough_candidates(gray_roi: np.ndarray, min_radius: int, max_radius: int) -> List[Tuple[float, float, float]]:
    """Try several Hough thresholds so small and large circles can both be found."""
    if min_radius < 1 or max_radius < min_radius:
        return []
    
    blur = cv2.GaussianBlur(gray_roi, (5, 5), 1.2)
    candidates: List[Tuple[float, float, float]] = []

    # The thresholds are algorithm settings, not feature/sample values.
    for p2 in (8, 10, 12, 15, 18, 22, 26, 30, 35):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(3.0, float(min_radius)),
            param1=80,
            param2=float(p2),
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            for c in np.round(circles[0]).astype(np.float64):
                candidates.append((float(c[0]), float(c[1]), float(c[2])))

    return candidates

import cv2
import numpy as np


def average_radius_from_edges(
    edges: np.ndarray,
    initial_circle: Tuple[float, float, float],
    max_distance_px: float = 20.0,
    angle_bins: int = 360,
    trim_fraction: float = 0.05,
):
    """
    Use an initial Hough circle only as a search guide.

    For each angular direction around the Hough center:
      - find Canny edge pixels near the Hough radius
      - select the edge closest to the Hough radius
      - collect one actual edge point per angular bin

    After selecting the edge points:
      - calculate the centroid of the selected points
      - use the centroid as the final center
      - calculate each point's radius from that centroid
      - robustly average the radii

    Returns:
        (cx, cy, average_radius, support, selected_points)

    where:
        cx, cy          = centroid of selected edge points
        average_radius  = average radial distance from centroid
        support         = fraction of angular bins containing an edge
        selected_points = Nx2 array of actual selected edge pixels
    """

    # --------------------------------------------------
    # Initial Hough circle
    # Used ONLY as a search/reference guide.
    # --------------------------------------------------

    hough_cx, hough_cy, initial_r = initial_circle

    ys, xs = np.nonzero(edges)

    if len(xs) < 5:
        return None

    # --------------------------------------------------
    # Radius and angle of every Canny edge pixel
    # relative to the Hough center.
    # --------------------------------------------------

    dx = xs.astype(np.float64) - hough_cx
    dy = ys.astype(np.float64) - hough_cy

    radii = np.sqrt(dx * dx + dy * dy)

    angles = np.arctan2(dy, dx)

    # Convert [-pi, pi] -> [0, 2pi)
    angles = np.mod(angles, 2.0 * np.pi)

    # --------------------------------------------------
    # Keep only pixels in an annulus around Hough circle.
    # --------------------------------------------------

    print("All Canny edge pixels:", len(radii))

    keep = np.abs(radii - initial_r) <= max_distance_px

    xs = xs[keep]
    ys = ys[keep]
    radii = radii[keep]
    angles = angles[keep]

    print(
        "Edge pixels near Hough circle:",
        np.count_nonzero(keep)
    )

    if len(radii) < 5:
        return None

    # --------------------------------------------------
    # Put edge pixels into angular bins.
    # --------------------------------------------------

    bin_ids = np.floor(
        angles / (2.0 * np.pi) * angle_bins
    ).astype(np.int32)

    bin_ids = np.clip(
        bin_ids,
        0,
        angle_bins - 1
    )

    selected_points = []

    # --------------------------------------------------
    # Select one actual boundary point per angle.
    # --------------------------------------------------

    for bin_id in range(angle_bins):

        idx = np.where(bin_ids == bin_id)[0]

        if len(idx) == 0:
            continue

        candidate_radii = radii[idx]

        # --------------------------------------------------
        # Select the edge closest to the Hough prediction.
        #
        # Hough is only a guide here. The selected pixel
        # itself becomes the actual measured boundary.
        # --------------------------------------------------

        best_local = idx[
            np.argmin(
                np.abs(candidate_radii - initial_r)
            )
        ]

        selected_points.append(
            (
                xs[best_local],
                ys[best_local]
            )
        )

    if len(selected_points) < 5:
        return None

    selected_points = np.asarray(
        selected_points,
        dtype=np.float64
    )

    # --------------------------------------------------
    # Calculate centroid of the ACTUAL selected points.
    #
    # Hough center is no longer used as the final center.
    # --------------------------------------------------

    cx = np.mean(selected_points[:, 0])
    cy = np.mean(selected_points[:, 1])

    # --------------------------------------------------
    # Calculate radius of every selected point relative
    # to the centroid.
    # --------------------------------------------------

    dx = selected_points[:, 0] - cx
    dy = selected_points[:, 1] - cy

    selected_radii = np.sqrt(
        dx * dx + dy * dy
    )

    # --------------------------------------------------
    # Robust averaging
    # --------------------------------------------------

    sorted_r = np.sort(selected_radii)

    trim = int(
        len(sorted_r) * trim_fraction
    )

    if trim > 0 and len(sorted_r) > 2 * trim:
        radius_for_average = sorted_r[
            trim:-trim
        ]
    else:
        radius_for_average = sorted_r

    average_radius = np.mean(
        radius_for_average
    )

    # --------------------------------------------------
    # Fraction of circumference where we found
    # a usable edge.
    # --------------------------------------------------

    support = (
        len(selected_points)
        / float(angle_bins)
    )

    return (
        float(cx),
        float(cy),
        float(average_radius),
        float(support),
        selected_points
    )
def estimate_threshold_from_edges(gray, transition_fraction=0.3):
    """
    Estimate threshold for one expanded circle ROI.

    Valid edges:
        black -> white : accepted
        black -> gray  : accepted

    Rejected:
        gray -> white

    Main criterion:
        The DARK side of the gradient must be close to the
        black background.

    The threshold is calculated relative to each detected
    black-side transition instead of using a fixed +15/+50.

    transition_fraction:
        0.0 -> very close to dark side
        0.5 -> middle of transition
        0.7 -> pushed toward bright side
        1.0 -> bright side
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # 1. Blur
    # -------------------------------------------------

    img = cv2.GaussianBlur(gray, (5, 5), 0)

    # -------------------------------------------------
    # 2. Estimate black background
    # -------------------------------------------------

    # Keep this close to your original implementation.
    bg_est = float(np.percentile(img, 20))

    # IMPORTANT:
    # Do NOT make this strongly dependent on the whole
    # ROI intensity range.
    #
    # Your original value of 30 was behaving better.
    bg_band = 30.0

    bg_limit = bg_est + bg_band

    # -------------------------------------------------
    # 3. Gradient
    # -------------------------------------------------

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    mag = cv2.magnitude(gx, gy)

    # Keep your original strong-edge selection.
    grad_threshold = np.percentile(mag, 85)

    ys, xs = np.where(mag >= grad_threshold)

    h, w = img.shape

    # -------------------------------------------------
    # 4. Store valid transitions
    # -------------------------------------------------

    edge_thresholds = []

    dark_values = []
    bright_values = []
    edge_weights = []

    # -------------------------------------------------
    # 5. Inspect both sides of every gradient
    # -------------------------------------------------

    for y, x in zip(ys, xs):

        m = float(mag[y, x])

        if m < 1e-6:
            continue

        # Unit vector along gradient
        dx = float(gx[y, x]) / m
        dy = float(gy[y, x]) / m

        # Keep the original sampling distance.
        step = 2.0

        x1 = int(round(x + dx * step))
        y1 = int(round(y + dy * step))

        x2 = int(round(x - dx * step))
        y2 = int(round(y - dy * step))

        if (
            x1 < 0 or x1 >= w or
            y1 < 0 or y1 >= h or
            x2 < 0 or x2 >= w or
            y2 < 0 or y2 >= h
        ):
            continue

        v1 = float(img[y1, x1])
        v2 = float(img[y2, x2])

        dark = min(v1, v2)
        bright = max(v1, v2)

        contrast = bright - dark

        # Ignore weak transitions
        if contrast < 15:
            continue

        # -------------------------------------------------
        # IMPORTANT:
        #
        # ONLY use the dark side to decide whether this
        # is a valid edge.
        #
        # This means:
        #
        # black -> gray  KEEP
        # black -> white KEEP
        # gray  -> white REJECT
        #
        # because gray -> white has a dark side that is
        # not close enough to the black background.
        # -------------------------------------------------

        if dark > bg_limit:
            continue

        # -------------------------------------------------
        # 6. Normalize the LOCAL transition
        # -------------------------------------------------

        #
        # Old:
        #
        #     threshold = dark_upper + 15
        #
        # New:
        #
        #     threshold = dark + fraction*(bright-dark)
        #
        # This makes the threshold relative to the
        # transition instead of using an absolute offset.
        #

        local_threshold = (
            dark
            + transition_fraction * contrast
        )

        edge_thresholds.append(local_threshold)

        dark_values.append(dark)
        bright_values.append(bright)

        # -------------------------------------------------
        # 7. Weight by how close dark is to background
        # -------------------------------------------------

        #
        # This is important.
        #
        # Example:
        #
        # bg_est = 25
        #
        # edge A:
        # dark = 26
        #
        # edge B:
        # dark = 50
        #
        # Edge A should have much more influence.
        #

        distance_from_bg = abs(dark - bg_est)

        closeness = max(
            0.0,
            1.0 - distance_from_bg / bg_band
        )

        # Keep some influence from every valid edge,
        # but strongly favor black-side edges.
        weight = 0.2 + 0.8 * closeness

        # Also include gradient strength.
        weight *= m

        edge_weights.append(weight)

    # -------------------------------------------------
    # 8. Fallback
    # -------------------------------------------------

    if len(edge_thresholds) < 10:

        threshold = int(
            np.clip(
                bg_est + 20,
                0,
                255
            )
        )

        return threshold, {
            "background": bg_est,
            "bg_limit": bg_limit,
            "num_edges": len(edge_thresholds),
            "method": "fallback"
        }

    # -------------------------------------------------
    # 9. Convert to arrays
    # -------------------------------------------------

    edge_thresholds = np.asarray(
        edge_thresholds,
        dtype=np.float32
    )

    dark_values = np.asarray(
        dark_values,
        dtype=np.float32
    )

    bright_values = np.asarray(
        bright_values,
        dtype=np.float32
    )

    edge_weights = np.asarray(
        edge_weights,
        dtype=np.float32
    )

    # -------------------------------------------------
    # 10. Weighted median
    # -------------------------------------------------

    #
    # Rather than averaging, use a weighted median.
    #
    # This prevents a few strange gradients from pulling
    # the threshold too far.
    #

    order = np.argsort(edge_thresholds)

    sorted_thresholds = edge_thresholds[order]
    sorted_weights = edge_weights[order]

    cumulative = np.cumsum(sorted_weights)

    total_weight = cumulative[-1]

    index = np.searchsorted(
        cumulative,
        0.50 * total_weight
    )

    index = min(
        index,
        len(sorted_thresholds) - 1
    )

    threshold = float(sorted_thresholds[index])

    # -------------------------------------------------
    # 11. Diagnostics
    # -------------------------------------------------

    dark_upper = float(
        np.percentile(dark_values, 95)
    )

    bright_lower = float(
        np.percentile(bright_values, 10)
    )

    return int(
        np.clip(threshold, 0, 255)
    ), {
        "background": bg_est,
        "bg_limit": bg_limit,

        "dark_upper": dark_upper,
        "bright_lower": bright_lower,

        "threshold_p25": float(
            np.percentile(edge_thresholds, 25)
        ),

        "threshold_median": float(
            np.percentile(edge_thresholds, 50)
        ),

        "threshold_p75": float(
            np.percentile(edge_thresholds, 75)
        ),

        "transition_fraction": transition_fraction,

        "num_edges": len(edge_thresholds),

        "method": "normalized_black_side_gradient"
    }
def find_dark_edges(gray, grad_percentile=85, dark_limit=70):
    """
    Find strong grayscale transitions where one side is dark.

    Returns:
        points: Nx2 array of (x, y)
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)

    grad_threshold = np.percentile(
        magnitude,
        grad_percentile
    )

    ys, xs = np.where(magnitude >= grad_threshold)

    h, w = gray.shape

    selected = []

    for y, x in zip(ys, xs):

        mag = magnitude[y, x]

        if mag < 1:
            continue

        # Gradient direction
        nx = gx[y, x] / mag
        ny = gy[y, x] / mag

        # Sample on both sides of transition
        step = 2.0

        x_dark_side = int(round(x - nx * step))
        y_dark_side = int(round(y - ny * step))

        x_other_side = int(round(x + nx * step))
        y_other_side = int(round(y + ny * step))

        if (
            x_dark_side < 0 or x_dark_side >= w or
            y_dark_side < 0 or y_dark_side >= h or
            x_other_side < 0 or x_other_side >= w or
            y_other_side < 0 or y_other_side >= h
        ):
            continue

        v1 = float(img[y_dark_side, x_dark_side])
        v2 = float(img[y_other_side, x_other_side])

        dark = min(v1, v2)
        bright = max(v1, v2)

        contrast = bright - dark

        # Must be a real transition
        if contrast < 20:
            continue

        # -------------------------------------------------
        # IMPORTANT:
        # One side must actually be dark.
        # -------------------------------------------------
        if dark > dark_limit:
            continue

        # -------------------------------------------------
        # Put point at the transition itself.
        #
        # x,y is approximately the center of the gradient.
        # Move slightly toward the dark side.
        # -------------------------------------------------
        edge_x = x - nx * 0.5
        edge_y = y - ny * 0.5

        selected.append((edge_x, edge_y))

    if not selected:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(selected, dtype=np.float32)

def detect_shape_polarity(
    gray: np.ndarray,
    annotation: Dict[str, Any],
) -> str:
    """
    Determine whether the annotated shape is a hole or a solid object.

    The decision is based only on grayscale intensity:

        HOLE:
            center is brighter than the surrounding region

        SOLID:
            center is darker than the surrounding region

    Returns:
        "hole" or "solid"
    """
    b = bbox_edges(annotation)

    h, w = gray.shape[:2]

    x1 = max(0, int(round(b["left"])))
    y1 = max(0, int(round(b["top"])))
    x2 = min(w, int(round(b["right"])))
    y2 = min(h, int(round(b["bottom"])))

    if x2 <= x1 or y2 <= y1:
        return "hole"

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return "hole"

    rh, rw = roi.shape[:2]

    # Center region.
    # Keep it well away from the actual edge.
    cx1 = int(rw * 0.30)
    cx2 = int(rw * 0.70)
    cy1 = int(rh * 0.30)
    cy2 = int(rh * 0.70)

    center = roi[cy1:cy2, cx1:cx2]

    # Outer ring of the annotation ROI.
    # Avoid the corners because they can contain unrelated background.
    ring_mask = np.zeros((rh, rw), dtype=np.uint8)

    margin = max(1, int(round(min(rw, rh) * 0.10)))

    ring_mask[:margin, :] = 1
    ring_mask[-margin:, :] = 1
    ring_mask[:, :margin] = 1
    ring_mask[:, -margin:] = 1

    ring = roi[ring_mask.astype(bool)]

    if center.size == 0 or ring.size == 0:
        return "hole"

    center_value = float(np.median(center))
    outside_value = float(np.median(ring))

    if center_value >= outside_value:
        return "hole"

    return "solid"
def detect_circle_from_annotation(
    gray: np.ndarray,
    annotation: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    Circle annotation method:
      1. read left/top/width/height from JSON
      2. make annotation ROI
      3. find an image circle inside that ROI
      4. refine the detected circle from image edges

    The JSON bbox is never replaced by a global/tag-based search.
    """
    shape_type = detect_shape_polarity(gray, annotation)

    if shape_type == "solid":
        gray = cv2.bitwise_not(gray)

    b = bbox_edges(annotation)
    # x1, y1, x2, y2 = clip_bbox(annotation, gray.shape)
    # roi = gray[y1:y2, x1:x2]
    # if roi.size == 0:
    #     return None
    roi_width = b["right"] - b["left"]
    roi_height = b["bottom"] - b["top"]

    expand_x = roi_width * 0.10
    expand_y = roi_height * 0.10

    expanded_left = b["left"] - expand_x
    expanded_top = b["top"] - expand_y
    expanded_right = b["right"] + expand_x
    expanded_bottom = b["bottom"] + expand_y

    # Clip expanded ROI to image boundaries
    h, w = gray.shape[:2]

    x1 = max(0, int(round(expanded_left)))
    y1 = max(0, int(round(expanded_top)))
    x2 = min(w, int(round(expanded_right)))
    y2 = min(h, int(round(expanded_bottom)))

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    h, w = roi.shape[:2]
    # Keep candidate radii tied to this annotation's own bbox.
    annotation_radius_x = max(0.5, b["width"] / 2.0)
    annotation_radius_y = max(0.5, b["height"] / 2.0)
    annotation_radius = (annotation_radius_x + annotation_radius_y) / 2.0
    min_radius = max(1, int(round(min(annotation_radius_x, annotation_radius_y) * 0.30)))
    max_radius = max(min_radius, int(round(max(annotation_radius_x, annotation_radius_y) * 1.30)))
    max_radius = min(max_radius, max(1, min(h, w) // 2 + 3))

    candidates = hough_candidates(roi, min_radius, max_radius)
    # threshold = get_circle_threshold(roi)
    threshold,info = estimate_threshold_from_edges(roi)

    print("binary threshold: ",threshold)
    _, roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
    # # blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)
    
    edges = cv2.Canny(roi, 40, 120)
    # edges = circle_edges(roi)

    # 
    # filename = f"binary_{annotation['tag_id']}.png"
    

    annotation_cx = b["center_x"] - x1
    annotation_cy = b["center_y"] - y1
    annotation_r = annotation_radius

    best = None
    best_score = -float("inf")

    # De-duplicate Hough results before scoring.
    unique: List[Tuple[float, float, float]] = []
    for c in candidates:
        if not any(px_distance(c[0] - q[0], c[1] - q[1]) < 2 and abs(c[2] - q[2]) < 2 for q in unique):
            unique.append(c)

    for cx, cy, r in unique:
        center_error = px_distance(cx - annotation_cx, cy - annotation_cy)
        radius_error = abs(r - annotation_r)
        support = circle_edge_support(edges, cx, cy, r)

        # Prefer circles that fit the annotation ROI and have strong image-edge support.
        score = (
            3.0 * support
            - 0.015 * center_error
            - 0.020 * radius_error
        )
        if score > best_score:
            best_score = score
            best = (cx, cy, r)

    if best is None:
        # Edge-only fallback: fit a circle to points near the JSON annotation circle.
        print(filename)
        print("edge only fallback")
        ys, xs = np.nonzero(edges)
        if len(xs) >= 3:
            d = np.sqrt((xs - annotation_cx) ** 2 + (ys - annotation_cy) ** 2)
            keep = np.abs(d - annotation_r) <= max(3.0, annotation_r * 0.20)
            points = np.column_stack((xs[keep], ys[keep]))
            fitted = fit_circle_least_squares(points)
            if fitted is not None:
                best = fitted
        

    if best is None:
        return None
    
    selected_points = None

    refined = average_radius_from_edges(
    edges,
    best,
    max_distance_px=max(
        8.0,
        annotation_r * 0.20
    ),
    angle_bins=360,
    trim_fraction=0.05
)
    if refined is None:
        cx, cy, r = best

        support = circle_edge_support(
            edges,
            cx,
            cy,
            r
        )

    else:
        cx, cy, r, support, selected_points = refined
    # refined = refine_circle_from_edges(edges, best, max_distance_px=max(2.0, annotation_r * 0.08))
    # # refined = None
    # if refined is None:
    #     cx, cy, r = best
    #     support = circle_edge_support(edges, cx, cy, r)
    # else:
    #     cx, cy, r, support = refined

    if selected_points is not None:
        contour_global = selected_points.copy()

        contour_global[:, 0] += x1
        contour_global[:, 1] += y1
    else:
        contour_global = None

    return {
        "center_x": float(cx + x1),
        "center_y": float(cy + y1),
        "radius_px": float(r),
        "diameter_px": float(2.0 * r),
        "edge_support": float(support),
        "contour_points": (
        contour_global.tolist()
        if contour_global is not None
        else None
    ),
        "roi_left": float(x1),
        "roi_top": float(y1),
        "roi_right": float(x2),
        "roi_bottom": float(y2),
    }


# -----------------------------------------------------------------------------
# Drawing methods selected by annotation shape
# -----------------------------------------------------------------------------

def draw_circle_annotation(
    overlay: np.ndarray,
    annotation: Dict[str, Any],
    detected: Optional[Dict[str, Any]],
    index: int,
    measured_mm: Optional[float],
) -> None:
    """Draw the red annotation box, green detected contour and mm measurement."""
    b = bbox_edges(annotation)

    cv2.rectangle(
        overlay,
        (int(round(b["left"])), int(round(b["top"]))),
        (int(round(b["right"])), int(round(b["bottom"]))),
        (0, 0, 255),
        1,
    )

    if detected is None:
        return

    points = detected.get("contour_points")
    if points is not None and len(points) >= 2:
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [contour], True, (0, 255, 0), 1)

    if measured_mm is not None:
        text = f"{index}: {measured_mm:.2f} mm"
        cv2.putText(
            overlay,
            text,
            (int(round(b["left"])), max(20, int(round(b["top"])) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_rect_annotation(
    overlay: np.ndarray,
    annotation: Dict[str, Any],
    detected: Optional[Dict[str, Any]],
    index: int,
    width_mm: Optional[float],
    height_mm: Optional[float],
) -> None:
    """Draw the red annotation box, green detected edges and mm dimensions."""
    b = bbox_edges(annotation)

    cv2.rectangle(
        overlay,
        (int(round(b["left"])), int(round(b["top"]))),
        (int(round(b["right"])), int(round(b["bottom"]))),
        (0, 0, 255),
        1,
    )

    if detected is None:
        return

    for px, py in detected.get("edge_points", []):
        x = int(round(px + detected["roi_x"]))
        y = int(round(py + detected["roi_y"]))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            overlay[y, x] = (0, 255, 0)

    if width_mm is not None and height_mm is not None:
        text = f"{index}: {width_mm:.2f} x {height_mm:.2f} mm"
        cv2.putText(
            overlay,
            text,
            (int(round(b["left"])), max(20, int(round(b["top"])) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_polygon_annotation(
    overlay: np.ndarray,
    annotation: Dict[str, Any],
) -> None:
    """Draw only the red JSON/reference polygon; no labels or pixel values."""
    points = points_from_annotation(annotation)
    if len(points) < 3:
        return

    pts = np.asarray(
        [[int(round(x)), int(round(y))] for x, y in points],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], True, (0, 0, 255), 1)


def draw_detected_polygon(
    overlay: np.ndarray,
    detected: Dict[str, Any],
) -> None:
    """Draw only the green image-detected polygon; no labels."""
    points = detected.get("points", [])
    if len(points) < 3:
        return

    pts = np.asarray(
        [[int(round(x)), int(round(y))] for x, y in points],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 1)


def draw_measurement_reference(
    image: np.ndarray,
    annotation: Dict[str, Any],
    text: Optional[str],
) -> None:
    """Draw the annotation geometry in green with its final mm value only."""
    shape = shape_name(annotation)

    if shape == "polygon":
        points = points_from_annotation(annotation)
        if len(points) >= 3:
            pts = np.asarray(
                [[int(round(x)), int(round(y))] for x, y in points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 1)
            b = bbox_edges(annotation)
        else:
            return
    else:
        b = bbox_edges(annotation)
        cv2.rectangle(
            image,
            (int(round(b["left"])), int(round(b["top"]))),
            (int(round(b["right"])), int(round(b["bottom"]))),
            (0, 255, 0),
            1,
        )

    if text:
        cv2.putText(
            image,
            text,
            (int(round(b["left"])), max(20, int(round(b["top"])) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


# -----------------------------------------------------------------------------
# Engineering value extraction (generic; no feature IDs are used)
# -----------------------------------------------------------------------------

def expected_values(feature: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Read engineering nominal/tolerance values generically from the JSON."""
    result = {
        "nominal_diameter_mm": None,
        "nominal_radius_mm": None,
        "lower_diameter_mm": None,
        "upper_diameter_mm": None,
        "nominal_value_mm": None,
        "lower_value_mm": None,
        "upper_value_mm": None,
    }

    diameter = feature.get("diameter")
    if isinstance(diameter, dict):
        result["nominal_diameter_mm"] = number(diameter.get("nominal"))
        low = number(diameter.get("lower_deviation"))
        high = number(diameter.get("upper_deviation"))
        nominal = result["nominal_diameter_mm"]
        if nominal is not None and low is not None:
            result["lower_diameter_mm"] = nominal + low
        if nominal is not None and high is not None:
            result["upper_diameter_mm"] = nominal + high

    holes = feature.get("holes")
    if result["nominal_diameter_mm"] is None and isinstance(holes, dict):
        hd = holes.get("diameter")
        if isinstance(hd, dict):
            result["nominal_diameter_mm"] = number(hd.get("nominal"))
            low = number(hd.get("lower_deviation"))
            high = number(hd.get("upper_deviation"))
            nominal = result["nominal_diameter_mm"]
            if nominal is not None and low is not None:
                result["lower_diameter_mm"] = nominal + low
            if nominal is not None and high is not None:
                result["upper_diameter_mm"] = nominal + high

    radius = feature.get("radius")
    if isinstance(radius, dict):
        result["nominal_radius_mm"] = number(radius.get("nominal"))

    value = feature.get("value")
    if isinstance(value, dict):
        result["nominal_value_mm"] = number(value.get("nominal"))
        low = number(value.get("lower_deviation"))
        high = number(value.get("upper_deviation"))
        nominal = result["nominal_value_mm"]
        if nominal is not None and low is not None:
            result["lower_value_mm"] = nominal + low
        if nominal is not None and high is not None:
            result["upper_value_mm"] = nominal + high

    return result

# def detect_rectangle_from_annotation(
#     gray: np.ndarray,
#     annotation: Dict[str, Any],
# ) -> Optional[Dict[str, float]]:
#     """
#     Detect a rectangle from the image using the JSON annotation bbox
#     as the search ROI.

#     The JSON bbox is used only to define the search region.
#     The rectangle itself is detected from image data.
#     """

#     b = bbox_edges(annotation)

#     # ------------------------------------------------------------------
#     # Expand annotation ROI by 10%, same approach used for circles
#     # ------------------------------------------------------------------

#     roi_width = b["right"] - b["left"]
#     roi_height = b["bottom"] - b["top"]

#     expand_x = roi_width * 0.10
#     expand_y = roi_height * 0.10

#     expanded_left = b["left"] - expand_x
#     expanded_top = b["top"] - expand_y
#     expanded_right = b["right"] + expand_x
#     expanded_bottom = b["bottom"] + expand_y

#     h_img, w_img = gray.shape[:2]

#     x1 = max(0, int(round(expanded_left)))
#     y1 = max(0, int(round(expanded_top)))
#     x2 = min(w_img, int(round(expanded_right)))
#     y2 = min(h_img, int(round(expanded_bottom)))

#     roi = gray[y1:y2, x1:x2]

#     if roi.size == 0:
#         return None

#     # ------------------------------------------------------------------
#     # Threshold
#     # ------------------------------------------------------------------

#     _, binary = cv2.threshold(
#         roi,
#         90,
#         255,
#         cv2.THRESH_BINARY_INV,
#     )

#     # Small cleanup
#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_RECT,
#         (3, 3),
#     )

#     binary = cv2.morphologyEx(
#         binary,
#         cv2.MORPH_CLOSE,
#         kernel,
#     )

#     # ------------------------------------------------------------------
#     # Find contours
#     # ------------------------------------------------------------------

#     contours, _ = cv2.findContours(
#         binary,
#         cv2.RETR_LIST,
#         cv2.CHAIN_APPROX_NONE,
#     )

#     if not contours:
#         return None

#     # JSON annotation center, in ROI coordinates
#     annotation_cx = b["center_x"] - x1
#     annotation_cy = b["center_y"] - y1

#     best = None
#     best_score = -float("inf")

#     annotation_area = max(
#         1.0,
#         b["width"] * b["height"],
#     )

#     # ------------------------------------------------------------------
#     # Examine every contour
#     # ------------------------------------------------------------------

#     for contour in contours:

#         area = cv2.contourArea(contour)

#         if area <= 0:
#             continue

#         perimeter = cv2.arcLength(
#             contour,
#             True,
#         )

#         if perimeter <= 0:
#             continue

#         # Approximate contour with polygon
#         epsilon = 0.02 * perimeter

#         approx = cv2.approxPolyDP(
#             contour,
#             epsilon,
#             True,
#         )

#         # We want a 4-sided contour
#         if len(approx) != 4:
#             continue

#         if not cv2.isContourConvex(approx):
#             continue

#         rx, ry, rw, rh = cv2.boundingRect(approx)

#         if rw <= 1 or rh <= 1:
#             continue

#         rect_cx = rx + rw / 2.0
#         rect_cy = ry + rh / 2.0

#         center_error = px_distance(
#             rect_cx - annotation_cx,
#             rect_cy - annotation_cy,
#         )

#         contour_area_ratio = area / annotation_area

#         # Reject contours that are clearly unrelated
#         if contour_area_ratio < 0.20:
#             continue

#         if contour_area_ratio > 5.0:
#             continue

#         # Compare detected size with annotation size
#         width_error = abs(rw - b["width"])
#         height_error = abs(rh - b["height"])

#         # Score:
#         # - closer center = better
#         # - closer dimensions = better
#         # - area closer to annotation = better
#         score = (
#             -0.020 * center_error
#             -0.020 * width_error
#             -0.020 * height_error
#             -0.50 * abs(math.log(max(contour_area_ratio, 1e-6)))
#         )

#         if score > best_score:
#             best_score = score
#             best = (
#                 float(rx),
#                 float(ry),
#                 float(rw),
#                 float(rh),
#             )

#     if best is None:
#         return None

#     rx, ry, rw, rh = best

#     # Convert ROI coordinates back to full-image coordinates
#     detected_left = rx + x1
#     detected_top = ry + y1
#     detected_right = rx + rw + x1
#     detected_bottom = ry + rh + y1

#     detected_cx = (
#         detected_left + detected_right
#     ) / 2.0

#     detected_cy = (
#         detected_top + detected_bottom
#     ) / 2.0

#     return {
#         "left": float(detected_left),
#         "top": float(detected_top),
#         "right": float(detected_right),
#         "bottom": float(detected_bottom),
#         "width": float(rw),
#         "height": float(rh),
#         "center_x": float(detected_cx),
#         "center_y": float(detected_cy),
#     }
# def detect_rectangle_from_annotation(
#     gray: np.ndarray,
#     annotation: Dict[str, Any],
#     roi_expand: float = 0.10,
# ) -> Optional[Dict[str, Any]]:
#     """
#     Detect a rectangle using the same approach as polygon detection.

#     Rectangle annotation does NOT contain points.
#     Therefore:

#         1. JSON bbox = original ROI
#         2. Expanded ROI is used for image processing
#         3. The 4 corners of the ORIGINAL ROI bbox are used as
#            approximate corner windows inside the expanded ROI
#         4. Contour points are detected inside the expanded ROI
#         5. Approximate corner locations are used only to split/trim
#            rounded corner regions
#         6. Straight lines are fitted to the remaining contour points
#         7. Adjacent lines are intersected to obtain theoretical corners

#     JSON dimensions are NOT used for candidate scoring.
#     """

#     # ------------------------------------------------------------
#     # 1. ORIGINAL ROI FROM JSON
#     # ------------------------------------------------------------
#     b = bbox_edges(annotation)

#     orig_left = float(b["left"])
#     orig_top = float(b["top"])
#     orig_right = float(b["right"])
#     orig_bottom = float(b["bottom"])

#     orig_width = orig_right - orig_left
#     orig_height = orig_bottom - orig_top

#     if orig_width <= 0 or orig_height <= 0:
#         print("[RECT][ERROR] Invalid original ROI.")
#         return None

#     print(
#         f"[RECT] JSON ROI: "
#         f"L={orig_left:.1f}, "
#         f"T={orig_top:.1f}, "
#         f"R={orig_right:.1f}, "
#         f"B={orig_bottom:.1f}"
#     )

#     # ------------------------------------------------------------
#     # 2. EXPANDED ROI
#     # ------------------------------------------------------------
#     expand_x = orig_width * roi_expand
#     expand_y = orig_height * roi_expand

#     x1 = max(0, int(np.floor(orig_left - expand_x)))
#     y1 = max(0, int(np.floor(orig_top - expand_y)))

#     x2 = min(gray.shape[1], int(np.ceil(orig_right + expand_x)))
#     y2 = min(gray.shape[0], int(np.ceil(orig_bottom + expand_y)))

#     if x2 <= x1 or y2 <= y1:
#         print("[RECT][ERROR] Expanded ROI is invalid.")
#         return None

#     roi = gray[y1:y2, x1:x2].copy()

#     print(
#         f"[RECT] Expanded ROI: "
#         f"x={x1}:{x2}, y={y1}:{y2}"
#     )

#     # ------------------------------------------------------------
#     # 3. ORIGINAL ROI CORNERS
#     #
#     # IMPORTANT:
#     # These are NOT the expanded ROI corners.
#     #
#     # They are the four corners of the ORIGINAL JSON bbox,
#     # converted into coordinates relative to the expanded ROI.
#     # ------------------------------------------------------------
#     reference_corners = np.array(
#         [
#             [orig_left,  orig_top],       # top-left
#             [orig_right, orig_top],       # top-right
#             [orig_right, orig_bottom],    # bottom-right
#             [orig_left,  orig_bottom],    # bottom-left
#         ],
#         dtype=np.float32,
#     )

#     reference_corners_roi = reference_corners.copy()
#     reference_corners_roi[:, 0] -= x1
#     reference_corners_roi[:, 1] -= y1

#     # ------------------------------------------------------------
#     # 4. THRESHOLD
#     # ------------------------------------------------------------
#     threshold,info = estimate_threshold_from_edges(roi)
#     # threshold = threshold - 10

#     BLACK_BIAS = 0
#     threshold = max(0, threshold - BLACK_BIAS)
#     _, binary = cv2.threshold(
#         roi,
#         threshold,
#         255,
#         cv2.THRESH_BINARY
#     )
#     # binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

#     print("[RECT] Threshold completed.")

#     # ------------------------------------------------------------
#     # 5. MORPHOLOGICAL CLOSING
#     # ------------------------------------------------------------
#     kernel = np.ones((3, 3), np.uint8)

#     binary = cv2.morphologyEx(
#         binary,
#         cv2.MORPH_CLOSE,
#         kernel,
#     )

#     print("[RECT] Morphological closing completed.")

#     # ------------------------------------------------------------
#     # 6. FIND CONTOURS
#     # ------------------------------------------------------------
#     contours, _ = cv2.findContours(
#         binary,
#         cv2.RETR_LIST,
#         cv2.CHAIN_APPROX_NONE,
#     )
    

#     print(f"[RECT] Total contours found: {len(contours)}")

#     if not contours:
#         print("[RECT][REJECT] No contours found.")
#         return None

#     # ------------------------------------------------------------
#     # 7. SELECT THE MAIN CONTOUR
#     #
#     # No JSON geometry matching.
#     #
#     # We simply use the largest useful contour from the expanded
#     # image-derived ROI.
#     # ------------------------------------------------------------
#     valid_contours = []

#     for idx, contour in enumerate(contours):

#         if contour is None or len(contour) < 20:
#             continue

#         area = abs(cv2.contourArea(contour))

#         if area <= 0:
#             continue

#         bx, by, bw, bh = cv2.boundingRect(contour)

#         print(
#             f"[RECT][CONTOUR {idx}] "
#             f"points={len(contour)}, "
#             f"bbox={bw:.1f}x{bh:.1f}, "
#             f"area={area:.1f}"
#         )

#         valid_contours.append(
#             (
#                 area,
#                 contour,
#             )
#         )

#     if not valid_contours:
#         print("[RECT][REJECT] No usable contour found.")
#         return None

#     valid_contours.sort(
#         key=lambda item: item[0],
#         reverse=True,
#     )

#     selected_area, selected_contour = valid_contours[0]

#     contour = selected_contour.reshape(-1, 2).astype(np.float32)

#     print(
#         f"[RECT] Selected contour: "
#         f"points={len(contour)}, "
#         f"area={selected_area:.1f}"
#     )

#     # ------------------------------------------------------------
#     # 8. FIND APPROXIMATE CORNER LOCATIONS
#     #
#     # The ORIGINAL ROI corners are used as windows.
#     #
#     # They are NOT treated as the actual measured corners.
#     # We only search locally around each one for the nearest
#     # contour point.
#     # ------------------------------------------------------------
#     corner_window_x = max(8.0, orig_width * 0.20)
#     corner_window_y = max(8.0, orig_height * 0.20)

#     approximate_corner_indices = []

#     for corner_id, ref in enumerate(reference_corners_roi):

#         dx = np.abs(contour[:, 0] - ref[0])
#         dy = np.abs(contour[:, 1] - ref[1])

#         local_mask = (
#             (dx <= corner_window_x) &
#             (dy <= corner_window_y)
#         )

#         local_indices = np.where(local_mask)[0]

#         if len(local_indices) == 0:
#             print(
#                 f"[RECT][CORNER {corner_id}] "
#                 f"No contour points inside window."
#             )
#             approximate_corner_indices = []
#             break

#         local_points = contour[local_indices]

#         distances = np.sqrt(
#             (local_points[:, 0] - ref[0]) ** 2 +
#             (local_points[:, 1] - ref[1]) ** 2
#         )

#         best_local = int(
#             local_indices[np.argmin(distances)]
#         )

#         approximate_corner_indices.append(best_local)

#         p = contour[best_local]

#         print(
#             f"[RECT][CORNER {corner_id}] "
#             f"reference=({ref[0]:.1f},{ref[1]:.1f}) "
#             f"approx=({p[0]:.1f},{p[1]:.1f})"
#         )

#     if len(approximate_corner_indices) != 4:
#         print(
#             "[RECT][REJECT] "
#             "Could not locate all 4 approximate corners."
#         )
#         return None

#     # ------------------------------------------------------------
#     # 9. ORDER CORNERS ALONG THE CONTOUR
#     # ------------------------------------------------------------
#     approximate_corner_indices = np.asarray(
#         approximate_corner_indices,
#         dtype=np.int32,
#     )

#     # Make sure indices are unique.
#     if len(np.unique(approximate_corner_indices)) != 4:
#         print(
#             "[RECT][REJECT] "
#             "Duplicate approximate corner indices."
#         )
#         return None

#     # ------------------------------------------------------------
#     # 10. SPLIT CONTOUR INTO FOUR EDGE REGIONS
#     #
#     # Exactly the same concept as polygon processing.
#     # ------------------------------------------------------------
#     order = np.argsort(approximate_corner_indices)

#     ordered_indices = approximate_corner_indices[order]

#     edge_regions = []

#     n = len(contour)

#     for i in range(4):

#         start_idx = int(ordered_indices[i])
#         end_idx = int(
#             ordered_indices[(i + 1) % 4]
#         )

#         if i < 3:
#             if end_idx > start_idx:
#                 edge_points = contour[
#                     start_idx:end_idx + 1
#                 ]
#             else:
#                 edge_points = contour[
#                     end_idx:start_idx + 1
#                 ][::-1]

#         else:
#             if end_idx >= start_idx:
#                 edge_points = contour[
#                     start_idx:end_idx + 1
#                 ]
#             else:
#                 edge_points = np.vstack(
#                     [
#                         contour[start_idx:],
#                         contour[:end_idx + 1],
#                     ]
#                 )

#         if len(edge_points) < 10:
#             print(
#                 f"[RECT][EDGE {i}] "
#                 f"Too few points: {len(edge_points)}"
#             )
#             return None

#         edge_regions.append(edge_points)

#     # ------------------------------------------------------------
#     # 11. TRIM ROUNDED CORNER PORTIONS
#     #
#     # The approximate corner locations define where the rounded
#     # regions are. The actual straight edge is fitted only from
#     # the remaining contour points.
#     # ------------------------------------------------------------
#     trimmed_edges = []

#     for i, edge_points in enumerate(edge_regions):

#         trimmed = _trim_rounded_corners(
#             edge_points,
#             trim_fraction=0.20,
#         )

#         if trimmed is None or len(trimmed) < 10:
#             print(
#                 f"[RECT][EDGE {i}] "
#                 f"Not enough points after trimming."
#             )
#             return None

#         trimmed_edges.append(trimmed)

#         print(
#             f"[RECT][EDGE {i}] "
#             f"raw={len(edge_points)} "
#             f"trimmed={len(trimmed)}"
#         )

#     # ------------------------------------------------------------
#     # 12. FIT STRAIGHT LINE TO EACH EDGE
#     # ------------------------------------------------------------
#     lines = []
#     fit_errors = []

#     for i, edge_points in enumerate(trimmed_edges):

#         line = _fit_line_to_points(edge_points)

#         if line is None:
#             print(
#                 f"[RECT][EDGE {i}] "
#                 f"Line fitting failed."
#             )
#             return None

#         error = _line_fit_error(
#             edge_points,
#             line,
#         )

#         lines.append(line)
#         fit_errors.append(float(error))

#         print(
#             f"[RECT][EDGE {i}] "
#             f"line fit error={error:.4f}px"
#         )

#     # ------------------------------------------------------------
#     # 13. INTERSECT ADJACENT LINES
#     #
#     # These are the theoretical rectangle corners.
#     # ------------------------------------------------------------
#     detected_points = []

#     for i in range(4):

#         line_a = lines[i]
#         line_b = lines[(i + 1) % 4]

#         intersection = _line_intersection(
#             line_a,
#             line_b,
#         )

#         if intersection is None:
#             print(
#                 f"[RECT][REJECT] "
#                 f"Could not intersect edges "
#                 f"{i} and {(i + 1) % 4}."
#             )
#             return None

#         detected_points.append(
#             intersection
#         )

#     detected_points = np.asarray(
#         detected_points,
#         dtype=np.float32,
#     )

#     # ------------------------------------------------------------
#     # 14. CONVERT ROI COORDINATES -> FULL IMAGE
#     # ------------------------------------------------------------
#     detected_points[:, 0] += x1
#     detected_points[:, 1] += y1

#     # ------------------------------------------------------------
#     # 15. DETECTED BOUNDING BOX FROM THEORETICAL CORNERS
#     # ------------------------------------------------------------
#     detected_left = float(
#         np.min(detected_points[:, 0])
#     )

#     detected_top = float(
#         np.min(detected_points[:, 1])
#     )

#     detected_right = float(
#         np.max(detected_points[:, 0])
#     )

#     detected_bottom = float(
#         np.max(detected_points[:, 1])
#     )

#     detected_width = (
#         detected_right - detected_left
#     )

#     detected_height = (
#         detected_bottom - detected_top
#     )

#     detected_center_x = (
#         detected_left + detected_right
#     ) / 2.0

#     detected_center_y = (
#         detected_top + detected_bottom
#     ) / 2.0

#     # ------------------------------------------------------------
#     # 16. EDGE LENGTHS
#     # ------------------------------------------------------------
#     edge_lengths = []

#     for i in range(4):

#         p1 = detected_points[i]
#         p2 = detected_points[(i + 1) % 4]

#         length = float(
#             np.linalg.norm(p2 - p1)
#         )

#         edge_lengths.append(length)

#     # ------------------------------------------------------------
#     # 17. RESULT
#     # ------------------------------------------------------------
#     return {
#         "left": detected_left,
#         "top": detected_top,
#         "right": detected_right,
#         "bottom": detected_bottom,
#         "width": detected_width,
#         "height": detected_height,
#         "center_x": detected_center_x,
#         "center_y": detected_center_y,

#         "points": detected_points.tolist(),

#         "edge_lengths_px": edge_lengths,

#         "lines": lines,

#         "fit_errors_px": fit_errors,

#         "edge_regions": [
#             edge.tolist()
#             for edge in trimmed_edges
#         ],

#         "approximate_corner_points": [
#             contour[idx].tolist()
#             for idx in approximate_corner_indices
#         ],

#         "contour": contour.tolist(),

#         "roi": roi,
#         "binary": binary,

#         "roi_x": x1,
#         "roi_y": y1,
#     }
def detect_rectangle_from_annotation(
    gray: np.ndarray,
    annotation: Dict[str, Any],
    roi_expand: float = 0.10,
) -> Optional[Dict[str, Any]]:
    """
    Detect rectangle dimensions directly from grayscale intensity.

    No thresholding, contours, line fitting, or line intersections
    are used.

    Processing:
        1. JSON bbox defines the original ROI.
        2. ROI is expanded by roi_expand.
        3. Only the center portion of the expanded ROI is used,
           avoiding rounded/corner regions.
        4. Width is measured row-wise:
               left dark-touching edge
               right dark-touching edge
               width = right - left
        5. Height is measured column-wise:
               top dark-touching edge
               bottom dark-touching edge
               height = bottom - top
        6. White->gray transitions are rejected because their
           dark side is not sufficiently dark.
        7. All valid measurements are averaged.

    Returns the same general result structure used by the previous
    rectangle detector.
    """

    # ------------------------------------------------------------
    # 1. ORIGINAL ROI FROM JSON
    # ------------------------------------------------------------
    shape_type = detect_shape_polarity(gray, annotation)

    if shape_type == "solid":
        gray = cv2.bitwise_not(gray)
    b = bbox_edges(annotation)

    orig_left = float(b["left"])
    orig_top = float(b["top"])
    orig_right = float(b["right"])
    orig_bottom = float(b["bottom"])

    orig_width = orig_right - orig_left
    orig_height = orig_bottom - orig_top

    if orig_width <= 0 or orig_height <= 0:
        print("[RECT][ERROR] Invalid original ROI.")
        return None

    print(
        f"[RECT] JSON ROI: "
        f"L={orig_left:.1f}, "
        f"T={orig_top:.1f}, "
        f"R={orig_right:.1f}, "
        f"B={orig_bottom:.1f}"
    )

    # ------------------------------------------------------------
    # 2. EXPAND ROI
    # ------------------------------------------------------------

    expand_x = orig_width * roi_expand
    expand_y = orig_height * roi_expand

    x1 = max(
        0,
        int(np.floor(orig_left - expand_x))
    )

    y1 = max(
        0,
        int(np.floor(orig_top - expand_y))
    )

    x2 = min(
        gray.shape[1],
        int(np.ceil(orig_right + expand_x))
    )

    y2 = min(
        gray.shape[0],
        int(np.ceil(orig_bottom + expand_y))
    )

    if x2 <= x1 or y2 <= y1:
        print("[RECT][ERROR] Expanded ROI is invalid.")
        return None

    roi = gray[y1:y2, x1:x2].copy()

    print(
        f"[RECT] Expanded ROI: "
        f"x={x1}:{x2}, y={y1}:{y2}"
    )

    # ------------------------------------------------------------
    # 3. CONVERT TO GRAYSCALE IF REQUIRED
    # ------------------------------------------------------------

    rotation_angle = 0.0

    if roi.ndim == 3:
        roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

    # ------------------------------------------------------------
    # ESTIMATE ROTATION FROM TOP EDGE
    # ------------------------------------------------------------

    angle_img = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    ).astype(np.float32)

    ah, aw = angle_img.shape

    DARK_LIMIT = 200.0
    MIN_CONTRAST = 20.0
    X_EDGE_INSET_PX = 2.0

# Keep Y unchanged for now
    Y_EDGE_INSET_PX = 1.0

    top_edge_points = []

    # Avoid rounded corners.
    # Only use the central 40% of the rectangle.
    x_start = int(aw * 0.30)
    x_end = int(aw * 0.70)

    # Scan vertically at each x position.
    # The first valid dark transition is the top outer edge.
    for x in range(x_start, x_end):

        profile = angle_img[:, x]

        diff = np.diff(profile)

        candidates = []

        for i, d in enumerate(diff):

            contrast = abs(float(d))

            if contrast < MIN_CONTRAST:
                continue

            v1 = float(profile[i])
            v2 = float(profile[i + 1])

            dark = min(v1, v2)

            if dark > DARK_LIMIT:
                continue

            candidates.append(i + 0.5)

        if candidates:

            y = min(candidates)

            top_edge_points.append(
                (float(x), float(y))
            )

    # ------------------------------------------------------------
    # FIT LINE TO TOP EDGE
    # ------------------------------------------------------------

    if len(top_edge_points) >= 10:

        points = np.asarray(
            top_edge_points,
            dtype=np.float32
        )

        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx = float(vx)
        vy = float(vy)

        # Angle of top edge relative to image horizontal
        rotation_angle = np.degrees(
            np.arctan2(vy, vx)
        )

        print(
            f"[RECT] Top edge points: "
            f"{len(top_edge_points)}"
        )

        print(
            f"[RECT] Top edge angle: "
            f"{rotation_angle:.3f} degrees"
        )

    else:

        print(
            f"[RECT] Not enough top edge points: "
            f"{len(top_edge_points)}"
        )

    # ------------------------------------------------------------
    # ROTATE WHOLE ROI
    # ------------------------------------------------------------
    M = None
    if abs(rotation_angle) > 0.01:

        h0, w0 = roi.shape[:2]

        center = (
            w0 / 2.0,
            h0 / 2.0
        )

        M = cv2.getRotationMatrix2D(
            center,
            rotation_angle,
            1.0
        )

        roi = cv2.warpAffine(
            roi,
            M,
            (w0, h0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        print(
            f"[RECT] Applied rotation: "
            f"{rotation_angle:.3f} degrees"
        )

    # ------------------------------------------------------------
    # END ANGLE CORRECTION
    #
    # YOUR ORIGINAL CODE CONTINUES HERE
    # ------------------------------------------------------------

    
    # ============================================================
    # 6. SMALL BLUR
    #
    # EXISTING CODE
    # ============================================================


    # ------------------------------------------------------------
    # 4. SMALL BLUR
    # ------------------------------------------------------------

    roi_float = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    ).astype(np.float32)
    # roi_float = roi
    # roi_float = roi.astype(np.float32)
    h, w = roi_float.shape

    # ------------------------------------------------------------
    # 5. CENTER REGION ONLY
    #
    # Ignore corners.
    #
    # center_percent = 0.60 means:
    #
    #      20% removed from each end
    #
    #          |---- 60% ----|
    #       XXX|              |XXX
    #
    # ------------------------------------------------------------

    center_percent = 0.30

    y_margin = int(
        round(h * (1.0 - center_percent) / 2.0)
    )

    x_margin = int(
        round(w * (1.0 - center_percent) / 2.0)
    )

    y_start = y_margin
    y_end = h - y_margin

    x_start = x_margin
    x_end = w - x_margin

    if y_end <= y_start or x_end <= x_start:
        print("[RECT][REJECT] Center region is invalid.")
        return None

    print(
        f"[RECT] Center region: "
        f"x={x_start}:{x_end}, "
        f"y={y_start}:{y_end}"
    )

    # ------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------

    DARK_LIMIT = 200.0
    MIN_CONTRAST = 20.0

    # ============================================================
    # HELPER
    # ============================================================

    def find_dark_transitions(profile):
        """
        Find strong intensity transitions where at least one
        side is sufficiently dark.

        Returns:
            list of (position, contrast, dark_value)
        """

        profile = np.asarray(
            profile,
            dtype=np.float32
        )

        if len(profile) < 3:
            return []

        diff = np.diff(profile)

        candidates = []

        for i, d in enumerate(diff):

            contrast = abs(float(d))

            if contrast < MIN_CONTRAST:
                continue

            v1 = float(profile[i])
            v2 = float(profile[i + 1])

            dark = min(v1, v2)

            # ----------------------------------------------------
            # IMPORTANT
            #
            # BLACK -> WHITE  accepted
            # BLACK -> GRAY   accepted
            # GRAY  -> WHITE  rejected
            # ----------------------------------------------------

            if dark > DARK_LIMIT:
                continue

            # Transition lies between these two pixels.
            position = i + 0.5

            candidates.append(
                (
                    position,
                    contrast,
                    dark
                )
            )

        return candidates

    # ============================================================
    # 6. WIDTH
    #
    # Row-wise.
    #
    # For every row in the center portion:
    #
    #       |<-------- width -------->|
    #
    #       dark edge             dark edge
    #          ↓                       ↓
    #
    #     -------------------------------
    #
    # ============================================================

    width_samples = []
    width_edge_points = []
    left_x_samples = []
    right_x_samples = []
    for y in range(y_start, y_end):

        profile = roi_float[y, :]

        transitions = find_dark_transitions(profile)

        if len(transitions) < 2:
            continue

        # --------------------------------------------------------
        # Left side:
        # nearest dark transition to left side of ROI
        # --------------------------------------------------------

        left_candidates = [
            t
            for t in transitions
            if t[0] < w * 0.5
        ]

        # --------------------------------------------------------
        # Right side:
        # nearest dark transition to right side of ROI
        # --------------------------------------------------------

        right_candidates = [
            t
            for t in transitions
            if t[0] >= w * 0.5
        ]

        if not left_candidates or not right_candidates:
            continue

        left_edge = min(
            left_candidates,
            key=lambda t: t[0]
        )

        right_edge = min(
            right_candidates,
            key=lambda t: (w - 1) - t[0]
        )
        
        left_x_samples.append(left_edge[0])
        right_x_samples.append(right_edge[0])
        if len(width_samples) < 5:

            lx = int(round(left_edge[0]))
            rx = int(round(right_edge[0]))

            print("\n[LEFT PROFILE]")
            for xx in range(max(0, lx - 10), min(w, lx + 11)):
                print(
                    f"x={xx:4d} "
                    f"I={roi_float[y, xx]:6.1f}"
                )

            print("\n[RIGHT PROFILE]")
            for xx in range(max(0, rx - 10), min(w, rx + 11)):
                print(
                    f"x={xx:4d} "
                    f"I={roi_float[y, xx]:6.1f}"
                )

        # width = (
        #     right_edge[0] -
        #     left_edge[0]
        # )
        # ---------------------------------------------------------
# Shift the detected edges inward before measuring
        # ---------------------------------------------------------
        left_edge_x = left_edge[0] + X_EDGE_INSET_PX
        right_edge_x = right_edge[0] - X_EDGE_INSET_PX

        width = right_edge_x - left_edge_x

        # Keep the adjusted coordinates for debugging/output
        left_edge = (left_edge_x, left_edge[1], left_edge[2])
        right_edge = (right_edge_x, right_edge[1], right_edge[2])


        if width <= 0:
            continue
            width_samples = []
        
        width_samples.append(width)
        width_edge_points.append(
            (left_edge[0], float(y))
        )

        width_edge_points.append(
            (right_edge[0], float(y))
        )
    print(
    f"[RECT] Left edge mean : {np.mean(left_x_samples):.4f}px"
)

    print(
        f"[RECT] Right edge mean: {np.mean(right_x_samples):.4f}px"
    )

    print(
        f"[RECT] Measured span  : "
        f"{np.mean(right_x_samples) - np.mean(left_x_samples):.4f}px"
    )

    print(
        f"[RECT] Left std       : {np.std(left_x_samples):.4f}px"
    )

    print(
        f"[RECT] Right std      : {np.std(right_x_samples):.4f}px"
    )

    # ============================================================
    # 7. HEIGHT
    #
    # Column-wise.
    #
    # For every column in the center portion:
    #
    #          dark edge
    #             ↓
    #       ----------------
    #       |              |
    #       |              |
    #       |              |
    #       ----------------
    #             ↑
    #          dark edge
    #
    # ============================================================

    height_samples = []
    # height_samples = []

    height_edge_points = []

    for x in range(x_start, x_end):

        profile = roi_float[:, x]

        transitions = find_dark_transitions(profile)

        if len(transitions) < 2:
            continue

        # --------------------------------------------------------
        # Top side
        # --------------------------------------------------------

        top_candidates = [
            t
            for t in transitions
            if t[0] < h * 0.5
        ]

        # --------------------------------------------------------
        # Bottom side
        # --------------------------------------------------------

        bottom_candidates = [
            t
            for t in transitions
            if t[0] >= h * 0.5
        ]

        if not top_candidates or not bottom_candidates:
            continue

        top_edge = min(
            top_candidates,
            key=lambda t: t[0]
        )

        bottom_edge = min(
            bottom_candidates,
            key=lambda t: (h - 1) - t[0]
        )

        top_edge_y = top_edge[0] + Y_EDGE_INSET_PX
        bottom_edge_y = bottom_edge[0] - Y_EDGE_INSET_PX

        height = bottom_edge_y - top_edge_y

        # Keep the adjusted coordinates for debugging/output
        top_edge = (top_edge_y, top_edge[1], top_edge[2])
        bottom_edge = (bottom_edge_y, bottom_edge[1], bottom_edge[2])


        if height <= 0:
            continue

        height_samples.append(height)
        height_edge_points.append(
            (float(x), top_edge[0])
        )

        height_edge_points.append(
            (float(x), bottom_edge[0])
        )

    # ============================================================
    # 8. CHECK SAMPLE COUNT
    # ============================================================

    if len(width_samples) < 10:
        print(
            f"[RECT][REJECT] "
            f"Too few width samples: "
            f"{len(width_samples)}"
        )
        return None

    if len(height_samples) < 10:
        print(
            f"[RECT][REJECT] "
            f"Too few height samples: "
            f"{len(height_samples)}"
        )
        return None
    if len(width_samples) == 0:
        print("\n========== CANDIDATES ==========")

        print("LEFT:")
        for t in left_candidates:
            print(
                f"x={t[0]:.1f}, "
                f"contrast={t[1]:.1f}, "
                f"dark={t[2]:.1f}"
            )

        print("RIGHT:")
        for t in right_candidates:
            print(
                f"x={t[0]:.1f}, "
                f"contrast={t[1]:.1f}, "
                f"dark={t[2]:.1f}"
            )
    

    # ============================================================
    # 9. REMOVE OUTLIERS
    # ============================================================

    def robust_mean(values):

        values = np.asarray(
            values,
            dtype=np.float64
        )

        if len(values) < 5:
            return float(np.mean(values))

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        valid = values[
            (values >= lower) &
            (values <= upper)
        ]

        if len(valid) == 0:
            valid = values

        return float(np.median(valid))

    # ============================================================
    # 10. FINAL WIDTH / HEIGHT
    # ============================================================

    detected_width = robust_mean(
        width_samples
    )

    detected_height = robust_mean(
        height_samples
    )

    print(
        f"[RECT] Width: "
        f"{detected_width:.4f}px "
        f"from {len(width_samples)} samples"
    )

    print(
        f"[RECT] Height: "
        f"{detected_height:.4f}px "
        f"from {len(height_samples)} samples"
    )

    # ============================================================
    # 11. CENTER
    #
    # Use original annotation center.
    # The intensity method measures dimensions, not position.
    # ============================================================

    detected_center_x = (
        orig_left + orig_right
    ) / 2.0

    detected_center_y = (
        orig_top + orig_bottom
    ) / 2.0

    detected_left = (
        detected_center_x -
        detected_width / 2.0
    )

    detected_right = (
        detected_center_x +
        detected_width / 2.0
    )

    detected_top = (
        detected_center_y -
        detected_height / 2.0
    )

    detected_bottom = (
        detected_center_y +
        detected_height / 2.0
    )

    # ============================================================
    # 12. CONVERT EDGE POINTS BACK TO ORIGINAL ROI COORDINATES
    # ============================================================


    rotated_edge_points = (
        width_edge_points +
        height_edge_points
    )

    if M is not None and rotated_edge_points:

        # Convert affine matrix to 3x3 homogeneous matrix
        M3 = np.vstack([
            M,
            [0.0, 0.0, 1.0]
        ])

        # Inverse transformation
        M_inv = np.linalg.inv(M3)

        points = np.asarray(
            rotated_edge_points,
            dtype=np.float64
        )

        ones = np.ones(
            (len(points), 1),
            dtype=np.float64
        )

        points_h = np.hstack([
            points,
            ones
        ])

        original_points_h = (
            points_h @ M_inv.T
        )

        original_edge_points = [
            (
                float(p[0]),
                float(p[1])
            )
            for p in original_points_h
        ]

    else:
        original_edge_points = rotated_edge_points
    return {
        "left": detected_left,
        "top": detected_top,
        "right": detected_right,
        "bottom": detected_bottom,

        "width": detected_width,
        "height": detected_height,

        "center_x": detected_center_x,
        "center_y": detected_center_y,

        "points": [
            [detected_left, detected_top],
            [detected_right, detected_top],
            [detected_right, detected_bottom],
            [detected_left, detected_bottom],
        ],

        "edge_lengths_px": [
            detected_width,
            detected_height,
            detected_width,
            detected_height,
        ],

        "fit_errors_px": [],

        "edge_regions": [],

        "approximate_corner_points": [],

        "contour": [],

        "roi": roi,

        "binary": None,

        "roi_x": x1,
        "roi_y": y1,

        "width_samples": width_samples,
        "height_samples": height_samples,

        "num_width_samples": len(width_samples),
        "num_height_samples": len(height_samples),
        "edge_points": (
        # width_edge_points +
        # height_edge_points
        original_edge_points
    ),
    }
def _polygon_signed_area(points: List[Tuple[float, float]]) -> float:
    """Signed shoelace area in pixel^2."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid of polygon vertices (used for matching candidates)."""
    if not points:
        return 0.0, 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


def _best_polygon_order(
    detected: List[Tuple[float, float]],
    reference: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Put detected vertices into the same cyclic order as the JSON points.

    The JSON point order is preserved.  Both clockwise and counter-clockwise
    possibilities are tested, including every cyclic starting point.
    """
    n = len(reference)
    if len(detected) != n or n == 0:
        return detected

    ref = np.asarray(reference, dtype=np.float64)
    det = np.asarray(detected, dtype=np.float64)

    # Compare after translating both polygons to their centroid so the score
    # is based on vertex correspondence rather than absolute image position.
    ref_c = ref - np.mean(ref, axis=0)
    det_c = det - np.mean(det, axis=0)

    ref_norm = np.linalg.norm(ref_c)
    det_norm = np.linalg.norm(det_c)
    if ref_norm > 0:
        ref_c = ref_c / ref_norm
    if det_norm > 0:
        det_c = det_c / det_norm

    best = None
    best_error = float("inf")

    for reverse in (False, True):
        base = det_c[::-1] if reverse else det_c
        base_points = detected[::-1] if reverse else detected

        for shift in range(n):
            candidate = np.roll(base, -shift, axis=0)
            error = float(np.sum((ref_c - candidate) ** 2))
            if error < best_error:
                best_error = error
                best = list(np.roll(np.asarray(base_points, dtype=np.float64), -shift, axis=0))

    if best is None:
        return detected

    return [(float(x), float(y)) for x, y in best]


def _approx_polygon_with_vertex_count(
    contour: np.ndarray,
    target_count: int,
) -> List[np.ndarray]:
    """Return contour approximations having exactly target_count vertices."""
    if contour is None or len(contour) < target_count or target_count < 3:
        return []

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return []

    candidates: List[np.ndarray] = []

    # Fine-to-coarse epsilon sweep.  This avoids hard-coding one epsilon for
    # every polygon shape and lets the annotation's vertex count drive it.
    for fraction in np.linspace(0.002, 0.10, 80):
        approx = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if len(approx) == target_count:
            if cv2.isContourConvex(approx) or target_count >= 5:
                candidates.append(approx)

    return candidates

def _fit_line_to_points(
    points: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit an infinite straight line to 2D contour points.

    Returns:
        (vx, vy, x0, y0)

    where:
        (vx, vy) = unit direction vector of the line
        (x0, y0) = one point lying on the line
    """
    if points is None or len(points) < 2:
        return None

    pts = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    try:
        vx, vy, x0, y0 = cv2.fitLine(
            pts,
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        )
    except cv2.error:
        return None

    # cv2.fitLine() returns 1-element NumPy arrays.
    vx = float(np.asarray(vx).reshape(-1)[0])
    vy = float(np.asarray(vy).reshape(-1)[0])
    x0 = float(np.asarray(x0).reshape(-1)[0])
    y0 = float(np.asarray(y0).reshape(-1)[0])

    return vx, vy, x0, y0

def _line_intersection(
    line1: Tuple[float, float, float, float],
    line2: Tuple[float, float, float, float],
    parallel_epsilon: float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection of two infinite lines.

    Each line is represented as:
        (vx, vy, x0, y0)

    where:
        (vx, vy) = direction
        (x0, y0) = point on line
    """
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    cross = vx1 * vy2 - vy1 * vx2

    if abs(cross) < parallel_epsilon:
        return None

    dx = x2 - x1
    dy = y2 - y1

    t = (dx * vy2 - dy * vx2) / cross

    ix = x1 + t * vx1
    iy = y1 + t * vy1

    return float(ix), float(iy)


def _polygon_edge_indices(
    contour: np.ndarray,
    approx: np.ndarray,
    min_corner_gap: float = 5.0,
) -> List[np.ndarray]:
    """
    Split contour points into straight-edge regions using the
    vertices of a preliminary polygon approximation.

    The rounded corner itself is excluded from line fitting.

    Returns one point array for each polygon edge.
    """
    contour_points = np.asarray(
        contour,
        dtype=np.float64
    ).reshape(-1, 2)

    approx_points = np.asarray(
        approx,
        dtype=np.float64
    ).reshape(-1, 2)

    if len(contour_points) < 3 or len(approx_points) < 3:
        return []

    n = len(approx_points)

    # For each approximate vertex, find the closest contour index.
    vertex_indices = []

    for vertex in approx_points:
        distances = np.linalg.norm(
            contour_points - vertex,
            axis=1,
        )

        vertex_indices.append(
            int(np.argmin(distances))
        )

    # Make sure vertices follow contour order.
    # The approximation normally already follows contour order.
    ordered = []

    for idx in vertex_indices:
        if not ordered or idx != ordered[-1]:
            ordered.append(idx)

    if len(ordered) != n:
        return []

    edge_points = []

    for i in range(n):
        start_idx = ordered[i]
        end_idx = ordered[(i + 1) % n]

        if start_idx <= end_idx:
            indices = list(range(start_idx, end_idx + 1))
        else:
            indices = (
                list(range(start_idx, len(contour_points)))
                + list(range(0, end_idx + 1))
            )

        pts = contour_points[indices]

        if len(pts) < 2:
            return []

        edge_points.append(pts)

    return edge_points


def _trim_rounded_corners(
    edge_points: np.ndarray,
    trim_fraction: float = 0.20,
) -> np.ndarray:
    """
    Remove the portions of an edge nearest to its two rounded corners.

    Only the central portion of the contour edge is retained for
    straight-line fitting.
    """
    points = np.asarray(
        edge_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    if len(points) < 5:
        return points

    trim_fraction = max(
        0.0,
        min(0.45, trim_fraction)
    )

    trim = int(len(points) * trim_fraction)

    if trim * 2 >= len(points) - 2:
        return points

    return points[trim:len(points) - trim]


def _line_fit_error(
    points: np.ndarray,
    line: Tuple[float, float, float, float],
) -> float:
    """
    Mean perpendicular distance of points from a fitted line.
    Lower is better.
    """
    vx, vy, x0, y0 = line

    pts = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 2)

    if len(pts) == 0:
        return float("inf")

    # Cross-product distance from point to infinite line.
    distances = np.abs(
        (pts[:, 0] - x0) * vy
        - (pts[:, 1] - y0) * vx
    )

    return float(np.mean(distances))


def _detect_straight_polygon_edges(
    contour: np.ndarray,
    expected_vertices: int,
) -> Optional[Dict[str, Any]]:
    """
    From one selected contour:

      1. obtain a preliminary polygon approximation,
      2. split the contour into edge regions,
      3. remove rounded-corner portions,
      4. fit one straight line to each edge,
      5. calculate theoretical vertices from adjacent line intersections.

    The contour itself supplies all measured geometry.
    """
    if contour is None or len(contour) < expected_vertices:
        return None

    perimeter = float(cv2.arcLength(contour, True))

    if perimeter <= 1.0:
        return None

    # Get candidate polygon approximations.
    approximations = _approx_polygon_with_vertex_count(
        contour,
        expected_vertices,
    )

    if not approximations:
        return None

    best_result = None
    best_fit_error = float("inf")

    for approx in approximations:

        edge_regions = _polygon_edge_indices(
            contour,
            approx,
        )

        if len(edge_regions) != expected_vertices:
            continue

        lines = []
        fit_errors = []

        valid = True

        for edge_points in edge_regions:

            # Remove the rounded portions near both corners.
            straight_points = _trim_rounded_corners(
                edge_points,
                trim_fraction=0.20,
            )

            if len(straight_points) < 2:
                valid = False
                break

            line = _fit_line_to_points(
                straight_points
            )

            if line is None:
                valid = False
                break

            error = _line_fit_error(
                straight_points,
                line,
            )

            lines.append(line)
            fit_errors.append(error)

        if not valid or len(lines) != expected_vertices:
            continue

        # Calculate theoretical vertices:
        #
        # L1 ∩ L2 -> V1
        # L2 ∩ L3 -> V2
        # ...
        # Ln ∩ L1 -> Vn
        vertices = []

        for i in range(expected_vertices):
            line_a = lines[i]
            line_b = lines[
                (i + 1) % expected_vertices
            ]

            vertex = _line_intersection(
                line_a,
                line_b,
            )

            if vertex is None:
                valid = False
                break

            vertices.append(vertex)

        if not valid:
            continue

        mean_fit_error = float(
            np.mean(fit_errors)
        )

        if mean_fit_error < best_fit_error:
            best_fit_error = mean_fit_error

            best_result = {
                "lines": lines,
                "vertices": vertices,
                "edge_regions": edge_regions,
                "fit_errors": fit_errors,
                "mean_fit_error": mean_fit_error,
                "approximation": approx.copy(),
            }

    return best_result
def detect_polygon_from_annotation(
    gray: np.ndarray,
    annotation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Detect a polygon from the camera image.

    JSON is used ONLY for:
      1. defining the general search ROI,
      2. defining the expected number of polygon vertices.

    The actual polygon geometry is obtained from the image contour.

    For rounded corners:
      - contour points are used to identify the straight edge regions,
      - rounded corner portions are excluded,
      - straight lines are fitted to the remaining contour points,
      - adjacent fitted lines are intersected to obtain theoretical vertices.
    """

    # ------------------------------------------------------------
    # 1. Read reference points only to obtain:
    #    - general polygon location
    #    - expected number of vertices
    # ------------------------------------------------------------

    reference = points_from_annotation(annotation)

    if len(reference) < 3:
        return None

    expected_vertices = len(reference)

    # ------------------------------------------------------------
    # 2. JSON bounding box = ONLY general search location
    # ------------------------------------------------------------

    b = bbox_edges(annotation)

    width = max(1.0, b["width"])
    height = max(1.0, b["height"])

    # Keep your existing ROI expansion.
    ex = 0.30 * width
    ey = 0.30 * height

    h_img, w_img = gray.shape[:2]

    x1 = max(
        0,
        int(math.floor(b["left"] - ex))
    )

    y1 = max(
        0,
        int(math.floor(b["top"] - ey))
    )

    x2 = min(
        w_img,
        int(math.ceil(b["right"] + ex))
    )

    y2 = min(
        h_img,
        int(math.ceil(b["bottom"] + ey))
    )

    if x2 <= x1 or y2 <= y1:
        return None

    # ------------------------------------------------------------
    # 3. Crop ROI
    # ------------------------------------------------------------

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # ------------------------------------------------------------
    # 4. Threshold
    #
    # KEEPING YOUR EXISTING METHOD.
    # This is only being used to obtain the contour.
    # It is NOT used directly for final vertex measurement.
    # ------------------------------------------------------------

    _, binary = cv2.threshold(
        roi,
        90,
        255,
        cv2.THRESH_BINARY,
    )

    # ------------------------------------------------------------
    # 6. Find contours
    #
    # KEEPING YOUR EXISTING CONTOUR STAGE.
    # ------------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    # ------------------------------------------------------------
    # 8. Select the correct WHOLE contour
    #
    # IMPORTANT:
    #
    # We no longer use:
    #   - reference center
    #   - reference point distances
    #   - matchShapes()
    #   - reference area ratio
    #
    # The JSON is only giving us the general ROI.
    #
    # We select the contour using:
    #   - reasonable size
    #   - expected number of vertices
    #   - contour location inside the ROI
    #
    # The actual edge geometry comes later from the contour.
    # ------------------------------------------------------------

    contour_candidates = []

    roi_area = float(roi.shape[0] * roi.shape[1])

    for contour in contours:

        area = abs(
            float(cv2.contourArea(contour))
        )

        if area <= 1.0:
            continue

        # Ignore extremely tiny contours.
        if area < 0.01 * roi_area:
            continue

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        if perimeter <= 1.0:
            continue

        # Get a preliminary approximation only to determine
        # whether this contour can represent the required
        # number of vertices.
        approximations = (
            _approx_polygon_with_vertex_count(
                contour,
                expected_vertices,
            )
        )

        if not approximations:
            continue

        # Use the largest-area approximation as the candidate.
        best_approx_for_contour = max(
            approximations,
            key=lambda a: abs(
                float(cv2.contourArea(a))
            ),
        )

        approx_area = abs(
            float(
                cv2.contourArea(
                    best_approx_for_contour
                )
            )
        )

        # Basic contour compactness.
        # This is image-derived, not JSON-derived.
        compactness = (
            (4.0 * math.pi * area)
            / max(perimeter * perimeter, 1e-9)
        )

        contour_candidates.append(
            {
                "contour": contour,
                "area": area,
                "perimeter": perimeter,
                "approximation": (
                    best_approx_for_contour
                ),
                "approx_area": approx_area,
                "compactness": compactness,
            }
        )

    if not contour_candidates:
        return None

    # ------------------------------------------------------------
    # 9. From the contour candidates, find one that produces
    #    the best six straight edges.
    #
    # This is now the important selection stage.
    # ------------------------------------------------------------

    best_contour_result = None
    best_edge_fit_error = float("inf")

    for candidate in contour_candidates:

        contour = candidate["contour"]

        edge_result = (
            _detect_straight_polygon_edges(
                contour,
                expected_vertices,
            )
        )

        if edge_result is None:
            continue

        mean_fit_error = (
            edge_result["mean_fit_error"]
        )

        # Lower line-fitting error means the contour
        # contains stronger straight edge regions.
        if mean_fit_error < best_edge_fit_error:

            best_edge_fit_error = (
                mean_fit_error
            )

            best_contour_result = {
                "contour": contour,
                "area": candidate["area"],
                "perimeter": candidate["perimeter"],
                "edge_result": edge_result,
            }

    if best_contour_result is None:
        return None

    # ------------------------------------------------------------
    # 10. Get the six fitted straight lines
    # ------------------------------------------------------------

    selected_contour = (
        best_contour_result["contour"]
    )

    edge_result = (
        best_contour_result["edge_result"]
    )

    lines = edge_result["lines"]

    edge_regions = edge_result[
        "edge_regions"
    ]

    fit_errors = edge_result[
        "fit_errors"
    ]

    # ------------------------------------------------------------
    # 11. Get theoretical vertices
    #
    # These are NOT contour points.
    #
    # They are intersections of adjacent fitted lines.
    # ------------------------------------------------------------

    detected_points_roi = (
        edge_result["vertices"]
    )

    # ------------------------------------------------------------
    # 12. Convert ROI coordinates to full-image coordinates
    # ------------------------------------------------------------

    detected_points = [
        (
            float(x + x1),
            float(y + y1),
        )
        for x, y in detected_points_roi
    ]

    # ------------------------------------------------------------
    # 13. Keep the same logical point ordering as before.
    #
    # IMPORTANT:
    #
    # This is only for ordering/reporting.
    # It is NOT used for detection or scoring.
    # ------------------------------------------------------------

    detected_points = _best_polygon_order(
        detected_points,
        reference,
    )

    # ------------------------------------------------------------
    # 14. Calculate final polygon geometry
    # ------------------------------------------------------------

    detected_np = np.asarray(
        detected_points,
        dtype=np.float64,
    )

    area_px2 = abs(
        float(
            cv2.contourArea(
                detected_np.astype(
                    np.float32
                )
            )
        )
    )

    perimeter_px = 0.0

    edge_lengths_px: List[float] = []

    for i in range(
        len(detected_points)
    ):

        x_a, y_a = (
            detected_points[i]
        )

        x_b, y_b = (
            detected_points[
                (i + 1) % len(detected_points)
            ]
        )

        d = px_distance(
            x_b - x_a,
            y_b - y_a,
        )

        edge_lengths_px.append(d)

        perimeter_px += d

    # ------------------------------------------------------------
    # 15. Bounding box of THEORETICAL vertices
    # ------------------------------------------------------------

    left = float(
        np.min(detected_np[:, 0])
    )

    top = float(
        np.min(detected_np[:, 1])
    )

    right = float(
        np.max(detected_np[:, 0])
    )

    bottom = float(
        np.max(detected_np[:, 1])
    )

    # ------------------------------------------------------------
    # 16. Visualization of fitted lines + theoretical vertices
    # ------------------------------------------------------------

    detection_overlay = cv2.cvtColor(
        roi,
        cv2.COLOR_GRAY2BGR,
    )

    # Draw original contour in green.
    cv2.drawContours(
        detection_overlay,
        [selected_contour],
        -1,
        (0, 255, 0),
        1,
    )

    # Draw fitted infinite lines.
    line_length = max(
        roi.shape[0],
        roi.shape[1],
    ) * 2.0

    for line in lines:

        vx, vy, x0, y0 = line

        p1 = (
            int(round(
                x0 - vx * line_length
            )),
            int(round(
                y0 - vy * line_length
            )),
        )

        p2 = (
            int(round(
                x0 + vx * line_length
            )),
            int(round(
                y0 + vy * line_length
            )),
        )

        cv2.line(
            detection_overlay,
            p1,
            p2,
            (255, 0, 0),
            2,
        )

    # Draw theoretical vertices in red.
    for i, (x, y) in enumerate(
        detected_points_roi
    ):

        px = int(round(x))
        py = int(round(y))

        cv2.circle(
            detection_overlay,
            (px, py),
            5,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            detection_overlay,
            f"V{i + 1}",
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 17. Return result
    # ------------------------------------------------------------

    return {
        "points": detected_points,

        "edge_lengths_px": edge_lengths_px,

        "perimeter_px": perimeter_px,

        "area_px2": area_px2,

        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,

        "width": right - left,
        "height": bottom - top,

        "center_x": float(
            np.mean(
                detected_np[:, 0]
            )
        ),

        "center_y": float(
            np.mean(
                detected_np[:, 1]
            )
        ),

        "roi": roi,
        "binary": binary,

        # Selected contour
        "contour": selected_contour,

        # Fitted straight edges
        "lines": lines,

        # Points used to fit each edge
        "edge_regions": edge_regions,

        # Straight-line fitting error for each edge
        "edge_fit_errors": fit_errors,

        "mean_edge_fit_error": float(
            edge_result["mean_fit_error"]
        ),

        # Kept as "score" so the rest of your
        # existing code doesn't break.
        #
        # Higher score = better straight-line fit.
        "score": float(
            -edge_result["mean_fit_error"]
        ),
    }

# -----------------------------------------------------------------------------
# Per-shape processing
# -----------------------------------------------------------------------------

def circle_result(
    feature: Dict[str, Any],
    annotation: Dict[str, Any],
    annotation_index: int,
    detected: Optional[Dict[str, float]],
    x_scale: float,
    y_scale: float,
) -> Dict[str, Any]:
    b = bbox_edges(annotation)
    expected = expected_values(feature)

    result: Dict[str, Any] = {
        "feature_id": feature.get("id"),
        "feature_type": feature.get("type"),
        "annotation_index": annotation_index,
        "annotation_shape": shape_name(annotation),
        "annotation_tag_name": annotation.get("tag_name"),
        "annotation_tag_id": annotation.get("tag_id"),
        "annotation_left_px": b["left"],
        "annotation_top_px": b["top"],
        "annotation_right_px": b["right"],
        "annotation_bottom_px": b["bottom"],
        "annotation_width_px": b["width"],
        "annotation_height_px": b["height"],
        "annotation_center_x_px": b["center_x"],
        "annotation_center_y_px": b["center_y"],
        "annotation_radius_x_px": b["width"] / 2.0,
        "annotation_radius_y_px": b["height"] / 2.0,
        "annotation_radius_px": (b["width"] / 2.0 + b["height"] / 2.0) / 2.0,
        "annotation_diameter_x_px": b["width"],
        "annotation_diameter_y_px": b["height"],
        "annotation_diameter_px": (b["width"] + b["height"]) / 2.0,
        "detected_center_x_px": None,
        "detected_center_y_px": None,
        "detected_radius_px": None,
        "detected_diameter_px": None,
        "detected_diameter_mm_x": None,
        "detected_diameter_mm_y": None,
        "detected_diameter_mm": None,
        "edge_support": None,
        "nominal_diameter_mm": expected["nominal_diameter_mm"],
        "lower_limit_mm": expected["lower_diameter_mm"],
        "upper_limit_mm": expected["upper_diameter_mm"],
        "status": "NOT_FOUND",
    }

    if detected is None:
        return result

    d_px = detected["diameter_px"]
    d_x_mm = d_px * x_scale
    d_y_mm = d_px * y_scale
    d_mm = (d_x_mm + d_y_mm) / 2.0

    result.update(
        {
            "detected_center_x_px": detected["center_x"],
            "detected_center_y_px": detected["center_y"],
            "detected_radius_px": detected["radius_px"],
            "detected_diameter_px": d_px,
            "detected_diameter_mm_x": d_x_mm,
            "detected_diameter_mm_y": d_y_mm,
            "detected_diameter_mm": d_mm,
            "edge_support": detected["edge_support"],
            "status": "MEASURED",
        }
    )

    low = expected["lower_diameter_mm"]
    high = expected["upper_diameter_mm"]
    if expected["nominal_diameter_mm"] is not None and low is not None and high is not None:
        result["status"] = "PASS" if low <= d_mm <= high else "FAIL"

    return result


def rectangle_result(
    feature: Dict[str, Any],
    annotation: Dict[str, Any],
    annotation_index: int,
    detected: Optional[Dict[str, float]],
    x_scale: float,
    y_scale: float,
) -> Dict[str, Any]:

    b = bbox_edges(annotation)

    result = {
        "feature_id": feature.get("id"),
        "feature_type": feature.get("type"),
        "annotation_index": annotation_index,
        "annotation_shape": shape_name(annotation),
        "annotation_tag_name": annotation.get("tag_name"),
        "annotation_tag_id": annotation.get("tag_id"),

        # JSON reference
        "annotation_left_px": b["left"],
        "annotation_top_px": b["top"],
        "annotation_right_px": b["right"],
        "annotation_bottom_px": b["bottom"],
        "annotation_width_px": b["width"],
        "annotation_height_px": b["height"],

        # Image detected
        "detected_left_px": None,
        "detected_top_px": None,
        "detected_right_px": None,
        "detected_bottom_px": None,
        "detected_width_px": None,
        "detected_height_px": None,
        "detected_center_x_px": None,
        "detected_center_y_px": None,

        "width_mm": None,
        "height_mm": None,
        "diagonal_mm": None,

        "status": "NOT_FOUND",
    }

    if detected is None:
        return result
    width_mm = detected["width"] * x_scale
    height_mm = detected["height"] * y_scale

    diagonal_mm = math.hypot(
        width_mm,
        height_mm,
    ) 
    result.update(
        {
            "detected_left_px": detected["left"],
            "detected_top_px": detected["top"],
            "detected_right_px": detected["right"],
            "detected_bottom_px": detected["bottom"],
            "detected_width_px": detected["width"],
            "detected_height_px": detected["height"],
            "detected_center_x_px": detected["center_x"],
            "detected_center_y_px": detected["center_y"],
            "width_mm": width_mm,
            "height_mm": height_mm,
            "diagonal_mm": diagonal_mm,
            "status": "MEASURED",
        }
    )

    return result

def polygon_result(
    feature: Dict[str, Any],
    annotation: Dict[str, Any],
    annotation_index: int,
    detected: Optional[Dict[str, Any]],
    x_scale: float,
    y_scale: float,
) -> Dict[str, Any]:
    """Create polygon measurement results from detected image vertices."""
    reference = points_from_annotation(annotation)

    result: Dict[str, Any] = {
        "feature_id": feature.get("id"),
        "feature_type": feature.get("type"),
        "annotation_index": annotation_index,
        "annotation_shape": shape_name(annotation),
        "annotation_tag_name": annotation.get("tag_name"),
        "annotation_tag_id": annotation.get("tag_id"),
        "point_count": len(reference),
        "points_px": [{"x": x, "y": y} for x, y in reference],
        "detected_points_px": None,
        "edge_lengths_px": [],
        "edge_lengths_mm": [],
        "perimeter_px": None,
        "perimeter_mm": None,
        "area_px2": None,
        "area_mm2": None,
        "detected_left_px": None,
        "detected_top_px": None,
        "detected_right_px": None,
        "detected_bottom_px": None,
        "detected_width_px": None,
        "detected_height_px": None,
        "detected_width_mm": None,
        "detected_height_mm": None,
        "detected_center_x_px": None,
        "detected_center_y_px": None,
        "detection_score": None,
        "status": "NOT_FOUND",
    }

    if detected is None:
        return result

    detected_points = detected["points"]
    edge_lengths_px = list(detected["edge_lengths_px"])
    edge_lengths_mm = [
        convert_dxdy_to_mm(
            detected_points[(i + 1) % len(detected_points)][0] - detected_points[i][0],
            detected_points[(i + 1) % len(detected_points)][1] - detected_points[i][1],
            x_scale,
            y_scale,
        )
        for i in range(len(detected_points))
    ]

    perimeter_px = float(sum(edge_lengths_px))
    perimeter_mm = float(sum(edge_lengths_mm))
    area_px2 = float(detected["area_px2"])
    area_mm2 = area_px2 * x_scale * y_scale

    result.update(
        {
            "detected_points_px": [
                {"x": float(x), "y": float(y)}
                for x, y in detected_points
            ],
            "edge_lengths_px": edge_lengths_px,
            "edge_lengths_mm": edge_lengths_mm,
            "perimeter_px": perimeter_px,
            "perimeter_mm": perimeter_mm,
            "area_px2": area_px2,
            "area_mm2": area_mm2,
            "detected_left_px": detected["left"],
            "detected_top_px": detected["top"],
            "detected_right_px": detected["right"],
            "detected_bottom_px": detected["bottom"],
            "detected_width_px": detected["width"],
            "detected_height_px": detected["height"],
            "detected_width_mm": detected["width"] * x_scale,
            "detected_height_mm": detected["height"] * y_scale,
            "detected_center_x_px": detected["center_x"],
            "detected_center_y_px": detected["center_y"],
            "detection_score": detected["score"],
            "status": "MEASURED",
        }
    )

    return result


# -----------------------------------------------------------------------------
# JSON loading / validation
# -----------------------------------------------------------------------------

def load_drawing(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        drawing = json.load(f)

    if not isinstance(drawing, dict):
        raise ValueError("Drawing JSON root must be an object.")
    if not isinstance(drawing.get("drawingFeatures"), list):
        raise ValueError("Drawing JSON must contain drawingFeatures[].")
    return drawing


def iter_annotations(drawing: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
    """Yield EVERY annotation, with no tag-based filtering."""
    for feature in drawing["drawingFeatures"]:
        annotations = feature.get("annotations", []) or []
        for index, annotation in enumerate(annotations, start=1):
            if not isinstance(annotation, dict):
                continue
            yield feature, index, annotation


def make_output_row_for_unsupported(
    feature: Dict[str, Any], annotation: Dict[str, Any], index: int
) -> Dict[str, Any]:
    return {
        "feature_id": feature.get("id"),
        "feature_type": feature.get("type"),
        "annotation_index": index,
        "annotation_shape": shape_name(annotation),
        "annotation_tag_name": annotation.get("tag_name"),
        "annotation_tag_id": annotation.get("tag_id"),
        "status": "UNSUPPORTED_SHAPE",
    }


# -----------------------------------------------------------------------------
# Fixed runtime inputs
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

JSON_PATH = BASE_DIR / "rect1.json"
IMAGE_PATH = BASE_DIR /"captures"/"rect1.jpg"
CALIBRATION_PATH = BASE_DIR / "calibration" / "results" / "camera_calibration.json"

# -----------------------------------------------------------------------------
# Dynamic camera-to-sample distance
# -----------------------------------------------------------------------------
# The physical setup has a known total camera-to-reference/base length.
# The product occupies part of that length.
#
# Therefore:
#
#     camera_to_sample_distance_mm
#         = total_length_mm - product_height_mm
#
# Example:
#     total length   = 1000 mm
#     product height = 445 mm
#     camera/sample distance = 555 mm
#
# The resulting camera_to_sample_distance_mm is then used for the
# mm-per-pixel calculation.
#
# Do NOT hard-code the product height or the final distance.
# They are supplied at runtime.


# Output is created automatically here:
OUTPUT_DIR = BASE_DIR / "results" / "drawing_validation"


def _read_positive_mm(prompt: str) -> float:
    """Read a positive measurement in millimetres from the console."""
    while True:
        value = input(prompt).strip()
        try:
            value_mm = float(value)
        except ValueError:
            print("Please enter a numeric value in mm, for example 250.")
            continue

        if value_mm <= 0:
            print("Value must be greater than 0 mm.")
            continue

        return value_mm


def parse_args() -> argparse.Namespace:
    """
    Use fixed project paths and a fixed camera-to-base height of 438 mm.

    The product height is entered from the console for now.
    Later, the frontend can pass this same product height directly.

    Camera-to-sample distance:
        distance_mm = 438.0 - product_height_mm
    """

    parser = argparse.ArgumentParser(
        description="Dynamic Buhler annotation-shape validation"
    )
    # Keep normal script compatibility; product height is intentionally
    # requested from the console for now.
    parser.parse_args()


    TOTAL_CAMERA_TO_BASE_MM = 438

    print("\nCamera setup:")
    print(f"  Total camera-to-base height : {TOTAL_CAMERA_TO_BASE_MM:.3f} mm")

    product_height_mm = _read_positive_mm(
        "Enter PRODUCT SAMPLE height (mm): "
    )

    if product_height_mm >= TOTAL_CAMERA_TO_BASE_MM:
        raise ValueError(
            "Product height must be smaller than the fixed camera-to-base height.\n"
            f"Camera-to-base : {TOTAL_CAMERA_TO_BASE_MM:.3f} mm\n"
            f"Product height : {product_height_mm:.3f} mm"
        )

    distance_mm = TOTAL_CAMERA_TO_BASE_MM - product_height_mm

    args = argparse.Namespace(
        json=JSON_PATH,
        image=IMAGE_PATH,
        calibration=CALIBRATION_PATH,
        total_length_mm=TOTAL_CAMERA_TO_BASE_MM,
        product_height_mm=product_height_mm,
        distance_mm=distance_mm,
        output_dir=OUTPUT_DIR,
    )

    for label, path in (
        ("JSON", args.json),
        ("image", args.image),
        ("calibration", args.calibration),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} file not found:\n{path}\n\n"
                "Check the project folder structure and filename."
            )

    return args
def draw_detected_rectangle(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Draw the detected rectangle on the image.

    Green color: BGR (0, 255, 0)
    Thickness: 1 pixel
    """

    output = image.copy()

    if points is None or len(points) != 4:
        return output

    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        output,
        [pts],
        isClosed=True,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return output
def main() -> None:
    args = parse_args()

    print("\nLoading project inputs:")
    print("  JSON                    :", args.json)
    print("  Image                   :", args.image)
    print("  Calibration             :", args.calibration)
    print("  Total length (mm)       :", args.total_length_mm)
    print("  Product height (mm)     :", args.product_height_mm)
    print("  Camera-to-sample (mm)   :", args.distance_mm)
    print("  Output                  :", args.output_dir)
    print()

    drawing = load_drawing(args.json)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # Local smoothing
    # blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # # Local detail / edge component
    # detail = gray.astype(np.float32) - blur.astype(np.float32)

    # # Strengthen local transitions
    # strength = 2.0

    # enhanced = gray.astype(np.float32) + strength * detail

    # gray = np.clip(enhanced, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (9, 9), 0)

# Local detail / edge component
    detail = gray.astype(np.float32) - blur.astype(np.float32)

    # Strengthen local transitions
    strength = 2.0

    enhanced = gray.astype(np.float32) + strength * detail

    gray = np.clip(enhanced, 0, 255).astype(np.uint8)
    overlay = image.copy()
    annotated_boxes = image.copy()

    camera_matrix, distortion, fx, fy = load_calibration(args.calibration)
    x_scale, y_scale = mm_per_pixel(args.distance_mm, fx, fy)

    # We intentionally do not undistort/warp the annotation coordinates here.
    # They are treated as coordinates in the supplied camera image, exactly as requested.

    output_dir = args.output_dir or (args.calibration.parent / "drawing_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep every generated overlay image in one dedicated folder.
    # The input image name and timestamp are used so a new run never
    # overwrites an overlay from an earlier run.
    overlay_dir = output_dir / "overlay_images"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(args.image).stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    overlay_path = overlay_dir / f"{image_stem}_overlay_{run_timestamp}.png"
    annotated_boxes_path = overlay_dir / f"{image_stem}_annotations_{run_timestamp}.png"

    # Reports keep their existing locations/filenames.
    json_path = output_dir / "drawing_validation_report.json"

    results: List[Dict[str, Any]] = []

    print("=" * 110)
    print("BUHLER DYNAMIC ANNOTATION VALIDATION")
    print("=" * 110)
    print("Part number       :", drawing.get("part_number"))
    print("Drawing JSON      :", args.json)
    print("Image             :", args.image)
    print("Calibration       :", args.calibration)
    print("Total length (mm) :", f"{args.total_length_mm:.3f}")
    print("Product height    :", f"{args.product_height_mm:.3f} mm")
    print("Distance (mm)     :", f"{args.distance_mm:.3f}")
    print("Formula           : total length - product height")
    print("Image resolution  :", f"{image.shape[1]} x {image.shape[0]}")
    print("fx / fy (px)      :", f"{fx:.6f} / {fy:.6f}")
    print("X scale (mm/px)   :", f"{x_scale:.9f}")
    print("Y scale (mm/px)   :", f"{y_scale:.9f}")
    print("Coordinate source : drawingFeatures[].annotations[]")
    print("Shape dispatch     : annotation.shape ONLY")
    print("CVAT              : NOT USED")
    print("=" * 110)

    shape_counts: Dict[str, int] = {}

    for feature, index, annotation in iter_annotations(drawing):
        shape = shape_name(annotation)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        label = f"{feature.get('id', 'FEATURE')}-{index}"

        print(f"\n{label}")
        print("  shape :", shape)
        print("  tag   :", annotation.get("tag_name"), "/", annotation.get("tag_id"))

        if shape == "circle":
            b = bbox_edges(annotation)
            print(
                "  JSON  :",
                f"left={b['left']:.6f}, top={b['top']:.6f}, "
                f"right={b['right']:.6f}, bottom={b['bottom']:.6f}, "
                f"width={b['width']:.6f}, height={b['height']:.6f}"
            )
            print(
                "  JSON circle:",
                f"center=({b['center_x']:.6f},{b['center_y']:.6f}), "
                f"radius_x={b['width']/2.0:.6f}px, "
                f"radius_y={b['height']/2.0:.6f}px, "
                f"radius={(b['width']/2.0 + b['height']/2.0)/2.0:.6f}px"
            )

            detected = detect_circle_from_annotation(gray, annotation)
            # draw_circle_annotation(overlay, annotation, detected, label)
            row = circle_result(feature, annotation, index, detected, x_scale, y_scale)
            draw_circle_annotation(
                overlay,
                annotation,
                detected,
                index + 1,
                row["detected_diameter_mm"],
            )
            draw_measurement_reference(
                annotated_boxes,
                annotation,
                (
                    f"{index + 1}: {row['detected_diameter_mm']:.2f} mm"
                    if row["detected_diameter_mm"] is not None
                    else None
                ),
            )
            results.append(row)

            if detected is None:
                print("  IMAGE : circle fit FAILED")
            else:
                print(
                    "  IMAGE :",
                    f"center=({detected['center_x']:.3f},{detected['center_y']:.3f})px, "
                    f"radius={detected['radius_px']:.3f}px, "
                    f"diameter={detected['diameter_px']:.3f}px, "
                    f"edge_support={detected['edge_support']:.3f}"
                )
                print(
                    "  SIZE  :",
                    f"diameter_x={row['detected_diameter_mm_x']:.3f}mm, "
                    f"diameter_y={row['detected_diameter_mm_y']:.3f}mm, "
                    f"mean={row['detected_diameter_mm']:.3f}mm"
                )

        elif shape in {"rect", "rectangle", "square"}:
            b = bbox_edges(annotation)
            detected = detect_rectangle_from_annotation(gray, annotation)
            row = rectangle_result(
                feature,
                annotation,
                index,
                detected,
                x_scale,
                y_scale,
            )
            draw_rect_annotation(
                overlay,
                annotation,
                detected,
                index + 1,
                row["width_mm"],
                row["height_mm"],
            )
            draw_measurement_reference(
                annotated_boxes,
                annotation,
                (
                    f"{index + 1}: {row['width_mm']:.2f} x {row['height_mm']:.2f} mm"
                    if row["width_mm"] is not None and row["height_mm"] is not None
                    else None
                ),
            )
            results.append(row)
            print(
                "  RECT  :",
                f"left={b['left']:.6f}, top={b['top']:.6f}, "
                f"right={b['right']:.6f}, bottom={b['bottom']:.6f}, "
                f"width={b['width']:.6f}px, height={b['height']:.6f}px"
            )
            if row["width_mm"] is not None and row["height_mm"] is not None:
                print(
                    "  SIZE  :",
                    f"width={row['width_mm']:.3f}mm, height={row['height_mm']:.3f}mm"
                )
            else:
                print("  SIZE  : rectangle detection FAILED")

        elif shape == "polygon":
            points = points_from_annotation(annotation)

            # RED = JSON/reference polygon and its original point order
            draw_polygon_annotation(overlay, annotation)

            # GREEN = polygon detected from the actual camera image
            detected = detect_polygon_from_annotation(gray, annotation)
            if detected is not None:
                draw_detected_polygon(overlay, detected)

            row = polygon_result(
                feature,
                annotation,
                index,
                detected,
                x_scale,
                y_scale,
            )
            polygon_text = None
            if row["perimeter_mm"] is not None:
                polygon_text = f"{index + 1}: {row['perimeter_mm']:.2f} mm"
            draw_measurement_reference(annotated_boxes, annotation, polygon_text)
            if detected is not None:
                draw_measurement_text = polygon_text
                if draw_measurement_text:
                    b = bbox_edges(annotation)
                    cv2.putText(
                        overlay,
                        draw_measurement_text,
                        (int(round(b["left"])), max(20, int(round(b["top"])) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            results.append(row)

            print("  POLYGON: point_count=", len(points))
            for i, (x, y) in enumerate(points, start=1):
                print(f"    JSON P{i}: ({x:.6f}, {y:.6f})")

            if detected is None:
                print("  IMAGE : polygon detection FAILED")
            else:
                for i, (x, y) in enumerate(detected["points"], start=1):
                    print(f"    DETECTED P{i}: ({x:.6f}, {y:.6f})")
                print(
                    "  SIZE  :",
                    f"width={row['detected_width_mm']:.3f}mm, "
                    f"height={row['detected_height_mm']:.3f}mm, "
                    f"perimeter={row['perimeter_mm']:.3f}mm, "
                    f"area={row['area_mm2']:.3f}mm^2"
                )
                for i, edge_mm in enumerate(row["edge_lengths_mm"], start=1):
                    print(f"    EDGE {i}: {edge_mm:.3f} mm")

        else:
            print("  WARNING: unsupported annotation.shape -> preserved, not discarded")
            results.append(make_output_row_for_unsupported(feature, annotation, index))

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(annotated_boxes_path), annotated_boxes)

    report = {
        "drawingId": drawing.get("drawingId"),
        "part_number": drawing.get("part_number"),
        "image": str(args.image),
        "calibration": str(args.calibration),
        "total_length_mm": args.total_length_mm,
        "product_height_mm": args.product_height_mm,
        "distance_mm": args.distance_mm,
        "distance_calculation": "total_length_mm - product_height_mm",
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "fx_px": fx,
        "fy_px": fy,
        "x_scale_mm_per_px": x_scale,
        "y_scale_mm_per_px": y_scale,
        "coordinate_source": "drawingFeatures[].annotations[]",
        "dispatch_rule": "annotation.shape only",
        "cvat_used": False,
        "shape_counts": shape_counts,
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 110)
    print("VALIDATION COMPLETE")
    print("=" * 110)
    print("Annotation counts by shape:", shape_counts)
    print("Overlay folder:", overlay_dir)
    print("Overlay       :", overlay_path)
    print("Annotations   :", annotated_boxes_path)
    print("JSON          :", json_path)
    print("=" * 110)


def run_metrology_inspection(image_path: Path, json_path: Path, product_height_mm: float):
    print(f"Running metrology on image: {image_path}")
    print(f"Using drawing JSON config: {json_path}")
    TOTAL_CAMERA_TO_BASE_MM = 560.0

    if product_height_mm >= TOTAL_CAMERA_TO_BASE_MM:
        raise ValueError(
            "Product height must be smaller than the fixed camera-to-base height.\n"
            f"Camera-to-base : {TOTAL_CAMERA_TO_BASE_MM:.3f} mm\n"
            f"Product height : {product_height_mm:.3f} mm"
        )

    distance_mm = TOTAL_CAMERA_TO_BASE_MM - product_height_mm

    args = argparse.Namespace(
        json=json_path,
        image=Path(image_path),
        calibration=CALIBRATION_PATH,
        total_length_mm=TOTAL_CAMERA_TO_BASE_MM,
        product_height_mm=product_height_mm,
        distance_mm=distance_mm,
        output_dir=OUTPUT_DIR,
    )

    for label, path in (
        ("JSON", args.json),
        ("image", args.image),
        ("calibration", args.calibration),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found:\n{path}")

    drawing = load_drawing(args.json)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize separate canvases for the two required pictures
    overlay = image.copy()            # Picture 1: Detected edges + red boxes
    annotated_boxes = image.copy()    # Picture 2: Green reference boxes + mm values only

    camera_matrix, distortion, fx, fy = load_calibration(args.calibration)
    x_scale, y_scale = mm_per_pixel(args.distance_mm, fx, fy)

    # Output directories and dynamic timestamped paths
    output_dir = args.output_dir or (args.calibration.parent / "drawing_validation")
    overlay_dir = output_dir / "overlay_images"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(args.image).stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    overlay_path = overlay_dir / f"{image_stem}_overlay_{run_timestamp}.png"
    annotated_boxes_path = overlay_dir / f"{image_stem}_annotations_{run_timestamp}.png"
    csv_path = output_dir / f"{image_stem}_report_{run_timestamp}.csv"
    json_path = output_dir / f"{image_stem}_report_{run_timestamp}.json"

    results: List[Dict[str, Any]] = []

    print("=" * 110)
    print("BUHLER DYNAMIC ANNOTATION VALIDATION")
    print("=" * 110)
    print("Part number      :", drawing.get("part_number"))
    print("Drawing JSON     :", args.json)
    print("Image            :", args.image)
    print("Calibration      :", args.calibration)
    print("Total length (mm) :", f"{args.total_length_mm:.3f}")
    print("Product height    :", f"{args.product_height_mm:.3f} mm")
    print("Distance (mm)     :", f"{args.distance_mm:.3f}")
    print("Formula          : total length - product height")
    print("Image resolution  :", f"{image.shape[1]} x {image.shape[0]}")
    print("fx / fy (px)      :", f"{fx:.6f} / {fy:.6f}")
    print("X scale (mm/px)   :", f"{x_scale:.9f}")
    print("Y scale (mm/px)   :", f"{y_scale:.9f}")
    print("Coordinate source : drawingFeatures[].annotations[]")
    print("Shape dispatch    : annotation.shape ONLY")
    print("CVAT              : NOT USED")
    print("=" * 110)

    shape_counts: Dict[str, int] = {}

    for feature, index, annotation in iter_annotations(drawing):
        shape = shape_name(annotation)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        label = f"{feature.get('id', 'FEATURE')}-{index}"

        print(f"\n{label}")
        print("  shape :", shape)
        print("  tag   :", annotation.get("tag_name"), "/", annotation.get("tag_id"))

        if shape == "circle":
            detected = detect_circle_from_annotation(gray, annotation)
            row = circle_result(feature, annotation, index, detected, x_scale, y_scale)
            
            # 1. Overlay image: Red boxes + detected green edges/contours
            draw_circle_annotation(overlay, annotation, detected, label, row["detected_diameter_mm"])
            
            # 2. Annotated boxes image: Green reference boxes + mm value printed only
            draw_measurement_reference(
                annotated_boxes,
                annotation,
                (
                    f"{index}: {row['detected_diameter_mm']:.2f} mm"
                    if row["detected_diameter_mm"] is not None
                    else None
                ),
            )
            results.append(row)

        elif shape in {"rect", "rectangle", "square"}:
            detected = detect_rectangle_from_annotation(gray, annotation)
            row = rectangle_result(feature, annotation, index, detected, x_scale, y_scale)
            
            # 1. Overlay image
            draw_rect_annotation(overlay, annotation, detected, label, row["width_mm"], row["height_mm"])
            
            # 2. Annotated boxes image
            draw_measurement_reference(
                annotated_boxes,
                annotation,
                (
                    f"{index}: {row['width_mm']:.2f} x {row['height_mm']:.2f} mm"
                    if row["width_mm"] is not None and row["height_mm"] is not None
                    else None
                ),
            )
            results.append(row)

        elif shape == "polygon":
            # 1. Overlay image
            draw_polygon_annotation(overlay, annotation, label)
            detected = detect_polygon_from_annotation(gray, annotation)
            if detected is not None:
                draw_detected_polygon(overlay, detected, label)

            row = polygon_result(feature, annotation, index, detected, x_scale, y_scale)
            
            # 2. Annotated boxes image
            polygon_text = f"{index}: {row['perimeter_mm']:.2f} mm" if row["perimeter_mm"] is not None else None
            draw_measurement_reference(annotated_boxes, annotation, polygon_text)
            
            results.append(row)

        else:
            print("  WARNING: unsupported annotation.shape -> preserved, not discarded")
            results.append(make_output_row_for_unsupported(feature, annotation, index))

 # Save images to disk and verify
    success_overlay = cv2.imwrite(str(overlay_path), overlay)
    success_boxes = cv2.imwrite(str(annotated_boxes_path), annotated_boxes)

    if not success_overlay:
        print(f"ERROR: Failed to save overlay image to {overlay_path}")
    if not success_boxes:
        print(f"ERROR: Failed to save annotated boxes image to {annotated_boxes_path}")

    # [REMOVED] CSV rows processing and csv_path file writing have been eliminated here.

    # Generate Base64 strings for both images
    overlay_base64 = None
    if overlay_path.exists():
        with open(overlay_path, "rb") as img_file:
            overlay_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    annotated_boxes_base64 = None
    if annotated_boxes_path.exists():
        with open(annotated_boxes_path, "rb") as img_file:
            annotated_boxes_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # Save full detailed report locally to disk (JSON only)
    full_report = {
        "drawingId": drawing.get("drawingId"),
        "part_number": drawing.get("part_number"),
        "image": str(args.image),
        "calibration": str(args.calibration),
        "total_length_mm": args.total_length_mm,
        "product_height_mm": args.product_height_mm,
        "distance_mm": args.distance_mm,
        "distance_calculation": "total_length_mm - product_height_mm",
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "fx_px": fx,
        "fy_px": fy,
        "x_scale_mm_per_px": x_scale,
        "y_scale_mm_per_px": y_scale,
        "coordinate_source": "drawingFeatures[].annotations[]",
        "dispatch_rule": "annotation.shape only",
        "cvat_used": False,
        "shape_counts": shape_counts,
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    valid_results = [
        {k: v for k, v in row.items() if not k.startswith("annotation_")}
        for row in results 
        if row.get("status") in {"MEASURED", "PASS", "FAIL"}
    ]

    api_report = {
        "drawingId": drawing.get("drawingId"),
        "part_number": drawing.get("part_number"),
        "product_height_mm": args.product_height_mm,
        "distance_mm": args.distance_mm,
        "shape_counts": shape_counts,
        "results": valid_results,
        "overlay_image_path": str(overlay_path),
        "annotated_boxes_path": str(annotated_boxes_path),
        "overlay_image_base64": overlay_base64,
        "annotated_boxes_base64": annotated_boxes_base64,
    }

    print("\n" + "=" * 110)
    print("VALIDATION COMPLETE")
    print("=" * 110)
    print("Annotation counts by shape:", shape_counts)
    print(f"Overlay             : \"{overlay_path}\"")
    print(f"Annotated Boxes     : \"{annotated_boxes_path}\"")
    print(f"JSON                : \"{json_path}\"")
    print("=" * 110)

    return api_report







# from __future__ import annotations

# import argparse
# import base64
# import csv
# import json
# import math
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
# import cv2
# import numpy as np


# def number(value: Any, default: Optional[float] = None) -> Optional[float]:
#     if value is None:
#         return default
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return default


# def shape_name(annotation: Dict[str, Any]) -> str:
#     """The annotation shape is the ONLY dispatch key."""
#     return str(annotation.get("shape", "")).strip().lower()


# def annotation_bbox(annotation: Dict[str, Any]) -> Tuple[float, float, float, float]:
#     left = number(annotation.get("left"), 0.0)
#     top = number(annotation.get("top"), 0.0)
#     width = number(annotation.get("width"), 0.0)
#     height = number(annotation.get("height"), 0.0)
#     assert left is not None and top is not None and width is not None and height is not None
#     return left, top, width, height


# def bbox_edges(annotation: Dict[str, Any]) -> Dict[str, float]:
#     left, top, width, height = annotation_bbox(annotation)
#     return {
#         "left": left,
#         "top": top,
#         "right": left + width,
#         "bottom": top + height,
#         "width": width,
#         "height": height,
#         "center_x": left + width / 2.0,
#         "center_y": top + height / 2.0,
#     }


# def clip_bbox(
#     annotation: Dict[str, Any], image_shape: Sequence[int]
# ) -> Tuple[int, int, int, int]:
#     """Return x1,y1,x2,y2 clipped to the camera image."""
#     h, w = image_shape[:2]
#     b = bbox_edges(annotation)
#     x1 = max(0, min(w - 1, int(math.floor(b["left"]))))
#     y1 = max(0, min(h - 1, int(math.floor(b["top"]))))
#     x2 = max(x1 + 1, min(w, int(math.ceil(b["right"]))))
#     y2 = max(y1 + 1, min(h, int(math.ceil(b["bottom"]))))
#     return x1, y1, x2, y2


# def points_from_annotation(annotation: Dict[str, Any]) -> List[Tuple[float, float]]:
#     points = []
#     for p in annotation.get("points", []) or []:
#         x = number(p.get("x")) if isinstance(p, dict) else None
#         y = number(p.get("y")) if isinstance(p, dict) else None
#         if x is not None and y is not None:
#             points.append((x, y))
#     return points


# def mm_per_pixel(distance_mm: float, fx: float, fy: float) -> Tuple[float, float]:
#     return distance_mm / fx, distance_mm / fy


# def px_distance(dx: float, dy: float) -> float:
#     return math.hypot(dx, dy)


# def convert_dxdy_to_mm(dx_px: float, dy_px: float, x_scale: float, y_scale: float) -> float:
#     return math.hypot(dx_px * x_scale, dy_px * y_scale)


# def load_calibration(path: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
#     """Load camera calibration from a JSON file."""

#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Camera matrix
#     if "camera_matrix" not in data:
#         raise KeyError("Could not find 'camera_matrix' in calibration JSON.")

#     # Distortion coefficients
#     if "distortion_coefficients" not in data:
#         raise KeyError(
#             "Could not find 'distortion_coefficients' in calibration JSON."
#         )

#     camera_matrix = np.asarray(
#         data["camera_matrix"],
#         dtype=np.float64
#     )

#     distortion = np.asarray(
#         data["distortion_coefficients"],
#         dtype=np.float64
#     )

#     # Validate camera matrix
#     if camera_matrix.shape != (3, 3):
#         raise ValueError(
#             f"Invalid camera matrix shape {camera_matrix.shape}; "
#             "expected (3, 3)."
#         )

#     # Get focal lengths directly from camera matrix
#     fx = float(camera_matrix[0, 0])
#     fy = float(camera_matrix[1, 1])

#     if fx <= 0 or fy <= 0:
#         raise ValueError(
#             "Calibration contains invalid fx/fy values."
#         )

#     print(f"Calibration camera matrix key : camera_matrix")
#     print(f"Calibration distortion key    : distortion_coefficients")
#     print(f"Calibration JSON camera       : {data.get('camera', 'N/A')}")
#     print(f"Calibration JSON lens         : {data.get('lens', 'N/A')}")
#     print(f"Calibration fx                : {fx}")
#     print(f"Calibration fy                : {fy}")
#     print(f"Calibration reprojection err  : "
#           f"{data.get('reprojection_error_pixels', 'N/A')} px")

#     return camera_matrix, distortion, fx, fy
# # -----------------------------------------------------------------------------
# # Circle method
# # -----------------------------------------------------------------------------

# def fit_circle_least_squares(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
#     """Algebraic least-squares circle fit: x^2+y^2+Ax+By+C=0."""
#     if points is None or len(points) < 3:
#         return None

#     pts = np.asarray(points, dtype=np.float64)
#     x = pts[:, 0]
#     y = pts[:, 1]
#     A = np.column_stack((x, y, np.ones_like(x)))
#     b = -(x * x + y * y)

#     try:
#         coef, *_ = np.linalg.lstsq(A, b, rcond=None)
#     except np.linalg.LinAlgError:
#         return None

#     a, bb, c = coef
#     cx = -a / 2.0
#     cy = -bb / 2.0
#     r2 = cx * cx + cy * cy - c
#     if r2 <= 0 or not np.isfinite(r2):
#         return None
#     r = math.sqrt(r2)
#     return float(cx), float(cy), float(r)


# def circle_edge_support(
#     edge_image: np.ndarray,
#     cx: float,
#     cy: float,
#     radius: float,
#     band_px: float = 1.0,
# ) -> float:
#     """Fraction of sampled circumference points supported by image edges."""
#     if radius <= 0:
#         return 0.0

#     angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
#     cos_a = np.cos(angles)
#     sin_a = np.sin(angles)
#     support = 0

#     h, w = edge_image.shape[:2]
#     for c, s in zip(cos_a, sin_a):
#         x = int(round(cx + radius * c))
#         y = int(round(cy + radius * s))
#         found = False
#         rr = max(1, int(math.ceil(band_px)))
#         for yy in range(max(0, y - rr), min(h, y + rr + 1)):
#             for xx in range(max(0, x - rr), min(w, x + rr + 1)):
#                 if edge_image[yy, xx] != 0:
#                     found = True
#                     break
#             if found:
#                 break
#         support += int(found)

#     return support / len(angles)


# def refine_circle_from_edges(
#     edge_image: np.ndarray,
#     initial: Tuple[float, float, float],
#     max_distance_px: float = 3.0,
# ) -> Optional[Tuple[float, float, float, float]]:
#     """
#     Refine a Hough circle using edge points near its circumference.
#     Returns cx,cy,radius,edge_support.
#     """
#     cx0, cy0, r0 = initial
#     ys, xs = np.nonzero(edge_image)
#     if len(xs) < 3:
#         return None

#     d = np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2)
#     keep = np.abs(d - r0) <= max_distance_px
#     pts = np.column_stack((xs[keep], ys[keep]))
#     if len(pts) < 3:
#         return None

#     # Reject extreme point counts caused by unrelated edges.
#     if len(pts) > 12000:
#         step = int(math.ceil(len(pts) / 12000))
#         pts = pts[::step]

#     fitted = fit_circle_least_squares(pts)
#     if fitted is None:
#         return None

#     cx, cy, radius = fitted
#     support = circle_edge_support(edge_image, cx, cy, radius)
#     return cx, cy, radius, support


# def hough_candidates(gray_roi: np.ndarray, min_radius: int, max_radius: int) -> List[Tuple[float, float, float]]:
#     """Try several Hough thresholds so small and large circles can both be found."""
#     if min_radius < 1 or max_radius < min_radius:
#         return []
    
#     blur = cv2.GaussianBlur(gray_roi, (5, 5), 1.2)
#     candidates: List[Tuple[float, float, float]] = []

#     # The thresholds are algorithm settings, not feature/sample values.
#     for p2 in (8, 10, 12, 15, 18, 22, 26, 30, 35):
#         circles = cv2.HoughCircles(
#             blur,
#             cv2.HOUGH_GRADIENT,
#             dp=1.0,
#             minDist=max(3.0, float(min_radius)),
#             param1=80,
#             param2=float(p2),
#             minRadius=min_radius,
#             maxRadius=max_radius,
#         )
#         if circles is not None:
#             for c in np.round(circles[0]).astype(np.float64):
#                 candidates.append((float(c[0]), float(c[1]), float(c[2])))

#     return candidates


# def detect_circle_from_annotation(
#     gray: np.ndarray,
#     annotation: Dict[str, Any],
# ) -> Optional[Dict[str, float]]:
#     """
#     Circle annotation method:
#       1. read left/top/width/height from JSON
#       2. make annotation ROI
#       3. find an image circle inside that ROI
#       4. refine the detected circle from image edges

#     The JSON bbox is never replaced by a global/tag-based search.
#     """
#     b = bbox_edges(annotation)
#     # x1, y1, x2, y2 = clip_bbox(annotation, gray.shape)
#     # roi = gray[y1:y2, x1:x2]
#     # if roi.size == 0:
#     #     return None
#     roi_width = b["right"] - b["left"]
#     roi_height = b["bottom"] - b["top"]

#     expand_x = roi_width * 0.10
#     expand_y = roi_height * 0.10

#     expanded_left = b["left"] - expand_x
#     expanded_top = b["top"] - expand_y
#     expanded_right = b["right"] + expand_x
#     expanded_bottom = b["bottom"] + expand_y

#     # Clip expanded ROI to image boundaries
#     h, w = gray.shape[:2]

#     x1 = max(0, int(round(expanded_left)))
#     y1 = max(0, int(round(expanded_top)))
#     x2 = min(w, int(round(expanded_right)))
#     y2 = min(h, int(round(expanded_bottom)))

#     roi = gray[y1:y2, x1:x2]

#     if roi.size == 0:
#         return None

#     h, w = roi.shape[:2]
#     # Keep candidate radii tied to this annotation's own bbox.
#     annotation_radius_x = max(0.5, b["width"] / 2.0)
#     annotation_radius_y = max(0.5, b["height"] / 2.0)
#     annotation_radius = (annotation_radius_x + annotation_radius_y) / 2.0
#     min_radius = max(1, int(round(min(annotation_radius_x, annotation_radius_y) * 0.30)))
#     max_radius = max(min_radius, int(round(max(annotation_radius_x, annotation_radius_y) * 1.30)))
#     max_radius = min(max_radius, max(1, min(h, w) // 2 + 3))

#     candidates = hough_candidates(roi, min_radius, max_radius)
    
#     _, roi = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY_INV)
#     # blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)
    
#     edges = cv2.Canny(roi, 40, 120)
#     # 
#     filename = f"binary_{annotation['tag_id']}.png"
#     filename_1 = f"edges_{annotation['tag_id']}.png"
#     cv2.imwrite(filename, roi)
#     cv2.imwrite(filename_1, edges)
    

#     annotation_cx = b["center_x"] - x1
#     annotation_cy = b["center_y"] - y1
#     annotation_r = annotation_radius

#     best = None
#     best_score = -float("inf")

#     # De-duplicate Hough results before scoring.
#     unique: List[Tuple[float, float, float]] = []
#     for c in candidates:
#         if not any(px_distance(c[0] - q[0], c[1] - q[1]) < 2 and abs(c[2] - q[2]) < 2 for q in unique):
#             unique.append(c)

#     for cx, cy, r in unique:
#         center_error = px_distance(cx - annotation_cx, cy - annotation_cy)
#         radius_error = abs(r - annotation_r)
#         support = circle_edge_support(edges, cx, cy, r)

#         # Prefer circles that fit the annotation ROI and have strong image-edge support.
#         score = (
#             3.0 * support
#             - 0.015 * center_error
#             - 0.020 * radius_error
#         )
#         if score > best_score:
#             best_score = score
#             best = (cx, cy, r)

#     if best is None:
#         # Edge-only fallback: fit a circle to points near the JSON annotation circle.
#         print(filename)
#         print("edge only fallback")
#         ys, xs = np.nonzero(edges)
#         if len(xs) >= 3:
#             d = np.sqrt((xs - annotation_cx) ** 2 + (ys - annotation_cy) ** 2)
#             keep = np.abs(d - annotation_r) <= max(3.0, annotation_r * 0.20)
#             points = np.column_stack((xs[keep], ys[keep]))
#             fitted = fit_circle_least_squares(points)
#             if fitted is not None:
#                 best = fitted
        

#     if best is None:
#         return None

#     refined = refine_circle_from_edges(edges, best, max_distance_px=max(2.0, annotation_r * 0.08))
#     # refined = None
#     if refined is None:
#         cx, cy, r = best
#         support = circle_edge_support(edges, cx, cy, r)
#     else:
#         cx, cy, r, support = refined

#     return {
#         "center_x": float(cx + x1),
#         "center_y": float(cy + y1),
#         "radius_px": float(r),
#         "diameter_px": float(2.0 * r),
#         "edge_support": float(support),
#         "roi_left": float(x1),
#         "roi_top": float(y1),
#         "roi_right": float(x2),
#         "roi_bottom": float(y2),
#     }


# # -----------------------------------------------------------------------------
# # Drawing methods selected by annotation shape
# # -----------------------------------------------------------------------------

# def draw_cross(image: np.ndarray, x: float, y: float, size: int = 8) -> None:
#     p = (int(round(x)), int(round(y)))
#     cv2.drawMarker(image, p, (0, 0, 255), cv2.MARKER_CROSS, size * 2, 1)


# def draw_label(image: np.ndarray, text: str, x: float, y: float) -> None:
#     cv2.putText(
#         image,
#         text,
#         (max(0, int(round(x))), max(15, int(round(y)) - 6)),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.45,
#         (0, 0, 255),
#         1,
#         cv2.LINE_AA,
#     )


# def draw_circle_annotation(
#     overlay: np.ndarray,
#     annotation: Dict[str, Any],
#     detected: Optional[Dict[str, float]],
#     label: str,
# ) -> None:
#     # JSON geometry is drawn as a thin red reference box/cross.
#     b = bbox_edges(annotation)
#     cv2.rectangle(
#         overlay,
#         (int(round(b["left"])), int(round(b["top"]))),
#         (int(round(b["right"])), int(round(b["bottom"]))),
#         (0, 0, 255),
#         1,
#     )
#     draw_cross(overlay, b["center_x"], b["center_y"])

#     if detected is not None:
#         # Image-fitted circle is GREEN, not a square.
#         cv2.circle(
#             overlay,
#             (int(round(detected["center_x"])), int(round(detected["center_y"]))),
#             max(1, int(round(detected["radius_px"]))),
#             (0, 255, 0),
#             1
#         )
#         draw_cross(overlay, detected["center_x"], detected["center_y"], size=6)
#         text = f"{label} C({detected['center_x']:.1f},{detected['center_y']:.1f}) R={detected['radius_px']:.2f}px"
#     else:
#         text = f"{label} CIRCLE NOT FIT"

#     draw_label(overlay, text, b["left"], b["top"])


# # def draw_rect_annotation(
# #     overlay: np.ndarray,
# #     annotation: Dict[str, Any],
# #     label: str,
# # ) -> None:
# #     b = bbox_edges(annotation)
# #     cv2.rectangle(
# #         overlay,
# #         (int(round(b["left"])), int(round(b["top"]))),
# #         (int(round(b["right"])), int(round(b["bottom"]))),
# #         (0, 0, 255),
# #         1,
# #     )
# #     draw_cross(overlay, b["center_x"], b["center_y"])
# #     draw_label(
# #         overlay,
# #         f"{label} L={b['left']:.1f} T={b['top']:.1f} R={b['right']:.1f} B={b['bottom']:.1f}",
# #         b["left"],
# #         b["top"],
# #     )
# def draw_rect_annotation(
#     overlay: np.ndarray,
#     annotation: Dict[str, Any],
#     detected: Optional[Dict[str, float]],
#     label: str,
# ) -> None:
#     # JSON geometry is drawn as a thin red reference box/cross.
#     b = bbox_edges(annotation)

#     cv2.rectangle(
#         overlay,
#         (int(round(b["left"])), int(round(b["top"]))),
#         (int(round(b["right"])), int(round(b["bottom"]))),
#         (0, 0, 255),
#         1,
#     )

#     draw_cross(
#         overlay,
#         b["center_x"],
#         b["center_y"],
#     )

#     if detected is not None:
#         # Image-detected rectangle is GREEN.
#         cv2.rectangle(
#             overlay,
#             (
#                 int(round(detected["left"])),
#                 int(round(detected["top"])),
#             ),
#             (
#                 int(round(detected["right"])),
#                 int(round(detected["bottom"])),
#             ),
#             (0, 255, 0),
#             1,
#         )

#         draw_cross(
#             overlay,
#             detected["center_x"],
#             detected["center_y"],
#             size=6,
#         )

#         text = (
#             f"{label} "
#             f"L={detected['left']:.1f} "
#             f"T={detected['top']:.1f} "
#             f"R={detected['right']:.1f} "
#             f"B={detected['bottom']:.1f}"
#         )
#     else:
#         text = f"{label} RECTANGLE NOT DETECTED"

#     draw_label(
#         overlay,
#         text,
#         b["left"],
#         b["top"],
#     )

# def draw_polygon_annotation(
#     overlay: np.ndarray,
#     annotation: Dict[str, Any],
#     label: str,
# ) -> None:
#     points = points_from_annotation(annotation)
#     if not points:
#         # Polygon without points is invalid for this method.
#         b = bbox_edges(annotation)
#         draw_label(overlay, f"{label} POLYGON: NO POINTS", b["left"], b["top"])
#         return

#     pts = np.asarray([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
#     cv2.polylines(overlay, [pts.reshape((-1, 1, 2))], True, (0, 0, 255), 1)
#     for i, (x, y) in enumerate(points):
#         cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
#         cv2.putText(
#             overlay,
#             str(i + 1),
#             (int(round(x)) + 4, int(round(y)) - 4),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.4,
#             (0, 0, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     cx = float(np.mean([p[0] for p in points]))
#     cy = float(np.mean([p[1] for p in points]))
#     draw_label(overlay, f"{label} POLYGON {len(points)} pts", cx, cy)


# # -----------------------------------------------------------------------------
# # Engineering value extraction (generic; no feature IDs are used)
# # -----------------------------------------------------------------------------

# def expected_values(feature: Dict[str, Any]) -> Dict[str, Optional[float]]:
#     """Read engineering nominal/tolerance values generically from the JSON."""
#     result = {
#         "nominal_diameter_mm": None,
#         "nominal_radius_mm": None,
#         "lower_diameter_mm": None,
#         "upper_diameter_mm": None,
#         "nominal_value_mm": None,
#         "lower_value_mm": None,
#         "upper_value_mm": None,
#     }

#     diameter = feature.get("diameter")
#     if isinstance(diameter, dict):
#         result["nominal_diameter_mm"] = number(diameter.get("nominal"))
#         low = number(diameter.get("lower_deviation"))
#         high = number(diameter.get("upper_deviation"))
#         nominal = result["nominal_diameter_mm"]
#         if nominal is not None and low is not None:
#             result["lower_diameter_mm"] = nominal + low
#         if nominal is not None and high is not None:
#             result["upper_diameter_mm"] = nominal + high

#     holes = feature.get("holes")
#     if result["nominal_diameter_mm"] is None and isinstance(holes, dict):
#         hd = holes.get("diameter")
#         if isinstance(hd, dict):
#             result["nominal_diameter_mm"] = number(hd.get("nominal"))
#             low = number(hd.get("lower_deviation"))
#             high = number(hd.get("upper_deviation"))
#             nominal = result["nominal_diameter_mm"]
#             if nominal is not None and low is not None:
#                 result["lower_diameter_mm"] = nominal + low
#             if nominal is not None and high is not None:
#                 result["upper_diameter_mm"] = nominal + high

#     radius = feature.get("radius")
#     if isinstance(radius, dict):
#         result["nominal_radius_mm"] = number(radius.get("nominal"))

#     value = feature.get("value")
#     if isinstance(value, dict):
#         result["nominal_value_mm"] = number(value.get("nominal"))
#         low = number(value.get("lower_deviation"))
#         high = number(value.get("upper_deviation"))
#         nominal = result["nominal_value_mm"]
#         if nominal is not None and low is not None:
#             result["lower_value_mm"] = nominal + low
#         if nominal is not None and high is not None:
#             result["upper_value_mm"] = nominal + high

#     return result

# def detect_rectangle_from_annotation(
#     gray: np.ndarray,
#     annotation: Dict[str, Any],
#     roi_expand: float = 0.10,
# ) -> Optional[Dict[str, Any]]:
#     """
#     Detect a rectangle using the same approach as polygon detection.

#     Rectangle annotation does NOT contain points.
#     Therefore:

#         1. JSON bbox = original ROI
#         2. Expanded ROI is used for image processing
#         3. The 4 corners of the ORIGINAL ROI bbox are used as
#            approximate corner windows inside the expanded ROI
#         4. Contour points are detected inside the expanded ROI
#         5. Approximate corner locations are used only to split/trim
#            rounded corner regions
#         6. Straight lines are fitted to the remaining contour points
#         7. Adjacent lines are intersected to obtain theoretical corners

#     JSON dimensions are NOT used for candidate scoring.
#     """

#     # ------------------------------------------------------------
#     # 1. ORIGINAL ROI FROM JSON
#     # ------------------------------------------------------------
#     b = bbox_edges(annotation)

#     orig_left = float(b["left"])
#     orig_top = float(b["top"])
#     orig_right = float(b["right"])
#     orig_bottom = float(b["bottom"])

#     orig_width = orig_right - orig_left
#     orig_height = orig_bottom - orig_top

#     if orig_width <= 0 or orig_height <= 0:
#         print("[RECT][ERROR] Invalid original ROI.")
#         return None

#     print(
#         f"[RECT] JSON ROI: "
#         f"L={orig_left:.1f}, "
#         f"T={orig_top:.1f}, "
#         f"R={orig_right:.1f}, "
#         f"B={orig_bottom:.1f}"
#     )

#     # ------------------------------------------------------------
#     # 2. EXPANDED ROI
#     # ------------------------------------------------------------
#     expand_x = orig_width * roi_expand
#     expand_y = orig_height * roi_expand

#     x1 = max(0, int(np.floor(orig_left - expand_x)))
#     y1 = max(0, int(np.floor(orig_top - expand_y)))

#     x2 = min(gray.shape[1], int(np.ceil(orig_right + expand_x)))
#     y2 = min(gray.shape[0], int(np.ceil(orig_bottom + expand_y)))

#     if x2 <= x1 or y2 <= y1:
#         print("[RECT][ERROR] Expanded ROI is invalid.")
#         return None

#     roi = gray[y1:y2, x1:x2].copy()

#     print(
#         f"[RECT] Expanded ROI: "
#         f"x={x1}:{x2}, y={y1}:{y2}"
#     )

#     # ------------------------------------------------------------
#     # 3. ORIGINAL ROI CORNERS
#     #
#     # IMPORTANT:
#     # These are NOT the expanded ROI corners.
#     #
#     # They are the four corners of the ORIGINAL JSON bbox,
#     # converted into coordinates relative to the expanded ROI.
#     # ------------------------------------------------------------
#     reference_corners = np.array(
#         [
#             [orig_left,  orig_top],       # top-left
#             [orig_right, orig_top],       # top-right
#             [orig_right, orig_bottom],    # bottom-right
#             [orig_left,  orig_bottom],    # bottom-left
#         ],
#         dtype=np.float32,
#     )

#     reference_corners_roi = reference_corners.copy()
#     reference_corners_roi[:, 0] -= x1
#     reference_corners_roi[:, 1] -= y1

#     # ------------------------------------------------------------
#     # 4. THRESHOLD
#     # ------------------------------------------------------------
#     _, binary = cv2.threshold(
#         roi,
#         100,
#         255,
#         cv2.THRESH_BINARY
#     )

#     print("[RECT] Threshold completed.")

#     # ------------------------------------------------------------
#     # 5. MORPHOLOGICAL CLOSING
#     # ------------------------------------------------------------
#     kernel = np.ones((3, 3), np.uint8)

#     binary = cv2.morphologyEx(
#         binary,
#         cv2.MORPH_CLOSE,
#         kernel,
#     )

#     print("[RECT] Morphological closing completed.")

#     # ------------------------------------------------------------
#     # 6. FIND CONTOURS
#     # ------------------------------------------------------------
#     contours, _ = cv2.findContours(
#         binary,
#         cv2.RETR_LIST,
#         cv2.CHAIN_APPROX_NONE,
#     )

#     print(f"[RECT] Total contours found: {len(contours)}")

#     if not contours:
#         print("[RECT][REJECT] No contours found.")
#         return None

#     # ------------------------------------------------------------
#     # 7. SELECT THE MAIN CONTOUR
#     #
#     # No JSON geometry matching.
#     #
#     # We simply use the largest useful contour from the expanded
#     # image-derived ROI.
#     # ------------------------------------------------------------
#     valid_contours = []

#     for idx, contour in enumerate(contours):

#         if contour is None or len(contour) < 20:
#             continue

#         area = abs(cv2.contourArea(contour))

#         if area <= 0:
#             continue

#         bx, by, bw, bh = cv2.boundingRect(contour)

#         print(
#             f"[RECT][CONTOUR {idx}] "
#             f"points={len(contour)}, "
#             f"bbox={bw:.1f}x{bh:.1f}, "
#             f"area={area:.1f}"
#         )

#         valid_contours.append(
#             (
#                 area,
#                 contour,
#             )
#         )

#     if not valid_contours:
#         print("[RECT][REJECT] No usable contour found.")
#         return None

#     valid_contours.sort(
#         key=lambda item: item[0],
#         reverse=True,
#     )

#     selected_area, selected_contour = valid_contours[0]

#     contour = selected_contour.reshape(-1, 2).astype(np.float32)

#     print(
#         f"[RECT] Selected contour: "
#         f"points={len(contour)}, "
#         f"area={selected_area:.1f}"
#     )

#     # ------------------------------------------------------------
#     # 8. FIND APPROXIMATE CORNER LOCATIONS
#     #
#     # The ORIGINAL ROI corners are used as windows.
#     #
#     # They are NOT treated as the actual measured corners.
#     # We only search locally around each one for the nearest
#     # contour point.
#     # ------------------------------------------------------------
#     corner_window_x = max(8.0, orig_width * 0.20)
#     corner_window_y = max(8.0, orig_height * 0.20)

#     approximate_corner_indices = []

#     for corner_id, ref in enumerate(reference_corners_roi):

#         dx = np.abs(contour[:, 0] - ref[0])
#         dy = np.abs(contour[:, 1] - ref[1])

#         local_mask = (
#             (dx <= corner_window_x) &
#             (dy <= corner_window_y)
#         )

#         local_indices = np.where(local_mask)[0]

#         if len(local_indices) == 0:
#             print(
#                 f"[RECT][CORNER {corner_id}] "
#                 f"No contour points inside window."
#             )
#             approximate_corner_indices = []
#             break

#         local_points = contour[local_indices]

#         distances = np.sqrt(
#             (local_points[:, 0] - ref[0]) ** 2 +
#             (local_points[:, 1] - ref[1]) ** 2
#         )

#         best_local = int(
#             local_indices[np.argmin(distances)]
#         )

#         approximate_corner_indices.append(best_local)

#         p = contour[best_local]

#         print(
#             f"[RECT][CORNER {corner_id}] "
#             f"reference=({ref[0]:.1f},{ref[1]:.1f}) "
#             f"approx=({p[0]:.1f},{p[1]:.1f})"
#         )

#     if len(approximate_corner_indices) != 4:
#         print(
#             "[RECT][REJECT] "
#             "Could not locate all 4 approximate corners."
#         )
#         return None

#     # ------------------------------------------------------------
#     # 9. ORDER CORNERS ALONG THE CONTOUR
#     # ------------------------------------------------------------
#     approximate_corner_indices = np.asarray(
#         approximate_corner_indices,
#         dtype=np.int32,
#     )

#     # Make sure indices are unique.
#     if len(np.unique(approximate_corner_indices)) != 4:
#         print(
#             "[RECT][REJECT] "
#             "Duplicate approximate corner indices."
#         )
#         return None

#     # ------------------------------------------------------------
#     # 10. SPLIT CONTOUR INTO FOUR EDGE REGIONS
#     #
#     # Exactly the same concept as polygon processing.
#     # ------------------------------------------------------------
#     order = np.argsort(approximate_corner_indices)

#     ordered_indices = approximate_corner_indices[order]

#     edge_regions = []

#     n = len(contour)

#     for i in range(4):

#         start_idx = int(ordered_indices[i])
#         end_idx = int(
#             ordered_indices[(i + 1) % 4]
#         )

#         if i < 3:
#             if end_idx > start_idx:
#                 edge_points = contour[
#                     start_idx:end_idx + 1
#                 ]
#             else:
#                 edge_points = contour[
#                     end_idx:start_idx + 1
#                 ][::-1]

#         else:
#             if end_idx >= start_idx:
#                 edge_points = contour[
#                     start_idx:end_idx + 1
#                 ]
#             else:
#                 edge_points = np.vstack(
#                     [
#                         contour[start_idx:],
#                         contour[:end_idx + 1],
#                     ]
#                 )

#         if len(edge_points) < 10:
#             print(
#                 f"[RECT][EDGE {i}] "
#                 f"Too few points: {len(edge_points)}"
#             )
#             return None

#         edge_regions.append(edge_points)

#     # ------------------------------------------------------------
#     # 11. TRIM ROUNDED CORNER PORTIONS
#     #
#     # The approximate corner locations define where the rounded
#     # regions are. The actual straight edge is fitted only from
#     # the remaining contour points.
#     # ------------------------------------------------------------
#     trimmed_edges = []

#     for i, edge_points in enumerate(edge_regions):

#         trimmed = _trim_rounded_corners(
#             edge_points,
#             trim_fraction=0.20,
#         )

#         if trimmed is None or len(trimmed) < 10:
#             print(
#                 f"[RECT][EDGE {i}] "
#                 f"Not enough points after trimming."
#             )
#             return None

#         trimmed_edges.append(trimmed)

#         print(
#             f"[RECT][EDGE {i}] "
#             f"raw={len(edge_points)} "
#             f"trimmed={len(trimmed)}"
#         )

#     # ------------------------------------------------------------
#     # 12. FIT STRAIGHT LINE TO EACH EDGE
#     # ------------------------------------------------------------
#     lines = []
#     fit_errors = []

#     for i, edge_points in enumerate(trimmed_edges):

#         line = _fit_line_to_points(edge_points)

#         if line is None:
#             print(
#                 f"[RECT][EDGE {i}] "
#                 f"Line fitting failed."
#             )
#             return None

#         error = _line_fit_error(
#             edge_points,
#             line,
#         )

#         lines.append(line)
#         fit_errors.append(float(error))

#         print(
#             f"[RECT][EDGE {i}] "
#             f"line fit error={error:.4f}px"
#         )

#     # ------------------------------------------------------------
#     # 13. INTERSECT ADJACENT LINES
#     #
#     # These are the theoretical rectangle corners.
#     # ------------------------------------------------------------
#     detected_points = []

#     for i in range(4):

#         line_a = lines[i]
#         line_b = lines[(i + 1) % 4]

#         intersection = _line_intersection(
#             line_a,
#             line_b,
#         )

#         if intersection is None:
#             print(
#                 f"[RECT][REJECT] "
#                 f"Could not intersect edges "
#                 f"{i} and {(i + 1) % 4}."
#             )
#             return None

#         detected_points.append(
#             intersection
#         )

#     detected_points = np.asarray(
#         detected_points,
#         dtype=np.float32,
#     )

#     # ------------------------------------------------------------
#     # 14. CONVERT ROI COORDINATES -> FULL IMAGE
#     # ------------------------------------------------------------
#     detected_points[:, 0] += x1
#     detected_points[:, 1] += y1

#     # ------------------------------------------------------------
#     # 15. DETECTED BOUNDING BOX FROM THEORETICAL CORNERS
#     # ------------------------------------------------------------
#     detected_left = float(
#         np.min(detected_points[:, 0])
#     )

#     detected_top = float(
#         np.min(detected_points[:, 1])
#     )

#     detected_right = float(
#         np.max(detected_points[:, 0])
#     )

#     detected_bottom = float(
#         np.max(detected_points[:, 1])
#     )

#     detected_width = (
#         detected_right - detected_left
#     )

#     detected_height = (
#         detected_bottom - detected_top
#     )

#     detected_center_x = (
#         detected_left + detected_right
#     ) / 2.0

#     detected_center_y = (
#         detected_top + detected_bottom
#     ) / 2.0

#     # ------------------------------------------------------------
#     # 16. EDGE LENGTHS
#     # ------------------------------------------------------------
#     edge_lengths = []

#     for i in range(4):

#         p1 = detected_points[i]
#         p2 = detected_points[(i + 1) % 4]

#         length = float(
#             np.linalg.norm(p2 - p1)
#         )

#         edge_lengths.append(length)

#     # ------------------------------------------------------------
#     # 17. RESULT
#     # ------------------------------------------------------------
#     return {
#         "left": detected_left,
#         "top": detected_top,
#         "right": detected_right,
#         "bottom": detected_bottom,
#         "width": detected_width,
#         "height": detected_height,
#         "center_x": detected_center_x,
#         "center_y": detected_center_y,

#         "points": detected_points.tolist(),

#         "edge_lengths_px": edge_lengths,

#         "lines": lines,

#         "fit_errors_px": fit_errors,

#         "edge_regions": [
#             edge.tolist()
#             for edge in trimmed_edges
#         ],

#         "approximate_corner_points": [
#             contour[idx].tolist()
#             for idx in approximate_corner_indices
#         ],

#         "contour": contour.tolist(),

#         "roi": roi,
#         "binary": binary,

#         "roi_x": x1,
#         "roi_y": y1,
#     }
# def _polygon_signed_area(points: List[Tuple[float, float]]) -> float:
#     """Signed shoelace area in pixel^2."""
#     if len(points) < 3:
#         return 0.0
#     area = 0.0
#     for i, (x1, y1) in enumerate(points):
#         x2, y2 = points[(i + 1) % len(points)]
#         area += x1 * y2 - x2 * y1
#     return 0.5 * area


# def _polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
#     """Centroid of polygon vertices (used for matching candidates)."""
#     if not points:
#         return 0.0, 0.0
#     arr = np.asarray(points, dtype=np.float64)
#     return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


# def _best_polygon_order(
#     detected: List[Tuple[float, float]],
#     reference: List[Tuple[float, float]],
# ) -> List[Tuple[float, float]]:
#     """
#     Put detected vertices into the same cyclic order as the JSON points.

#     The JSON point order is preserved.  Both clockwise and counter-clockwise
#     possibilities are tested, including every cyclic starting point.
#     """
#     n = len(reference)
#     if len(detected) != n or n == 0:
#         return detected

#     ref = np.asarray(reference, dtype=np.float64)
#     det = np.asarray(detected, dtype=np.float64)

#     # Compare after translating both polygons to their centroid so the score
#     # is based on vertex correspondence rather than absolute image position.
#     ref_c = ref - np.mean(ref, axis=0)
#     det_c = det - np.mean(det, axis=0)

#     ref_norm = np.linalg.norm(ref_c)
#     det_norm = np.linalg.norm(det_c)
#     if ref_norm > 0:
#         ref_c = ref_c / ref_norm
#     if det_norm > 0:
#         det_c = det_c / det_norm

#     best = None
#     best_error = float("inf")

#     for reverse in (False, True):
#         base = det_c[::-1] if reverse else det_c
#         base_points = detected[::-1] if reverse else detected

#         for shift in range(n):
#             candidate = np.roll(base, -shift, axis=0)
#             error = float(np.sum((ref_c - candidate) ** 2))
#             if error < best_error:
#                 best_error = error
#                 best = list(np.roll(np.asarray(base_points, dtype=np.float64), -shift, axis=0))

#     if best is None:
#         return detected

#     return [(float(x), float(y)) for x, y in best]


# def _approx_polygon_with_vertex_count(
#     contour: np.ndarray,
#     target_count: int,
# ) -> List[np.ndarray]:
#     """Return contour approximations having exactly target_count vertices."""
#     if contour is None or len(contour) < target_count or target_count < 3:
#         return []

#     perimeter = cv2.arcLength(contour, True)
#     if perimeter <= 0:
#         return []

#     candidates: List[np.ndarray] = []

#     # Fine-to-coarse epsilon sweep.  This avoids hard-coding one epsilon for
#     # every polygon shape and lets the annotation's vertex count drive it.
#     for fraction in np.linspace(0.002, 0.10, 80):
#         approx = cv2.approxPolyDP(contour, fraction * perimeter, True)
#         if len(approx) == target_count:
#             if cv2.isContourConvex(approx) or target_count >= 5:
#                 candidates.append(approx)

#     return candidates

# def _fit_line_to_points(
#     points: np.ndarray,
# ) -> Optional[Tuple[float, float, float, float]]:
#     """
#     Fit an infinite straight line to 2D contour points.

#     Returns:
#         (vx, vy, x0, y0)

#     where:
#         (vx, vy) = unit direction vector of the line
#         (x0, y0) = one point lying on the line
#     """
#     if points is None or len(points) < 2:
#         return None

#     pts = np.asarray(
#         points,
#         dtype=np.float32,
#     ).reshape(-1, 1, 2)

#     try:
#         vx, vy, x0, y0 = cv2.fitLine(
#             pts,
#             cv2.DIST_L2,
#             0,
#             0.01,
#             0.01,
#         )
#     except cv2.error:
#         return None

#     # cv2.fitLine() returns 1-element NumPy arrays.
#     vx = float(np.asarray(vx).reshape(-1)[0])
#     vy = float(np.asarray(vy).reshape(-1)[0])
#     x0 = float(np.asarray(x0).reshape(-1)[0])
#     y0 = float(np.asarray(y0).reshape(-1)[0])

#     return vx, vy, x0, y0

# def _line_intersection(
#     line1: Tuple[float, float, float, float],
#     line2: Tuple[float, float, float, float],
#     parallel_epsilon: float = 1e-6,
# ) -> Optional[Tuple[float, float]]:
#     """
#     Calculate the intersection of two infinite lines.

#     Each line is represented as:
#         (vx, vy, x0, y0)

#     where:
#         (vx, vy) = direction
#         (x0, y0) = point on line
#     """
#     vx1, vy1, x1, y1 = line1
#     vx2, vy2, x2, y2 = line2

#     cross = vx1 * vy2 - vy1 * vx2

#     if abs(cross) < parallel_epsilon:
#         return None

#     dx = x2 - x1
#     dy = y2 - y1

#     t = (dx * vy2 - dy * vx2) / cross

#     ix = x1 + t * vx1
#     iy = y1 + t * vy1

#     return float(ix), float(iy)


# def _polygon_edge_indices(
#     contour: np.ndarray,
#     approx: np.ndarray,
#     min_corner_gap: float = 5.0,
# ) -> List[np.ndarray]:
#     """
#     Split contour points into straight-edge regions using the
#     vertices of a preliminary polygon approximation.

#     The rounded corner itself is excluded from line fitting.

#     Returns one point array for each polygon edge.
#     """
#     contour_points = np.asarray(
#         contour,
#         dtype=np.float64
#     ).reshape(-1, 2)

#     approx_points = np.asarray(
#         approx,
#         dtype=np.float64
#     ).reshape(-1, 2)

#     if len(contour_points) < 3 or len(approx_points) < 3:
#         return []

#     n = len(approx_points)

#     # For each approximate vertex, find the closest contour index.
#     vertex_indices = []

#     for vertex in approx_points:
#         distances = np.linalg.norm(
#             contour_points - vertex,
#             axis=1,
#         )

#         vertex_indices.append(
#             int(np.argmin(distances))
#         )

#     # Make sure vertices follow contour order.
#     # The approximation normally already follows contour order.
#     ordered = []

#     for idx in vertex_indices:
#         if not ordered or idx != ordered[-1]:
#             ordered.append(idx)

#     if len(ordered) != n:
#         return []

#     edge_points = []

#     for i in range(n):
#         start_idx = ordered[i]
#         end_idx = ordered[(i + 1) % n]

#         if start_idx <= end_idx:
#             indices = list(range(start_idx, end_idx + 1))
#         else:
#             indices = (
#                 list(range(start_idx, len(contour_points)))
#                 + list(range(0, end_idx + 1))
#             )

#         pts = contour_points[indices]

#         if len(pts) < 2:
#             return []

#         edge_points.append(pts)

#     return edge_points


# def _trim_rounded_corners(
#     edge_points: np.ndarray,
#     trim_fraction: float = 0.20,
# ) -> np.ndarray:
#     """
#     Remove the portions of an edge nearest to its two rounded corners.

#     Only the central portion of the contour edge is retained for
#     straight-line fitting.
#     """
#     points = np.asarray(
#         edge_points,
#         dtype=np.float64,
#     ).reshape(-1, 2)

#     if len(points) < 5:
#         return points

#     trim_fraction = max(
#         0.0,
#         min(0.45, trim_fraction)
#     )

#     trim = int(len(points) * trim_fraction)

#     if trim * 2 >= len(points) - 2:
#         return points

#     return points[trim:len(points) - trim]


# def _line_fit_error(
#     points: np.ndarray,
#     line: Tuple[float, float, float, float],
# ) -> float:
#     """
#     Mean perpendicular distance of points from a fitted line.
#     Lower is better.
#     """
#     vx, vy, x0, y0 = line

#     pts = np.asarray(
#         points,
#         dtype=np.float64,
#     ).reshape(-1, 2)

#     if len(pts) == 0:
#         return float("inf")

#     # Cross-product distance from point to infinite line.
#     distances = np.abs(
#         (pts[:, 0] - x0) * vy
#         - (pts[:, 1] - y0) * vx
#     )

#     return float(np.mean(distances))


# def _detect_straight_polygon_edges(
#     contour: np.ndarray,
#     expected_vertices: int,
# ) -> Optional[Dict[str, Any]]:
#     """
#     From one selected contour:

#       1. obtain a preliminary polygon approximation,
#       2. split the contour into edge regions,
#       3. remove rounded-corner portions,
#       4. fit one straight line to each edge,
#       5. calculate theoretical vertices from adjacent line intersections.

#     The contour itself supplies all measured geometry.
#     """
#     if contour is None or len(contour) < expected_vertices:
#         return None

#     perimeter = float(cv2.arcLength(contour, True))

#     if perimeter <= 1.0:
#         return None

#     # Get candidate polygon approximations.
#     approximations = _approx_polygon_with_vertex_count(
#         contour,
#         expected_vertices,
#     )

#     if not approximations:
#         return None

#     best_result = None
#     best_fit_error = float("inf")

#     for approx in approximations:

#         edge_regions = _polygon_edge_indices(
#             contour,
#             approx,
#         )

#         if len(edge_regions) != expected_vertices:
#             continue

#         lines = []
#         fit_errors = []

#         valid = True

#         for edge_points in edge_regions:

#             # Remove the rounded portions near both corners.
#             straight_points = _trim_rounded_corners(
#                 edge_points,
#                 trim_fraction=0.20,
#             )

#             if len(straight_points) < 2:
#                 valid = False
#                 break

#             line = _fit_line_to_points(
#                 straight_points
#             )

#             if line is None:
#                 valid = False
#                 break

#             error = _line_fit_error(
#                 straight_points,
#                 line,
#             )

#             lines.append(line)
#             fit_errors.append(error)

#         if not valid or len(lines) != expected_vertices:
#             continue

#         # Calculate theoretical vertices:
#         #
#         # L1 ∩ L2 -> V1
#         # L2 ∩ L3 -> V2
#         # ...
#         # Ln ∩ L1 -> Vn
#         vertices = []

#         for i in range(expected_vertices):
#             line_a = lines[i]
#             line_b = lines[
#                 (i + 1) % expected_vertices
#             ]

#             vertex = _line_intersection(
#                 line_a,
#                 line_b,
#             )

#             if vertex is None:
#                 valid = False
#                 break

#             vertices.append(vertex)

#         if not valid:
#             continue

#         mean_fit_error = float(
#             np.mean(fit_errors)
#         )

#         if mean_fit_error < best_fit_error:
#             best_fit_error = mean_fit_error

#             best_result = {
#                 "lines": lines,
#                 "vertices": vertices,
#                 "edge_regions": edge_regions,
#                 "fit_errors": fit_errors,
#                 "mean_fit_error": mean_fit_error,
#                 "approximation": approx.copy(),
#             }

#     return best_result

# def detect_polygon_from_annotation(
#     gray: np.ndarray,
#     annotation: Dict[str, Any],
# ) -> Optional[Dict[str, Any]]:
#     """
#     Detect a polygon from the camera image.

#     JSON is used ONLY for:
#       1. defining the general search ROI,
#       2. defining the expected number of polygon vertices.

#     The actual polygon geometry is obtained from the image contour.

#     For rounded corners:
#       - contour points are used to identify the straight edge regions,
#       - rounded corner portions are excluded,
#       - straight lines are fitted to the remaining contour points,
#       - adjacent fitted lines are intersected to obtain theoretical vertices.
#     """

#     # ------------------------------------------------------------
#     # 1. Read reference points only to obtain:
#     #    - general polygon location
#     #    - expected number of vertices
#     # ------------------------------------------------------------

#     reference = points_from_annotation(annotation)

#     if len(reference) < 3:
#         return None

#     expected_vertices = len(reference)

#     # ------------------------------------------------------------
#     # 2. JSON bounding box = ONLY general search location
#     # ------------------------------------------------------------

#     b = bbox_edges(annotation)

#     width = max(1.0, b["width"])
#     height = max(1.0, b["height"])

#     # Keep your existing ROI expansion.
#     ex = 0.30 * width
#     ey = 0.30 * height

#     h_img, w_img = gray.shape[:2]

#     x1 = max(
#         0,
#         int(math.floor(b["left"] - ex))
#     )

#     y1 = max(
#         0,
#         int(math.floor(b["top"] - ey))
#     )

#     x2 = min(
#         w_img,
#         int(math.ceil(b["right"] + ex))
#     )

#     y2 = min(
#         h_img,
#         int(math.ceil(b["bottom"] + ey))
#     )

#     if x2 <= x1 or y2 <= y1:
#         return None

#     # ------------------------------------------------------------
#     # 3. Crop ROI
#     # ------------------------------------------------------------

#     roi = gray[y1:y2, x1:x2]

#     if roi.size == 0:
#         return None

#     # ------------------------------------------------------------
#     # 4. Threshold
#     #
#     # KEEPING YOUR EXISTING METHOD.
#     # This is only being used to obtain the contour.
#     # It is NOT used directly for final vertex measurement.
#     # ------------------------------------------------------------

#     _, binary = cv2.threshold(
#         roi,
#         90,
#         255,
#         cv2.THRESH_BINARY,
#     )

#     # ------------------------------------------------------------
#     # 5. Save debugging images
#     # ------------------------------------------------------------

#     tag = annotation.get(
#         "tag_id",
#         "polygon",
#     )

#     cv2.imwrite(
#         str(
#             OUTPUT_DIR /
#             f"binary_{tag}.png"
#         ),
#         binary,
#     )

#     cv2.imwrite(
#         str(
#             OUTPUT_DIR /
#             f"roi_{tag}.png"
#         ),
#         roi,
#     )

#     # ------------------------------------------------------------
#     # 6. Find contours
#     #
#     # KEEPING YOUR EXISTING CONTOUR STAGE.
#     # ------------------------------------------------------------

#     contours, _ = cv2.findContours(
#         binary,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_NONE,
#     )

#     # ------------------------------------------------------------
#     # 7. Draw all contours for debugging
#     # ------------------------------------------------------------

#     contours_overlay = cv2.cvtColor(
#         roi,
#         cv2.COLOR_GRAY2BGR,
#     )

#     cv2.drawContours(
#         contours_overlay,
#         contours,
#         -1,
#         (0, 255, 0),
#         1,
#     )

#     cv2.imwrite(
#         str(
#             OUTPUT_DIR /
#             f"contours_overlay_{tag}.png"
#         ),
#         contours_overlay,
#     )

#     if not contours:
#         return None

#     # ------------------------------------------------------------
#     # 8. Select the correct WHOLE contour
#     #
#     # IMPORTANT:
#     #
#     # We no longer use:
#     #   - reference center
#     #   - reference point distances
#     #   - matchShapes()
#     #   - reference area ratio
#     #
#     # The JSON is only giving us the general ROI.
#     #
#     # We select the contour using:
#     #   - reasonable size
#     #   - expected number of vertices
#     #   - contour location inside the ROI
#     #
#     # The actual edge geometry comes later from the contour.
#     # ------------------------------------------------------------

#     contour_candidates = []

#     roi_area = float(roi.shape[0] * roi.shape[1])

#     for contour in contours:

#         area = abs(
#             float(cv2.contourArea(contour))
#         )

#         if area <= 1.0:
#             continue

#         # Ignore extremely tiny contours.
#         if area < 0.01 * roi_area:
#             continue

#         perimeter = float(
#             cv2.arcLength(
#                 contour,
#                 True,
#             )
#         )

#         if perimeter <= 1.0:
#             continue

#         # Get a preliminary approximation only to determine
#         # whether this contour can represent the required
#         # number of vertices.
#         approximations = (
#             _approx_polygon_with_vertex_count(
#                 contour,
#                 expected_vertices,
#             )
#         )

#         if not approximations:
#             continue

#         # Use the largest-area approximation as the candidate.
#         best_approx_for_contour = max(
#             approximations,
#             key=lambda a: abs(
#                 float(cv2.contourArea(a))
#             ),
#         )

#         approx_area = abs(
#             float(
#                 cv2.contourArea(
#                     best_approx_for_contour
#                 )
#             )
#         )

#         # Basic contour compactness.
#         # This is image-derived, not JSON-derived.
#         compactness = (
#             (4.0 * math.pi * area)
#             / max(perimeter * perimeter, 1e-9)
#         )

#         contour_candidates.append(
#             {
#                 "contour": contour,
#                 "area": area,
#                 "perimeter": perimeter,
#                 "approximation": (
#                     best_approx_for_contour
#                 ),
#                 "approx_area": approx_area,
#                 "compactness": compactness,
#             }
#         )

#     if not contour_candidates:
#         return None

#     # ------------------------------------------------------------
#     # 9. From the contour candidates, find one that produces
#     #    the best six straight edges.
#     #
#     # This is now the important selection stage.
#     # ------------------------------------------------------------

#     best_contour_result = None
#     best_edge_fit_error = float("inf")

#     for candidate in contour_candidates:

#         contour = candidate["contour"]

#         edge_result = (
#             _detect_straight_polygon_edges(
#                 contour,
#                 expected_vertices,
#             )
#         )

#         if edge_result is None:
#             continue

#         mean_fit_error = (
#             edge_result["mean_fit_error"]
#         )

#         # Lower line-fitting error means the contour
#         # contains stronger straight edge regions.
#         if mean_fit_error < best_edge_fit_error:

#             best_edge_fit_error = (
#                 mean_fit_error
#             )

#             best_contour_result = {
#                 "contour": contour,
#                 "area": candidate["area"],
#                 "perimeter": candidate["perimeter"],
#                 "edge_result": edge_result,
#             }

#     if best_contour_result is None:
#         return None

#     # ------------------------------------------------------------
#     # 10. Get the six fitted straight lines
#     # ------------------------------------------------------------

#     selected_contour = (
#         best_contour_result["contour"]
#     )

#     edge_result = (
#         best_contour_result["edge_result"]
#     )

#     lines = edge_result["lines"]

#     edge_regions = edge_result[
#         "edge_regions"
#     ]

#     fit_errors = edge_result[
#         "fit_errors"
#     ]

#     # 11. Get theoretical vertices

#     detected_points_roi = (
#         edge_result["vertices"]
#     )

#     # 12. Convert ROI coordinates to full-image coordinates

#     detected_points = [
#         (
#             float(x + x1),
#             float(y + y1),
#         )
#         for x, y in detected_points_roi
#     ]

#     detected_points = _best_polygon_order(
#         detected_points,
#         reference,
#     )

#     # ------------------------------------------------------------
#     # 14. Calculate final polygon geometry
#     # ------------------------------------------------------------

#     detected_np = np.asarray(
#         detected_points,
#         dtype=np.float64,
#     )

#     area_px2 = abs(
#         float(
#             cv2.contourArea(
#                 detected_np.astype(
#                     np.float32
#                 )
#             )
#         )
#     )

#     perimeter_px = 0.0

#     edge_lengths_px: List[float] = []

#     for i in range(
#         len(detected_points)
#     ):

#         x_a, y_a = (
#             detected_points[i]
#         )

#         x_b, y_b = (
#             detected_points[
#                 (i + 1) % len(detected_points)
#             ]
#         )

#         d = px_distance(
#             x_b - x_a,
#             y_b - y_a,
#         )

#         edge_lengths_px.append(d)

#         perimeter_px += d

#     # ------------------------------------------------------------
#     # 15. Bounding box of THEORETICAL vertices
#     # ------------------------------------------------------------

#     left = float(
#         np.min(detected_np[:, 0])
#     )

#     top = float(
#         np.min(detected_np[:, 1])
#     )

#     right = float(
#         np.max(detected_np[:, 0])
#     )

#     bottom = float(
#         np.max(detected_np[:, 1])
#     )

#     # ------------------------------------------------------------
#     # 16. Visualization of fitted lines + theoretical vertices
#     # ------------------------------------------------------------

#     detection_overlay = cv2.cvtColor(
#         roi,
#         cv2.COLOR_GRAY2BGR,
#     )

#     # Draw original contour in green.
#     cv2.drawContours(
#         detection_overlay,
#         [selected_contour],
#         -1,
#         (0, 255, 0),
#         1,
#     )

#     # Draw fitted infinite lines.
#     line_length = max(
#         roi.shape[0],
#         roi.shape[1],
#     ) * 2.0

#     for line in lines:

#         vx, vy, x0, y0 = line

#         p1 = (
#             int(round(
#                 x0 - vx * line_length
#             )),
#             int(round(
#                 y0 - vy * line_length
#             )),
#         )

#         p2 = (
#             int(round(
#                 x0 + vx * line_length
#             )),
#             int(round(
#                 y0 + vy * line_length
#             )),
#         )

#         cv2.line(
#             detection_overlay,
#             p1,
#             p2,
#             (255, 0, 0),
#             2,
#         )

#     # Draw theoretical vertices in red.
#     for i, (x, y) in enumerate(
#         detected_points_roi
#     ):

#         px = int(round(x))
#         py = int(round(y))

#         cv2.circle(
#             detection_overlay,
#             (px, py),
#             5,
#             (0, 0, 255),
#             -1,
#         )

#         cv2.putText(
#             detection_overlay,
#             f"V{i + 1}",
#             (px + 8, py - 8),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 255),
#             2,
#             cv2.LINE_AA,
#         )

#     cv2.imwrite(
#         str(
#             OUTPUT_DIR /
#             f"polygon_lines_{tag}.png"
#         ),
#         detection_overlay,
#     )

#     # ------------------------------------------------------------
#     # 17. Return result
#     # ------------------------------------------------------------

#     return {
#         "points": detected_points,

#         "edge_lengths_px": edge_lengths_px,

#         "perimeter_px": perimeter_px,

#         "area_px2": area_px2,

#         "left": left,
#         "top": top,
#         "right": right,
#         "bottom": bottom,

#         "width": right - left,
#         "height": bottom - top,

#         "center_x": float(
#             np.mean(
#                 detected_np[:, 0]
#             )
#         ),

#         "center_y": float(
#             np.mean(
#                 detected_np[:, 1]
#             )
#         ),

#         "roi": roi,
#         "binary": binary,

#         # Selected contour
#         "contour": selected_contour,

#         # Fitted straight edges
#         "lines": lines,

#         # Points used to fit each edge
#         "edge_regions": edge_regions,

#         # Straight-line fitting error for each edge
#         "edge_fit_errors": fit_errors,

#         "mean_edge_fit_error": float(
#             edge_result["mean_fit_error"]
#         ),

#         # Kept as "score" so the rest of your
#         # existing code doesn't break.
#         #
#         # Higher score = better straight-line fit.
#         "score": float(
#             -edge_result["mean_fit_error"]
#         ),
#     }

# def draw_detected_polygon(
#     overlay: np.ndarray,
#     detected: Dict[str, Any],
#     label: str,
# ) -> None:
#     """Draw the polygon actually detected from the image in green."""
#     points = detected.get("points", [])
#     if len(points) < 3:
#         return

#     pts = np.asarray(
#         [[int(round(x)), int(round(y))] for x, y in points],
#         dtype=np.int32,
#     ).reshape((-1, 1, 2))

#     cv2.polylines(
#         overlay,
#         [pts],
#         True,
#         (0, 255, 0),
#         1
#     )

#     for i, (x, y) in enumerate(points, start=1):
#         p = (int(round(x)), int(round(y)))
#         cv2.circle(overlay, p, 4, (0, 255, 0), -1)
#         cv2.putText(
#             overlay,
#             f"D{i}",
#             (p[0] + 5, p[1] + 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.4,
#             (0, 255, 0),
#             1,
#         )

#     draw_label(
#         overlay,
#         f"DETECTED {label} {len(points)} pts",
#         detected["center_x"],
#         detected["center_y"],
#     )

# # -----------------------------------------------------------------------------
# # Per-shape processing
# # -----------------------------------------------------------------------------

# # def circle_result(
# #     feature: Dict[str, Any],
# #     annotation: Dict[str, Any],
# #     annotation_index: int,
# #     detected: Optional[Dict[str, float]],
# #     x_scale: float,
# #     y_scale: float,
# # ) -> Dict[str, Any]:
# #     b = bbox_edges(annotation)
# #     expected = expected_values(feature)

# #     result: Dict[str, Any] = {
# #         "feature_id": feature.get("id"),
# #         "feature_type": feature.get("type"),
# #         "annotation_index": annotation_index,
# #         "annotation_shape": shape_name(annotation),
# #         "annotation_tag_name": annotation.get("tag_name"),
# #         "annotation_tag_id": annotation.get("tag_id"),
# #         "annotation_left_px": b["left"],
# #         "annotation_top_px": b["top"],
# #         "annotation_right_px": b["right"],
# #         "annotation_bottom_px": b["bottom"],
# #         "annotation_width_px": b["width"],
# #         "annotation_height_px": b["height"],
# #         "annotation_center_x_px": b["center_x"],
# #         "annotation_center_y_px": b["center_y"],
# #         "annotation_radius_x_px": b["width"] / 2.0,
# #         "annotation_radius_y_px": b["height"] / 2.0,
# #         "annotation_radius_px": (b["width"] / 2.0 + b["height"] / 2.0) / 2.0,
# #         "annotation_diameter_x_px": b["width"],
# #         "annotation_diameter_y_px": b["height"],
# #         "annotation_diameter_px": (b["width"] + b["height"]) / 2.0,
# #         "detected_center_x_px": None,
# #         "detected_center_y_px": None,
# #         "detected_radius_px": None,
# #         "detected_diameter_px": None,
# #         "detected_diameter_mm_x": None,
# #         "detected_diameter_mm_y": None,
# #         "detected_diameter_mm": None,
# #         "edge_support": None,
# #         "nominal_diameter_mm": expected["nominal_diameter_mm"],
# #         "lower_limit_mm": expected["lower_diameter_mm"],
# #         "upper_limit_mm": expected["upper_diameter_mm"],
# #         "status": "NOT_FOUND",
# #     }

# #     if detected is None:
# #         return result

# #     d_px = detected["diameter_px"]
# #     d_x_mm = d_px * x_scale
# #     d_y_mm = d_px * y_scale
# #     d_mm = (d_x_mm + d_y_mm) / 2.0

# #     result.update(
# #         {
# #             "detected_center_x_px": detected["center_x"],
# #             "detected_center_y_px": detected["center_y"],
# #             "detected_radius_px": detected["radius_px"],
# #             "detected_diameter_px": d_px,
# #             "detected_diameter_mm_x": d_x_mm,
# #             "detected_diameter_mm_y": d_y_mm,
# #             "detected_diameter_mm": d_mm,
# #             "edge_support": detected["edge_support"],
# #             "status": "MEASURED",
# #         }
# #     )

# #     low = expected["lower_diameter_mm"]
# #     high = expected["upper_diameter_mm"]
# #     if expected["nominal_diameter_mm"] is not None and low is not None and high is not None:
# #         result["status"] = "PASS" if low <= d_mm <= high else "FAIL"

# #     return result


# # def rectangle_result(
# #     feature: Dict[str, Any],
# #     annotation: Dict[str, Any],
# #     annotation_index: int,
# #     detected: Optional[Dict[str, float]],
# #     x_scale: float,
# #     y_scale: float,
# # ) -> Dict[str, Any]:

# #     b = bbox_edges(annotation)

# #     result = {
# #         "feature_id": feature.get("id"),
# #         "feature_type": feature.get("type"),
# #         "annotation_index": annotation_index,
# #         "annotation_shape": shape_name(annotation),
# #         "annotation_tag_name": annotation.get("tag_name"),
# #         "annotation_tag_id": annotation.get("tag_id"),

# #         # JSON reference
# #         "annotation_left_px": b["left"],
# #         "annotation_top_px": b["top"],
# #         "annotation_right_px": b["right"],
# #         "annotation_bottom_px": b["bottom"],
# #         "annotation_width_px": b["width"],
# #         "annotation_height_px": b["height"],

# #         # Image detected
# #         "detected_left_px": None,
# #         "detected_top_px": None,
# #         "detected_right_px": None,
# #         "detected_bottom_px": None,
# #         "detected_width_px": None,
# #         "detected_height_px": None,
# #         "detected_center_x_px": None,
# #         "detected_center_y_px": None,

# #         "width_mm": None,
# #         "height_mm": None,
# #         "diagonal_mm": None,

# #         "status": "NOT_FOUND",
# #     }

# #     if detected is None:
# #         return result
# #     width_mm = detected["width"] * x_scale
# #     height_mm = detected["height"] * y_scale

# #     diagonal_mm = math.hypot(
# #         width_mm,
# #         height_mm,
# #     ) 
# #     result.update(
# #         {
# #             "detected_left_px": detected["left"],
# #             "detected_top_px": detected["top"],
# #             "detected_right_px": detected["right"],
# #             "detected_bottom_px": detected["bottom"],
# #             "detected_width_px": detected["width"],
# #             "detected_height_px": detected["height"],
# #             "detected_center_x_px": detected["center_x"],
# #             "detected_center_y_px": detected["center_y"],
# #             "width_mm": width_mm,
# #             "height_mm": height_mm,
# #             "diagonal_mm": diagonal_mm,
# #             "status": "MEASURED",
# #         }
# #     )

# #     return result

# # def polygon_result(
# #     feature: Dict[str, Any],
# #     annotation: Dict[str, Any],
# #     annotation_index: int,
# #     detected: Optional[Dict[str, Any]],
# #     x_scale: float,
# #     y_scale: float,
# # ) -> Dict[str, Any]:
# #     """Create polygon measurement results from detected image vertices."""
# #     reference = points_from_annotation(annotation)

# #     result: Dict[str, Any] = {
# #         "feature_id": feature.get("id"),
# #         "feature_type": feature.get("type"),
# #         "annotation_index": annotation_index,
# #         "annotation_shape": shape_name(annotation),
# #         "annotation_tag_name": annotation.get("tag_name"),
# #         "annotation_tag_id": annotation.get("tag_id"),
# #         "point_count": len(reference),
# #         "points_px": [{"x": x, "y": y} for x, y in reference],
# #         "detected_points_px": None,
# #         "edge_lengths_px": [],
# #         "edge_lengths_mm": [],
# #         "perimeter_px": None,
# #         "perimeter_mm": None,
# #         "area_px2": None,
# #         "area_mm2": None,
# #         "detected_left_px": None,
# #         "detected_top_px": None,
# #         "detected_right_px": None,
# #         "detected_bottom_px": None,
# #         "detected_width_px": None,
# #         "detected_height_px": None,
# #         "detected_width_mm": None,
# #         "detected_height_mm": None,
# #         "detected_center_x_px": None,
# #         "detected_center_y_px": None,
# #         "detection_score": None,
# #         "status": "NOT_FOUND",
# #     }

# #     if detected is None:
# #         return result

# #     detected_points = detected["points"]
# #     edge_lengths_px = list(detected["edge_lengths_px"])
# #     edge_lengths_mm = [
# #         convert_dxdy_to_mm(
# #             detected_points[(i + 1) % len(detected_points)][0] - detected_points[i][0],
# #             detected_points[(i + 1) % len(detected_points)][1] - detected_points[i][1],
# #             x_scale,
# #             y_scale,
# #         )
# #         for i in range(len(detected_points))
# #     ]

# #     perimeter_px = float(sum(edge_lengths_px))
# #     perimeter_mm = float(sum(edge_lengths_mm))
# #     area_px2 = float(detected["area_px2"])
# #     area_mm2 = area_px2 * x_scale * y_scale

# #     result.update(
# #         {
# #             "detected_points_px": [
# #                 {"x": float(x), "y": float(y)}
# #                 for x, y in detected_points
# #             ],
# #             "edge_lengths_px": edge_lengths_px,
# #             "edge_lengths_mm": edge_lengths_mm,
# #             "perimeter_px": perimeter_px,
# #             "perimeter_mm": perimeter_mm,
# #             "area_px2": area_px2,
# #             "area_mm2": area_mm2,
# #             "detected_left_px": detected["left"],
# #             "detected_top_px": detected["top"],
# #             "detected_right_px": detected["right"],
# #             "detected_bottom_px": detected["bottom"],
# #             "detected_width_px": detected["width"],
# #             "detected_height_px": detected["height"],
# #             "detected_width_mm": detected["width"] * x_scale,
# #             "detected_height_mm": detected["height"] * y_scale,
# #             "detected_center_x_px": detected["center_x"],
# #             "detected_center_y_px": detected["center_y"],
# #             "detection_score": detected["score"],
# #             "status": "MEASURED",
# #         }
# #     )

# #     return result


# def circle_result(
#     feature: Dict[str, Any],
#     annotation: Dict[str, Any],
#     annotation_index: int,
#     detected: Optional[Dict[str, float]],
#     x_scale: float,
#     y_scale: float,
# ) -> Dict[str, Any]:
#     expected = expected_values(feature)
#     result: Dict[str, Any] = {
#         "feature_id": feature.get("id"),
#         "feature_type": feature.get("type"),
#         "annotation_shape": shape_name(annotation),
#         "annotation_tag_id": annotation.get("tag_id"),
#         "detected_center_x_px": None,
#         "detected_center_y_px": None,
#         "detected_radius_px": None,
#         "detected_diameter_px": None,
#         "detected_diameter_mm_x": None,
#         "detected_diameter_mm_y": None,
#         "detected_diameter_mm": None,
#         "edge_support": None,
#         "nominal_diameter_mm": expected["nominal_diameter_mm"],
#         "lower_limit_mm": expected["lower_diameter_mm"],
#         "upper_limit_mm": expected["upper_diameter_mm"],
#         "status": "NOT_FOUND",
#     }

#     if detected is None:
#         return result

#     d_px = detected["diameter_px"]
#     d_x_mm = d_px * x_scale
#     d_y_mm = d_px * y_scale
#     d_mm = (d_x_mm + d_y_mm) / 2.0

#     result.update({
#         "detected_center_x_px": detected["center_x"],
#         "detected_center_y_px": detected["center_y"],
#         "detected_radius_px": detected["radius_px"],
#         "detected_diameter_px": d_px,
#         "detected_diameter_mm_x": d_x_mm,
#         "detected_diameter_mm_y": d_y_mm,
#         "detected_diameter_mm": d_mm,
#         "edge_support": detected["edge_support"],
#         "status": "MEASURED",
#     })

#     low = expected["lower_diameter_mm"]
#     high = expected["upper_diameter_mm"]
#     if expected["nominal_diameter_mm"] is not None and low is not None and high is not None:
#         result["status"] = "PASS" if low <= d_mm <= high else "FAIL"

#     return result


# def rectangle_result(
#     feature: Dict[str, Any],
#     annotation: Dict[str, Any],
#     annotation_index: int,
#     detected: Optional[Dict[str, float]],
#     x_scale: float,
#     y_scale: float,
# ) -> Dict[str, Any]:
#     result = {
#         "feature_id": feature.get("id"),
#         "feature_type": feature.get("type"),
#         "annotation_shape": shape_name(annotation),
#         "annotation_tag_id": annotation.get("tag_id"),
#         "detected_left_px": None,
#         "detected_top_px": None,
#         "detected_right_px": None,
#         "detected_bottom_px": None,
#         "detected_width_px": None,
#         "detected_height_px": None,
#         "detected_center_x_px": None,
#         "detected_center_y_px": None,
#         "width_mm": None,
#         "height_mm": None,
#         "diagonal_mm": None,
#         "status": "NOT_FOUND",
#     }

#     if detected is None:
#         return result

#     width_mm = detected["width"] * x_scale
#     height_mm = detected["height"] * y_scale
#     diagonal_mm = math.hypot(width_mm, height_mm)

#     result.update({
#         "detected_left_px": detected["left"],
#         "detected_top_px": detected["top"],
#         "detected_right_px": detected["right"],
#         "detected_bottom_px": detected["bottom"],
#         "detected_width_px": detected["width"],
#         "detected_height_px": detected["height"],
#         "detected_center_x_px": detected["center_x"],
#         "detected_center_y_px": detected["center_y"],
#         "width_mm": width_mm,
#         "height_mm": height_mm,
#         "diagonal_mm": diagonal_mm,
#         "status": "MEASURED",
#     })

#     return result


# def polygon_result(
#     feature: Dict[str, Any],
#     annotation: Dict[str, Any],
#     annotation_index: int,
#     detected: Optional[Dict[str, Any]],
#     x_scale: float,
#     y_scale: float,
# ) -> Dict[str, Any]:
#     reference = points_from_annotation(annotation)
#     result: Dict[str, Any] = {
#         "feature_id": feature.get("id"),
#         "feature_type": feature.get("type"),
#         "annotation_shape": shape_name(annotation),
#         "annotation_tag_id": annotation.get("tag_id"),
#         "point_count": len(reference),
#         "detected_points_px": None,
#         "edge_lengths_px": [],
#         "edge_lengths_mm": [],
#         "perimeter_px": None,
#         "perimeter_mm": None,
#         "area_px2": None,
#         "area_mm2": None,
#         "detected_left_px": None,
#         "detected_top_px": None,
#         "detected_right_px": None,
#         "detected_bottom_px": None,
#         "detected_width_px": None,
#         "detected_height_px": None,
#         "detected_width_mm": None,
#         "detected_height_mm": None,
#         "detected_center_x_px": None,
#         "detected_center_y_px": None,
#         "detection_score": None,
#         "status": "NOT_FOUND",
#     }

#     if detected is None:
#         return result

#     detected_points = detected["points"]
#     edge_lengths_px = list(detected["edge_lengths_px"])
#     edge_lengths_mm = [
#         convert_dxdy_to_mm(
#             detected_points[(i + 1) % len(detected_points)][0] - detected_points[i][0],
#             detected_points[(i + 1) % len(detected_points)][1] - detected_points[i][1],
#             x_scale,
#             y_scale,
#         )
#         for i in range(len(detected_points))
#     ]

#     perimeter_px = float(sum(edge_lengths_px))
#     perimeter_mm = float(sum(edge_lengths_mm))
#     area_px2 = float(detected["area_px2"])
#     area_mm2 = area_px2 * x_scale * y_scale

#     result.update({
#         "detected_points_px": [{"x": float(x), "y": float(y)} for x, y in detected_points],
#         "edge_lengths_px": edge_lengths_px,
#         "edge_lengths_mm": edge_lengths_mm,
#         "perimeter_px": perimeter_px,
#         "perimeter_mm": perimeter_mm,
#         "area_px2": area_px2,
#         "area_mm2": area_mm2,
#         "detected_left_px": detected["left"],
#         "detected_top_px": detected["top"],
#         "detected_right_px": detected["right"],
#         "detected_bottom_px": detected["bottom"],
#         "detected_width_px": detected["width"],
#         "detected_height_px": detected["height"],
#         "detected_width_mm": detected["width"] * x_scale,
#         "detected_height_mm": detected["height"] * y_scale,
#         "detected_center_x_px": detected["center_x"],
#         "detected_center_y_px": detected["center_y"],
#         "detection_score": detected["score"],
#         "status": "MEASURED",
#     })

#     return result



# # -----------------------------------------------------------------------------
# # JSON loading / validation
# # -----------------------------------------------------------------------------

# def load_drawing(path: Path) -> Dict[str, Any]:
#     with path.open("r", encoding="utf-8") as f:
#         drawing = json.load(f)

#     if not isinstance(drawing, dict):
#         raise ValueError("Drawing JSON root must be an object.")
#     if not isinstance(drawing.get("drawingFeatures"), list):
#         raise ValueError("Drawing JSON must contain drawingFeatures[].")
#     return drawing


# def iter_annotations(drawing: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
#     """Yield EVERY annotation, with no tag-based filtering."""
#     for feature in drawing["drawingFeatures"]:
#         annotations = feature.get("annotations", []) or []
#         for index, annotation in enumerate(annotations, start=1):
#             if not isinstance(annotation, dict):
#                 continue
#             yield feature, index, annotation


# def make_output_row_for_unsupported(
#     feature: Dict[str, Any], annotation: Dict[str, Any], index: int
# ) -> Dict[str, Any]:
#     return {
#         "feature_id": feature.get("id"),
#         "feature_type": feature.get("type"),
#         "annotation_index": index,
#         "annotation_shape": shape_name(annotation),
#         "annotation_tag_name": annotation.get("tag_name"),
#         "annotation_tag_id": annotation.get("tag_id"),
#         "status": "UNSUPPORTED_SHAPE",
#     }


# BASE_DIR = Path(__file__).resolve().parent

# JSON_PATH = BASE_DIR / "drawing_dimensions_new.json"
# IMAGE_PATH = BASE_DIR / "captures" / "part20.png"
# CALIBRATION_PATH = BASE_DIR / "calibration" / "results" / "camera_calibration.json"
# OUTPUT_DIR = BASE_DIR / "results" / "drawing_validation"


# def _read_positive_mm(prompt: str) -> float:
#     """Read a positive measurement in millimetres from the console."""
#     while True:
#         value = input(prompt).strip()
#         try:
#             value_mm = float(value)
#         except ValueError:
#             print("Please enter a numeric value in mm, for example 250.")
#             continue

#         if value_mm <= 0:
#             print("Value must be greater than 0 mm.")
#             continue

#         return value_mm


# def parse_args() -> argparse.Namespace:
#     """
#     Use fixed project paths and a fixed camera-to-base height of 560 mm.

#     The product height is entered from the console for now.
#     Later, the frontend can pass this same product height directly.

#     Camera-to-sample distance:
#         distance_mm = 560.0 - product_height_mm
#     """

#     parser = argparse.ArgumentParser(
#         description="Dynamic Buhler annotation-shape validation"
#     )
#     # Keep normal script compatibility; product height is intentionally
#     # requested from the console for now.
#     parser.parse_args()

#     TOTAL_CAMERA_TO_BASE_MM = 560.0

#     print("\nCamera setup:")
#     print(f"  Total camera-to-base height : {TOTAL_CAMERA_TO_BASE_MM:.3f} mm")

#     product_height_mm = _read_positive_mm(
#         "Enter PRODUCT SAMPLE height (mm): "
#     )

#     if product_height_mm >= TOTAL_CAMERA_TO_BASE_MM:
#         raise ValueError(
#             "Product height must be smaller than the fixed camera-to-base height.\n"
#             f"Camera-to-base : {TOTAL_CAMERA_TO_BASE_MM:.3f} mm\n"
#             f"Product height : {product_height_mm:.3f} mm"
#         )

#     distance_mm = TOTAL_CAMERA_TO_BASE_MM - product_height_mm

#     args = argparse.Namespace(
#         json=JSON_PATH,
#         image=IMAGE_PATH,
#         calibration=CALIBRATION_PATH,
#         total_length_mm=TOTAL_CAMERA_TO_BASE_MM,
#         product_height_mm=product_height_mm,
#         distance_mm=distance_mm,
#         output_dir=OUTPUT_DIR,
#     )

#     for label, path in (
#         ("JSON", args.json),
#         ("image", args.image),
#         ("calibration", args.calibration),
#     ):
#         if not path.exists():
#             raise FileNotFoundError(
#                 f"{label} file not found:\n{path}\n\n"
#                 "Check the project folder structure and filename."
#             )

#     return args
# def draw_detected_rectangle(
#     image: np.ndarray,
#     points: np.ndarray,
# ) -> np.ndarray:
#     """
#     Draw the detected rectangle on the image.

#     Green color: BGR (0, 255, 0)
#     Thickness: 1 pixel
#     """

#     output = image.copy()

#     if points is None or len(points) != 4:
#         return output

#     pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)

#     cv2.polylines(
#         output,
#         [pts],
#         isClosed=True,
#         color=(0, 255, 0),
#         thickness=1,
#         lineType=cv2.LINE_AA,
#     )

#     return output
# def main() -> None:
#     args = parse_args()

#     print("\nLoading project inputs:")
#     print("  JSON                    :", args.json)
#     print("  Image                   :", args.image)
#     print("  Calibration             :", args.calibration)
#     print("  Total length (mm)       :", args.total_length_mm)
#     print("  Product height (mm)     :", args.product_height_mm)
#     print("  Camera-to-sample (mm)   :", args.distance_mm)
#     print("  Output                  :", args.output_dir)
#     print()

#     drawing = load_drawing(args.json)
#     image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {args.image}")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     overlay = image.copy()

#     camera_matrix, distortion, fx, fy = load_calibration(args.calibration)
#     x_scale, y_scale = mm_per_pixel(args.distance_mm, fx, fy)

#     # We intentionally do not undistort/warp the annotation coordinates here.
#     # They are treated as coordinates in the supplied camera image, exactly as requested.

#     output_dir = args.output_dir or (args.calibration.parent / "drawing_validation")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     overlay_path = output_dir / "drawing_validation_overlay.png"
#     csv_path = output_dir / "drawing_validation_report.csv"
#     json_path = output_dir / "drawing_validation_report.json"

#     results: List[Dict[str, Any]] = []

#     print("=" * 110)
#     print("BUHLER DYNAMIC ANNOTATION VALIDATION")
#     print("=" * 110)
#     print("Part number       :", drawing.get("part_number"))
#     print("Drawing JSON      :", args.json)
#     print("Image             :", args.image)
#     print("Calibration       :", args.calibration)
#     print("Total length (mm) :", f"{args.total_length_mm:.3f}")
#     print("Product height    :", f"{args.product_height_mm:.3f} mm")
#     print("Distance (mm)     :", f"{args.distance_mm:.3f}")
#     print("Formula           : total length - product height")
#     print("Image resolution  :", f"{image.shape[1]} x {image.shape[0]}")
#     print("fx / fy (px)      :", f"{fx:.6f} / {fy:.6f}")
#     print("X scale (mm/px)   :", f"{x_scale:.9f}")
#     print("Y scale (mm/px)   :", f"{y_scale:.9f}")
#     print("Coordinate source : drawingFeatures[].annotations[]")
#     print("Shape dispatch     : annotation.shape ONLY")
#     print("CVAT              : NOT USED")
#     print("=" * 110)

#     shape_counts: Dict[str, int] = {}

#     for feature, index, annotation in iter_annotations(drawing):
#         shape = shape_name(annotation)
#         shape_counts[shape] = shape_counts.get(shape, 0) + 1
#         label = f"{feature.get('id', 'FEATURE')}-{index}"

#         print(f"\n{label}")
#         print("  shape :", shape)
#         print("  tag   :", annotation.get("tag_name"), "/", annotation.get("tag_id"))

#         if shape == "circle":
#             b = bbox_edges(annotation)
#             print(
#                 "  JSON  :",
#                 f"left={b['left']:.6f}, top={b['top']:.6f}, "
#                 f"right={b['right']:.6f}, bottom={b['bottom']:.6f}, "
#                 f"width={b['width']:.6f}, height={b['height']:.6f}"
#             )
#             print(
#                 "  JSON circle:",
#                 f"center=({b['center_x']:.6f},{b['center_y']:.6f}), "
#                 f"radius_x={b['width']/2.0:.6f}px, "
#                 f"radius_y={b['height']/2.0:.6f}px, "
#                 f"radius={(b['width']/2.0 + b['height']/2.0)/2.0:.6f}px"
#             )

#             detected = detect_circle_from_annotation(gray, annotation)
#             draw_circle_annotation(overlay, annotation, detected, label)
#             row = circle_result(feature, annotation, index, detected, x_scale, y_scale)
#             results.append(row)

#             if detected is None:
#                 print("  IMAGE : circle fit FAILED")
#             else:
#                 print(
#                     "  IMAGE :",
#                     f"center=({detected['center_x']:.3f},{detected['center_y']:.3f})px, "
#                     f"radius={detected['radius_px']:.3f}px, "
#                     f"diameter={detected['diameter_px']:.3f}px, "
#                     f"edge_support={detected['edge_support']:.3f}"
#                 )
#                 print(
#                     "  SIZE  :",
#                     f"diameter_x={row['detected_diameter_mm_x']:.3f}mm, "
#                     f"diameter_y={row['detected_diameter_mm_y']:.3f}mm, "
#                     f"mean={row['detected_diameter_mm']:.3f}mm"
#                 )

#         elif shape in {"rect", "rectangle", "square"}:
#             b = bbox_edges(annotation)
#             detected = detect_rectangle_from_annotation(gray, annotation)
#             # draw_rect_annotation(overlay, annotation, label)
#             draw_rect_annotation(overlay, annotation,detected, label)

#             row = rectangle_result(
#                 feature,
#                 annotation,
#                 index,
#                 detected,
#                 x_scale,
#                 y_scale,
#             )
#             results.append(row)
#             print(
#                 "  RECT  :",
#                 f"left={b['left']:.6f}, top={b['top']:.6f}, "
#                 f"right={b['right']:.6f}, bottom={b['bottom']:.6f}, "
#                 f"width={b['width']:.6f}px, height={b['height']:.6f}px"
#             )
#             if row["width_mm"] is not None and row["height_mm"] is not None:
#                 print(
#                     "  SIZE  :",
#                     f"width={row['width_mm']:.3f}mm, height={row['height_mm']:.3f}mm"
#                 )
#             else:
#                 print("  SIZE  : rectangle detection FAILED")

#         elif shape == "polygon":
#             points = points_from_annotation(annotation)

#             # RED = JSON/reference polygon and its original point order
#             draw_polygon_annotation(overlay, annotation, label)

#             # GREEN = polygon detected from the actual camera image
#             detected = detect_polygon_from_annotation(gray, annotation)
#             if detected is not None:
#                 draw_detected_polygon(overlay, detected, label)

#             row = polygon_result(
#                 feature,
#                 annotation,
#                 index,
#                 detected,
#                 x_scale,
#                 y_scale,
#             )
#             results.append(row)

#             print("  POLYGON: point_count=", len(points))
#             for i, (x, y) in enumerate(points, start=1):
#                 print(f"    JSON P{i}: ({x:.6f}, {y:.6f})")

#             if detected is None:
#                 print("  IMAGE : polygon detection FAILED")
#             else:
#                 for i, (x, y) in enumerate(detected["points"], start=1):
#                     print(f"    DETECTED P{i}: ({x:.6f}, {y:.6f})")
#                 print(
#                     "  SIZE  :",
#                     f"width={row['detected_width_mm']:.3f}mm, "
#                     f"height={row['detected_height_mm']:.3f}mm, "
#                     f"perimeter={row['perimeter_mm']:.3f}mm, "
#                     f"area={row['area_mm2']:.3f}mm^2"
#                 )
#                 for i, edge_mm in enumerate(row["edge_lengths_mm"], start=1):
#                     print(f"    EDGE {i}: {edge_mm:.3f} mm")

#         else:
#             print("  WARNING: unsupported annotation.shape -> preserved, not discarded")
#             results.append(make_output_row_for_unsupported(feature, annotation, index))

#     cv2.imwrite(str(overlay_path), overlay)

#     # Flat CSV: include all common fields plus serialized variable fields.
#     csv_rows: List[Dict[str, Any]] = []
#     for row in results:
#         flat = dict(row)
#         if isinstance(flat.get("points_px"), list):
#             flat["points_px"] = json.dumps(flat["points_px"], ensure_ascii=False)
#         if isinstance(flat.get("edge_lengths_px"), list):
#             flat["edge_lengths_px"] = json.dumps(flat["edge_lengths_px"])
#         if isinstance(flat.get("edge_lengths_mm"), list):
#             flat["edge_lengths_mm"] = json.dumps(flat["edge_lengths_mm"])
#         csv_rows.append(flat)

#     csv_fields = sorted({key for row in csv_rows for key in row.keys()})
#     with csv_path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
#         writer.writeheader()
#         writer.writerows(csv_rows)

#     report = {
#         "drawingId": drawing.get("drawingId"),
#         "part_number": drawing.get("part_number"),
#         "image": str(args.image),
#         "calibration": str(args.calibration),
#         "total_length_mm": args.total_length_mm,
#         "product_height_mm": args.product_height_mm,
#         "distance_mm": args.distance_mm,
#         "distance_calculation": "total_length_mm - product_height_mm",
#         "camera_matrix": camera_matrix.tolist(),
#         "distortion_coefficients": distortion.tolist(),
#         "fx_px": fx,
#         "fy_px": fy,
#         "x_scale_mm_per_px": x_scale,
#         "y_scale_mm_per_px": y_scale,
#         "coordinate_source": "drawingFeatures[].annotations[]",
#         "dispatch_rule": "annotation.shape only",
#         "cvat_used": False,
#         "shape_counts": shape_counts,
#         "results": results,
#     }

#     with json_path.open("w", encoding="utf-8") as f:
#         json.dump(report, f, indent=4, ensure_ascii=False)

#     print("\n" + "=" * 110)
#     print("VALIDATION COMPLETE")
#     print("=" * 110)
#     print("Annotation counts by shape:", shape_counts)
#     print("Overlay :", overlay_path)
#     print("CSV     :", csv_path)
#     print("JSON    :", json_path)
#     print("=" * 110)


# def run_metrology_inspection(image_path: Path | str, product_height_mm: float) -> Dict[str, Any]:
#     TOTAL_CAMERA_TO_BASE_MM = 560.0

#     if product_height_mm >= TOTAL_CAMERA_TO_BASE_MM:
#         raise ValueError(
#             "Product height must be smaller than the fixed camera-to-base height.\n"
#             f"Camera-to-base : {TOTAL_CAMERA_TO_BASE_MM:.3f} mm\n"
#             f"Product height : {product_height_mm:.3f} mm"
#         )

#     distance_mm = TOTAL_CAMERA_TO_BASE_MM - product_height_mm

#     args = argparse.Namespace(
#         json=JSON_PATH,
#         image=Path(image_path),
#         calibration=CALIBRATION_PATH,
#         total_length_mm=TOTAL_CAMERA_TO_BASE_MM,
#         product_height_mm=product_height_mm,
#         distance_mm=distance_mm,
#         output_dir=OUTPUT_DIR,
#     )

#     for label, path in (
#         ("JSON", args.json),
#         ("image", args.image),
#         ("calibration", args.calibration),
#     ):
#         if not path.exists():
#             raise FileNotFoundError(f"{label} file not found:\n{path}")

#     drawing = load_drawing(args.json)
#     image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {args.image}")

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     overlay = image.copy()

#     camera_matrix, distortion, fx, fy = load_calibration(args.calibration)
#     x_scale, y_scale = mm_per_pixel(args.distance_mm, fx, fy)

#     output_dir = args.output_dir or (args.calibration.parent / "drawing_validation")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     overlay_path = output_dir / "drawing_validation_overlay.png"
#     csv_path = output_dir / "drawing_validation_report.csv"
#     json_path = output_dir / "drawing_validation_report.json"

#     results: List[Dict[str, Any]] = []

#     print("=" * 110)
#     print("BUHLER DYNAMIC ANNOTATION VALIDATION")
#     print("=" * 110)
#     print("Part number       :", drawing.get("part_number"))
#     print("Drawing JSON      :", args.json)
#     print("Image             :", args.image)
#     print("Calibration       :", args.calibration)
#     print("Total length (mm) :", f"{args.total_length_mm:.3f}")
#     print("Product height    :", f"{args.product_height_mm:.3f} mm")
#     print("Distance (mm)     :", f"{args.distance_mm:.3f}")
#     print("Formula           : total length - product height")
#     print("Image resolution  :", f"{image.shape[1]} x {image.shape[0]}")
#     print("fx / fy (px)      :", f"{fx:.6f} / {fy:.6f}")
#     print("X scale (mm/px)   :", f"{x_scale:.9f}")
#     print("Y scale (mm/px)   :", f"{y_scale:.9f}")
#     print("Coordinate source : drawingFeatures[].annotations[]")
#     print("Shape dispatch     : annotation.shape ONLY")
#     print("CVAT              : NOT USED")
#     print("=" * 110)

#     shape_counts: Dict[str, int] = {}

#     for feature, index, annotation in iter_annotations(drawing):
#         shape = shape_name(annotation)
#         shape_counts[shape] = shape_counts.get(shape, 0) + 1
#         label = f"{feature.get('id', 'FEATURE')}-{index}"

#         print(f"\n{label}")
#         print("  shape :", shape)
#         print("  tag   :", annotation.get("tag_name"), "/", annotation.get("tag_id"))

#         if shape == "circle":
#             b = bbox_edges(annotation)
#             print(
#                 "  JSON  :",
#                 f"left={b['left']:.6f}, top={b['top']:.6f}, "
#                 f"right={b['right']:.6f}, bottom={b['bottom']:.6f}, "
#                 f"width={b['width']:.6f}, height={b['height']:.6f}"
#             )
#             print(
#                 "  JSON circle:",
#                 f"center=({b['center_x']:.6f},{b['center_y']:.6f}), "
#                 f"radius_x={b['width']/2.0:.6f}px, "
#                 f"radius_y={b['height']/2.0:.6f}px, "
#                 f"radius={(b['width']/2.0 + b['height']/2.0)/2.0:.6f}px"
#             )

#             detected = detect_circle_from_annotation(gray, annotation)
#             draw_circle_annotation(overlay, annotation, detected, label)
#             row = circle_result(feature, annotation, index, detected, x_scale, y_scale)
#             results.append(row)

#             if detected is None:
#                 print("  IMAGE : circle fit FAILED")
#             else:
#                 print(
#                     "  IMAGE :",
#                     f"center=({detected['center_x']:.3f},{detected['center_y']:.3f})px, ",
#                     f"radius={detected['radius_px']:.3f}px, ",
#                     f"diameter={detected['diameter_px']:.3f}px, ",
#                     f"edge_support={detected['edge_support']:.3f}"
#                 )
#                 print(
#                     "  SIZE  :",
#                     f"diameter_x={row['detected_diameter_mm_x']:.3f}mm, ",
#                     f"diameter_y={row['detected_diameter_mm_y']:.3f}mm, ",
#                     f"mean={row['detected_diameter_mm']:.3f}mm"
#                 )

#         elif shape in {"rect", "rectangle", "square"}:
#             b = bbox_edges(annotation)
#             detected = detect_rectangle_from_annotation(gray, annotation)
#             draw_rect_annotation(overlay, annotation, detected, label)

#             row = rectangle_result(feature, annotation, index, detected, x_scale, y_scale)
#             results.append(row)
#             print(
#                 "  RECT  :",
#                 f"left={b['left']:.6f}, top={b['top']:.6f}, "
#                 f"right={b['right']:.6f}, bottom={b['bottom']:.6f}, "
#                 f"width={b['width']:.6f}px, height={b['height']:.6f}px"
#             )
#             if row["width_mm"] is not None and row["height_mm"] is not None:
#                 print(
#                     "  SIZE  :",
#                     f"width={row['width_mm']:.3f}mm, height={row['height_mm']:.3f}mm"
#                 )
#             else:
#                 print("  SIZE  : rectangle detection FAILED")

#         elif shape == "polygon":
#             points = points_from_annotation(annotation)
#             draw_polygon_annotation(overlay, annotation, label)
#             detected = detect_polygon_from_annotation(gray, annotation)
#             if detected is not None:
#                 draw_detected_polygon(overlay, detected, label)

#             row = polygon_result(feature, annotation, index, detected, x_scale, y_scale)
#             results.append(row)

#             print("  POLYGON: point_count=", len(points))
#             for i, (x, y) in enumerate(points, start=1):
#                 print(f"    JSON P{i}: ({x:.6f}, {y:.6f})")

#             if detected is None:
#                 print("  IMAGE : polygon detection FAILED")
#             else:
#                 for i, (x, y) in enumerate(detected["points"], start=1):
#                     print(f"    DETECTED P{i}: ({x:.6f}, {y:.6f})")
#                 print(
#                     "  SIZE  :",
#                     f"width={row['detected_width_mm']:.3f}mm, ",
#                     f"height={row['detected_height_mm']:.3f}mm, ",
#                     f"perimeter={row['perimeter_mm']:.3f}mm, ",
#                     f"area={row['area_mm2']:.3f}mm^2"
#                 )
#                 for i, edge_mm in enumerate(row["edge_lengths_mm"], start=1):
#                     print(f"    EDGE {i}: {edge_mm:.3f} mm")

#         else:
#             print("  WARNING: unsupported annotation.shape -> preserved, not discarded")
#             results.append(make_output_row_for_unsupported(feature, annotation, index))

#     cv2.imwrite(str(overlay_path), overlay)

#     csv_rows: List[Dict[str, Any]] = []
#     for row in results:
#         flat = dict(row)
#         if isinstance(flat.get("points_px"), list):
#             flat["points_px"] = json.dumps(flat["points_px"], ensure_ascii=False)
#         if isinstance(flat.get("edge_lengths_px"), list):
#             flat["edge_lengths_px"] = json.dumps(flat["edge_lengths_px"])
#         if isinstance(flat.get("edge_lengths_mm"), list):
#             flat["edge_lengths_mm"] = json.dumps(flat["edge_lengths_mm"])
#         csv_rows.append(flat)

#     csv_fields = sorted({key for row in csv_rows for key in row.keys()})
#     with csv_path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
#         writer.writeheader()
#         writer.writerows(csv_rows)

#    # Filter results to include ONLY successfully measured/detected items
#     valid_results = [
#         row for row in results 
#         if row.get("status") in {"MEASURED", "PASS", "FAIL"}
#     ]

# # 1. Generate Base64 binary string from the saved overlay image
#     overlay_base64 = None
#     if overlay_path.exists():
#         with open(overlay_path, "rb") as img_file:
#             overlay_base64 = base64.b64encode(img_file.read()).decode('utf-8')

#     # 2. Save full detailed report locally to disk (keeps everything for debugging)
#     full_report = {
#         "drawingId": drawing.get("drawingId"),
#         "part_number": drawing.get("part_number"),
#         "image": str(args.image),
#         "calibration": str(args.calibration),
#         "total_length_mm": args.total_length_mm,
#         "product_height_mm": args.product_height_mm,
#         "distance_mm": args.distance_mm,
#         "distance_calculation": "total_length_mm - product_height_mm",
#         "camera_matrix": camera_matrix.tolist(),
#         "distortion_coefficients": distortion.tolist(),
#         "fx_px": fx,
#         "fy_px": fy,
#         "x_scale_mm_per_px": x_scale,
#         "y_scale_mm_per_px": y_scale,
#         "coordinate_source": "drawingFeatures[].annotations[]",
#         "dispatch_rule": "annotation.shape only",
#         "cvat_used": False,
#         "shape_counts": shape_counts,
#         "results": results,
#     }

#     with json_path.open("w", encoding="utf-8") as f:
#         json.dump(full_report, f, indent=4, ensure_ascii=False)

#     # 3. Filter out failed items and strip 'annotation_' keys for the clean API response
#     valid_results = [
#         {k: v for k, v in row.items() if not k.startswith("annotation_")}
#         for row in results 
#         if row.get("status") in {"MEASURED", "PASS", "FAIL"}
#     ]

#     # 4. Clean API payload (contains ONLY detected metrics + base64 binary image string)
#     api_report = {
#         "drawingId": drawing.get("drawingId"),
#         "part_number": drawing.get("part_number"),
#         "product_height_mm": args.product_height_mm,
#         "distance_mm": args.distance_mm,
#         "shape_counts": shape_counts,
#         "results": valid_results,
#         "overlay_image_base64": overlay_base64,  # Actual base64 binary payload string
#     }

#     print("\n" + "=" * 110)
#     print("VALIDATION COMPLETE")
#     print("=" * 110)
#     print("Annotation counts by shape:", shape_counts)
#     print("Overlay :", overlay_path)
#     print("CSV     :", csv_path)
#     print("JSON    :", json_path)
#     print("=" * 110)

#     return api_report

#     # report = {
#     #     "drawingId": drawing.get("drawingId"),
#     #     "part_number": drawing.get("part_number"),
#     #     "image": str(args.image),
#     #     "calibration": str(args.calibration),
#     #     "total_length_mm": args.total_length_mm,
#     #     "product_height_mm": args.product_height_mm,
#     #     "distance_mm": args.distance_mm,
#     #     "distance_calculation": "total_length_mm - product_height_mm",
#     #     "camera_matrix": camera_matrix.tolist(),
#     #     "distortion_coefficients": distortion.tolist(),
#     #     "fx_px": fx,
#     #     "fy_px": fy,
#     #     "x_scale_mm_per_px": x_scale,
#     #     "y_scale_mm_per_px": y_scale,
#     #     "coordinate_source": "drawingFeatures[].annotations[]",
#     #     "dispatch_rule": "annotation.shape only",
#     #     "cvat_used": False,
#     #     "shape_counts": shape_counts,
#     #     "results": results,
#     # }

#     # with json_path.open("w", encoding="utf-8") as f:
#     #     json.dump(report, f, indent=4, ensure_ascii=False)

#     # print("\n" + "=" * 110)
#     # print("VALIDATION COMPLETE")
#     # print("=" * 110)
#     # print("Annotation counts by shape:", shape_counts)
#     # print("Overlay :", overlay_path)
#     # print("CSV     :", csv_path)
#     # print("JSON    :", json_path)
#     # print("=" * 110)

#     # return report