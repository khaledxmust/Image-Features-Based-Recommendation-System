"""Input validation and JSON-safe match statistics."""

import ast
import json
import math
from statistics import fmean


def parse_image_urls(value):
    """Accept the original imgs query format and JSON request objects."""
    if isinstance(value, str):
        if len(value) > 8192:
            raise ValueError("The imgs value is too long.")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError, TypeError, RecursionError) as error:
                raise ValueError("imgs must contain an img1 and img2 object.") from error
    if not isinstance(value, dict):
        raise ValueError("Provide an object with img1 and img2 URLs.")
    urls = []
    for key in ("img1", "img2"):
        url = value.get(key)
        if not isinstance(url, str) or not url.strip() or len(url) > 4096:
            raise ValueError(f"{key} must be a nonempty image URL of at most 4096 characters.")
        urls.append(url.strip())
    return tuple(urls)


def summarize_matches(confidences, inliers):
    """Keep fractional means and use null when an average has no observations."""
    values = [float(value) for value in confidences]
    mask = [bool(value) for value in inliers]
    if len(values) != len(mask):
        raise ValueError("Each tentative correspondence must have one inlier flag.")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Matcher confidence must be finite and between 0 and 1.")
    selected = [value for value, is_inlier in zip(values, mask) if is_inlier]
    return {
        "Points": len(values),
        "Matches": len(selected),
        "P.Confidence": round(fmean(values), 6) if values else None,
        "M.Confidence": round(fmean(selected), 6) if selected else None,
        "InlierRatio": round(len(selected) / len(values), 6) if values else 0.0,
    }
