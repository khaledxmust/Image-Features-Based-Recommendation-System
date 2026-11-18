"""Regression checks that do not download model weights."""

from io import BytesIO
import math
import unittest

from PIL import Image

from config import Settings
from src.v1.Functions import parse_image_urls, summarize_matches
from src.v1.images import decode_image


class MatchStatisticsTests(unittest.TestCase):
    def test_fractional_confidence_is_preserved(self):
        result = summarize_matches([0.9, 0.7, 0.2], [True, True, False])
        self.assertEqual(result["Points"], 3)
        self.assertEqual(result["Matches"], 2)
        self.assertEqual(result["M.Confidence"], 0.8)
        self.assertEqual(result["P.Confidence"], 0.6)

    def test_empty_and_no_inlier_results_are_json_safe(self):
        empty = summarize_matches([], [])
        self.assertIsNone(empty["P.Confidence"])
        self.assertIsNone(empty["M.Confidence"])
        self.assertEqual(empty["InlierRatio"], 0.0)
        self.assertIsNone(summarize_matches([0.5], [False])["M.Confidence"])

    def test_invalid_matcher_output_is_rejected(self):
        for confidence, mask in [([0.5], []), ([math.nan], [True]), ([1.5], [True])]:
            with self.assertRaises(ValueError):
                summarize_matches(confidence, mask)


class InputTests(unittest.TestCase):
    def test_json_and_original_query_format_are_supported(self):
        expected = ("https://example.org/a.jpg", "https://example.org/b.jpg")
        for data in [
            '{"img1":"https://example.org/a.jpg","img2":"https://example.org/b.jpg"}',
            "{'img1':'https://example.org/a.jpg','img2':'https://example.org/b.jpg'}",
        ]:
            self.assertEqual(parse_image_urls(data), expected)

    def test_missing_and_malformed_urls_are_rejected(self):
        for value in [None, [], "{", {"img1": "x"}, {"img1": 12, "img2": "y"}]:
            with self.assertRaises(ValueError):
                parse_image_urls(value)

    def test_grayscale_and_rgba_decode_to_rgb(self):
        for mode in ("L", "RGBA"):
            stream = BytesIO()
            Image.new(mode, (32, 48)).save(stream, "PNG")
            result = decode_image(stream.getvalue(), Settings())
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.size, (32, 48))

    def test_invalid_and_oversized_images_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_image(b"not an image", Settings())
        stream = BytesIO()
        Image.new("RGB", (32, 32)).save(stream, "PNG")
        with self.assertRaises(ValueError):
            decode_image(stream.getvalue(), Settings(max_image_pixels=100))


if __name__ == "__main__":
    unittest.main()
