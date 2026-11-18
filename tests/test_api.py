"""API contracts using a deterministic stub instead of neural inference."""

from io import BytesIO
import unittest

from PIL import Image

from config import Settings
from main import create_app


class StubMatcher:
    loaded = True

    def compare(self, image1, image2):
        assert image1.mode == image2.mode == "RGB"
        return {"Points": 10, "Matches": 8, "P.Confidence": 0.71, "M.Confidence": 0.83}


def image_file():
    stream = BytesIO()
    Image.new("RGB", (32, 32), "navy").save(stream, "PNG")
    stream.seek(0)
    return stream


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(Settings(), StubMatcher()).test_client()

    def test_upload_preserves_fractional_scores(self):
        response = self.client.post(
            "/similarity/v1/find",
            data={"img1": (image_file(), "first.png"), "img2": (image_file(), "second.png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["M.Confidence"], 0.83)

    def test_missing_image_has_clear_error(self):
        response = self.client.post(
            "/similarity/v1/find", data={"img1": (image_file(), "first.png")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("img2", response.json["error"])

    def test_urls_are_disabled_by_default(self):
        response = self.client.get("/similarity/v1/find", query_string={"imgs": "{}"})
        self.assertEqual(response.status_code, 400)

    def test_health_does_not_load_the_model(self):
        response = create_app(Settings()).test_client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["model_loaded"])

    def test_upload_limit_is_enforced(self):
        client = create_app(Settings(max_image_bytes=16), StubMatcher()).test_client()
        response = client.post(
            "/similarity/v1/find",
            data={"img1": (image_file(), "first.png"), "img2": (image_file(), "second.png")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
