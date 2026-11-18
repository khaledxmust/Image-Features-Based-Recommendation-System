"""Compare two local files using the same implementation as the API."""

import argparse
import json
from pathlib import Path

from config import Settings
from src.v1.images import decode_image


def main():
    parser = argparse.ArgumentParser(description="Compare two local images with LoFTR and MAGSAC.")
    parser.add_argument("img1", type=Path)
    parser.add_argument("img2", type=Path)
    args = parser.parse_args()
    settings = Settings.from_environment()
    images = []
    for path in (args.img1, args.img2):
        with path.open("rb") as handle:
            images.append(decode_image(handle.read(settings.max_image_bytes + 1), settings))
    from src.v1.Model import SimilarityService

    print(json.dumps(SimilarityService(settings).compare(*images), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
