# Image Features-Based Recommendation System

Compare product images through local visual correspondences. A Flask API uses Kornia's pretrained LoFTR matcher to locate corresponding image points, then applies OpenCV USAC_MAGSAC to filter them through an estimated fundamental matrix. The resulting match counts and confidences can support a downstream catalogue recommendation workflow.

The API supplies the pairwise matching component for a catalogue workflow. Product selection, ranking rules, and relevance evaluation belong to the downstream application. LoFTR uses pretrained upstream weights downloaded separately.

## Pipeline

1. Decode two JPEG, PNG, or WebP images, correct EXIF orientation, and convert to RGB.
2. Preserve the aspect ratio while bounding the longest side to 640 pixels and aligning dimensions to multiples of eight.
3. Convert to grayscale and find LoFTR correspondences.
4. Estimate a fundamental matrix with MAGSAC when there are at least eight correspondences.
5. Return tentative points, geometric inliers, fractional mean confidences, and an inlier ratio.

```text
Two images → RGB and grayscale preprocessing → LoFTR → MAGSAC → Match statistics
```

Geometric consistency is helpful when two images contain the same or closely related visual structure. Product photographs can contain planar scenes, low texture, repeated patterns, different viewpoints, or different garment shapes. These cases can make fundamental-matrix estimation fail or yield misleading matches. Calibrate ranking rules using labeled product pairs before using these statistics for recommendations.

## Setup

Use Python 3.11 in a fresh environment. Dependency ranges are provided. A full model run and a resolved environment lock remain to be validated.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS or Linux
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a particular CPU or CUDA build, install PyTorch using the command provided by the [official PyTorch installer](https://pytorch.org/get-started/locally/) before installing the remaining requirements. CPU inference is the default. CUDA inference requires a compatible machine and PyTorch installation.

Optionally copy `.env.example` to `.env`. For a trusted local checkpoint, set `LOFTR_WEIGHTS` to the path of an upstream LoFTR state dictionary or checkpoint with a `state_dict` field. Otherwise Kornia downloads its selected `outdoor` or `indoor` checkpoint on the first comparison and caches it through PyTorch. Startup and `/health` do not load model weights. The first comparison therefore needs network access or a locally available checkpoint and can take longer.

## Local demo

Generate two original synthetic images, then compare them through the same model path used by the API.

```bash
python demo/generate_images.py
python compare.py demo/generated/image1.png demo/generated/image2.png
```

The synthetic scene provides a pipeline smoke input. Evaluate recommendation relevance with labeled product pairs. You can also run `python compare.py first.jpg second.jpg` with your own permitted images. Model weights and generated images are excluded from version control.

## API

Run from the repository root.

```bash
python main.py
```

The local development server listens at `http://127.0.0.1:8002`. Upload two files with multipart field names `img1` and `img2`.

```bash
curl -X POST http://127.0.0.1:8002/similarity/v1/find \
  -F "img1=@demo/generated/image1.png" \
  -F "img2=@demo/generated/image2.png"
```

In Windows PowerShell use `curl.exe` and put the request on one line, or use PowerShell backticks for continuation. `/health` reports API availability and whether a matcher has been loaded. It does not perform an inference readiness check.

An illustrative response is shown below. Values here are explanatory.

```json
{
  "Points": 120,
  "Matches": 85,
  "P.Confidence": 0.72,
  "M.Confidence": 0.81,
  "InlierRatio": 0.708333,
  "GeometryEstimated": true
}
```

| Field | Meaning |
| --- | --- |
| `Points` | Tentative point correspondences reported by LoFTR |
| `Matches` | Correspondences retained by geometric verification |
| `P.Confidence` | Mean confidence across tentative correspondences, or `null` if none exist |
| `M.Confidence` | Mean confidence across inlier correspondences, or `null` if none exist |
| `InlierRatio` | Inlier count divided by tentative count, or zero when there are no correspondences |
| `GeometryEstimated` | Whether a finite fundamental matrix and matching inlier mask were estimated |

Confidence describes point correspondences. Product equivalence and recommendation relevance require separate evaluation. Counts depend on resizing and image content. Fewer than eight correspondences, degenerate geometry, and failed estimation return zero inliers and a `null` inlier confidence.

The original `GET /similarity/v1/find?imgs={...}` URL format is supported when exact HTTPS hostnames are configured in `IMAGE_URL_HOSTS`. Both JSON and the original Python-literal object syntax are accepted for the `imgs` query. A JSON POST with `img1` and `img2` URL fields is also supported in that mode. Hosts must resolve to public addresses, images have size and timeout limits, and HTTP redirects are rejected. Uploads work without URL configuration. Keep the development server local. An internet-facing service needs its own authentication, concurrency, and resource controls.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py config.py compare.py src demo tests
```

Tests cover fractional confidence, empty and invalid match statistics, image decoding and limits, input formats, upload responses, and lazy startup using a deterministic matcher stub. Full neural inference requires the declared dependencies and a compatible checkpoint. Assess model quality on a labeled collection with clearly separated evaluation pairs.

## Provenance and licensing

Adapted from the local `brantu-similarity` pairwise LoFTR and MAGSAC API. The portable service retains its response names and algorithm, with corrected fractional scores, empty-match handling, image decoding, aspect-ratio handling, and lazy model loading. Company-specific BigQuery catalogue synchronization remains outside this demo.

Confirm publication rights for adapted employer-related source before public distribution. A project license requires the owner's decision. Third-party code and weights retain their own terms.

## References

- [LoFTR paper and original implementation](https://zju3dv.github.io/loftr/)
- [Kornia LoFTR implementation](https://github.com/kornia/kornia/tree/main/kornia/feature/loftr)
- [OpenCV fundamental-matrix estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

LoFTR and Kornia provide the pretrained model. This project integrates them into a feature-matching service.
