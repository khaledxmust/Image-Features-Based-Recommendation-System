"""Pairwise upload API and compatibility URL endpoint."""

from threading import Lock

from flask import Blueprint, current_app, jsonify, request

from src.v1.Functions import parse_image_urls
from src.v1.images import decode_image, fetch_image

similarity_v1 = Blueprint("similarity_v1", __name__)
_service_lock = Lock()


def get_service():
    with _service_lock:
        if current_app.config["SIMILARITY_SERVICE"] is None:
            from src.v1.Model import SimilarityService

            current_app.config["SIMILARITY_SERVICE"] = SimilarityService(
                current_app.config["SIMILARITY_SETTINGS"]
            )
        return current_app.config["SIMILARITY_SERVICE"]


@similarity_v1.route("/find", methods=["GET", "POST"])
def find():
    settings = current_app.config["SIMILARITY_SETTINGS"]
    try:
        if request.method == "POST" and request.mimetype == "multipart/form-data":
            images = []
            for key in ("img1", "img2"):
                uploaded = request.files.get(key)
                if uploaded is None:
                    raise ValueError(f"Missing uploaded image {key}.")
                images.append(decode_image(uploaded.read(settings.max_image_bytes + 1), settings))
        else:
            if not settings.allowed_url_hosts:
                raise ValueError("Upload images with POST. URL mode requires IMAGE_URL_HOSTS configuration.")
            if request.method == "GET":
                payload = request.args.get("imgs")
            elif request.is_json:
                payload = request.get_json(silent=True)
            else:
                raise ValueError("Use multipart image uploads or a JSON object with image URLs.")
            urls = parse_image_urls(payload)
            images = [fetch_image(url, settings) for url in urls]
    except ValueError as error:
        return jsonify(error=str(error)), 400

    try:
        return jsonify(get_service().compare(*images))
    except (RuntimeError, ImportError, OSError, ValueError):
        current_app.logger.exception("Image matching failed")
        return jsonify(
            error="Model inference is unavailable. Check dependencies, checkpoint access, and device configuration."
        ), 503
