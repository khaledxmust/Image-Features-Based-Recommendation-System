"""Flask entry point for pairwise visual matching."""

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from config import Settings
from src.v1.routes import similarity_v1


def create_app(settings=None, service=None):
    """Create the API without loading weights or opening cloud connections."""
    app = Flask(__name__)
    app.config["SIMILARITY_SETTINGS"] = settings or Settings.from_environment()
    app.config["MAX_CONTENT_LENGTH"] = 2 * app.config["SIMILARITY_SETTINGS"].max_image_bytes + 65536
    app.config["SIMILARITY_SERVICE"] = service
    app.register_blueprint(similarity_v1, url_prefix="/similarity/v1")

    @app.get("/")
    def index():
        return jsonify(
            name="Image Features-Based Recommendation System",
            compare="POST /similarity/v1/find with img1 and img2 image files",
            health="GET /health",
        )

    @app.get("/health")
    def health():
        matcher = app.config["SIMILARITY_SERVICE"]
        return jsonify(status="ok", model_loaded=bool(matcher and matcher.loaded))

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):
        return jsonify(error="Uploaded request exceeds the configured size limit."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8002, debug=False)
