"""Decode uploaded images and optionally fetch allowlisted HTTPS images."""

from io import BytesIO
import ipaddress
import socket
from urllib.parse import urlsplit
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


def decode_image(data, settings):
    if not data or len(data) > settings.max_image_bytes:
        raise ValueError("Each image must be nonempty and within MAX_IMAGE_BYTES.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Supported image formats are JPEG, PNG, and WebP.")
                if image.width * image.height > settings.max_image_pixels:
                    raise ValueError("Image exceeds MAX_IMAGE_PIXELS.")
                if min(image.size) < 16:
                    raise ValueError("Images must be at least 16 pixels on each side.")
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("The supplied image cannot be decoded within the configured limits.") from error


def fetch_image(url, settings):
    """The original URL API is opt-in through exact hostname configuration."""
    import requests

    parsed = urlsplit(url)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid image URL port.") from error
    if (
        parsed.scheme != "https"
        or not host
        or host.lower() not in settings.allowed_url_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise ValueError("Use an HTTPS image URL from a hostname configured in IMAGE_URL_HOSTS.")
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError("Image host could not be resolved.") from error
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Image hosts must resolve to public IP addresses.")

    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(
                url, timeout=settings.download_timeout, stream=True, allow_redirects=False
            ) as response:
                if response.status_code != 200:
                    raise ValueError("Image URL must return HTTP 200 without redirects.")
                chunks = []
                length = 0
                for chunk in response.iter_content(chunk_size=65536):
                    length += len(chunk)
                    if length > settings.max_image_bytes:
                        raise ValueError("Downloaded image exceeds MAX_IMAGE_BYTES.")
                    chunks.append(chunk)
                return decode_image(b"".join(chunks), settings)
    except requests.RequestException as error:
        raise ValueError("Image download failed or timed out.") from error
