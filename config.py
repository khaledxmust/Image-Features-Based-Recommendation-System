"""Environment-backed settings with no service-account or cloud dependency."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    device: str = "cpu"
    pretrained: str = "outdoor"
    weights_path: str = ""
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 24_000_000
    max_image_side: int = 640
    allowed_url_hosts: tuple[str, ...] = ()
    download_timeout: float = 10.0

    def __post_init__(self):
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("LOFTR_DEVICE must be cpu, cuda, or auto.")
        if self.pretrained not in {"outdoor", "indoor"}:
            raise ValueError("LOFTR_PRETRAINED must be outdoor or indoor.")
        if self.max_image_bytes < 1 or self.max_image_pixels < 1:
            raise ValueError("Image byte and pixel limits must be positive.")
        if not 64 <= self.max_image_side <= 2048:
            raise ValueError("MAX_IMAGE_SIDE must be between 64 and 2048.")
        if not 0 < self.download_timeout <= 120:
            raise ValueError("IMAGE_DOWNLOAD_TIMEOUT must be between 0 and 120 seconds.")

    @classmethod
    def from_environment(cls):
        # An optional .env is local configuration, never required by imports.
        try:
            from dotenv import load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv(Path(__file__).parent / ".env")

        hosts = tuple(
            host.strip().lower()
            for host in os.getenv("IMAGE_URL_HOSTS", "").split(",")
            if host.strip()
        )
        return cls(
            device=os.getenv("LOFTR_DEVICE", "cpu"),
            pretrained=os.getenv("LOFTR_PRETRAINED", "outdoor"),
            weights_path=os.getenv("LOFTR_WEIGHTS", ""),
            max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
            max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "24000000")),
            max_image_side=int(os.getenv("MAX_IMAGE_SIDE", "640")),
            allowed_url_hosts=hosts,
            download_timeout=float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "10")),
        )
