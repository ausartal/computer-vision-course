"""Utilitas bersama praktikum Computer Vision 02-08."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"


def ensure_dirs(number: int) -> Path:
    DATA.mkdir(exist_ok=True)
    out = OUTPUT / f"praktikum_{number:02d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_sample(path: Path | None = None, size: tuple[int, int] = (640, 420)) -> Path:
    """Membuat citra RGB sintetis lengkap: gradien, bentuk, teks, dan detail."""
    DATA.mkdir(exist_ok=True)
    path = path or DATA / "citra_contoh.png"
    if path.exists():
        return path
    w, h = size
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    b = np.broadcast_to(45 + 150 * y, (h, w))
    g = np.broadcast_to(35 + 190 * x, (h, w))
    r = 35 + 170 * (1 - x[None, :]) * (1 - y)
    image = np.dstack((b, g, r)).clip(0, 255).astype(np.uint8)
    cv2.circle(image, (w // 4, h // 2), 75, (30, 210, 245), -1)
    cv2.rectangle(image, (w // 2, 70), (w - 55, h - 75), (210, 75, 55), -1)
    cv2.line(image, (25, h - 35), (w - 25, 30), (250, 250, 250), 5)
    cv2.putText(image, "COMPUTER VISION", (35, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return path


def load_image(input_path: str | None = None, grayscale: bool = False) -> np.ndarray:
    # Utamakan dataset asli yang diberikan pengguna, bukan citra sintetis.
    candidates = [p for p in sorted(DATA.iterdir())
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                  and p.name != "citra_contoh.png"] if DATA.exists() else []
    path = Path(input_path) if input_path else (candidates[0] if candidates else generate_sample())
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Citra tidak dapat dibaca: {path}")
    return image


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8)):
        raise OSError(f"Gagal menyimpan: {path}")


def gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def autolevel(channel: np.ndarray) -> np.ndarray:
    """Rentangkan nilai minimum-maksimum ke 0-255, aman untuk citra konstan."""
    values = channel.astype(np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        return np.zeros_like(channel)
    return np.round((values - lo) * 255.0 / (hi - lo)).astype(np.uint8)
