"""Praktikum 04: threshold dan kuantisasi gray/warna level 1-8 bit."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ensure_dirs, load_image, gray, save


def quantize(image, bits):
    """Kuantisasi seragam ke 2**bits tingkat; bits 1..8."""
    step = 256 // (2 ** bits)
    return ((image.astype(np.uint16) // step) * step).clip(0, 255).astype(np.uint8)


def main(input_path=None):
    out = ensure_dirs(4); color = load_image(input_path); grayscale = gray(color)
    for threshold in (100, 200, int(grayscale.mean())):
        binary = np.where(grayscale >= threshold, 255, 0).astype(np.uint8)
        save(out / f"threshold_{threshold}.png", binary)
    gray_results, color_results = [], []
    for bits in range(1, 9):
        qg, qc = quantize(grayscale, bits), quantize(color, bits)
        gray_results.append(qg); color_results.append(qc)
        save(out / f"gray_level_{bits}.png", qg)
        save(out / f"warna_level_{bits}.png", qc)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, bits, value in zip(axes.flat, range(8, 0, -1), reversed(gray_results)):
        ax.imshow(value, cmap="gray", vmin=0, vmax=255); ax.set_title(f"Gray level {bits}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "kuantisasi_gray_8_sampai_1.png", dpi=150); plt.close(fig)
    print("Semakin kecil bit, semakin sedikit tingkat intensitas dan makin kuat efek posterisasi.")
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(4, "Kuantisasi Citra", main)
