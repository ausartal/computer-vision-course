"""Praktikum 07: histogram equalization, CDF, dan perbandingan autolevel."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import autolevel, ensure_dirs, load_image, gray, save


def histogram_cdf(img):
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = hist.cumsum()
    return hist, cdf


def main(input_path=None):
    out = ensure_dirs(7); img = gray(load_image(input_path))
    equalized = cv2.equalizeHist(img); auto = autolevel(img)
    save(out / "equalized.png", equalized); save(out / "autolevel.png", auto)
    hist1, cdf1 = histogram_cdf(img); hist2, cdf2 = histogram_cdf(equalized)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, (value, hist, cdf, title) in enumerate(((img, hist1, cdf1, "Asli"),
                                                     (equalized, hist2, cdf2, "Equalisasi"))):
        axes[row, 0].imshow(value, cmap="gray", vmin=0, vmax=255); axes[row, 0].set_title(title); axes[row, 0].axis("off")
        axes[row, 1].bar(np.arange(256), hist, width=1); axes[row, 1].set_title(f"Histogram {title}")
        axes[row, 2].plot(cdf / cdf[-1]); axes[row, 2].set_title(f"CDF normalisasi {title}"); axes[row, 2].set_xlim(0, 255)
    fig.tight_layout(); fig.savefig(out / "histogram_equalization_cdf.png", dpi=150); plt.close(fig)
    print("Autolevel hanya merentangkan min-maks; equalization memetakan intensitas memakai CDF.")
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(7, "Histogram Equalization", main)
