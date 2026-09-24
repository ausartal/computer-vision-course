"""Praktikum 06: histogram brightness/contrast/invers dan autolevel gray/BGR."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import autolevel, ensure_dirs, load_image, gray, save


def main(input_path=None):
    out = ensure_dirs(6); color = load_image(input_path); img = gray(color); f = img.astype(np.float32)
    variants = {
        "brightness_-30": np.clip(f - 30, 0, 255).astype(np.uint8), "original": img,
        "brightness_+30": np.clip(f + 30, 0, 255).astype(np.uint8),
        "contrast_075": np.clip(f * .75, 0, 255).astype(np.uint8),
        "contrast_125": np.clip(f * 1.25, 0, 255).astype(np.uint8), "inverse": 255 - img,
    }
    gray_auto = autolevel(img)
    b, g, r = cv2.split(color)
    color_auto = cv2.merge(tuple(autolevel(c) for c in (b, g, r)))
    save(out / "autolevel_gray.png", gray_auto); save(out / "autolevel_bgr.png", color_auto)
    fig, axes = plt.subplots(len(variants), 2, figsize=(11, 18))
    for row, (name, value) in enumerate(variants.items()):
        axes[row, 0].imshow(value, cmap="gray", vmin=0, vmax=255); axes[row, 0].set_title(name); axes[row, 0].axis("off")
        axes[row, 1].hist(value.ravel(), bins=256, range=(0, 256)); axes[row, 1].set_xlim(0, 255); axes[row, 1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "histogram_transformasi.png", dpi=150); plt.close(fig)
    for name, value in variants.items(): save(out / f"{name}.png", value)
    print(f"Gray sebelum: min={img.min()}, max={img.max()}; sesudah: min={gray_auto.min()}, max={gray_auto.max()}")
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(6, "Histogram Citra & Auto Level", main)
