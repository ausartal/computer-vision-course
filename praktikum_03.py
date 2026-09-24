"""Praktikum 03: kanal BGR, grayscale, sepia, threshold global/adaptif."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ensure_dirs, load_image, save


def main(input_path=None):
    out = ensure_dirs(3); img = load_image(input_path)
    b, g, r = cv2.split(img); zero = np.zeros_like(b)
    layers = {"layer_b": cv2.merge((b, zero, zero)), "layer_g": cv2.merge((zero, g, zero)),
              "layer_r": cv2.merge((zero, zero, r))}
    gray_avg = np.mean(img.astype(np.float32), axis=2).astype(np.uint8)
    gray_min = np.min(img, axis=2).astype(np.uint8)
    gray_max = np.max(img, axis=2).astype(np.uint8)
    # Rumus tugas: B=r, G=1.8r, R=2r; clip mencegah overflow uint8.
    rf = r.astype(np.float32)
    sepia = cv2.merge((rf, 1.8 * rf, 2.0 * rf)).clip(0, 255).astype(np.uint8)
    gradient = np.tile(np.arange(256, dtype=np.uint8), (150, 1))
    threshold_modes = {
        "binary": cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY)[1],
        "binary_inv": cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY_INV)[1],
        "trunc": cv2.threshold(gradient, 127, 255, cv2.THRESH_TRUNC)[1],
        "tozero": cv2.threshold(gradient, 127, 255, cv2.THRESH_TOZERO)[1],
        "tozero_inv": cv2.threshold(gradient, 127, 255, cv2.THRESH_TOZERO_INV)[1],
    }
    adaptive = cv2.adaptiveThreshold(gray_avg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 5)
    all_images = {**layers, "gray_average": gray_avg, "gray_min": gray_min,
                  "gray_max": gray_max, "sepia": sepia, "adaptive_threshold": adaptive,
                  **{f"threshold_{k}": v for k, v in threshold_modes.items()}}
    for name, value in all_images.items(): save(out / f"{name}.png", value)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    shown = [img, *layers.values(), gray_avg, sepia, threshold_modes["binary"], adaptive]
    titles = ["Asli", "Layer B", "Layer G", "Layer R", "Gray rata-rata", "Sepia", "Biner", "Adaptif"]
    for ax, value, title in zip(axes.flat, shown, titles):
        ax.imshow(value if value.ndim == 2 else cv2.cvtColor(value, cv2.COLOR_BGR2RGB), cmap="gray")
        ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "ringkasan.png", dpi=150); plt.close(fig)
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(3, "Format Citra, Layer RGB & Grayscale", main)
