"""Praktikum 05: brightness, contrast, invers, gamma, log, inverse-log, root."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ensure_dirs, load_image, gray, save


def gamma_transform(img, gamma):
    return np.round(255 * (img.astype(np.float32) / 255) ** gamma).astype(np.uint8)


def main(input_path=None):
    out = ensure_dirs(5); img = gray(load_image(input_path)); f = img.astype(np.float32)
    results = {
        "original": img,
        "brightness_minus_30": np.clip(f - 30, 0, 255),
        "brightness_plus_30": np.clip(f + 30, 0, 255),
        "brightness_plus_255": np.clip(f + 255, 0, 255),
        "contrast_050": np.clip(f * .5, 0, 255), "contrast_075": np.clip(f * .75, 0, 255),
        "contrast_125": np.clip(f * 1.25, 0, 255), "contrast_150": np.clip(f * 1.5, 0, 255),
        "inverse_255": 255 - img, "inverse_128": np.clip(128 - f, 0, 255),
        "gamma_05": gamma_transform(img, .5), "gamma_20": gamma_transform(img, 2.0),
        "root": gamma_transform(img, .5),
    }
    results["log"] = np.round(255 * np.log1p(f) / np.log(256)).astype(np.uint8)
    results["inverse_log"] = np.round(np.expm1(f / 255 * np.log(256))).clip(0, 255).astype(np.uint8)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    results["fourier_log"] = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    for name, value in results.items(): save(out / f"{name}.png", value)
    names = ["original", "brightness_plus_30", "contrast_150", "inverse_255",
             "gamma_05", "gamma_20", "log", "inverse_log", "fourier_log"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, name in zip(axes.flat, names):
        ax.imshow(results[name], cmap="gray", vmin=0, vmax=255); ax.set_title(name); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "ringkasan.png", dpi=150); plt.close(fig)
    print("Brightness menggeser intensitas; contrast merenggangkan/merapatkan rentangnya.")
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(5, "Peningkatan Kualitas Citra", main)
