"""Praktikum 08: konvolusi, noise, reduksi noise, tepi, sharpness, sketsa."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ensure_dirs, load_image, gray, save


def edge_outputs(img, response):
    edge = cv2.convertScaleAbs(response)
    binary = cv2.threshold(edge, 60, 255, cv2.THRESH_BINARY)[1]
    return edge, binary, 255 - binary


def main(input_path=None):
    out = ensure_dirs(8); img = gray(load_image(input_path)); rng = np.random.default_rng(42)
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)
    saltpepper = img.copy(); mask = rng.random(img.shape)
    saltpepper[mask < .025] = 0; saltpepper[mask > .975] = 255
    results = {
        "original": img, "noise_gaussian": noisy, "noise_salt_pepper": saltpepper,
        "blur_5": cv2.blur(noisy, (5, 5)), "blur_11": cv2.blur(noisy, (11, 11)),
        "gaussian": cv2.GaussianBlur(noisy, (7, 7), 0), "median": cv2.medianBlur(saltpepper, 7),
    }
    kernels = {
        "kernel_4_edge": np.array([[0, -.5, 0], [-.5, 0, .5], [0, .5, 0]], np.float32),
        "kernel_4_sharp": np.array([[0, -.5, 0], [-.5, 1, .5], [0, .5, 0]], np.float32),
        "kernel_8_edge": np.array([[-1, -.5, 0], [-.5, 1, .5], [0, .5, 1]], np.float32),
        "kernel_8_sharp": np.array([[-1, -.5, 0], [-.5, 2, .5], [0, .5, 1]], np.float32),
    }
    for name, kernel in kernels.items(): results[name] = cv2.filter2D(img, cv2.CV_32F, kernel)
    prewitt_x = np.array([[-1, 0, 1]] * 3, np.float32); prewitt_y = prewitt_x.T
    responses = {
        "prewitt": cv2.magnitude(cv2.filter2D(img, cv2.CV_32F, prewitt_x), cv2.filter2D(img, cv2.CV_32F, prewitt_y)),
        "sobel": cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)),
        "laplacian": cv2.Laplacian(img, cv2.CV_32F, ksize=3), "canny": cv2.Canny(img, 100, 200),
    }
    for name, response in responses.items():
        edge, binary, sketch = edge_outputs(img, response)
        results[f"edge_{name}"] = edge; results[f"binary_{name}"] = binary; results[f"sketch_{name}"] = sketch
    lpf = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    high = img.astype(np.float32) - lpf
    results["sharp_ratio_2_1"] = np.clip((2 * lpf + high) / 2, 0, 255)
    results["sharp_ratio_1_2"] = np.clip((lpf + 4 * high) / 3, 0, 255)
    for name, value in results.items(): save(out / f"{name}.png", value)
    selected = ["original", "noise_gaussian", "gaussian", "median", "edge_prewitt", "edge_sobel",
                "edge_laplacian", "edge_canny", "sharp_ratio_2_1", "sharp_ratio_1_2", "sketch_sobel", "sketch_canny"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for ax, name in zip(axes.flat, selected):
        ax.imshow(np.clip(results[name], 0, 255), cmap="gray", vmin=0, vmax=255); ax.set_title(name); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "ringkasan.png", dpi=150); plt.close(fig)
    print("Prewitt berbobot seragam; Sobel memberi bobot 2 pada tetangga pusat sehingga lebih tahan noise.")
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(8, "Filtering, Noise, Tepi & Sharpness", main)
