"""Praktikum 02: akses piksel, copy, flip, dan rotasi citra."""
import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ensure_dirs, load_image, save


def main(input_path=None):
    out = ensure_dirs(2)
    img = load_image(input_path)
    h, w = img.shape[:2]
    print(f"Ukuran (tinggi, lebar, kanal): {img.shape}; dtype: {img.dtype}")
    y, x = min(136, h - 1), min(413, w - 1)
    print(f"Piksel BGR [{y}, {x}]: {img[y, x].tolist()}")

    copy_img = img.copy()
    copy_img[y:min(y + 10, h), x:min(x + 10, w)] = (0, 255, 0)
    horizontal = cv2.flip(img, 1)      # kiri-kanan
    vertical = cv2.flip(img, 0)        # atas-bawah
    both = cv2.flip(img, -1)           # keduanya
    cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
    rot45 = cv2.warpAffine(img, matrix, (w, h))
    for name, value in {"copy_set_pixel": copy_img, "flip_horizontal": horizontal,
                        "flip_vertical": vertical, "flip_both": both,
                        "rotate_cw": cw, "rotate_ccw": ccw, "rotate_45": rot45}.items():
        save(out / f"{name}.png", value)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, value, title in zip(axes.flat, [copy_img, horizontal, vertical, both],
                                ["Copy + set pixel", "Flip horizontal", "Flip vertical", "Flip H+V"]):
        ax.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "ringkasan.png", dpi=150); plt.close(fig)
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--batch", action="store_true"); args=p.parse_args()
    if args.batch: main(args.input)
    else:
        from vision_gui import launch
        launch(2, "Akses Piksel, Copy, Flip & Rotasi", main)
