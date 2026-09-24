"""Pertemuan 11: morfologi citra biner dan latihan thinning inisial 20x20."""
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ensure_dirs, load_image, gray, save


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Skeleton morfologis tanpa modul opencv-contrib."""
    skeleton = np.zeros_like(binary)
    work = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
    return skeleton


def create_initials(text: str = "AN") -> np.ndarray:
    """Membuat citra latihan dua inisial berukuran tepat 20x20 piksel."""
    canvas = np.zeros((20, 20), np.uint8)
    cv2.putText(canvas, text[:2].upper(), (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, 255, 1, cv2.LINE_8)
    return canvas


def main(input_path=None):
    out = ensure_dirs(11)
    image = load_image(input_path)
    grayscale = gray(image)
    # Otsu memilih threshold dari histogram agar cocok untuk dataset beragam.
    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernels = {
        "rect_3x3": cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        "cross_3x3": cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        "ellipse_5x5": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    }
    kernel = kernels["cross_3x3"]
    results = {
        "binary_otsu": binary,
        "dilasi": cv2.dilate(binary, kernel, iterations=1),
        "erosi": cv2.erode(binary, kernel, iterations=1),
        "opening": cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
        "morphological_gradient": cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel),
        "top_hat": cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel),
        "black_hat": cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel),
    }

    # Hit-or-Miss bekerja pada citra 0/1; 1 harus cocok objek dan -1 latar.
    hit_kernel = np.array([[-1, 1, -1], [0, 1, 0], [0, 1, 0]], np.int8)
    results["hit_or_miss"] = cv2.morphologyEx(binary // 255, cv2.MORPH_HITMISS, hit_kernel) * 255
    results["skeleton"] = skeletonize(binary)

    # Latihan modul: thinning pada dua inisial, masing-masing dalam citra 20x20.
    initials = create_initials("AN")
    results["inisial_AN_20x20"] = initials
    results["thinning_inisial_AN_20x20"] = skeletonize(initials)

    # Bandingkan bentuk structuring element pada operasi dilasi.
    for name, current_kernel in kernels.items():
        results[f"dilasi_{name}"] = cv2.dilate(binary, current_kernel)

    for name, value in results.items():
        save(out / f"{name}.png", value)

    selected = ["binary_otsu", "dilasi", "erosi", "opening", "closing", "hit_or_miss",
                "skeleton", "inisial_AN_20x20", "thinning_inisial_AN_20x20"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    titles = ["Biner Otsu", "Dilasi", "Erosi", "Opening", "Closing", "Hit-or-Miss",
              "Skeleton", "Inisial AN (20×20)", "Thinning AN"]
    for ax, name, title in zip(axes.flat, selected, titles):
        interpolation = "nearest" if "20x20" in name else "antialiased"
        ax.imshow(results[name], cmap="gray", vmin=0, vmax=255, interpolation=interpolation)
        ax.set_title(title); ax.axis("off")
    fig.suptitle("Pertemuan 11 — Morfologi Citra", fontsize=16)
    fig.tight_layout(); fig.savefig(out / "ringkasan.png", dpi=150); plt.close(fig)

    print("Dilasi memperluas objek; erosi menipiskan objek; opening menghapus detail kecil;")
    print("closing menutup celah kecil; skeleton mempertahankan topologi dalam garis tipis.")
    print(f"Hasil tersimpan di {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path gambar masukan")
    parser.add_argument("--batch", action="store_true", help="Jalankan tanpa GUI")
    args = parser.parse_args()
    if args.batch:
        main(args.input)
    else:
        from vision_gui import launch
        launch(11, "Morfologi Citra", main)
