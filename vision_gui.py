"""GUI modern bersama untuk Praktikum Computer Vision 02-08."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

from common import DATA, OUTPUT, autolevel, save

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OPERATIONS = {
    2: ["Copy + tanda piksel", "Flip horizontal", "Flip vertikal", "Flip horizontal + vertikal",
        "Rotasi 90° CW", "Rotasi 90° CCW", "Rotasi bebas"],
    3: ["Layer Blue", "Layer Green", "Layer Red", "Grayscale rata-rata", "Grayscale minimum",
        "Grayscale maksimum", "Sepia", "Threshold global", "Threshold adaptif"],
    4: ["Threshold", "Kuantisasi grayscale", "Kuantisasi warna"],
    5: ["Brightness", "Contrast", "Invers", "Gamma / Power", "Logaritmik", "Inverse log", "Root", "Spektrum Fourier"],
    6: ["Brightness", "Contrast", "Invers", "Autolevel grayscale", "Autolevel BGR", "Histogram grayscale"],
    7: ["Histogram Equalization", "Autolevel", "CLAHE", "Histogram + CDF"],
    8: ["Gaussian noise", "Salt & pepper noise", "Average blur", "Gaussian blur", "Median filter",
        "Prewitt", "Sobel", "Laplacian", "Canny", "Sharpness", "Sketsa"],
    11: ["Citra biner", "Dilasi", "Erosi", "Opening", "Closing", "Morphological gradient",
         "Top hat", "Black hat", "Hit-or-Miss", "Thinning / Skeleton"],
}

DEFAULTS = {
    2: "Panorama_Image.jpg", 3: "Fruit_Image.png", 4: "Face-Cream_Image.jpg",
    5: "Cat_Image.jpg", 6: "Architecture_Image.jpg", 7: "Face-Black_Image.jpg", 8: "Dog_Image.jpg",
    11: "Block_Image.jpg",
}


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def transform(number: int, operation: str, image: np.ndarray, value: float) -> np.ndarray:
    """Transformasi preview; value berasal dari slider 0..100."""
    img, g = image, _gray(image)
    if number == 2:
        if operation == "Copy + tanda piksel":
            result = img.copy(); h, w = result.shape[:2]; cv2.rectangle(result, (w//2-12, h//2-12), (w//2+12, h//2+12), (0,255,0), -1); return result
        if operation == "Flip horizontal": return cv2.flip(img, 1)
        if operation == "Flip vertikal": return cv2.flip(img, 0)
        if operation == "Flip horizontal + vertikal": return cv2.flip(img, -1)
        if operation == "Rotasi 90° CW": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if operation == "Rotasi 90° CCW": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img.shape[:2]; m = cv2.getRotationMatrix2D((w/2,h/2), value*3.6-180, 1); return cv2.warpAffine(img,m,(w,h))
    if number == 3:
        b,gc,r = cv2.split(img); z=np.zeros_like(b)
        if operation == "Layer Blue": return cv2.merge((b,z,z))
        if operation == "Layer Green": return cv2.merge((z,gc,z))
        if operation == "Layer Red": return cv2.merge((z,z,r))
        if operation == "Grayscale rata-rata": return np.mean(img.astype(float),axis=2).astype(np.uint8)
        if operation == "Grayscale minimum": return np.min(img,axis=2).astype(np.uint8)
        if operation == "Grayscale maksimum": return np.max(img,axis=2).astype(np.uint8)
        if operation == "Sepia":
            rf=r.astype(float); return cv2.merge((rf,1.8*rf,2*rf)).clip(0,255).astype(np.uint8)
        if operation == "Threshold global": return cv2.threshold(g,int(value*2.55),255,cv2.THRESH_BINARY)[1]
        block=max(3,int(value)//2*2+3); return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,5)
    if number == 4:
        if operation == "Threshold": return cv2.threshold(g,int(value*2.55),255,cv2.THRESH_BINARY)[1]
        bits=max(1,min(8,round(value*7/100)+1)); step=256//(2**bits); source=g if "grayscale" in operation else img
        return ((source.astype(np.uint16)//step)*step).clip(0,255).astype(np.uint8)
    if number == 5:
        f=g.astype(np.float32)
        if operation == "Brightness": return np.clip(f+(value-50)*2.5,0,255).astype(np.uint8)
        if operation == "Contrast": return np.clip(f*(value/50),0,255).astype(np.uint8)
        if operation == "Invers": return 255-g
        if operation in {"Gamma / Power","Root"}: gamma=.15+value/35 if operation.startswith("Gamma") else .5; return np.round(255*(f/255)**gamma).astype(np.uint8)
        if operation == "Logaritmik": return np.round(255*np.log1p(f)/np.log(256)).astype(np.uint8)
        if operation == "Inverse log": return np.expm1(f/255*np.log(256)).clip(0,255).astype(np.uint8)
        spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g)))); return cv2.normalize(spec,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if number == 6:
        f=g.astype(float)
        if operation == "Brightness": return np.clip(f+(value-50)*2.5,0,255).astype(np.uint8)
        if operation == "Contrast": return np.clip(f*(value/50),0,255).astype(np.uint8)
        if operation == "Invers": return 255-g
        if operation == "Autolevel grayscale": return autolevel(g)
        if operation == "Autolevel BGR": return cv2.merge(tuple(autolevel(c) for c in cv2.split(img)))
        canvas=np.full((420,640,3),245,np.uint8); hist=cv2.calcHist([g],[0],None,[256],[0,256]).ravel(); hist=hist/max(hist.max(),1)*350
        for x,y in enumerate(hist): cv2.line(canvas,(x*2+64,390),(x*2+64,390-int(y)),(230,110,35),1)
        return canvas
    if number == 7:
        if operation == "Histogram Equalization": return cv2.equalizeHist(g)
        if operation == "Autolevel": return autolevel(g)
        if operation == "CLAHE": return cv2.createCLAHE(clipLimit=max(1,value/15),tileGridSize=(8,8)).apply(g)
        canvas=np.full((420,640,3),245,np.uint8); hist=np.bincount(g.ravel(),minlength=256); cdf=hist.cumsum()/hist.sum()
        hn=hist/max(hist.max(),1)*320
        for x,y in enumerate(hn): cv2.line(canvas,(x*2+64,380),(x*2+64,380-int(y)),(225,120,45),1)
        pts=np.array([[x*2+64,380-int(cdf[x]*320)] for x in range(256)],np.int32); cv2.polylines(canvas,[pts],False,(45,80,220),2); return canvas
    if number == 11:
        threshold = int(value * 2.55)
        binary = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        size = max(3, (int(value) // 15) * 2 + 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
        if operation == "Citra biner": return binary
        if operation == "Dilasi": return cv2.dilate(binary, kernel, iterations=1)
        if operation == "Erosi": return cv2.erode(binary, kernel, iterations=1)
        if operation == "Opening": return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if operation == "Closing": return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if operation == "Morphological gradient": return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        if operation == "Top hat": return cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
        if operation == "Black hat": return cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
        if operation == "Hit-or-Miss":
            small = cv2.threshold(g, threshold, 1, cv2.THRESH_BINARY_INV)[1]
            hit_kernel = np.array([[-1, 1, -1], [0, 1, 0], [0, 1, 0]], np.int8)
            return cv2.morphologyEx(small, cv2.MORPH_HITMISS, hit_kernel) * 255
        skeleton = np.zeros_like(binary); work = binary.copy(); sk = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(work):
            eroded = cv2.erode(work, sk); opened = cv2.dilate(eroded, sk)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened)); work = eroded
        return skeleton
    # Praktikum 08
    rng=np.random.default_rng(42); k=max(3,int(value)//10*2+3)
    if operation == "Gaussian noise": return np.clip(g.astype(float)+rng.normal(0,max(1,value/2),g.shape),0,255).astype(np.uint8)
    if operation == "Salt & pepper noise":
        result=g.copy(); m=rng.random(g.shape); p=value/1000; result[m<p]=0; result[m>1-p]=255; return result
    if operation == "Average blur": return cv2.blur(g,(k,k))
    if operation == "Gaussian blur": return cv2.GaussianBlur(g,(k,k),0)
    if operation == "Median filter": return cv2.medianBlur(g,k)
    if operation == "Prewitt":
        px=np.array([[-1,0,1]]*3,np.float32); return cv2.convertScaleAbs(cv2.magnitude(cv2.filter2D(g,cv2.CV_32F,px),cv2.filter2D(g,cv2.CV_32F,px.T)))
    if operation == "Sobel": return cv2.convertScaleAbs(cv2.magnitude(cv2.Sobel(g,cv2.CV_32F,1,0),cv2.Sobel(g,cv2.CV_32F,0,1)))
    if operation == "Laplacian": return cv2.convertScaleAbs(cv2.Laplacian(g,cv2.CV_32F,ksize=3))
    if operation == "Canny": return cv2.Canny(g,max(1,int(value)),max(2,int(value*2)))
    if operation == "Sharpness":
        blur=cv2.GaussianBlur(g,(0,0),2); return cv2.addWeighted(g,1+value/50,blur,-value/50,0)
    edge=cv2.Canny(g,50,150); return 255-edge


class VisionGUI:
    BG="#10151f"; PANEL="#182131"; CARD="#222e40"; TEXT="#ecf3ff"; MUTED="#94a5bd"; ACCENT="#35c2ff"
    def __init__(self, number, title, batch_callback):
        self.number, self.title, self.batch_callback = number, title, batch_callback
        self.root=tk.Tk(); self.root.title(f"Praktikum {number:02d} • {title}"); self.root.geometry("1280x780"); self.root.minsize(1000,650); self.root.configure(bg=self.BG)
        self.files=[p for p in sorted(DATA.iterdir()) if p.suffix.lower() in IMAGE_TYPES and p.name!="citra_contoh.png"]
        self.path=None; self.image=None; self.result=None; self._photo=[]
        self._style(); self._layout(); self._load_default()

    def _style(self):
        s=ttk.Style(); s.theme_use("clam")
        s.configure("TCombobox",fieldbackground=self.CARD,background=self.CARD,foreground=self.TEXT,arrowcolor=self.ACCENT,padding=8)
        s.configure("Accent.TButton",background=self.ACCENT,foreground="#06111a",font=("Segoe UI Semibold",10),padding=10,borderwidth=0)
        s.map("Accent.TButton",background=[("active","#73d7ff")])
        s.configure("Dark.Horizontal.TScale",background=self.PANEL,troughcolor=self.CARD)

    def _layout(self):
        header=tk.Frame(self.root,bg=self.BG,height=82); header.pack(fill="x",padx=24,pady=(18,8)); header.pack_propagate(False)
        tk.Label(header,text=f"PRAKTIKUM {self.number:02d}",bg=self.BG,fg=self.ACCENT,font=("Segoe UI Semibold",11)).pack(anchor="w")
        tk.Label(header,text=self.title,bg=self.BG,fg=self.TEXT,font=("Segoe UI Semibold",24)).pack(anchor="w")
        body=tk.Frame(self.root,bg=self.BG); body.pack(fill="both",expand=True,padx=24,pady=(0,18))
        side=tk.Frame(body,bg=self.PANEL,width=260); side.pack(side="left",fill="y",padx=(0,14)); side.pack_propagate(False)
        tk.Label(side,text="DATASET",bg=self.PANEL,fg=self.MUTED,font=("Segoe UI Semibold",9)).pack(anchor="w",padx=18,pady=(20,7))
        self.dataset=ttk.Combobox(side,state="readonly",values=[p.name for p in self.files]); self.dataset.pack(fill="x",padx=18); self.dataset.bind("<<ComboboxSelected>>",lambda e:self.load(self.files[self.dataset.current()]))
        ttk.Button(side,text="Buka gambar lain",style="Accent.TButton",command=self.open_file).pack(fill="x",padx=18,pady=(10,22))
        tk.Label(side,text="OPERASI",bg=self.PANEL,fg=self.MUTED,font=("Segoe UI Semibold",9)).pack(anchor="w",padx=18,pady=(0,7))
        self.operation=ttk.Combobox(side,state="readonly",values=OPERATIONS[self.number]); self.operation.current(0); self.operation.pack(fill="x",padx=18); self.operation.bind("<<ComboboxSelected>>",lambda e:self.update())
        tk.Label(side,text="PARAMETER",bg=self.PANEL,fg=self.MUTED,font=("Segoe UI Semibold",9)).pack(anchor="w",padx=18,pady=(22,4))
        self.value=tk.DoubleVar(value=50); self.scale=ttk.Scale(side,from_=0,to=100,variable=self.value,style="Dark.Horizontal.TScale",command=lambda v:self.update()); self.scale.pack(fill="x",padx=18)
        self.value_label=tk.Label(side,text="50",bg=self.PANEL,fg=self.ACCENT,font=("Consolas",11)); self.value_label.pack(anchor="e",padx=18)
        ttk.Button(side,text="Simpan hasil",style="Accent.TButton",command=self.save_result).pack(fill="x",padx=18,pady=(28,8))
        ttk.Button(side,text="Jalankan semua eksperimen",command=self.run_all).pack(fill="x",padx=18)
        self.info=tk.Label(side,text="",bg=self.PANEL,fg=self.MUTED,font=("Segoe UI",9),justify="left",wraplength=220); self.info.pack(side="bottom",anchor="w",padx=18,pady=18)
        content=tk.Frame(body,bg=self.BG); content.pack(side="left",fill="both",expand=True)
        cards=tk.Frame(content,bg=self.BG); cards.pack(fill="both",expand=True)
        self.left=self._card(cards,"CITRA ASLI"); self.right=self._card(cards,"HASIL PROSES")
        self.status=tk.Label(content,text="Siap",bg=self.CARD,fg=self.MUTED,font=("Segoe UI",9),anchor="w",padx=12,pady=8); self.status.pack(fill="x",pady=(12,0))

    def _card(self,parent,title):
        frame=tk.Frame(parent,bg=self.CARD); frame.pack(side="left",fill="both",expand=True,padx=(0,7) if title.startswith("CITRA") else (7,0))
        tk.Label(frame,text=title,bg=self.CARD,fg=self.MUTED,font=("Segoe UI Semibold",9)).pack(anchor="w",padx=16,pady=(13,0))
        label=tk.Label(frame,bg=self.CARD); label.pack(fill="both",expand=True,padx=12,pady=12); return label

    def _load_default(self):
        if not self.files: messagebox.showerror("Dataset kosong",f"Tidak ada gambar di {DATA}"); return
        wanted=DEFAULTS.get(self.number); idx=next((i for i,p in enumerate(self.files) if p.name==wanted),0); self.dataset.current(idx); self.load(self.files[idx])

    def load(self,path):
        image=cv2.imread(str(path));
        if image is None: messagebox.showerror("Error",f"Gagal membaca {path}"); return
        self.path,self.image=Path(path),image; h,w=image.shape[:2]; self.info.config(text=f"{self.path.name}\n{w} × {h} piksel\n{image.dtype} • 3 kanal"); self.update()

    def open_file(self):
        p=filedialog.askopenfilename(filetypes=[("Image","*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")]);
        if p:self.load(Path(p))

    def update(self):
        if self.image is None:return
        value=self.value.get(); self.value_label.config(text=f"{value:.0f}"); self.result=transform(self.number,self.operation.get(),self.image,value)
        self._show(self.left,self.image); self._show(self.right,self.result); self.status.config(text=f"{self.operation.get()}  •  parameter {value:.0f}")

    def _show(self,label,array):
        rgb=cv2.cvtColor(array.astype(np.uint8),cv2.COLOR_GRAY2RGB) if array.ndim==2 else cv2.cvtColor(array.astype(np.uint8),cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb); image.thumbnail((510,540),Image.Resampling.LANCZOS); photo=ImageTk.PhotoImage(image); label.config(image=photo); label.image=photo

    def save_result(self):
        if self.result is None:return
        name=self.operation.get().lower().replace(" ","_").replace("/","_").replace("+","plus").replace("°","")
        target=OUTPUT/f"praktikum_{self.number:02d}"/f"gui_{self.path.stem}_{name}.png"; save(target,self.result); self.status.config(text=f"Tersimpan: {target}")

    def run_all(self):
        try:self.batch_callback(str(self.path)); messagebox.showinfo("Selesai",f"Semua eksperimen tersimpan di output/praktikum_{self.number:02d}")
        except Exception as exc:messagebox.showerror("Gagal",str(exc))

    def run(self): self.root.mainloop()


def launch(number, title, batch_callback):
    VisionGUI(number,title,batch_callback).run()
