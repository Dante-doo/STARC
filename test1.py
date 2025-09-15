#!/usr/bin/env python3
# hough_pipeline_strong.py
# pip install opencv-python scikit-image numpy

from pathlib import Path
import re
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks

RAW_DIR    = Path("raw")
ACC_DIR    = Path("hough/accumulator_points")
INV_DIR    = Path("hough/inversed")
LINES_DIR  = Path("hough/lines_only")
PANEL_DIR  = Path("hough/preview")
DBG_DIR    = Path("hough/debug")

for d in (ACC_DIR, INV_DIR, LINES_DIR, PANEL_DIR, DBG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# arquivos: full*.png|jpg|jpeg
files = []
for ext in ("png","jpg","jpeg","PNG","JPG","JPEG"):
    files.extend(RAW_DIR.glob(f"full*.{ext}"))

def num_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0

# ----------------- HYPERPARAMS (ajuste se precisar) -----------------
PERC_LOW, PERC_HIGH = 0.5, 99.9      # stretch por percentis
CLAHE_CLIP = 2.5                     # 0 desliga
CLAHE_TILE = (16,16)
STAR_Q = 99.85                       # mascara de estrelas muito brilhantes
STAR_DILATE = 7                      # px
MEDIAN_K = 3                         # denoise
BG_K = 31                            # mediana grande p/ “flatten” (ímpar)
LINE_KLEN = 31                       # tamanho do kernel de linha (ímpar)
LINE_THICK = 1                       # espessura da linha no kernel
LINE_ANGLES = list(range(0,180,5))   # ângulos do banco (graus)
RESP_PERC = 99.4                     # limiar do mapa de resposta (percentil)
CANNY_SIGMA = 0.33                   # auto-canny
OPEN_K = 3                           # abertura p/ limpar ruído (0 desliga)
CLOSE_K = 0                          # fechamento p/ ligar pontos (0 desliga)
NUM_PEAKS = 100
MIN_DIST = 20
MIN_ANG = 8
DOT_RADIUS = 3
LINE_THICKNESS = 1

# ----------------- FUNÇÕES -----------------
def percentile_stretch(u8, p_low=1.0, p_high=99.8):
    lo, hi = np.percentile(u8, [p_low, p_high])
    if hi <= lo: return u8
    out = np.clip((u8.astype(np.float32)-lo)*(255.0/(hi-lo)), 0, 255)
    return out.astype(np.uint8)

def apply_clahe(u8, clip=2.0, tile=(8,8)):
    if clip <= 0: return u8
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile)).apply(u8)

def inpaint_stars(u8, q=99.8, dilate_r=5):
    th = np.percentile(u8, q)
    m = (u8 >= th).astype(np.uint8)*255
    if dilate_r>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate_r+1,2*dilate_r+1))
        m = cv2.dilate(m,k)
    # inpaint para remover halos/arestas
    return cv2.inpaint(u8, m, 3, cv2.INPAINT_TELEA), m

def flatten_bg(u8, k=31):
    if k<=1 or k%2==0: return u8
    bg = cv2.medianBlur(u8, k)
    x = cv2.subtract(u8, bg)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def line_bank_response(u8, klen=31, thk=1, angles_deg=tuple(range(0,180,5))):
    # banco de filtros de linha (DC-removido) e max-response
    ksz = int(klen)|1
    resp_max = np.full(u8.shape, -1e9, dtype=np.float32)
    for ang in angles_deg:
        ker = np.zeros((ksz, ksz), np.float32)
        c = ksz//2
        p1 = (0, c); p2 = (ksz-1, c)
        M = cv2.getRotationMatrix2D((c,c), float(ang), 1.0)
        cv2.line(ker, p1, p2, 1.0, thk)
        # rotaciona o kernel
        ker = cv2.warpAffine(ker, M, (ksz, ksz), flags=cv2.INTER_NEAREST)
        ker -= ker.mean()  # remove DC
        if np.allclose(ker.std(), 0): continue
        ker /= (np.linalg.norm(ker) + 1e-8)
        r = cv2.filter2D(u8.astype(np.float32), -1, ker, borderType=cv2.BORDER_REFLECT)
        resp_max = np.maximum(resp_max, r)
    # normaliza para 0..255
    rmin, rmax = float(resp_max.min()), float(resp_max.max())
    if rmax <= rmin: 
        return np.zeros_like(u8)
    out = (resp_max - rmin) * (255.0/(rmax-rmin))
    return out.astype(np.uint8)

def auto_canny(u8, sigma=0.33):
    v = float(np.median(u8))
    lo = int(max(0,(1.0-sigma)*v)); hi = int(min(255,(1.0+sigma)*v))
    return cv2.Canny(u8, lo, hi, L2gradient=True)

def make_tile(img, title, tile_w=1200, content_h=900, title_h=60):
    if img.ndim==2: vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else: vis = img.copy()
    h,w = vis.shape[:2]
    s = min(tile_w/w, content_h/h)
    new = (max(1,int(w*s)), max(1,int(h*s)))
    vis = cv2.resize(vis, new, interpolation=cv2.INTER_NEAREST)
    tile = np.zeros((title_h+content_h, tile_w, 3), np.uint8)
    tile[0:title_h,:]=(35,35,35)
    cv2.putText(tile, title, (16,title_h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2,cv2.LINE_AA)
    y0 = title_h + (content_h - vis.shape[0])//2
    x0 = (tile_w - vis.shape[1])//2
    tile[y0:y0+vis.shape[0], x0:x0+vis.shape[1]] = vis
    return tile

def stack2x2(a,b,c,d):
    h1=max(a.shape[0],b.shape[0]); h2=max(c.shape[0],d.shape[0])
    if a.shape[0]!=h1: a=cv2.copyMakeBorder(a,0,h1-a.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    if b.shape[0]!=h1: b=cv2.copyMakeBorder(b,0,h1-b.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    if c.shape[0]!=h2: c=cv2.copyMakeBorder(c,0,h2-c.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    if d.shape[0]!=h2: d=cv2.copyMakeBorder(d,0,h2-d.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    row1=np.hstack([a,b]); row2=np.hstack([c,d])
    w1,w2=row1.shape[1],row2.shape[1]
    if w1<w2: row1=cv2.copyMakeBorder(row1,0,0,0,w2-w1,cv2.BORDER_CONSTANT,value=(0,0,0))
    elif w2<w1: row2=cv2.copyMakeBorder(row2,0,0,0,w1-w2,cv2.BORDER_CONSTANT,value=(0,0,0))
    return np.vstack([row1,row2])

# ----------------- LOOP -----------------
for img_path in sorted(files, key=num_key):
    name = img_path.stem
    print(f"[+] {img_path}")

    img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        print("    (skip: cannot read)"); continue

    # 1) stretch + clahe
    x = percentile_stretch(img0, PERC_LOW, PERC_HIGH)
    x = apply_clahe(x, CLAHE_CLIP, CLAHE_TILE)

    # 2) remove estrelas brilhantes (inpaint) + mediana + flatten bg
    x, star_mask = inpaint_stars(x, STAR_Q, STAR_DILATE)
    x = cv2.medianBlur(x, MEDIAN_K)
    x = flatten_bg(x, BG_K)

    # 3) banco de linhas orientadas (resposta)
    resp = line_bank_response(x, LINE_KLEN, LINE_THICK, LINE_ANGLES)
    thr = np.percentile(resp, RESP_PERC)
    resp_mask = (resp >= thr).astype(np.uint8) * 255

    # 4) edges (canny) + OR com resposta do banco
    edges = auto_canny(x, CANNY_SIGMA)
    edges = cv2.bitwise_or(edges, resp_mask)

    # morfologia
    if OPEN_K and OPEN_K>1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(OPEN_K,OPEN_K))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k, iterations=1)
    if CLOSE_K and CLOSE_K>1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(CLOSE_K,CLOSE_K))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

    cv2.imwrite(str(DBG_DIR / f"{name}_resp.png"), resp)
    cv2.imwrite(str(DBG_DIR / f"{name}_edges.png"), edges)

    # 5) Hough
    hspace, angles, dists = hough_line(edges>0)
    acc_img = np.zeros_like(hspace, dtype=np.uint8)
    acc, theta_peaks, rho_peaks = hough_line_peaks(
        hspace, angles, dists,
        num_peaks=NUM_PEAKS, min_distance=MIN_DIST, min_angle=MIN_ANG
    )
    for rho, theta in zip(rho_peaks, theta_peaks):
        r_idx = int(np.argmin(np.abs(dists - rho)))
        t_idx = int(np.argmin(np.abs(angles - theta)))
        cv2.circle(acc_img, (t_idx, r_idx), DOT_RADIUS, 255, -1)

    # fallback se nao achou nada
    if len(rho_peaks) == 0:
        print("    [!] Fallback: relaxando limiar e conectando")
        thr2 = np.percentile(resp, RESP_PERC - 0.7)
        resp_mask2 = (resp >= thr2).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        edges2 = cv2.dilate(resp_mask2, k, iterations=1)
        hspace, angles, dists = hough_line(edges2>0)
        acc_img = np.zeros_like(hspace, dtype=np.uint8)
        acc, theta_peaks, rho_peaks = hough_line_peaks(
            hspace, angles, dists,
            num_peaks=NUM_PEAKS*2, min_distance=max(5,MIN_DIST//2), min_angle=max(3,MIN_ANG//2)
        )
        for rho, theta in zip(rho_peaks, theta_peaks):
            r_idx = int(np.argmin(np.abs(dists - rho)))
            t_idx = int(np.argmin(np.abs(angles - theta)))
            cv2.circle(acc_img, (t_idx, r_idx), DOT_RADIUS, 255, -1)

    cv2.imwrite(str(ACC_DIR / f"{name}_acc_points.png"), acc_img)

    # 6) desenha linhas
    on_original = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    lines_only = np.zeros_like(img0, dtype=np.uint8)
    for rho, theta in zip(rho_peaks, theta_peaks):
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        p1 = (int(x0 + 20000*(-b)), int(y0 + 20000*(a)))
        p2 = (int(x0 - 20000*(-b)), int(y0 - 20000*(a)))
        cv2.line(on_original, p1, p2, (0,255,0), LINE_THICKNESS, cv2.LINE_AA)
        cv2.line(lines_only, p1, p2, 255, LINE_THICKNESS, cv2.LINE_AA)

    cv2.imwrite(str(INV_DIR / f"{name}_inversed.png"), on_original)
    cv2.imwrite(str(LINES_DIR / f"{name}_hough_lines.png"), lines_only)

    # 7) painel 2x2
    def tile(img, title): return make_tile(img, title, tile_w=1200, content_h=900)
    panel = stack2x2(
        tile(img0, "Original"),
        tile(acc_img, "Accumulator (points)"),
        tile(on_original, "Overlay (lines)"),
        tile(lines_only, "Lines only")
    )
    cv2.imwrite(str(PANEL_DIR / f"{name}_panel.png"), panel)

print("Done.")
