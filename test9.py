#!/usr/bin/env python3
# pip install opencv-python scikit-image numpy

from pathlib import Path
import re, math
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks

# ============ I/O ============
RAW_DIR = Path("raw")
OUT_DIR = Path("out")
for sub in ("overlay","lines","panel","debug"):
    (OUT_DIR/sub).mkdir(parents=True, exist_ok=True)

def list_inputs():
    files=[]
    for ext in ("png","PNG","jpg","JPG","jpeg","JPEG","tif","TIF","tiff","TIFF"):
        files.extend(RAW_DIR.glob(f"full*.{ext}"))
    def num_key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    return sorted(files, key=num_key)

# ============ PARAMS (ajuste aqui) ============
PROC_MAX_SIDE = 2048

# base
PERC_LOW, PERC_HIGH = 0.5, 99.9
CLAHE_CLIP, CLAHE_TILE = 2.5, (16,16)
BG_MED_K = 31
CANNY_SIGMA = 0.33

# coerência (tensor de estrutura)
COH_WIN        = 9
COH_THR_ABS    = 0.18
COH_THR_PERC   = 99.2
MAX_DIRS       = 3
ORI_TOL_DEG    = 12.0

# hough
NUM_PEAKS = 600
MIN_DIST  = 12
MIN_ANG   = 4

# validação (suporte na faixa da reta)
SUPPORT_BAND_PX      = 5
SUPPORT_MIN_HITS     = 400
SUPPORT_MIN_DENSITY  = 0.030
SUPPORT_MIN_MEAN_COH = 0.30   # média da coerência dentro dos hits
SUPPORT_MAX_ODISP    = 16.0   # p95 da diferença angular (em graus)

# extração do segmento (perfil 1D)
PROFILE_MAX_GAP = 24
PROFILE_MIN_LEN = 110

# artefatos verticais 1px
VERT_TOL        = 2
VERT_MIN_RUN_R  = 0.10
VERT_MAX_WIDTH  = 1

# ============ Utils ============
def percentile_stretch(u8, p_low, p_high):
    lo, hi = np.percentile(u8, [p_low, p_high])
    if hi <= lo: return u8
    x = np.clip((u8.astype(np.float32)-lo) * (255.0/(hi-lo)), 0, 255)
    return x.astype(np.uint8)

def apply_clahe(u8, clip, tile):
    if clip <= 0: return u8
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile)).apply(u8)

def flatten_bg(u8, k):
    if k<=1 or k%2==0: return u8
    bg = cv2.medianBlur(u8, k)
    x  = cv2.subtract(u8, bg)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def auto_canny(u8, sigma):
    v = float(np.median(u8))
    lo = int(max(0,(1.0-sigma)*v)); hi = int(min(255,(1.0+sigma)*v))
    if hi <= lo: hi = lo + 1
    return cv2.Canny(u8, lo, hi, L2gradient=True)

def detect_vertical_artifacts_strict(u8, tol=2, min_run_r=0.07, max_width=1):
    H, W = u8.shape
    min_run = max(16, int(min_run_r * H))
    white = (u8 >= (255 - tol)).astype(np.uint8)
    black = (u8 <= tol).astype(np.uint8)
    def width1(b):
        left = np.zeros_like(b);  left[:,1:]  = b[:,:-1]
        right= np.zeros_like(b);  right[:,:-1]= b[:,1:]
        return (b & (~left) & (~right)).astype(np.uint8)
    w1 = width1(white); b1 = width1(black)
    vker = cv2.getStructuringElement(cv2.MORPH_RECT,(1, min_run))
    w_long = cv2.morphologyEx(w1*255, cv2.MORPH_OPEN, vker)
    b_long = cv2.morphologyEx(b1*255, cv2.MORPH_OPEN, vker)
    mask = cv2.bitwise_or(w_long, b_long)
    if max_width > 1:
        hker = cv2.getStructuringElement(cv2.MORPH_RECT,(max_width+1,1))
        too_wide = cv2.morphologyEx((mask>0).astype(np.uint8)*255, cv2.MORPH_OPEN, hker)
        mask[too_wide>0] = 0
    return mask

# --- Structure tensor ---
def coherence_map(u8, win=9):
    f = u8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    k = int(win)|1
    Axx = cv2.GaussianBlur(gx*gx, (k,k), 0)
    Axy = cv2.GaussianBlur(gx*gy, (k,k), 0)
    Ayy = cv2.GaussianBlur(gy*gy, (k,k), 0)
    tmp = (Axx - Ayy)**2 + 4.0*(Axy**2)
    tmp = np.sqrt(np.maximum(tmp, 0))
    l1 = 0.5*((Axx + Ayy) + tmp)
    l2 = 0.5*((Axx + Ayy) - tmp)
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    ori = 0.5*np.degrees(np.arctan2(2*Axy, (Axx - Ayy))) % 180.0
    return np.clip(coh,0,1).astype(np.float32), ori.astype(np.float32)

def ang_diff_deg(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def find_orientation_modes(ori_deg, mask, k=3, sep=12.0):
    th = ori_deg[mask>0].ravel()
    if th.size == 0: return []
    hist, edges = np.histogram(th, bins=np.arange(0,181,1.0))
    peaks = []
    for _ in range(k):
        i = int(np.argmax(hist))
        if hist[i] == 0: break
        center = 0.5*(edges[i] + edges[i+1])
        peaks.append(center)
        a = int(max(0, i - int(sep))); b = int(min(len(hist), i + int(sep)+1))
        hist[a:b] = 0
    return peaks

# --- Validação por coerência ---
def validate_line(edges, coh, ori_deg, rho, theta_rad,
                  band=5, min_hits=400, min_density=0.03,
                  min_mean_coh=0.30, max_odisp=16.0, ori_tol=12.0):
    H, W = edges.shape
    diag = int(np.hypot(H, W))
    a, b = math.cos(theta_rad), math.sin(theta_rad)
    x0, y0 = a*rho, b*rho
    p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
    p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))

    band_mask = np.zeros((H,W), np.uint8)
    cv2.line(band_mask, p1, p2, 255, 2*band+1, cv2.LINE_8)

    d = np.vectorize(ang_diff_deg)(ori_deg, np.degrees(theta_rad)).astype(np.float32)
    ori_gate = (d <= ori_tol).astype(np.uint8)*255

    hits_mask = cv2.bitwise_and(band_mask, cv2.bitwise_and(edges, ori_gate))
    total = cv2.countNonZero(band_mask); hits = cv2.countNonZero(hits_mask)
    if hits==0 or total==0: return False, None

    density = hits/float(total)
    if hits < min_hits or density < min_density: return False, None

    # coerência média nos hits e dispersão angular
    ys, xs = np.where(hits_mask>0)
    mean_coh = float(coh[ys, xs].mean()) if xs.size else 0.0
    odisp = float(np.percentile(d[hits_mask>0], 95)) if xs.size else 180.0
    if mean_coh < min_mean_coh or odisp > max_odisp:
        return False, None

    return True, hits_mask

def extract_segment_from_hits(hit_mask, rho, theta_rad, max_gap=24, min_len=110):
    ys, xs = np.where(hit_mask>0)
    if xs.size < 2: return None
    vx, vy = -math.sin(theta_rad), math.cos(theta_rad)
    t = xs*vx + ys*vy
    idx = np.argsort(t); xs, ys, t = xs[idx], ys[idx], t[idx]

    start = 0; best = (0, 0, 0)
    for i in range(1, len(t)):
        if (t[i] - t[i-1]) > max_gap:
            span = t[i-1] - t[start]
            if span > best[0]: best = (span, start, i-1)
            start = i
    span = t[-1] - t[start]
    if span > best[0]: best = (span, start, len(t)-1)

    if best[0] < min_len: return None
    i0, i1 = best[1], best[2]
    return (int(xs[i0]), int(ys[i0])), (int(xs[i1]), int(ys[i1]))

def draw_panel(img_w, edges_g, overlay, lines_only, name):
    def tile_box(img, title, W=900, H=800, title_h=36):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
        h,w = vis.shape[:2]; s = min(W / w, H / h)
        vis = cv2.resize(vis, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
        tile = np.zeros((title_h+H, W, 3), np.uint8); tile[0:title_h,:] = (30,30,30)
        cv2.putText(tile, title, (12,title_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2, cv2.LINE_8)
        y0 = title_h + (H - vis.shape[0])//2; x0 = (W - vis.shape[1])//2
        tile[y0:y0+vis.shape[0], x0:x0+vis.shape[1]] = vis
        return tile
    A = tile_box(img_w,   "Preprocessed (weak): stretch+CLAHE+flatten")
    B = tile_box(edges_g, "Edges (union + coherence-orientation gate)")
    C = tile_box(overlay, "Overlay (white segments)")
    D = tile_box(lines_only, "Lines only (white)")
    h,w = A.shape[:2]; sep=8; margin=10
    H = 2*h + 3*margin + sep; W = 2*w + 3*margin + sep
    panel = np.zeros((H,W,3), np.uint8)
    panel[margin:margin+h, margin:margin+w] = A
    panel[margin:margin+h, margin+w+sep:margin+w+sep+w] = B
    panel[margin+h+sep:margin+h+sep+h, margin:margin+w] = C
    panel[margin+h+sep:margin+h+sep+h, margin+w+sep:margin+w+sep+w] = D
    cv2.imwrite(str(OUT_DIR / "panel" / f"{name}_panel.png"), panel)

# ============ MAIN ============
if __name__ == "__main__":
    for img_path in list_inputs():
        name = img_path.stem
        print(f"[+] {img_path}")

        # load + downscale
        img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img0 is None: print("   (skip)"); continue
        H0, W0 = img0.shape
        scale = min(1.0, PROC_MAX_SIDE / float(max(H0, W0)))
        img = cv2.resize(img0, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img0.copy()

        # limpa colunas verticais 1px
        vmask = detect_vertical_artifacts_strict(img, tol=VERT_TOL, min_run_r=VERT_MIN_RUN_R, max_width=VERT_MAX_WIDTH)
        if np.count_nonzero(vmask)>0: img = cv2.inpaint(img, vmask, 3, cv2.INPAINT_TELEA)

        # base
        base = percentile_stretch(img, PERC_LOW, PERC_HIGH)
        base = apply_clahe(base, CLAHE_CLIP, CLAHE_TILE)
        base = flatten_bg(base, BG_MED_K)
        edges = auto_canny(base, CANNY_SIGMA)

        # coerência + máscara
        coh, ori_deg = coherence_map(base, COH_WIN)
        thr_coh = max(COH_THR_ABS, np.percentile(coh, COH_THR_PERC))
        coh_mask = (coh >= thr_coh).astype(np.uint8)*255

        # orientações dominantes e gate
        dirs = find_orientation_modes(ori_deg, coh_mask, k=MAX_DIRS, sep=12.0)
        gated = np.zeros_like(edges)
        if len(dirs)==0:
            gated = cv2.bitwise_and(edges, coh_mask)
        else:
            for th in dirs:
                delta = np.vectorize(ang_diff_deg)(ori_deg, th).astype(np.float32)
                gate = (delta <= ORI_TOL_DEG).astype(np.uint8)*255
                gated |= cv2.bitwise_and(edges, cv2.bitwise_and(coh_mask, gate))

        # Hough nos pixels coerentes
        h, ang, rho = hough_line(gated>0)
        acc, thetas, rhos = hough_line_peaks(h, ang, rho, num_peaks=NUM_PEAKS, min_distance=MIN_DIST, min_angle=MIN_ANG)

        # escalas
        BAND_S         = max(3,  int(round(SUPPORT_BAND_PX * scale))) if scale<1.0 else SUPPORT_BAND_PX
        MIN_HITS_S     = max(120,int(round(SUPPORT_MIN_HITS*scale)))   if scale<1.0 else SUPPORT_MIN_HITS
        MIN_LEN_S      = max(40, int(round(PROFILE_MIN_LEN*scale)))    if scale<1.0 else PROFILE_MIN_LEN
        MAX_GAP_S      = max(12, int(round(PROFILE_MAX_GAP*scale)))    if scale<1.0 else PROFILE_MAX_GAP

        def run_selection(min_hits=MIN_HITS_S, min_den=SUPPORT_MIN_DENSITY,
                          min_mean=SUPPORT_MIN_MEAN_COH, odisp=SUPPORT_MAX_ODISP, ori_tol=ORI_TOL_DEG):
            segs=[]
            for rho_i, th_i in zip(rhos, thetas):
                ok, hits = validate_line(gated, coh, ori_deg, rho_i, th_i,
                                         band=BAND_S, min_hits=min_hits, min_density=min_den,
                                         min_mean_coh=min_mean, max_odisp=odisp, ori_tol=ori_tol)
                if not ok: continue
                seg = extract_segment_from_hits(hits, rho_i, th_i, max_gap=MAX_GAP_S, min_len=MIN_LEN_S)
                if seg is not None: segs.append(seg)
            return segs

        segments = run_selection()

        # fallback: relaxa limites se nada foi achado
        if len(segments)==0:
            segments = run_selection(
                min_hits=int(0.7*MIN_HITS_S),
                min_den=max(0.018, SUPPORT_MIN_DENSITY*0.7),
                min_mean=max(0.22, SUPPORT_MIN_MEAN_COH*0.8),
                odisp=SUPPORT_MAX_ODISP+4.0,
                ori_tol=ORI_TOL_DEG+2.0
            )

        # upsample para o tamanho original
        def up(pt): return (int(round(pt[0]/scale)), int(round(pt[1]/scale))) if scale<1.0 else pt
        segments_up = [(up(a), up(b)) for (a,b) in segments]

        # desenha
        overlay = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        lines   = np.zeros_like(img0, np.uint8)
        for p1,p2 in segments_up:
            cv2.line(overlay, p1, p2, (255,255,255), 3, cv2.LINE_8)
            cv2.line(lines,   p1, p2, 255,            3, cv2.LINE_8)

        # debug coerência
        coh_vis = (np.clip(coh,0,1)*255).astype(np.uint8)
        coh_vis = cv2.applyColorMap(coh_vis, cv2.COLORMAP_TURBO)
        if scale < 1.0: coh_vis = cv2.resize(coh_vis, (W0,H0), interpolation=cv2.INTER_CUBIC)

        # painel
        base_show = percentile_stretch(img, 0.5, 99.9)
        gated_show = cv2.resize(gated, (W0,H0), interpolation=cv2.INTER_NEAREST) if scale<1.0 else gated
        draw_panel(base_show, gated_show, overlay, lines, name)

        # saves
        cv2.imwrite(str(OUT_DIR/"overlay"/f"{name}_overlay.png"), overlay)
        cv2.imwrite(str(OUT_DIR/"lines"/f"{name}_lines.png"), lines)
        cv2.imwrite(str(OUT_DIR/"debug"/f"{name}_coherence.png"), coh_vis)
        print(f"   segments: {len(segments_up)}")
