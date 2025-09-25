#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install opencv-python scikit-image numpy scikit-learn

from pathlib import Path
import re, math
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import DBSCAN

# =========================
# I/O
# =========================
RAW_DIR = Path("raw")
OUT_DIR = Path("out")
for sub in ("overlay","lines","panel","debug","debug/patch"):
    (OUT_DIR/sub).mkdir(parents=True, exist_ok=True)

def list_inputs():
    files=[]
    for ext in ("png","PNG","jpg","JPG","jpeg","JPEG","tif","TIF","tiff","TIFF"):
        files.extend(RAW_DIR.glob(f"full*.{ext}"))
    def num_key(p: Path):
        m = re.search(r"(\d+)", p.stem); return int(m.group(1)) if m else 0
    return sorted(files, key=num_key)

# =========================
# PARÂMETROS (↑ sensível / ↓ conservador)
# =========================
PROC_MAX_SIDE = 2048
PATCH_SIZE = 224

# Preprocess
PERC_LOW, PERC_HIGH   = 0.5, 99.9
CLAHE_CLIP, CLAHE_TILE= 2.7, (16,16)
BG_MED_K              = 31
CANNY_SIGMA           = 0.33

# Bordas (hard zero só no seed)
BORDER_HARD_ZERO_PX = 8
BORDER_MARGIN_PX    = 12

# Halo
HALO_Q, HALO_GROW, ALPHA_HALO = 99.87, 21, 0.5

# Artefato vertical finíssimo
ARTV_MAX_THICK      = 2
ARTV_MIN_AREA       = 25
ARTV_CONNECT_LEN    = 61
ARTV_BORDER_XMARGIN = 3
ARTV_FULLH_TOL      = 3

# EXCLUSÃO EXATA 90°
REJECT_EXACT_VERTICAL = True

# --------- PROFILES do FULL (iguais ao seu código bom) ----------
PROFILES_FULL = [
    dict(LABEL="perm", COH_WIN=11, COH_THR_ABS=0.08, COH_THR_PERC=98.2, ORI_TOL_DEG=18.0, DIR_CLOSE_LEN=80),
    dict(LABEL="med",  COH_WIN=9,  COH_THR_ABS=0.18, COH_THR_PERC=99.2, ORI_TOL_DEG=12.0, DIR_CLOSE_LEN=55),
    dict(LABEL="rest", COH_WIN=9,  COH_THR_ABS=0.28, COH_THR_PERC=99.5, ORI_TOL_DEG=10.0, DIR_CLOSE_LEN=61),
]

# --------- PROFILES para PATCH (mais permissivos) ----------
PROFILES_PATCH = [
    dict(LABEL="perm", COH_WIN=11, COH_THR_ABS=0.06, COH_THR_PERC=98.0, ORI_TOL_DEG=22.0, DIR_CLOSE_LEN=60),
    dict(LABEL="med",  COH_WIN=9,  COH_THR_ABS=0.14, COH_THR_PERC=99.0, ORI_TOL_DEG=16.0, DIR_CLOSE_LEN=45),
    dict(LABEL="rest", COH_WIN=9,  COH_THR_ABS=0.22, COH_THR_PERC=99.4, ORI_TOL_DEG=14.0, DIR_CLOSE_LEN=45),
]

# Hough scans (FULL)
HOUGHP_GLOBAL_SCAN_FULL = [
    dict(rho=1, theta=np.pi/180, threshold=45, minLineLength=70,  maxLineGap=260),
    dict(rho=1, theta=np.pi/180, threshold=52, minLineLength=100, maxLineGap=200),
]
HOUGHP_LOCAL_SCAN_FULL = [
    dict(rho=1, theta=np.pi/180, threshold=40, minLineLength=50, maxLineGap=14),
    dict(rho=1, theta=np.pi/180, threshold=50, minLineLength=60, maxLineGap=10),
]

# Hough scans (PATCH)
HOUGHP_GLOBAL_SCAN_PATCH = [
    dict(rho=1, theta=np.pi/180, threshold=22, minLineLength=36, maxLineGap=110),
    dict(rho=1, theta=np.pi/180, threshold=27, minLineLength=48, maxLineGap=80),
]
HOUGHP_LOCAL_SCAN_PATCH = [
    dict(rho=1, theta=np.pi/180, threshold=20, minLineLength=28, maxLineGap=8),
    dict(rho=1, theta=np.pi/180, threshold=24, minLineLength=36, maxLineGap=6),
]

# Validação na faixa de suporte (FULL)
SUPPORT_BAND_PX_FULL      = 5
SUPPORT_MIN_DENSITY_FULL  = 0.028
SUPPORT_MIN_MEAN_COH_FULL = 0.27
SUPPORT_MAX_ODISP_FULL    = 16.0
ORI_TOL_VALID_FULL        = 14.0
BORDER_REJECT_FRAC_FULL   = 0.92

# Validação na faixa de suporte (PATCH, mais leve)
SUPPORT_BAND_PX_PATCH      = 7
SUPPORT_MIN_DENSITY_PATCH  = 0.012
SUPPORT_MIN_MEAN_COH_PATCH = 0.16
SUPPORT_MAX_ODISP_PATCH    = 18.0
ORI_TOL_VALID_PATCH        = 20.0
BORDER_REJECT_FRAC_PATCH   = 1.10   # >1 ⇒ desliga rejeição por borda

# Conectividade (FULL)
AREA_THRESHOLD_FULL = 3000
MIN_COMP_SIZE_FULL  = 500
CLOSE_K_SZ          = 3

# Conectividade (PATCH)
AREA_THRESHOLD_PATCH = 400
MIN_COMP_SIZE_PATCH  = 120

# Trim pelo suporte
PROFILE_MAX_GAP_BASE = 22
PROFILE_MIN_LEN_FULL = 40
PROFILE_MIN_LEN_PATCH= 24

# Debug
SAVE_PER_PATCH_PANELS = False   # pode virar True se quiser salvar ~2k painéis
MAX_PATCH_PANELS      = 200

# =========================
# Utils
# =========================
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
    if hi<=lo: hi=lo+1
    return cv2.Canny(u8, lo, hi, L2gradient=True)

def border_mask(shape, margin):
    H,W = shape
    bm = np.zeros((H,W), np.uint8)
    m = int(max(1, margin))
    bm[:m,:] = 255; bm[-m:,:] = 255; bm[:,:m] = 255; bm[:,-m:] = 255
    return bm

def bright_halo_mask(u8, q=99.87, grow=21):
    th = np.percentile(u8, q)
    bin_ = cv2.threshold(u8, th, 255, cv2.THRESH_BINARY)[1]
    if grow>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow+1,2*grow+1))
        bin_ = cv2.dilate(bin_, k, 1)
    return bin_

# Structure tensor
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
    peaks=[]
    for _ in range(k):
        i = int(np.argmax(hist))
        if hist[i]==0: break
        center = 0.5*(edges[i]+edges[i+1]); peaks.append(center)
        a = int(max(0, i-int(sep))); b = int(min(len(hist), i+int(sep)+1))
        hist[a:b]=0
    return peaks

def oriented_close_one(img_u8, theta_deg, length=61):
    ksz = int(length)|1; ker = np.zeros((ksz, ksz), np.uint8); c = ksz//2
    cv2.line(ker, (0,c), (ksz-1,c), 255, 1)
    M = cv2.getRotationMatrix2D((c,c), float(theta_deg), 1.0)
    ker = cv2.warpAffine(ker, M, (ksz,ksz), flags=cv2.INTER_NEAREST)
    ker = (ker>0).astype(np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, ker, iterations=1)

def remove_small_components(mask, min_size=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: out[labels==i]=255
    return out

def farthest_endpoints(lines_xyxy):
    pts = np.vstack([lines_xyxy[:, :2], lines_xyxy[:, 2:4]]).astype(np.float32)
    if len(pts) > 256:
        idx = np.linspace(0, len(pts)-1, 256).astype(int); pts = pts[idx]
    d2 = ((pts[:,None,:]-pts[None,:,:])**2).sum(-1)
    i,j = np.unravel_index(np.argmax(d2), d2.shape)
    return tuple(map(int, pts[i])), tuple(map(int, pts[j]))

# ========================= Artefato vertical
def detect_thin_vertical_artifacts(img_u8, edges_u8):
    H, W = img_u8.shape
    vert = oriented_close_one(edges_u8, 90.0, length=ARTV_CONNECT_LEN)
    vert = cv2.morphologyEx(vert, cv2.MORPH_OPEN, np.ones((3,1), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    out = np.zeros_like(img_u8, np.uint8)
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < ARTV_MIN_AREA: continue
        if w > max(1, ARTV_MAX_THICK): continue
        ys, xs = np.where(labels==i)
        if xs.size < 2: continue
        near_left  = x <= ARTV_BORDER_XMARGIN
        near_right = x + w >= W-1-ARTV_BORDER_XMARGIN
        if (near_left or near_right):
            full_h = (y <= ARTV_FULLH_TOL) and (y + h >= H-1-ARTV_FULLH_TOL)
            if not full_h: continue
        out[labels==i] = 255
    return out

# ---------- Hough helpers
def reject_exact_vertical_lines(L):
    if L is None: return None
    keep = []
    for x1,y1,x2,y2 in L:
        if x1 == x2:  # 90° exato
            continue
        keep.append([x1,y1,x2,y2])
    return np.array(keep) if keep else None

def houghp_fuse(mask, scans):
    fused = np.zeros_like(mask)
    lines_all = []
    for prm in scans:
        lines = cv2.HoughLinesP(mask, **prm)
        if lines is None: continue
        L = lines[:,0,:]
        if REJECT_EXACT_VERTICAL:
            L = reject_exact_vertical_lines(L)
            if L is None: continue
        lines_all.append(L)
        for x1,y1,x2,y2 in L:
            cv2.line(fused,(x1,y1),(x2,y2),255,1)
    return fused, (np.vstack(lines_all) if lines_all else None)

# ---------- validação + trim
def validate_line(edges_like, coh, ori_deg, rho, theta_rad,
                  halo=None, band=5, min_density=0.03, min_mean_coh=0.27,
                  max_odisp=16.0, ori_tol=12.0, border_reject_frac=0.92,
                  border_margin=BORDER_MARGIN_PX):
    H,W = edges_like.shape
    diag = int(np.hypot(H, W))
    a, b = math.cos(theta_rad), math.sin(theta_rad)
    x0, y0 = a*rho, b*rho
    p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
    p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))
    band_mask = np.zeros((H,W), np.uint8)
    cv2.line(band_mask, p1, p2, 255, 2*band+1, cv2.LINE_8)

    d = np.vectorize(ang_diff_deg)(ori_deg, np.degrees(theta_rad)).astype(np.float32)
    ori_gate = (d <= ori_tol).astype(np.uint8)*255

    hits_mask = cv2.bitwise_and(band_mask, cv2.bitwise_and(edges_like, ori_gate))
    hits = cv2.countNonZero(hits_mask); total = cv2.countNonZero(band_mask)
    if hits==0 or total==0: return False, None, None
    dens = hits/float(total)
    if dens < min_density: return False, None, None

    ys, xs = np.where(hits_mask>0)
    cvals = coh[ys, xs] if xs.size else np.array([0.0], np.float32)
    if halo is not None and xs.size:
        hmask = (halo[ys, xs] > 0).astype(np.float32)
        w = 1.0 - hmask * (1.0 - ALPHA_HALO)
        cvals = cvals * w
    mean_coh = float(cvals.mean())
    odisp    = float(np.percentile(d[hits_mask>0], 95)) if xs.size else 180.0
    if mean_coh < min_mean_coh or odisp > max_odisp: return False, None, None

    bm = border_mask((H,W), border_margin)
    if cv2.countNonZero(cv2.bitwise_and(hits_mask,bm))/float(hits) >= border_reject_frac:
        return False, None, None
    return True, hits_mask, dict(density=dens, mean_coh=mean_coh)

def trim_to_support(p1, p2, hits_mask, max_gap=16, min_len=40):
    ys, xs = np.where(hits_mask > 0)
    if xs.size < 5: 
        return p1, p2
    vx, vy = (p2[0]-p1[0]), (p2[1]-p1[1])
    n = math.hypot(vx, vy) + 1e-6
    ux, uy = vx/n, vy/n
    t = (xs - p1[0]) * ux + (ys - p1[1]) * uy
    t = np.sort(t)
    gaps = np.diff(t)
    start = 0
    best = (t[0], t[0])
    for i, g in enumerate(gaps, 1):
        if g > max_gap:
            if t[i-1] - t[start] > best[1] - best[0]:
                best = (t[start], t[i-1])
            start = i
    if t[-1] - t[start] > best[1] - best[0]:
        best = (t[start], t[-1])
    if best[1] - best[0] < min_len:
        return p1, p2
    a = (int(round(p1[0] + ux*best[0])), int(round(p1[1] + uy*best[0])))
    b = (int(round(p1[0] + ux*best[1])), int(round(p1[1] + uy*best[1])))
    return a, b

# ---------- mosaicos & tiles
def crop_to_grid(img, P):
    H,W = img.shape[:2]
    R, C = H//P, W//P
    Hc, Wc = R*P, C*P
    return img[:Hc,:Wc].copy(), R, C

def to_tiles(img, R, C, P):
    tiles=[]
    for r in range(R):
        for c in range(C):
            y, x = r*P, c*P
            tiles.append(img[y:y+P, x:x+P].copy())
    return tiles

def save_mosaic(tiles, grid_hw, out_path):
    R, C = grid_hw
    assert len(tiles) == R*C, f"tiles={len(tiles)} != R*C={R*C}"

    t0 = tiles[0]
    h, w = t0.shape[:2]
    ch = 1 if (t0.ndim == 2 or (t0.ndim == 3 and t0.shape[2] == 1)) else t0.shape[2]
    is_gray = (ch == 1)

    # canvas 2D para cinza, 3D para RGB
    canvas = np.zeros((R*h, C*w), np.uint8) if is_gray else np.zeros((R*h, C*w, ch), np.uint8)

    for i, t in enumerate(tiles):
        # normaliza canais do tile
        if t.ndim == 3 and is_gray:
            t = t[..., 0]                           # 3D -> 2D (pega o canal 0)
        elif t.ndim == 2 and not is_gray:
            t = np.repeat(t[..., None], ch, axis=2) # 2D -> 3D (repete canais)

        # normaliza tamanho do tile (corta/pad)
        th, tw = t.shape[:2]
        if (th, tw) != (h, w):
            t = t[:min(th, h), :min(tw, w)]
            pad_shape = (h, w) if is_gray else (h, w, ch)
            pad = np.zeros(pad_shape, np.uint8)
            pad[:t.shape[0], :t.shape[1]] = t
            t = pad

        r, c = divmod(i, C)
        y, x = r*h, c*w
        canvas[y:y+h, x:x+w] = t

    cv2.imwrite(str(out_path), canvas)


def draw_panel(img_w, accum, overlay, lines_only, name):
    DARK_GREEN = (22,40,22)
    W, H, TITLE = 900, 800, 36
    SEP, MARGIN = 8, 10

    def tile(img, title):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        h, w = vis.shape[:2]
        s = min(W / w, H / h)
        vis = cv2.resize(vis, (int(w * s), int(h * s)), cv2.INTER_AREA)

        box = np.zeros((TITLE + H, W, 3), np.uint8); box[:] = DARK_GREEN
        cv2.rectangle(box, (0, 0), (W - 1, TITLE - 1), (30, 30, 30), -1)
        cv2.putText(box, title, (12, TITLE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_8)
        y0 = TITLE + (H - vis.shape[0]) // 2
        x0 = (W - vis.shape[1]) // 2
        box[y0:y0 + vis.shape[0], x0:x0 + vis.shape[1]] = vis
        cv2.rectangle(box, (0, 0), (W - 1, TITLE + H - 1), DARK_GREEN, 2)
        return box

    A = tile(img_w,   "Preprocessed (stretch+CLAHE+flatten)")
    B = tile(accum,   "Coherence (full)")
    C = tile(overlay, "Overlay (white segments)")
    D = tile(lines_only, "Lines only (white)")

    h, w = A.shape[:2]
    panel = np.zeros((2*h + 3*MARGIN + SEP, 2*w + 3*MARGIN + SEP, 3), np.uint8)
    panel[:] = DARK_GREEN

    panel[MARGIN:MARGIN+h,                     MARGIN:MARGIN+w] = A
    panel[MARGIN:MARGIN+h,                     MARGIN+w+SEP:MARGIN+w+SEP+w] = B
    panel[MARGIN+h+SEP:MARGIN+h+SEP+h,         MARGIN:MARGIN+w] = C
    panel[MARGIN+h+SEP:MARGIN+h+SEP+h,         MARGIN+w+SEP:MARGIN+w+SEP+w] = D

    cv2.imwrite(str(OUT_DIR/"panel"/f"{name}_panel.png"), panel)


# =========================
# Segment extraction (reutilizado, com knobs)
# =========================
def extract_segments_from_seed(seed, coh, ori, halo, scale,
                               area_thresh,
                               hough_local_scans,
                               support_band_px,
                               support_min_density,
                               support_min_mean_coh,
                               support_max_odisp,
                               ori_tol_valid,
                               border_reject_frac,
                               profile_min_len):
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accum_valid = np.zeros_like(seed)
    segments = []
    rejected90 = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < (area_thresh*scale*scale if scale<1.0 else area_thresh): 
            continue
        mask = np.zeros_like(seed); cv2.drawContours(mask,[cnt],-1,255,cv2.FILLED)
        loc_mask, loc_lines = houghp_fuse(mask, hough_local_scans)
        if loc_lines is None: 
            continue
        L = loc_lines
        ang = (np.degrees(np.arctan2(L[:,3]-L[:,1], L[:,2]-L[:,0]))%180.0).reshape(-1,1)
        labels = DBSCAN(eps=5, min_samples=1).fit(ang).labels_
        uniq = np.unique(labels)
        if len(uniq) >= 5:  # muito bagunçado
            continue
        for lb in uniq:
            sel = L[labels==lb]
            if len(sel)==0: continue
            cl_mask = np.zeros_like(mask)
            for x1,y1,x2,y2 in sel: cv2.line(cl_mask,(x1,y1),(x2,y2),255,1)
            cl_cnts,_ = cv2.findContours(cl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c2 in cl_cnts:
                if cv2.contourArea(c2) < (area_thresh*scale*scale if scale<1.0 else area_thresh):
                    continue
                cm = np.zeros_like(mask); cv2.drawContours(cm,[c2],-1,255,cv2.FILLED)
                f = []
                for x1,y1,x2,y2 in sel:
                    if cm[y1,x1]>0 and cm[y2,x2]>0: f.append([x1,y1,x2,y2])
                if not f: continue
                f = np.array(f)
                p1,p2 = farthest_endpoints(f)
                if REJECT_EXACT_VERTICAL and (p1[0] == p2[0]):
                    rejected90 += 1
                    continue
                dx = p2[0]-p1[0]; dy=p2[1]-p1[1]
                th = math.atan2(dy, dx) + np.pi/2.0
                rho = p1[0]*math.cos(th) + p1[1]*math.sin(th)
                ok,hits,stats = validate_line(
                    seed, coh, ori, rho, th,
                    halo=halo, band=support_band_px,
                    min_density=support_min_density,
                    min_mean_coh=support_min_mean_coh,
                    max_odisp=support_max_odisp,
                    ori_tol=ori_tol_valid,
                    border_reject_frac=border_reject_frac,
                    border_margin=BORDER_MARGIN_PX
                )
                if not ok: 
                    continue
                dens = stats["density"]
                max_gap = max(10, int(round(PROFILE_MAX_GAP_BASE * (1.0 + 0.7*(0.12 - min(0.12, dens))/0.12))))
                a,b = trim_to_support(p1,p2,hits,max_gap=max_gap,min_len=profile_min_len)
                segments.append((a,b))
                accum_valid |= hits
    return segments, accum_valid, rejected90

# =========================
# PROCESSO POR PATCH (usando dirs_full)
# =========================
def process_patch(edges, coh, ori, halo, dirs_full, scale):
    # knobs modo patch
    area_thresh=AREA_THRESHOLD_PATCH
    min_comp=MIN_COMP_SIZE_PATCH
    profiles=PROFILES_PATCH
    scans_global=HOUGHP_GLOBAL_SCAN_PATCH
    scans_local =HOUGHP_LOCAL_SCAN_PATCH

    accum_all = np.zeros_like(edges)
    segments_all = []
    killed90_total = 0
    ramp_log = []

    # usa só dirs_full; se nada nascer, tenta 2 modos locais
    tried_dirs = list(dirs_full)

    # fallback (até 2 modos locais)
    if len(tried_dirs)==0:
        th_local=[]
    else:
        th_local=[]
    # threshold por patch (variável por perfil)
    for th_deg in tried_dirs:
        found=False
        for prof in profiles:
            thr = max(prof["COH_THR_ABS"], np.percentile(coh, prof["COH_THR_PERC"]))
            delta = np.vectorize(ang_diff_deg)(ori, th_deg).astype(np.float32)
            gate  = (delta <= prof["ORI_TOL_DEG"]).astype(np.uint8)*255
            cmk   = (coh >= thr).astype(np.uint8)*255
            ed = cv2.bitwise_and(edges, cv2.bitwise_and(cmk, gate))
            if np.count_nonzero(ed)==0:
                continue
            ed = oriented_close_one(ed, th_deg, length=prof["DIR_CLOSE_LEN"])
            hmask,_ = houghp_fuse(ed, scans_global)
            seed = cv2.bitwise_or(ed, hmask)
            seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((CLOSE_K_SZ,CLOSE_K_SZ), np.uint8))
            seed = remove_small_components(seed, min_size=int(round(min_comp*scale)) if scale<1.0 else min_comp)

            segs, accum_valid, killed90 = extract_segments_from_seed(
                seed, coh, ori, halo, scale,
                area_thresh=area_thresh,
                hough_local_scans=scans_local,
                support_band_px=SUPPORT_BAND_PX_PATCH,
                support_min_density=SUPPORT_MIN_DENSITY_PATCH,
                support_min_mean_coh=SUPPORT_MIN_MEAN_COH_PATCH,
                support_max_odisp=SUPPORT_MAX_ODISP_PATCH,
                ori_tol_valid=ORI_TOL_VALID_PATCH,
                border_reject_frac=BORDER_REJECT_FRAC_PATCH,
                profile_min_len=PROFILE_MIN_LEN_PATCH
            )
            killed90_total += killed90
            if segs:
                segments_all.extend(segs); accum_all |= accum_valid
                ramp_log.append(f"{th_deg:.1f}°→{prof['LABEL']}")
                found=True
                break
        if not found:
            ramp_log.append(f"{th_deg:.1f}°→none")

    # fallback local (se nada passou)
    if not segments_all:
        # detecta dois picos locais simples
        cmk0 = (coh >= max(0.06, np.percentile(coh, 98.0))).astype(np.uint8)*255
        hist_dirs = find_orientation_modes(ori, cmk0, k=2, sep=12.0)
        for th_deg in hist_dirs:
            thr = max(0.10, np.percentile(coh, 98.8))
            delta = np.vectorize(ang_diff_deg)(ori, th_deg).astype(np.float32)
            gate  = (delta <= 22.0).astype(np.uint8)*255
            cmk   = (coh >= thr).astype(np.uint8)*255
            ed = cv2.bitwise_and(edges, cv2.bitwise_and(cmk, gate))
            if np.count_nonzero(ed)==0: continue
            ed = oriented_close_one(ed, th_deg, length=45)
            hmask,_ = houghp_fuse(ed, scans_global)
            seed = cv2.bitwise_or(ed, hmask)
            seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((CLOSE_K_SZ,CLOSE_K_SZ), np.uint8))
            seed = remove_small_components(seed, min_size=int(round(min_comp*scale)) if scale<1.0 else min_comp)
            segs, accum_valid, killed90 = extract_segments_from_seed(
                seed, coh, ori, halo, scale,
                area_thresh=area_thresh, hough_local_scans=scans_local,
                support_band_px=SUPPORT_BAND_PX_PATCH,
                support_min_density=SUPPORT_MIN_DENSITY_PATCH,
                support_min_mean_coh=SUPPORT_MIN_MEAN_COH_PATCH,
                support_max_odisp=SUPPORT_MAX_ODISP_PATCH,
                ori_tol_valid=ORI_TOL_VALID_PATCH,
                border_reject_frac=BORDER_REJECT_FRAC_PATCH,
                profile_min_len=PROFILE_MIN_LEN_PATCH
            )
            killed90_total += killed90
            if segs:
                segments_all.extend(segs); accum_all |= accum_valid
                ramp_log.append(f"fallback {th_deg:.1f}°")
                break

    debug_maps = dict()
    debug_maps['accum']=accum_all
    return segments_all, debug_maps, ramp_log, killed90_total

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for img_path in list_inputs():
        name = img_path.stem
        print(f"[+] {img_path}")

        img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img0 is None: print("   (skip)"); continue
        H0, W0 = img0.shape
        scale = min(1.0, PROC_MAX_SIDE / float(max(H0, W0)))
        img = cv2.resize(img0, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_AREA) if scale<1.0 else img0.copy()

        # 0) pré-filtro rápido + remoção coluna finíssima
        base_det = percentile_stretch(img, PERC_LOW, PERC_HIGH)
        base_det = apply_clahe(base_det, CLAHE_CLIP, CLAHE_TILE)
        base_det = flatten_bg(base_det, BG_MED_K)
        edges_det= auto_canny(base_det, CANNY_SIGMA)
        art_mask = detect_thin_vertical_artifacts(base_det, edges_det)
        if np.count_nonzero(art_mask) > 0:
            img = cv2.inpaint(img, art_mask, 3, cv2.INPAINT_TELEA)

        # 1) Preprocess definitivo (FULL)
        base_full = percentile_stretch(img, PERC_LOW, PERC_HIGH)
        base_full = apply_clahe(base_full, CLAHE_CLIP, CLAHE_TILE)
        base_full = flatten_bg(base_full, BG_MED_K)
        edges_full= auto_canny(base_full, CANNY_SIGMA)

        bm_hard_full = border_mask(base_full.shape, int(round(BORDER_HARD_ZERO_PX*scale)) if scale<1.0 else BORDER_HARD_ZERO_PX)
        halo_full    = bright_halo_mask(base_full, q=HALO_Q, grow=int(round(HALO_GROW*scale)) if scale<1.0 else HALO_GROW)

        # 2) Coerência/ori (FULL) + dirs prior
        p0 = PROFILES_FULL[0]
        coh_full, ori_full = coherence_map(base_full, p0["COH_WIN"])
        thr_full = max(p0["COH_THR_ABS"], np.percentile(coh_full, p0["COH_THR_PERC"]))
        cmk_full = (coh_full >= thr_full).astype(np.uint8)*255
        cmk_full[bm_hard_full>0] = 0; cmk_full[halo_full>0] = 0

        dirs_full = find_orientation_modes(ori_full, cmk_full, k=3, sep=12.0)
        if not dirs_full:
            # fallback simples pelo Hough do full
            h_, ang_, rho_ = hough_line(edges_full>0)
            _, ths, _ = hough_line_peaks(h_, ang_, rho_, num_peaks=3, min_distance=10, min_angle=4)
            dirs_full = [float((np.degrees(t)%180.0)) for t in ths]
        print(f"   dirs_full={['%.1f' % d for d in dirs_full]}")

        # 3) Fatiar mapas (coh/ori/edges/halo) em grade 47×47 (224px)
        base_full_c, R, C = crop_to_grid(base_full, PATCH_SIZE)
        edges_full_c,_,_   = crop_to_grid(edges_full, PATCH_SIZE)
        halo_full_c,_,_    = crop_to_grid(halo_full, PATCH_SIZE)
        coh_full_c,_,_     = crop_to_grid(coh_full, PATCH_SIZE)
        ori_full_c,_,_     = crop_to_grid(ori_full, PATCH_SIZE)

        print(f"   grid detectado: {R} x {C} (patch={PATCH_SIZE})")

        edges_tiles = to_tiles(edges_full_c, R, C, PATCH_SIZE)
        coh_tiles   = to_tiles(coh_full_c,   R, C, PATCH_SIZE)
        ori_tiles   = to_tiles(ori_full_c,   R, C, PATCH_SIZE)
        halo_tiles  = to_tiles(halo_full_c,  R, C, PATCH_SIZE)
        base_tiles  = to_tiles(base_full_c,  R, C, PATCH_SIZE)

        # 4) Hough por patch (usando dirs_full)
        inv_tiles = []            # linhas brancas
        accum_tiles = []          # validação hits
        ed_tiles = []             # opcional (vazio aqui)
        seed_tiles = []
        hmask_tiles = []
        coh_u8_tiles = [(255*np.clip(t,0,1)).astype(np.uint8) for t in coh_tiles]

        total_segments=0; killed90_total=0
        saved_panels=0

        for i in range(R*C):
            r, c = divmod(i, C)
            edges = edges_tiles[i]
            coh   = coh_tiles[i]
            ori   = ori_tiles[i]
            halo  = halo_tiles[i]

            # borda "hard zero" leve no patch (opcional desativar)
            # (no modo patch a rejeição por borda é desligada na validação)
            # bm = border_mask(edges.shape, 2); edges = cv2.bitwise_and(edges, cv2.bitwise_not(bm))

            segs, dbg, ramp, killed90 = process_patch(edges, coh, ori, halo, dirs_full, scale=1.0)
            killed90_total += killed90
            total_segments += len(segs)

            # desenha linhas no tile
            lines_tile = np.zeros_like(edges, np.uint8)
            for a,b in segs:
                cv2.line(lines_tile, a, b, 255, 2, cv2.LINE_8)
            inv_tiles.append(lines_tile)
            accum_tiles.append(dbg['accum'])

            # debug por patch (opcional)
            if SAVE_PER_PATCH_PANELS and saved_panels < MAX_PATCH_PANELS and len(segs)>0:
                vis = cv2.cvtColor(base_tiles[i], cv2.COLOR_GRAY2BGR)
                for a,b in segs: cv2.line(vis, a, b, (255,255,255), 2, cv2.LINE_8)
                cv2.imwrite(str(OUT_DIR/f"debug/patch/{name}_r{r:02d}_c{c:02d}_overlay.png"), vis)
                cv2.imwrite(str(OUT_DIR/f"debug/patch/{name}_r{r:02d}_c{c:02d}_lines.png"), lines_tile)
                saved_panels += 1

            print(f"   patch {r:02d},{c:02d} | segs={len(segs)} | ramp={','.join(ramp) if ramp else 'n/a'}")

        # 5) Mosaicos e overlay
        save_mosaic(coh_u8_tiles, (R,C), OUT_DIR/"debug"/f"{name}_coh_mosaic.png")
        save_mosaic(accum_tiles,  (R,C), OUT_DIR/"debug"/f"{name}_accum_mosaic.png")
        save_mosaic(inv_tiles,    (R,C), OUT_DIR/"lines"/f"{name}_inverse_mosaic.png")

        # reconstruir full "lines-only" e overlay
        Hc, Wc = R*PATCH_SIZE, C*PATCH_SIZE
        lines_full = cv2.imread(str(OUT_DIR/"lines"/f"{name}_inverse_mosaic.png"), cv2.IMREAD_GRAYSCALE)
        lines_full = lines_full if lines_full is not None else np.zeros((Hc,Wc), np.uint8)

        overlay = cv2.cvtColor(base_full_c, cv2.COLOR_GRAY2BGR)
        overlay[lines_full>0] = (255,255,255)

        # painel
        draw_panel(base_full_c, (255*np.clip(coh_full_c,0,1)).astype(np.uint8), overlay, lines_full, name)

        cv2.imwrite(str(OUT_DIR/"overlay"/f"{name}_overlay.png"), overlay)
        cv2.imwrite(str(OUT_DIR/"lines"/f"{name}_lines.png"),   lines_full)

        print(f"   segments(total): {total_segments} | killed_exact_90: {killed90_total}")
        print(f"   debug: coh_mosaic / accum_mosaic | lines: inverse_mosaic + full")
