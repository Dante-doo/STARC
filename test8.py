#!/usr/bin/env python3
# test.py
# pip install opencv-python scikit-image numpy

from pathlib import Path
import re, math
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks

# =================== Paths ===================
RAW_DIR    = Path("raw")
ACC_DIR    = Path("hough/accumulator_points")
INV_DIR    = Path("hough/inversed")
LINES_DIR  = Path("hough/lines_only")
PANEL_DIR  = Path("hough/preview")
DBG_DIR    = Path("hough/debug")
for d in (ACC_DIR, INV_DIR, LINES_DIR, PANEL_DIR, DBG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# PNG only
files = []
for ext in ("png","PNG"):
    files.extend(RAW_DIR.glob(f"full*.{ext}"))

def num_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0

# =================== Hyperparams (base) ===================
# Downscale para acelerar (resto reescala automaticamente)
PROC_MAX_SIDE = 2048  # lado máx. no processamento

# Normalização
PERC_LOW, PERC_HIGH    = 0.5, 99.9
CLAHE_CLIP, CLAHE_TILE = 2.5, (16,16)

# Weak branch
STAR_Q, STAR_DILATE    = 99.85, 7
MEDIAN_K, BG_K         = 3, 31

# Small-star removal
SMALL_STAR_Q       = 99.0
SMALL_STAR_MAX_A   = 30      # px^2 (reescalado)
SMALL_STAR_DILATE  = 1

# Big-star exclusion (halos) → EXCLUIR do processamento
BIG_STAR_Q         = 99.85
BIG_STAR_MIN_A     = 400     # px^2 (reescalado)
BIG_STAR_GROW_PX   = 25      # px (reescalado)

# Banco de linhas
LINE_KLEN, LINE_THICK = 31, 1
LINE_ANGLES           = list(range(0,180,5))
# Strong extras
LINE_THICK_STRONG   = 4
LINE_ANGLES_STRONG  = list(range(0,180,3))
RESP_PERC_WEAK      = 99.4
RESP_PERC_STRONG    = 98.0

# Bordas & morfologia
CANNY_SIGMA = 0.33
OPEN_K, CLOSE_K, GATE_K = 3, 5, 5

# Fechamento direcional
USE_DIR_CLOSE   = True
DIR_CLOSE_LEN   = 45
DIR_CLOSE_STEP  = 10
DIR_CLOSE_LEN_STRONG = 61

# Hough & desenho
NUM_PEAKS, MIN_DIST, MIN_ANG = 400, 20, 8
DOT_RADIUS, DRAW_THICK = 3, 3   # linhas brancas 255

# Filtro de suporte geométrico (pós-Hough)
SUPPORT_BAND      = 4
SUPPORT_MIN_RATIO = 0.04
SUPPORT_MIN_ABS   = 900
MAX_LINES_TO_CHECK= 120

# Segmento finito (não reta infinita)
SEG_BAND            = 3
SEG_MIN_LEN         = 50
SEG_DILATE_ALONG    = 5
SEG_ENDPOINT_RADIUS = 5

# --- Refinos locais e perfil 1-D (NOVO) ---
ANGLE_SWEEP_DEG   = 2.4   # varredura em torno do theta do Hough
ANGLE_STEP_DEG    = 0.2
ORI_TOL_DEG       = 14.0  # tolerância de orientação local
PROFILE_BAND_MULT = 1.0   # multiplica o SEG_BAND_S
PROFILE_WIN       = 33    # suavização 1-D ao longo da linha
PROFILE_MIN_DENS  = 0.18  # fração mínima por seção transversal
PROFILE_MAX_GAP   = 16   # hiatos tolerados (px)
MIN_MEAN_COH      = 0.22  # coerência média mínima

# Preview layout
TILE_W, CONTENT_H, SEP, SEP_COLOR, TITLE_H = 1200, 900, 10, (45,45,45), 56

# Artefatos verticais (colunas 1px brancas/pretas sem nuance)
VERT_TOL        = 2
VERT_MIN_RUN_R  = 0.07   # fração da altura
VERT_MAX_WIDTH  = 1

# =================== Utils ===================
def percentile_stretch(u8, p_low=1.0, p_high=99.8):
    lo, hi = np.percentile(u8, [p_low, p_high])
    if hi <= lo: return u8
    out = np.clip((u8.astype(np.float32)-lo)*(255.0/(hi-lo)), 0, 255)
    return out.astype(np.uint8)

def apply_clahe(u8, clip=2.0, tile=(8,8)):
    if clip <= 0: return u8
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile)).apply(u8)

def auto_canny(u8, sigma=0.33):
    v = float(np.median(u8))
    lo = int(max(0,(1.0-sigma)*v)); hi = int(min(255,(1.0+sigma)*v))
    if hi <= lo: hi = lo + 1
    return cv2.Canny(u8, lo, hi, L2gradient=True)

def flatten_bg(u8, k=31):
    if k<=1 or k%2==0: return u8
    bg = cv2.medianBlur(u8, k)
    x  = cv2.subtract(u8, bg)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def inpaint_stars(u8, q=99.8, dilate_r=5):
    th = np.percentile(u8, q)
    m = (u8 >= th).astype(np.uint8)*255
    if dilate_r>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate_r+1,2*dilate_r+1))
        m = cv2.dilate(m,k)
    return cv2.inpaint(u8, m, 3, cv2.INPAINT_TELEA), m

def remove_small_stars_inpaint(u8, q, max_a, dilate_px):
    th = np.percentile(u8, q)
    bin_ = cv2.threshold(u8, th, 255, cv2.THRESH_BINARY)[1]
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    mask = np.zeros_like(u8, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= max_a:
            mask[lab==i] = 255
    if dilate_px>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate_px+1,2*dilate_px+1))
        mask = cv2.dilate(mask, k, 1)
    out = cv2.inpaint(u8, mask, 3, cv2.INPAINT_TELEA)
    return out, mask

def detect_big_stars(u8, q, min_a, grow):
    th = np.percentile(u8, q)
    bin_ = cv2.threshold(u8, th, 255, cv2.THRESH_BINARY)[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    mask = np.zeros_like(u8, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_a:
            mask[lab==i] = 255
    if grow>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grow+1,2*grow+1))
        mask = cv2.dilate(mask, k, 1)
    return mask

def detect_vertical_artifacts_strict(u8, tol=VERT_TOL, min_run_r=VERT_MIN_RUN_R, max_width=VERT_MAX_WIDTH):
    """Detecta segmentos verticais (1 px) totalmente brancos ou pretos (sem nuance)."""
    H, W = u8.shape
    min_run = max(16, int(min_run_r * H))

    white = (u8 >= (255 - tol)).astype(np.uint8)
    black = (u8 <= tol).astype(np.uint8)

    def width1(b):
        left = np.zeros_like(b);  left[:,1:]  = b[:,:-1]
        right= np.zeros_like(b);  right[:,:-1]= b[:,1:]
        return (b & (~left) & (~right)).astype(np.uint8)

    w1 = width1(white)
    b1 = width1(black)

    vker = cv2.getStructuringElement(cv2.MORPH_RECT,(1, min_run))
    w_long = cv2.morphologyEx(w1*255, cv2.MORPH_OPEN, vker)
    b_long = cv2.morphologyEx(b1*255, cv2.MORPH_OPEN, vker)

    mask = cv2.bitwise_or(w_long, b_long)

    if max_width > 1:
        hker = cv2.getStructuringElement(cv2.MORPH_RECT,(max_width+1,1))
        too_wide = cv2.morphologyEx((mask>0).astype(np.uint8)*255, cv2.MORPH_OPEN, hker)
        mask[too_wide>0] = 0
    return mask

def line_bank_response(u8, klen=31, thk=1, angles_deg=tuple(range(0,180,5))):
    ksz = int(klen)|1
    resp_max = np.full(u8.shape, -1e9, dtype=np.float32)
    for ang in angles_deg:
        ker = np.zeros((ksz, ksz), np.float32)
        c = ksz//2
        cv2.line(ker, (0,c), (ksz-1,c), 1.0, thk)
        M = cv2.getRotationMatrix2D((c,c), float(ang), 1.0)
        ker = cv2.warpAffine(ker, M, (ksz, ksz), flags=cv2.INTER_NEAREST)
        ker -= ker.mean()
        n = np.linalg.norm(ker)
        if n < 1e-8: 
            continue
        ker /= n
        r = cv2.filter2D(u8.astype(np.float32), -1, ker, borderType=cv2.BORDER_REFLECT)
        resp_max = np.maximum(resp_max, r)
    rmin, rmax = float(resp_max.min()), float(resp_max.max())
    if rmax <= rmin: 
        return np.zeros_like(u8)
    out = (resp_max - rmin) * (255.0/(rmax-rmin))
    return out.astype(np.uint8)

def tile_box(img, title, tile_w=TILE_W, content_h=CONTENT_H, title_h=TITLE_H):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
    h, w = vis.shape[:2]
    s = min(tile_w / w, content_h / h)
    new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_AREA if new_w < w or new_h < h else cv2.INTER_NEAREST
    vis = cv2.resize(vis, (new_w, new_h), interpolation=interp)
    tile = np.zeros((title_h+content_h, tile_w, 3), np.uint8)
    tile[0:title_h,:] = (30,30,30)
    cv2.putText(tile, title, (16,title_h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255), 2, cv2.LINE_8)
    y0 = title_h + (content_h - vis.shape[0])//2;  x0 = (tile_w - vis.shape[1])//2
    tile[y0:y0+vis.shape[0], x0:x0+vis.shape[1]] = vis
    return tile

def stack_grid(tiles, rows, cols, sep=SEP, color=SEP_COLOR, margin=SEP):
    assert len(tiles)==rows*cols
    h,w = tiles[0].shape[:2]
    H = rows*h + (rows+1)*margin + (rows-1)*sep
    W = cols*w + (cols+1)*margin + (cols-1)*sep
    canvas = np.zeros((H,W,3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            y = margin + r*(h+sep); x = margin + c*(w+sep)
            canvas[y:y+h, x:x+w] = tiles[r*cols+c]
            cv2.rectangle(canvas, (x,y), (x+w-1,y+h-1), color, 1, cv2.LINE_8)
    return canvas

def render_acc_preview(angles, dists, rho_peaks, theta_peaks, max_w=TILE_W, min_px=3):
    H = len(dists); W = len(angles)
    if H==0 or W==0:
        return np.zeros((CONTENT_H, TILE_W, 3), np.uint8)
    scale = max(1, int(min(max_w/W, (CONTENT_H)/H)))
    canvas = np.zeros((H*scale, W*scale), np.uint8)
    deg = np.degrees(angles)%180.0; ticks=[]
    for t in (0,15,30,45,60,75,90,105,120,135,150,165):
        idx = int(np.argmin(np.abs(deg - t))); ticks.append((idx,t))
    for idx,_ in ticks: canvas[:, idx*scale:idx*scale+1] = 60
    rad = max(min_px, int(0.002*max(H,W)*scale))
    for rho, theta in zip(rho_peaks, theta_peaks):
        r_idx = int(np.argmin(np.abs(dists - rho))); t_idx = int(np.argmin(np.abs(angles - theta)))
        cv2.circle(canvas, (t_idx*scale, r_idx*scale), rad, 255, -1, cv2.LINE_8)
    vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis, "theta (deg)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,200,200), 2, cv2.LINE_8)
    cv2.putText(vis, "rho", (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,200,200), 2, cv2.LINE_8)
    for idx,t in ticks: cv2.putText(vis, f"{int(t)}", (idx*scale+4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(220,220,220),1, cv2.LINE_8)
    return vis

def directional_close(img, length=31, step=10):
    ksz = int(length) | 1
    accum = np.zeros_like(img)
    for ang in range(0, 180, int(step)):
        ker = np.zeros((ksz, ksz), np.uint8)
        c = ksz // 2
        cv2.line(ker, (0, c), (ksz-1, c), 255, 1)
        M = cv2.getRotationMatrix2D((c, c), float(ang), 1.0)
        ker = cv2.warpAffine(ker, M, (ksz, ksz), flags=cv2.INTER_NEAREST)
        ker = (ker > 0).astype(np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker, iterations=1)
        accum = cv2.bitwise_or(accum, closed)
    return accum

def merge_peaks(rhoA, thA, accA, rhoB, thB, accB, deg_tol=1.0, rho_tol=15.0):
    rho = list(rhoA) + list(rhoB)
    th  = list(thA)  + list(thB)
    ac  = list(accA) + list(accB)
    order = np.argsort(ac)[::-1]
    kept_rho, kept_th = [], []
    for i in order:
        r, t = rho[i], th[i]
        ok = True
        for rr, tt in zip(kept_rho, kept_th):
            if abs(np.degrees(t-tt)) <= deg_tol and abs(r-rr) <= rho_tol:
                ok = False; break
        if ok:
            kept_rho.append(r); kept_th.append(t)
        if len(kept_rho) >= NUM_PEAKS: break
    return np.array(kept_rho), np.array(kept_th)

def filter_lines_by_support(edges, rhos, thetas, band, min_ratio, min_abs, max_keep=MAX_LINES_TO_CHECK):
    H, W = edges.shape
    diag = int(np.hypot(H, W))
    scores = []
    for i in range(min(len(rhos), max_keep)):
        rho, theta = rhos[i], thetas[i]
        mask = np.zeros((H,W), np.uint8)
        a, b = math.cos(theta), math.sin(theta)
        x0, y0 = a*rho, b*rho
        p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
        p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))
        cv2.line(mask, p1, p2, 255, 2*band+1, cv2.LINE_8)
        hits = cv2.countNonZero(cv2.bitwise_and(mask, edges))
        total = cv2.countNonZero(mask)
        ratio = 0 if total==0 else hits/float(total)
        score = hits * (1.0 + 0.5*ratio)
        if hits >= min_abs and ratio >= min_ratio:
            scores.append((score, rho, theta))
    if not scores:
        return np.array([]), np.array([])
    scores.sort(reverse=True)
    r_out = np.array([s[1] for s in scores])
    t_out = np.array([s[2] for s in scores])
    return r_out, t_out

def extract_segment_from_edges(edges, rho, theta, band, min_len, dilate_along=5, endpoint_r=5):
    H, W = edges.shape
    diag = int(np.hypot(H, W))
    a, b = math.cos(theta), math.sin(theta)
    x0, y0 = a*rho, b*rho
    p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
    p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))

    band_mask = np.zeros_like(edges, np.uint8)
    cv2.line(band_mask, p1, p2, 255, 2*band+1, cv2.LINE_8)

    ys, xs = np.where(band_mask>0)
    if xs.size == 0: 
        return None
    y0r, y1r = max(0, ys.min()-2), min(H, ys.max()+3)
    x0r, x1r = max(0, xs.min()-2), min(W, xs.max()+3)

    band_roi  = band_mask[y0r:y1r, x0r:x1r]
    edges_roi = edges[y0r:y1r, x0r:x1r]
    hits = cv2.bitwise_and(band_roi, edges_roi)

    if dilate_along>0:
        hits = cv2.dilate(hits, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)

    n, lab, stats, _ = cv2.connectedComponentsWithStats((hits>0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return None

    vx, vy = -math.sin(theta), math.cos(theta)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10: 
            continue
        ys_i, xs_i = np.where(lab==i)
        xs_i = xs_i + x0r; ys_i = ys_i + y0r
        t = xs_i*vx + ys_i*vy
        span = t.max() - t.min()
        if best is None or span > best[0]:
            best = (span, t.min(), t.max(), xs_i, ys_i)

    if best is None or best[0] < min_len:
        return None

    _, tmin, tmax, xs_i, ys_i = best
    t = xs_i*vx + ys_i*vy
    p_a_idx = np.argmin(np.abs(t - tmin))
    p_b_idx = np.argmin(np.abs(t - tmax))
    pA = (int(xs_i[p_a_idx]), int(ys_i[p_a_idx]))
    pB = (int(xs_i[p_b_idx]), int(ys_i[p_b_idx]))
    return pA, pB

# ============ Escalonadores ============
def scale_len(val, scale, odd=False, minv=1):
    k = max(minv, int(round(val*scale)))
    if odd: k |= 1
    return k

def scale_area(val, scale, minv=1):
    return max(minv, int(round(val*(scale*scale))))

# ---------- Métricas + Auto-tune + Poda ----------
def measure_scene(img_u8):
    H, W = img_u8.shape
    area = float(H*W)

    th = np.percentile(img_u8, 99.0)
    bin_ = cv2.threshold(img_u8, th, 255, cv2.THRESH_BINARY)[1]
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, 8)
    small = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= 25)
    small_star_density = small / area

    thb = np.percentile(img_u8, 99.8)
    binb = cv2.threshold(img_u8, thb, 255, cv2.THRESH_BINARY)[1]
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(binb, 8)
    big_area = sum(int(stats2[i, cv2.CC_STAT_AREA]) for i in range(1, n2)
                   if stats2[i, cv2.CC_STAT_AREA] >= 300)
    big_star_frac = big_area / area

    rough = cv2.medianBlur(img_u8, 3)
    rough = flatten_bg(rough, 31)
    ed = auto_canny(rough, 0.33)
    ed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    h, ang, rho = hough_line(ed>0)
    if h.size == 0 or np.all(h==0):
        peak_ratio = 0.0
    else:
        vmax = float(h.max())
        med  = float(np.median(h[h>0])) if np.count_nonzero(h) else 1.0
        peak_ratio = vmax / max(1.0, med)

    return dict(small_star_density=small_star_density,
                big_star_frac=big_star_frac,
                peak_ratio=peak_ratio)

def auto_tune(metrics, scale):
    m = metrics
    ov = {}

    star_heavy   = (m["small_star_density"] > 2.0e-4) or (m["big_star_frac"] > 0.015)
    strong_like  = (m["peak_ratio"] >= 12.0)
    weak_like    = (6.0 <= m["peak_ratio"] < 12.0)

    ov["GATE_K"] = 5
    ov["SUPPORT_BAND"] = max(3, int(round(4*scale)))
    ov["SUPPORT_MIN_RATIO"] = 0.04
    ov["SUPPORT_MIN_ABS"]   = max(200, int(round(900*scale)))
    ov["SEG_MIN_LEN"]       = max(20, int(round(50*scale)))
    ov["SEG_DILATE_ALONG"]  = 5
    ov["DIR_CLOSE_LEN"]         = max(31, int(round(45*scale)))
    ov["DIR_CLOSE_LEN_STRONG"]  = max(41, int(round(61*scale)))
    ov["RESP_PERC_WEAK"]   = 99.4
    ov["RESP_PERC_STRONG"] = 98.0
    ov["LINE_THICK_STRONG"]= 4
    ov["COH_THR"] = 0.18

    if star_heavy:
        ov.update({
            "SMALL_STAR_Q": 98.7,
            "SMALL_STAR_MAX_A": 25,
            "BIG_STAR_Q": 99.87,
            "BIG_STAR_MIN_A": 600,
            "BIG_STAR_GROW_PX": max(35, int(round(45*scale))),
            "GATE_K": 3,
            "RESP_PERC_WEAK": 99.6,
            "SUPPORT_MIN_RATIO": 0.06,
            "SUPPORT_MIN_ABS":  max(400, int(round(1400*scale))),
            "SEG_MIN_LEN":      max(40, int(round(70*scale))),
            "COH_THR": 0.26
        })

    if strong_like:
        ov.update({
            "RESP_PERC_STRONG": 97.5,
            "LINE_THICK_STRONG": 5,
            "DIR_CLOSE_LEN_STRONG": max(51, int(round(71*scale))),
            "SUPPORT_BAND": max(4, int(round(5*scale))),
            "SUPPORT_MIN_RATIO": 0.035,
            "SUPPORT_MIN_ABS":   max(200, int(round(800*scale))),
            "SEG_MIN_LEN":       max(30, int(round(60*scale))),
            "SEG_DILATE_ALONG":  6,
            "COH_THR": max(0.20, ov["COH_THR"])
        })
    elif weak_like and not star_heavy:
        ov.update({
            "RESP_PERC_WEAK": 98.9,
            "DIR_CLOSE_LEN":  max(41, int(round(55*scale))),
            "CLOSE_K":        max(5, int(round(7*scale))),
            "SUPPORT_MIN_ABS":max(200, int(round(700*scale))),
            "SUPPORT_MIN_RATIO": 0.035,
            "SEG_MIN_LEN":    max(25, int(round(45*scale))),
            "SEG_DILATE_ALONG": 6,
            "COH_THR": 0.20
        })

    ov["NO_TRAIL_GUARD"] = (m["peak_ratio"] < 5.0 and star_heavy)
    return ov

def prune_by_theta_bins(rhos, thetas, edges, band=3, bin_w_deg=1.5, keep_bins=2):
    if len(rhos)==0: return np.array([]), np.array([]), []
    deg = (np.degrees(thetas) % 180.0)
    bins = np.arange(0, 180+bin_w_deg, bin_w_deg)
    which = np.digitize(deg, bins) - 1
    H, W = edges.shape
    diag = int(np.hypot(H, W))
    hits = []
    for rho, theta in zip(rhos, thetas):
        a, b = math.cos(theta), math.sin(theta)
        x0, y0 = a*rho, b*rho
        p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
        p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))
        m = np.zeros((H,W), np.uint8)
        cv2.line(m, p1, p2, 255, 2*band+1, cv2.LINE_8)
        hits.append(cv2.countNonZero(cv2.bitwise_and(m, edges)))
    hits = np.array(hits)

    bin_sum = {}
    deg_bins = np.digitize(deg, bins) - 1
    for i, b in enumerate(deg_bins):
        bin_sum[b] = bin_sum.get(b, 0) + hits[i]
    keep = sorted(bin_sum.keys(), key=lambda k: bin_sum[k], reverse=True)[:max(1,keep_bins)]
    idx = [i for i,b in enumerate(deg_bins) if b in keep]
    return rhos[idx], thetas[idx], hits[idx].tolist()

def reject_1px_vertical_segments(lines_pts, gray_img, tol=2, min_run=0.07):
    H = gray_img.shape[0]
    out = []
    for p1, p2 in lines_pts:
        x1,y1 = p1; x2,y2 = p2
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dx <= 1 and dy >= int(min_run*H):
            x = x1
            col = gray_img[min(y1,y2):max(y1,y2)+1, max(0,min(x,gray_img.shape[1]-1))]
            if col.size>0 and (np.all(col <= tol) or np.all(col >= 255-tol)):
                continue
        out.append((p1,p2))
    return out

def merge_colinear_segments(segs, ang_tol_deg=1.2, gap_max=40):
    if len(segs)<2: return segs
    def angle(p1,p2):
        return (np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))%180.0)
    def dist(a,b):
        return float(np.hypot(a[0]-b[0], a[1]-b[1]))
    used=[False]*len(segs); out=[]
    for i,(p1a,p2a) in enumerate(segs):
        if used[i]: continue
        Ai=angle(p1a,p2a); P=[p1a,p2a]; used[i]=True
        changed=True
        while changed:
            changed=False
            for j,(p1b,p2b) in enumerate(segs):
                if used[j]: continue
                Aj=angle(p1b,p2b)
                if min(abs(Ai-Aj), 180-abs(Ai-Aj))<=ang_tol_deg:
                    dd=min(dist(P[0],p1b),dist(P[0],p2b),dist(P[1],p1b),dist(P[1],p2b))
                    if dd<=gap_max:
                        C=[P[0],P[1],p1b,p2b]
                        best=None; bestd=-1.0
                        for a in range(4):
                            for b in range(a+1,4):
                                d=dist(C[a],C[b])
                                if d>bestd: bestd=d; best=(C[a],C[b])
                        P=[best[0],best[1]]; used[j]=True; changed=True
        out.append((P[0],P[1]))
    return out

# --------- Coherence gate (agora retorna coh_map também) ---------
def coherence_gate(u8, win=9, thr=0.20):
    f = u8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    k = win | 1
    Axx = cv2.GaussianBlur(gx*gx, (k,k), 0)
    Axy = cv2.GaussianBlur(gx*gy, (k,k), 0)
    Ayy = cv2.GaussianBlur(gy*gy, (k,k), 0)
    tmp = (Axx - Ayy)**2 + 4.0*(Axy**2)
    tmp = np.sqrt(np.maximum(tmp, 0))
    l1 = 0.5*((Axx + Ayy) + tmp)
    l2 = 0.5*((Axx + Ayy) - tmp)
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    ori = 0.5*np.degrees(np.arctan2(2*Axy, (Axx - Ayy))) % 180.0
    m = (coh >= float(thr)).astype(np.uint8)*255
    vis = (np.clip(coh,0,1)*255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    return m, vis, ori.astype(np.float32), coh.astype(np.float32)

# --------- Auxiliares NOVAS para o refino e o perfil ---------
def norm01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn: 
        return np.zeros_like(x, np.float32)
    return (x - mn) / (mx - mn)

def build_weight_map(edges_u8, coh_map, resp_map):
    m = (edges_u8 > 0).astype(np.float32)
    c = np.clip(coh_map, 0, 1).astype(np.float32)
    r = norm01(resp_map)
    w = m * (0.2 + 0.4*c + 0.4*r)
    return w

def weighted_rho_hist(xs, ys, weights, theta, est_rho=None, search=50, bin_size=1.0):
    rho_vals = xs*np.cos(theta) + ys*np.sin(theta)
    w = weights
    if est_rho is not None:
        sel = np.abs(rho_vals - est_rho) <= float(search)
        rho_vals = rho_vals[sel]
        w = w[sel]
        if rho_vals.size < 30:
            return est_rho, None
    if rho_vals.size < 20:
        return est_rho, None
    rmin, rmax = float(rho_vals.min()), float(rho_vals.max())
    nbins = max(50, int(np.ceil((rmax - rmin) / bin_size)))
    hist, edges = np.histogram(rho_vals, bins=nbins, range=(rmin, rmax), weights=w)
    k = int(np.argmax(hist))
    rho_peak = 0.5 * (edges[k] + edges[k+1])
    return rho_peak, hist.max()

def refine_theta_rho_local(edges_u8, coh_map, resp_map, theta_hint, ori_deg,
                           angle_sweep=np.radians(ANGLE_SWEEP_DEG),
                           angle_step=np.radians(ANGLE_STEP_DEG),
                           ori_tol_deg=ORI_TOL_DEG):
    ys, xs = np.where(edges_u8 > 0)
    if xs.size < 50:
        return None, None
    # gate por orientação local (opcional)
    if ori_deg is not None:
        ori_loc = ori_deg[ys, xs]
        d = np.abs((ori_loc - np.degrees(theta_hint) + 90.0) % 180.0 - 90.0)
        keep = d <= ori_tol_deg
        if keep.sum() >= 30:
            xs, ys = xs[keep], ys[keep]
    w_all = build_weight_map(edges_u8, coh_map, resp_map)
    weights = w_all[ys, xs]

    best = (None, None, -1.0)
    for th in np.arange(theta_hint - angle_sweep, theta_hint + angle_sweep + 1e-6, angle_step):
        rho_peak, peak_val = weighted_rho_hist(xs, ys, weights, th, est_rho=None, bin_size=1.0)
        if rho_peak is None:
            continue
        # continuidade aproximada num band estreito
        band = 2
        H, W = edges_u8.shape
        diag = int(np.hypot(H, W))
        a, b = np.cos(th), np.sin(th)
        x0, y0 = a*rho_peak, b*rho_peak
        p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
        p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))
        band_mask = np.zeros_like(edges_u8, np.uint8)
        cv2.line(band_mask, p1, p2, 255, 2*band+1, cv2.LINE_8)
        hits = cv2.countNonZero(cv2.bitwise_and(band_mask, edges_u8))
        score = float(peak_val if peak_val is not None else 0.0) + 0.5*hits
        if score > best[2]:
            best = (th, rho_peak, score)
    return best[0], best[1]

def extract_segment_by_profile(edges_u8, rho, theta, band_px, win=PROFILE_WIN,
                               min_dens=PROFILE_MIN_DENS, max_gap=PROFILE_MAX_GAP):
    H, W = edges_u8.shape
    diag = int(np.hypot(H, W))
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho
    p1 = (int(x0 + diag*(-b)), int(y0 + diag*(a)))
    p2 = (int(x0 - diag*(-b)), int(y0 - diag*(a)))

    band_mask = np.zeros_like(edges_u8, np.uint8)
    cv2.line(band_mask, p1, p2, 255, 2*band_px+1, cv2.LINE_8)
    M = cv2.bitwise_and(band_mask, edges_u8)

    vx, vy = -np.sin(theta), np.cos(theta)
    ys, xs = np.where(M > 0)
    if xs.size < 20:
        return None
    t = xs*vx + ys*vy
    tmin, tmax = float(t.min()), float(t.max())
    nbins = max(200, int((tmax - tmin) / 1.0))
    hist, edges = np.histogram(t, bins=nbins, range=(tmin, tmax))

    k = max(3, int(win)|1)
    ker = np.ones(k, np.float32)/float(k)
    prof = np.convolve(hist.astype(np.float32), ker, mode='same')
    prof_norm = prof / float(max(1, 2*band_px+1))
    thr = float(min_dens)
    on = (prof_norm >= thr).astype(np.uint8)

    gap = 0; best=(0, -1, -1)
    i0 = None
    for i, v in enumerate(on):
        if v:
            if i0 is None: i0 = i
            gap = 0
        else:
            if i0 is not None:
                gap += 1
                if gap > max_gap:
                    L = i - gap - i0
                    if L > best[0]:
                        best = (L, i0, i-gap)
                    i0 = None; gap = 0
    if i0 is not None:
        L = len(on) - i0
        if L > best[0]:
            best = (L, i0, len(on)-1)
    if best[0] <= 0:
        return None

    tA = 0.5*(edges[best[1]] + edges[best[1]+1])
    tB = 0.5*(edges[best[2]] + edges[best[2]+1])

    def clamp_pt(x, y):
        return (int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1)))
    pA = clamp_pt(x0 + tA*vx, y0 + tA*vy)
    pB = clamp_pt(x0 + tB*vx, y0 + tB*vy)
    if pA == pB:
        return None
    return pA, pB

def fit_line_ransac_weighted(edges_u8, coh_map, resp_map, dist_th=3.0, iters=400):
    ys, xs = np.where(edges_u8 > 0)
    if xs.size < 50: 
        return None, None
    w = build_weight_map(edges_u8, coh_map, resp_map)[ys, xs]
    ps = w / (w.sum() + 1e-6)
    best = (None, None, -1.0)
    H, W = edges_u8.shape
    for _ in range(iters):
        i, j = np.random.choice(len(xs), size=2, replace=False, p=ps)
        x1,y1 = xs[i], ys[i]; x2,y2 = xs[j], ys[j]
        if x1==x2 and y1==y2: 
            continue
        a = y1 - y2; b = x2 - x1; c = x1*y2 - x2*y1
        theta = np.arctan2(b, a) % np.pi
        rho = (x1*np.cos(theta) + y1*np.sin(theta))
        d = np.abs(a*xs + b*ys + c) / (np.hypot(a,b) + 1e-6)
        inl = d <= dist_th
        score = float((w[inl]).sum())
        if score > best[2]:
            best = (theta, rho, score)
    return best[0], best[1]

def ang_diff_deg(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def dominant_angle(thetas_rad, weights=None):
    if len(thetas_rad) == 0:
        return None
    th = np.degrees(np.asarray(thetas_rad)) % 180.0
    if weights is None:
        weights = np.ones_like(th)
    bins = np.arange(0, 180+1.0, 1.0)
    hist, _ = np.histogram(th, bins=bins, weights=weights)
    idx = np.argmax(hist)
    th_star_deg = 0.5 * (bins[idx] + bins[idx+1])
    return np.radians(th_star_deg)

def oriented_close_one(img_u8, theta_deg, length=61):
    ksz = int(length) | 1
    ker = np.zeros((ksz, ksz), np.uint8)
    c = ksz // 2
    cv2.line(ker, (0, c), (ksz-1, c), 255, 1)
    M = cv2.getRotationMatrix2D((c, c), float(theta_deg), 1.0)
    ker = cv2.warpAffine(ker, M, (ksz, ksz), flags=cv2.INTER_NEAREST)
    ker = (ker > 0).astype(np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, ker, iterations=1)

# =================== Main ===================
for img_path in sorted(files, key=num_key):
    name = img_path.stem
    print(f"[+] {img_path}")

    # ---------- Carrega original ----------
    img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        print("    (skip: cannot read)"); continue
    H0, W0 = img0.shape
    scale = min(1.0, PROC_MAX_SIDE / float(max(H0, W0)))

    # ---------- Remove artefatos verticais ----------
    vmask_strict = detect_vertical_artifacts_strict(img0, tol=VERT_TOL, min_run_r=VERT_MIN_RUN_R, max_width=VERT_MAX_WIDTH)
    if np.count_nonzero(vmask_strict):
        img0_clean = cv2.inpaint(img0, vmask_strict, 3, cv2.INPAINT_TELEA)
    else:
        img0_clean = img0.copy()
    cv2.imwrite(str(DBG_DIR / f"{name}_vertical_artifacts_mask.png"), vmask_strict)

    # ---------- Downscale ----------
    if scale < 1.0:
        newW, newH = int(round(W0*scale)), int(round(H0*scale))
        base_orig = cv2.resize(img0_clean, (newW, newH), interpolation=cv2.INTER_AREA)
    else:
        base_orig = img0_clean.copy()
    H, W = base_orig.shape

    # Escala de parâmetros base
    MEDIAN_K_S  = scale_len(MEDIAN_K, scale, odd=True, minv=3)
    BG_K_S      = scale_len(BG_K, scale, odd=True, minv=15)
    LINE_KLEN_S = scale_len(LINE_KLEN, scale, odd=True, minv=15)
    LINE_KLEN_STRONG_S = scale_len(LINE_KLEN, scale, odd=True, minv=15)
    OPEN_K_S    = max(1, scale_len(OPEN_K, scale))
    CLOSE_K_S   = max(1, scale_len(CLOSE_K, scale))
    GATE_K_S    = max(1, scale_len(GATE_K, scale))
    DIR_CLOSE_LEN_S        = scale_len(DIR_CLOSE_LEN, scale, odd=True, minv=41)
    DIR_CLOSE_LEN_STRONG_S = scale_len(DIR_CLOSE_LEN_STRONG, scale, odd=True, minv=51)
    SUPPORT_BAND_S = max(1, scale_len(SUPPORT_BAND, scale))
    SEG_BAND_S     = max(1, scale_len(SEG_BAND, scale))
    SEG_MIN_LEN_S  = max(20, scale_len(SEG_MIN_LEN, scale))
    SUPPORT_MIN_ABS_S = max(200, int(round(SUPPORT_MIN_ABS * scale)))
    SMALL_STAR_MAX_A_S = scale_area(SMALL_STAR_MAX_A, scale, minv=8)
    BIG_STAR_MIN_A_S   = scale_area(BIG_STAR_MIN_A,   scale, minv=80)
    BIG_STAR_GROW_PX_S = scale_len(BIG_STAR_GROW_PX,  scale, minv=5)

    # ---------- Enhance base ----------
    base = percentile_stretch(base_orig, PERC_LOW, PERC_HIGH)
    base = apply_clahe(base, CLAHE_CLIP, CLAHE_TILE)

    # ---------- Métricas + auto-tune ----------
    metrics = measure_scene(base)
    ov = auto_tune(metrics, scale)

    # Overrides
    SMALL_STAR_Q           = ov.get("SMALL_STAR_Q", SMALL_STAR_Q)
    SMALL_STAR_MAX_A_S     = scale_area(ov.get("SMALL_STAR_MAX_A", SMALL_STAR_MAX_A), scale, minv=8)
    BIG_STAR_Q             = ov.get("BIG_STAR_Q", BIG_STAR_Q)
    BIG_STAR_MIN_A_S       = scale_area(ov.get("BIG_STAR_MIN_A", BIG_STAR_MIN_A), scale, minv=80)
    BIG_STAR_GROW_PX_S     = scale_len(ov.get("BIG_STAR_GROW_PX", BIG_STAR_GROW_PX), scale, minv=5)
    GATE_K_S               = max(1, scale_len(ov.get("GATE_K", GATE_K), scale))
    RESP_PERC_WEAK         = ov.get("RESP_PERC_WEAK", RESP_PERC_WEAK)
    RESP_PERC_STRONG       = ov.get("RESP_PERC_STRONG", RESP_PERC_STRONG)
    LINE_THICK_STRONG      = ov.get("LINE_THICK_STRONG", LINE_THICK_STRONG)
    DIR_CLOSE_LEN_S        = scale_len(ov.get("DIR_CLOSE_LEN", DIR_CLOSE_LEN), scale, odd=True, minv=31)
    DIR_CLOSE_LEN_STRONG_S = scale_len(ov.get("DIR_CLOSE_LEN_STRONG", DIR_CLOSE_LEN_STRONG), scale, odd=True, minv=41)
    SUPPORT_BAND_S         = max(1, scale_len(ov.get("SUPPORT_BAND", SUPPORT_BAND), scale))
    SUPPORT_MIN_RATIO      = ov.get("SUPPORT_MIN_RATIO", SUPPORT_MIN_RATIO)
    SUPPORT_MIN_ABS_S      = max(200, int(round(ov.get("SUPPORT_MIN_ABS", SUPPORT_MIN_ABS))))
    SEG_MIN_LEN_S          = max(20, int(round(ov.get("SEG_MIN_LEN", SEG_MIN_LEN))))
    SEG_DILATE_ALONG       = ov.get("SEG_DILATE_ALONG", SEG_DILATE_ALONG)
    NO_TRAIL_GUARD         = ov.get("NO_TRAIL_GUARD", False)
    CLOSE_K_S              = max(1, scale_len(ov.get("CLOSE_K", CLOSE_K), scale))
    COH_THR                = float(ov.get("COH_THR", 0.18))

    if not NO_TRAIL_GUARD and RESP_PERC_STRONG >= 98.0:
        SMALL_STAR_Q = min(SMALL_STAR_Q, 98.8)
        SMALL_STAR_MAX_A_S = max(SMALL_STAR_MAX_A_S, 30)

    # ---------- Big stars / halos (excluir) ----------
    big_mask = detect_big_stars(base, BIG_STAR_Q, BIG_STAR_MIN_A_S, BIG_STAR_GROW_PX_S)
    cv2.imwrite(str(DBG_DIR / f"{name}_big_star_mask.png"), big_mask)

    # ---------- Weak branch ----------
    xw, _ = inpaint_stars(base, STAR_Q, max(1, scale_len(STAR_DILATE, scale)))
    xw, _ = remove_small_stars_inpaint(xw, SMALL_STAR_Q, SMALL_STAR_MAX_A_S, max(0, scale_len(SMALL_STAR_DILATE, scale)))
    xw = cv2.medianBlur(xw, MEDIAN_K_S)
    xw = flatten_bg(xw, BG_K_S)
    resp_w = line_bank_response(xw, LINE_KLEN_S, LINE_THICK, LINE_ANGLES)
    thr_w  = np.percentile(resp_w, RESP_PERC_WEAK)
    mask_w = (resp_w >= thr_w).astype(np.uint8) * 255
    edges_w = auto_canny(xw, CANNY_SIGMA)

    # ---------- Strong branch ----------
    xs = cv2.medianBlur(base, MEDIAN_K_S)
    xs = flatten_bg(xs, BG_K_S)
    resp_s1 = line_bank_response(xs, LINE_KLEN_S, LINE_THICK,        LINE_ANGLES)
    resp_s2 = line_bank_response(xs, LINE_KLEN_STRONG_S, LINE_THICK_STRONG, LINE_ANGLES_STRONG)
    resp_s  = np.maximum(resp_s1, resp_s2)
    thr_s   = np.percentile(resp_s, RESP_PERC_STRONG)
    mask_s  = (resp_s >= thr_s).astype(np.uint8) * 255
    edges_s_pos = auto_canny(xs, CANNY_SIGMA)
    edges_s_neg = auto_canny(255 - xs, CANNY_SIGMA)

    # ---------- União + gating ----------
    mask_union = cv2.bitwise_or(mask_w, mask_s)
    mask_union[big_mask>0] = 0
    gate = cv2.dilate(mask_union, cv2.getStructuringElement(cv2.MORPH_RECT,(GATE_K_S,GATE_K_S)), 1)
    edges_all_raw = cv2.bitwise_or(edges_w, cv2.bitwise_or(edges_s_pos, edges_s_neg))
    edges_all_raw[big_mask>0] = 0
    edges_gated   = cv2.bitwise_and(edges_all_raw, gate)

    # --- Coherence gate ---
    coh_mask, coh_vis, ori_deg, coh_map = coherence_gate(base, win=9, thr=COH_THR)
    cv2.imwrite(str(DBG_DIR / f"{name}_coherence.png"), coh_vis)
    edges_union = cv2.bitwise_and(edges_gated, coh_mask)

    # ---------- Morfologia + fechamento direcional ----------
    if OPEN_K_S>1:
        edges_union = cv2.morphologyEx(edges_union, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(OPEN_K_S,OPEN_K_S)), 1)
    if CLOSE_K_S>1:
        edges_union = cv2.morphologyEx(edges_union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(CLOSE_K_S,CLOSE_K_S)), 1)
    if USE_DIR_CLOSE:
        edges_union = directional_close(edges_union, length=DIR_CLOSE_LEN_S, step=DIR_CLOSE_STEP)
    if np.count_nonzero(edges_union) < 4000:
        edges_union = directional_close(edges_union, length=DIR_CLOSE_LEN_S+10, step=5)

    # Strong-only (para Hough auxiliar)
    edges_strong = cv2.bitwise_or(mask_s, cv2.bitwise_or(edges_s_pos, edges_s_neg))
    edges_strong[big_mask>0] = 0
    if OPEN_K_S>1:
        edges_strong = cv2.morphologyEx(edges_strong, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(OPEN_K_S,OPEN_K_S)), 1)
    if CLOSE_K_S>1:
        edges_strong = cv2.morphologyEx(edges_strong, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(CLOSE_K_S,CLOSE_K_S)), 1)
    if USE_DIR_CLOSE:
        edges_strong = directional_close(edges_strong, length=DIR_CLOSE_LEN_STRONG_S, step=DIR_CLOSE_STEP)

    # Debug reduzido
    cv2.imwrite(str(DBG_DIR / f"{name}_edges_union.png"),  edges_union)
    cv2.imwrite(str(DBG_DIR / f"{name}_edges_strong.png"), edges_strong)

    # ---------- Hough ----------
    def run_hough(edges):
        h, ang, rho = hough_line(edges>0)
        acc, th, r  = hough_line_peaks(h, ang, rho, num_peaks=NUM_PEAKS, min_distance=MIN_DIST, min_angle=MIN_ANG)
        return h, ang, rho, acc, th, r

    hU, angU, rhoU, accU, thU, rU = run_hough(edges_union)
    hS, angS, rhoS, accS, thS, rS = run_hough(edges_strong)

    rho_peaks, theta_peaks = merge_peaks(rU, thU, accU, rS, thS, accS, deg_tol=1.0, rho_tol=15.0)
    rho_peaks, theta_peaks = filter_lines_by_support(
        edges_union, rho_peaks, theta_peaks,
        band=SUPPORT_BAND_S, min_ratio=SUPPORT_MIN_RATIO, min_abs=SUPPORT_MIN_ABS_S
    )

    # ---------- Poda por direção adaptativa ----------
    rho_tmp, th_tmp, hits = prune_by_theta_bins(
        rho_peaks, theta_peaks, edges_union, band=SUPPORT_BAND_S,
        bin_w_deg=1.5, keep_bins=3
    )
    if len(hits)>=3 and max(hits) > 2.0*sorted(hits)[-2]:
        keep_bins0 = 1
    else:
        keep_bins0 = 2 if len(rho_peaks)>10 else 1
    rho_peaks, theta_peaks, _ = prune_by_theta_bins(
        rho_peaks, theta_peaks, edges_union, band=SUPPORT_BAND_S,
        bin_w_deg=1.5, keep_bins=keep_bins0
    )

    # ---------- Pós-processo orientado ----------
    theta_star = dominant_angle(theta_peaks) if len(theta_peaks)>0 else None

    edges_for_fit = edges_union.copy()
    if theta_star is not None and ori_deg is not None:
        delta = np.abs(np.vectorize(ang_diff_deg)(ori_deg, np.degrees(theta_star)))
        ori_gate = (delta <= 12.0).astype(np.uint8) * 255
        edges_for_fit = cv2.bitwise_and(edges_for_fit, ori_gate)
    if theta_star is not None:
        edges_for_fit = oriented_close_one(edges_for_fit, np.degrees(theta_star), length=max(41, DIR_CLOSE_LEN_S+20))

    # ---------- Accumulator preview ----------
    acc_img_pts = np.zeros_like(hU, dtype=np.uint8)
    for rho, theta in zip(rho_peaks, theta_peaks):
        r_idx = int(np.argmin(np.abs(rhoU - rho))); t_idx = int(np.argmin(np.abs(angU - theta)))
        cv2.circle(acc_img_pts, (t_idx, r_idx), DOT_RADIUS, 255, -1, cv2.LINE_8)
    cv2.imwrite(str(ACC_DIR / f"{name}_acc_points.png"), acc_img_pts)
    acc_preview = render_acc_preview(angU, rhoU, rho_peaks, theta_peaks, max_w=TILE_W, min_px=3)

    # ---------- Traçar as linhas (NOVO fluxo) ----------
    resp_map = np.maximum(resp_s.astype(np.float32), resp_w.astype(np.float32))

    segments_scaled = []
    if len(theta_peaks) > 0:
        for rho, th in zip(rho_peaks, theta_peaks):
            th0 = theta_star if theta_star is not None else th
            th_ref, rho_ref = refine_theta_rho_local(
                edges_for_fit, coh_map, resp_map, th0, ori_deg,
                angle_sweep=np.radians(ANGLE_SWEEP_DEG),
                angle_step=np.radians(ANGLE_STEP_DEG),
                ori_tol_deg=ORI_TOL_DEG
            )
            if th_ref is None:
                th_ref, rho_ref = fit_line_ransac_weighted(edges_for_fit, coh_map, resp_map)
                if th_ref is None:
                    continue

            band_prof = max(1, int(round(PROFILE_BAND_MULT * SEG_BAND_S)))
            seg = extract_segment_by_profile(
                edges_for_fit, rho_ref, th_ref,
                band_px=band_prof, win=PROFILE_WIN,
                min_dens=PROFILE_MIN_DENS, max_gap=PROFILE_MAX_GAP
            )
            if seg is None:
                seg = extract_segment_from_edges(
                    edges_for_fit, rho_ref, th_ref,
                    band=SEG_BAND_S, min_len=SEG_MIN_LEN_S,
                    dilate_along=SEG_DILATE_ALONG, endpoint_r=SEG_ENDPOINT_RADIUS
                )
            if seg is None:
                Ht, Wt = edges_for_fit.shape
                diag = int(np.hypot(Ht, Wt))
                a, b = np.cos(th_ref), np.sin(th_ref)
                x0, y0 = a*rho_ref, b*rho_ref
                seg = ((int(x0 + diag*(-b)), int(y0 + diag*(a))),
                       (int(x0 - diag*(-b)), int(y0 - diag*(a))))
            # checagem de coerência média (no scale de processamento)
            p1, p2 = seg
            mask = np.zeros_like(edges_for_fit, np.uint8)
            cv2.line(mask, p1, p2, 255, DRAW_THICK, cv2.LINE_8)
            cm = float(coh_map[mask>0].mean()) if np.any(mask>0) else 0.0
            if cm >= MIN_MEAN_COH:
                segments_scaled.append(seg)
    else:
        th_ref, rho_ref = fit_line_ransac_weighted(edges_for_fit, coh_map, resp_map)
        if th_ref is not None:
            band_prof = max(1, int(round(PROFILE_BAND_MULT * SEG_BAND_S)))
            seg = extract_segment_by_profile(edges_for_fit, rho_ref, th_ref,
                                             band_px=band_prof, win=PROFILE_WIN,
                                             min_dens=PROFILE_MIN_DENS, max_gap=PROFILE_MAX_GAP)
            if seg is not None:
                segments_scaled.append(seg)

    # Reescalar p/ original
    def up(pt):
        if scale >= 1.0: return pt
        return (int(round(pt[0]/scale)), int(round(pt[1]/scale)))
    segments_orig = [(up(p1), up(p2)) for (p1,p2) in segments_scaled]

    # Fusão + rejeições finais
    segments_orig = merge_colinear_segments(segments_orig, ang_tol_deg=1.0, gap_max=40)
    segments_orig = reject_1px_vertical_segments(segments_orig, img0, tol=2, min_run=0.10)

    # ---------- Desenho ----------
    if NO_TRAIL_GUARD and len(segments_orig) == 0:
        on_original = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        lines_only  = np.zeros_like(img0, dtype=np.uint8)
    else:
        on_original = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        lines_only  = np.zeros_like(img0, dtype=np.uint8)
        for p1, p2 in segments_orig:
            cv2.line(on_original, p1, p2, (255,255,255), DRAW_THICK, cv2.LINE_8)
            cv2.line(lines_only,  p1, p2, 255,            DRAW_THICK, cv2.LINE_8)

    # ---------- Saves ----------
    cv2.imwrite(str(INV_DIR   / f"{name}_inversed.png"), on_original)
    cv2.imwrite(str(LINES_DIR / f"{name}_hough_lines.png"), lines_only)

    # Panel
    def T(img, title): return tile_box(img, title, TILE_W, CONTENT_H, TITLE_H)
    panel = stack_grid(
        [
            T(base_orig,   "Original (downscaled)"),
            T(acc_preview, "Accumulator (points, rho–theta scaled)"),
            T(xw,          "Preprocessed (weak): stretch+CLAHE+inpaint+flatten+small-star"),
            T(edges_union, "Edges (union + coherence + dir-close)"),
            T(on_original, "Overlay (white segments)"),
            T(lines_only,  "Lines only (white)")
        ],
        rows=3, cols=2, sep=SEP, color=SEP_COLOR, margin=SEP
    )
    cv2.imwrite(str(PANEL_DIR / f"{name}_panel.png"), panel)

print("Done.")
