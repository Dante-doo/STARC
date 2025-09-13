#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Hough for satellite trails:
- Pre-clean (column/row artifacts + bright blobs), adaptive Canny by percentiles.
- Angle veto (penalize near vertical/horizontal if artifacts present).
- Accumulator → auto peaks → sub-bin refinement → local on-off search.
- Quality gates (length, continuity, width) + pruning.
- Emergency fallback if 0 lines (liberal settings).
- Save accumulator(points), inverse(full), ROI mask/CSV; per-patch Hough heatmaps.
"""

from pathlib import Path
import glob, csv, math
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks

from code.config import *  # import all constants

# =============================== utils =======================================
def ensure_dir(p: Path):
    if p.suffix: p.parent.mkdir(parents=True, exist_ok=True)
    else: p.mkdir(parents=True, exist_ok=True)

def theta_grid_full():
    return np.linspace(-np.pi/2, np.pi/2, int(HOUGH_ACC_NUM_THETA), endpoint=False, dtype=np.float64)

def restrict_theta(centers_rad, delta_deg):
    full = theta_grid_full()
    if not centers_rad: return full
    delta = math.radians(float(PATCH_THETA_DELTA_DEG if delta_deg is None else delta_deg))
    diff = np.abs(np.angle(np.exp(1j*(full[:,None] - np.asarray(centers_rad)[None,:]))))
    mask = (diff <= delta).any(axis=1)
    return full[mask] if mask.any() else full

def rotate_angles(angles_rad, rot_deg):
    r = math.radians(rot_deg)
    a = (np.asarray(angles_rad) + r + np.pi/2) % np.pi - np.pi/2
    return a.tolist()

# ============================ artifact pre-clean ==============================
def mad(x):
    x = np.asarray(x, np.float64)
    m = np.median(x); return np.median(np.abs(x - m)) + 1e-9

def flag_runs(mask, min_run):
    m = mask.astype(np.uint8)
    if m.sum() == 0: return np.zeros_like(mask, bool)
    idx = np.where(np.diff(np.r_[0, m, 0]) != 0)[0]
    runs = [(idx[i], idx[i+1]) for i in range(0, len(idx), 2)]
    out = np.zeros_like(mask, bool)
    for a,b in runs:
        if (b-a) >= int(min_run):
            out[a:b] = True
    return out

def detect_colrow_artifacts(img):
    col_med = np.median(img, axis=0); row_med = np.median(img, axis=1)
    g_med = np.median(img)
    col_dev = np.abs(col_med - g_med); row_dev = np.abs(row_med - g_med)
    c_mask = col_dev > ARTIF_KMAD * mad(col_med)
    r_mask = row_dev > ARTIF_KMAD * mad(row_med)
    c_bad = flag_runs(c_mask, ARTIF_MIN_RUN)
    r_bad = flag_runs(r_mask, ARTIF_MIN_RUN)
    return c_bad, r_bad

def mask_blobs(img):
    b1 = cv2.GaussianBlur(img, (0,0), 1.2)
    b2 = cv2.GaussianBlur(img, (0,0), 3.0)
    dog = cv2.subtract(b1, b2)
    thr = np.median(dog) + 5.0 * mad(dog)
    m = (dog > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return cv2.dilate(m, k, iterations=1)

def remove_background(img):
    bg = cv2.medianBlur(img, 31)
    return cv2.subtract(img, bg)

def apply_artifact_masks(bw_edges, col_bad, row_bad):
    out = bw_edges.copy()
    if col_bad.any(): out[:, col_bad] = 0
    if row_bad.any(): out[row_bad, :] = 0
    return out

def angle_penalty(angles, col_bad, row_bad):
    w = np.ones_like(angles, dtype=np.float32)
    if col_bad.any():
        mask = np.abs(np.degrees(angles)) <= ANGLE_VETO_DEG
        w[mask] *= ANGLE_VETO_WEIGHT
    if row_bad.any():
        mask = (np.abs(np.degrees(np.abs(angles) - np.pi/2)) <= ANGLE_VETO_DEG)
        w[mask] *= ANGLE_VETO_WEIGHT
    return w

# =============================== preproc =====================================
def apply_clahe(img, enable=True):
    if not enable: return img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)

def morph_open_close(bw, k_open=0, k_close=0):
    out = bw
    if k_open and k_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if k_close and k_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out

def grad_mag(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def edges_from_percentiles(img):
    g = grad_mag(img)
    lo = np.percentile(g, CANNY_PERC_LOW)
    hi = np.percentile(g, CANNY_PERC_HIGH)
    lo = max(5.0, float(lo)); hi = min(255.0, float(hi))
    b = cv2.Canny(img, lo, hi, L2gradient=True)
    return b, g

def edges_from_percentiles_fallback(img):
    g = grad_mag(img)
    lo = np.percentile(g, CANNY_PERC_LOW_FALLBACK)
    hi = np.percentile(g, CANNY_PERC_HIGH_FALLBACK)
    lo = max(5.0, float(lo)); hi = min(255.0, float(hi))
    b = cv2.Canny(img, lo, hi, L2gradient=True)
    return b, g

def make_edges_variant(gray, variant):
    base = apply_clahe(remove_background(gray), enable=bool(variant.get("clahe", True)))
    blob_m = mask_blobs(base)
    if variant.get("edge") == "canny":
        bw, gmag = edges_from_percentiles(base)
    else:
        gmag = grad_mag(base)
        mag_u8 = cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bw = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(blob_m))
    bw = morph_open_close(bw, variant.get("open",0), variant.get("close",0))
    return bw, gmag

# =============================== scoring =====================================
def heatmap_score(H: np.ndarray, topk=64, w_peak=0.7, w_anis=0.3) -> float:
    X = H.astype(np.float32)
    m = float(X.mean()) + 1e-9
    k = max(1, min(int(topk), X.size))
    s_peak = float(np.partition(X.ravel(), -k)[-k:].mean()) / m
    col = X.sum(axis=0)
    s_aniso = float(col.max() / (col.mean() + 1e-9))
    return w_peak*s_peak + w_anis*s_aniso

# ============================ Hough & peaks ==================================
def hough_accumulator(bw_bool: np.ndarray, theta: np.ndarray):
    H, angles, dists = hough_line(bw_bool, theta=np.ascontiguousarray(theta))
    return H, angles, dists

def select_peaks_auto(H, angles, dists):
    p_lo, p_hi = float(HPOINTS_PMIN), float(HPOINTS_PMAX)
    step_deg = 180.0 / float(len(angles))
    min_angle_bins = max(1, int(round(float(HPOINTS_MIN_ANGLE_DEG) / step_deg)))
    min_dist_bins  = max(1, int(HPOINTS_MIN_DIST_RHO))
    best = ([], [], [])
    for _ in range(int(HPOINTS_BIN_STEPS)):
        p = 0.5*(p_lo + p_hi)
        thr = np.percentile(H, p)
        hv, th, rh = hough_line_peaks(H, angles, dists,
                                      threshold=thr,
                                      num_peaks=int(HPOINTS_MAX_PEAKS),
                                      min_distance=min_dist_bins,
                                      min_angle=min_angle_bins)
        k = len(hv); best = (hv, th, rh)
        if k < HPOINTS_K_MIN:   p_hi = max(p_lo, p - 0.01)
        elif k > HPOINTS_K_MAX: p_lo = min(p_hi, p + 0.01)
        else: break
    return best

# --------- sub-bin refinement -------------------------------------------------
def refine_peaks_subbin(H, angles, dists, thetas, rhos, win=7):
    if len(thetas) == 0: return rhos, thetas
    win = int(win); win += (win % 2 == 0)
    ah, rh = len(angles), len(dists)
    angs = np.asarray(angles); dss = np.asarray(dists)
    out_r, out_t = [], []
    for t, r in zip(thetas, rhos):
        ai = int(np.clip(np.searchsorted(angs, t), 0, ah-1))
        ri = int(np.clip(np.searchsorted(dss, r),  0, rh-1))
        a0,a1 = max(0, ai-win//2), min(ah, ai+win//2+1)
        r0,r1 = max(0, ri-win//2), min(rh, ri+win//2+1)
        W = H[r0:r1, a0:a1].astype(np.float64)
        if W.size == 0 or W.max() <= 0:
            out_r.append(r); out_t.append(t); continue
        Rs = dss[r0:r1]; Ts = angs[a0:a1]
        RR, TT = np.meshgrid(Rs, Ts, indexing="ij")
        s = W.sum()
        out_r.append(float((W*RR).sum()/s))
        out_t.append(float((W*TT).sum()/s))
    return out_r, out_t

# ------------------------- line geometry & sampling --------------------------
def endpoints_from_rho_theta(rho: float, theta: float, w: int, h: int):
    p0x, p0y = rho*np.cos(theta), rho*np.sin(theta)
    vx, vy   = -np.sin(theta), np.cos(theta)
    pts = []
    if abs(vx) > 1e-9:
        t = (0 - p0x)/vx;     y = p0y + t*vy
        if 0 <= y <= h-1: pts.append((0, int(round(y))))
        t = ((w-1) - p0x)/vx; y = p0y + t*vy
        if 0 <= y <= h-1: pts.append((w-1, int(round(y))))
    if abs(vy) > 1e-9:
        t = (0 - p0y)/vy;     x = p0x + t*vx
        if 0 <= x <= w-1: pts.append((int(round(x)), 0))
        t = ((h-1) - p0y)/vy; x = p0x + t*vx
        if 0 <= x <= w-1: pts.append((int(round(x)), h-1))
    if len(pts) < 2: return None
    uniq = []
    for p in pts:
        if p not in uniq: uniq.append(p)
        if len(uniq) == 2: break
    return uniq if len(uniq) == 2 else None

def sample_line(img: np.ndarray, p0, p1, step: float = 1.0) -> np.ndarray:
    x0,y0 = p0; x1,y1 = p1
    n = int(max(abs(x1-x0), abs(y1-y0)) / max(1.0, step)) + 1
    xs = np.clip(np.rint(np.linspace(x0, x1, n)).astype(int), 0, img.shape[1]-1)
    ys = np.clip(np.rint(np.linspace(y0, y1, n)).astype(int), 0, img.shape[0]-1)
    return img[ys, xs]

def line_score_onoff(grad_mag_img: np.ndarray, rho: float, theta: float, w: int, h: int,
                     offset_px: int, minlen_px: int):
    pts = endpoints_from_rho_theta(rho, theta, w, h)
    if pts is None: return -1e9, 0, None
    (x0,y0),(x1,y1) = pts
    L = int(np.hypot(x1-x0, y1-y0))
    if L < int(minlen_px): return -1e9, L, pts
    n = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    p0 = np.array([x0,y0], dtype=np.float64); p1 = np.array([x1,y1], dtype=np.float64)
    p0a=(p0+n*offset_px).astype(int); p1a=(p1+n*offset_px).astype(int)
    p0b=(p0-n*offset_px).astype(int); p1b=(p1-n*offset_px).astype(int)
    on  = sample_line(grad_mag_img, (x0,y0),(x1,y1)).mean()
    off = 0.5*( sample_line(grad_mag_img, tuple(p0a),tuple(p1a)).mean() +
                sample_line(grad_mag_img, tuple(p0b),tuple(p1b)).mean() )
    return float(on - off), L, pts

def local_search_onoff(grad_mag_img, rho, theta, w, h):
    dr  = int(HOUGH_REFINE_LOCAL_DRHO_PX)
    dtd = float(HOUGH_REFINE_LOCAL_DTHETA_DEG)
    step_t = float(HOUGH_REFINE_LOCAL_STEP_THETA)
    best = (-1e9, rho, theta)
    ts = np.arange(-dtd, dtd+1e-9, step_t, dtype=np.float64)
    for dtheta_deg in ts:
        th = theta + math.radians(dtheta_deg)
        for r in range(-dr, dr+1):
            sc, L, _ = line_score_onoff(grad_mag_img, rho + r, th, w, h,
                                        offset_px=int(HOUGH_ONOFF_OFFSET_PX),
                                        minlen_px=int(HOUGH_LINE_MINLEN_FULL))
            if sc > best[0]:
                best = (sc, rho + r, th)
    return best[1], best[2], best[0]

# ---------------------------- extra gates ------------------------------------
def continuity_fraction(grad_mag_img, pts, thr_pct=90.0):
    thr = np.percentile(grad_mag_img, float(thr_pct))
    vals = sample_line(grad_mag_img, pts[0], pts[1])
    return float((vals >= thr).mean())

def normal_profile_width(grad_mag_img, pts, theta, rel_thr=0.5,
                         half=8, n_samples=30):
    (x0,y0),(x1,y1) = pts
    xs = np.linspace(x0, x1, n_samples); ys = np.linspace(y0, y1, n_samples)
    nvec = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    widths = []
    for x,y in zip(xs, ys):
        t = np.linspace(-half, half, 2*half+1)
        xx = np.clip(np.rint(x + t*nvec[0]).astype(int), 0, grad_mag_img.shape[1]-1)
        yy = np.clip(np.rint(y + t*nvec[1]).astype(int), 0, grad_mag_img.shape[0]-1)
        p = grad_mag_img[yy, xx]
        m = float(p.max())
        if m <= 1e-6: continue
        mask = (p >= rel_thr*m).astype(np.uint8)
        widths.append(mask.sum())
    if not widths: return 0.0
    return float(np.mean(widths))

# ----------------------------- rendering -------------------------------------
def accumulator_points_image(H, angles, dists, thetas, rhos):
    out = H.astype(np.float32)
    out -= out.min()
    if out.max() > 0: out /= out.max()
    out *= float(HPOINTS_BG_ALPHA)
    if len(rhos):
        ai = np.searchsorted(angles, np.asarray(thetas)); ai = np.clip(ai, 0, len(angles)-1)
        ri = np.searchsorted(dists,  np.asarray(rhos));   ri = np.clip(ri, 0, len(dists)-1)
        r = int(HPOINTS_DOT_RADIUS)
        for y, x in zip(ri, ai):
            if r <= 0: out[y, x] = 1.0
            else: cv2.circle(out, (int(x), int(y)), r, 1.0, -1)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def draw_inverse_lines(gray_full, rhos, thetas, thickness=2):
    out = gray_full.copy()
    h, w = out.shape
    for rho, th in zip(rhos, thetas):
        pts = endpoints_from_rho_theta(float(rho), float(th), w, h)
        if pts is not None:
            cv2.line(out, pts[0], pts[1], 255, int(thickness), cv2.LINE_AA)
    return out

def roi_mask_from_lines(size_hw, lines_r_t, band_half_px: int):
    h, w = size_hw
    m = np.zeros((h, w), np.uint8)
    thick = max(1, int(2*band_half_px + 1))
    for rho, th in lines_r_t:
        pts = endpoints_from_rho_theta(float(rho), float(th), w, h)
        if pts is not None:
            cv2.line(m, pts[0], pts[1], 255, thick, cv2.LINE_AA)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

# ============================== FULL pipeline ================================
def run_variant_pipeline(gray, variant, theta_full, col_bad, row_bad):
    bw, gmag = make_edges_variant(gray, variant)
    bw = apply_artifact_masks(bw, col_bad, row_bad)
    H, angles, dists = hough_accumulator(bw.astype(bool), theta_full)
    wcol = angle_penalty(angles, col_bad, row_bad)
    Hf = H.astype(np.float32, copy=False)
    Hw = Hf * wcol[None, :]
    score = heatmap_score(Hw, topk=HOUGH_SCORE_TOPK,
                          w_peak=HOUGH_SCORE_W_PEAK, w_anis=HOUGH_SCORE_W_ANIS)
    return Hw, angles, dists, gmag, score, variant

def emergency_candidates(img, theta_full):
    """Very liberal path: no angle penalty, strong closing, low thresholds."""
    base = apply_clahe(remove_background(img), enable=True)
    bw, _ = edges_from_percentiles_fallback(base)
    bw = morph_open_close(bw, 0, 7)
    H, angles, dists = hough_accumulator(bw.astype(bool), theta_full)
    hvals, thetas, rhos = select_peaks_auto(H, angles, dists)
    gmag = grad_mag(img)
    w, h = img.shape[1], img.shape[0]
    cand = []
    for rho, th in zip(rhos, thetas):
        rho2, th2, sc = local_search_onoff(gmag, float(rho), float(th), w, h)
        pts = endpoints_from_rho_theta(rho2, th2, w, h)
        if pts is None: continue
        L = int(np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]))
        if L < int(HOUGH_EMERG_MINLEN): continue
        cand.append((sc, rho2, th2, pts))
    if not cand: return [], [], []
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:int(HOUGH_EMERG_TOPK)]
    return [c[1] for c in cand], [c[2] for c in cand], [c[3] for c in cand]

def process_full_image(full_path: Path):
    stem = full_path.stem
    img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] read fail: {full_path}"); return None

    ds = int(HOUGH_FULL_DOWNSCALE)
    work = img if ds == 1 else cv2.resize(img, (img.shape[1]//ds, img.shape[0]//ds), interpolation=cv2.INTER_AREA)
    theta_full = theta_grid_full()

    col_bad, row_bad = detect_colrow_artifacts(work)

    # choose best variant by score
    best = (-1, None, None, None, None, None)
    for v in HOUGH_AUTO_VARIANTS:
        H, angles, dists, gmag, s, vv = run_variant_pipeline(work, v, theta_full, col_bad, row_bad)
        if s > best[0]: best = (s, H, angles, dists, gmag, vv)
    _, H, angles, dists, gmag_work, variant = best

    # peaks + sub-bin refine
    hvals, thetas, rhos = select_peaks_auto(H, angles, dists)
    rhos, thetas = refine_peaks_subbin(H, angles, dists, thetas, rhos, win=int(HOUGH_REFINE_PEAK_WINDOW))
    if ds != 1: rhos = [float(r)*ds for r in rhos]

    gmag_full = grad_mag(img)

    # local search + pruning with gates
    w, h = img.shape[1], img.shape[0]
    cand = []
    for rho, th in zip(rhos, thetas):
        rho2, th2, sc = local_search_onoff(gmag_full, float(rho), float(th), w, h)
        pts = endpoints_from_rho_theta(rho2, th2, w, h)
        if pts is None: continue
        L = int(np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]))
        if L < int(HOUGH_LINE_MINLEN_FULL): continue
        cont = continuity_fraction(gmag_full, pts, thr_pct=CONTINUITY_THR_PCT)
        if cont < float(CONTINUITY_MIN_FRAC): continue
        width = normal_profile_width(gmag_full, pts, th,
                                     rel_thr=float(WIDTH_REL_THR),
                                     half=int(WIDTH_PROFILE_HALF),
                                     n_samples=int(WIDTH_SAMPLES_ALONG))
        if not (float(WIDTH_MIN_PX) <= width <= float(WIDTH_MAX_PX)): continue
        cand.append((sc, rho2, th2, pts))

    # if nothing kept, run relaxed fallback; if still nothing, emergency
    if not cand:
        # relaxed: fallback thresholds + stronger closing + keep by percentile
        base = apply_clahe(remove_background(work), enable=bool(variant.get("clahe", True)))
        bw_fb, _ = edges_from_percentiles_fallback(base)
        bw_fb = morph_open_close(bw_fb, variant.get("open",0), max(5, int(variant.get("close",0))))
        bw_fb = apply_artifact_masks(bw_fb, col_bad, row_bad)
        H2, angles2, dists2 = hough_accumulator(bw_fb.astype(bool), theta_full)
        wcol2 = angle_penalty(angles2, col_bad, row_bad)
        H2 = H2.astype(np.float32, copy=False); H2 *= wcol2[None, :]
        hvals, thetas, rhos = select_peaks_auto(H2, angles2, dists2)
        rhos, thetas = refine_peaks_subbin(H2, angles2, dists2, thetas, rhos, win=int(HOUGH_REFINE_PEAK_WINDOW))
        if ds != 1: rhos = [float(r)*ds for r in rhos]
        for rho, th in zip(rhos, thetas):
            rho2, th2, sc = local_search_onoff(gmag_full, float(rho), float(th), w, h)
            pts = endpoints_from_rho_theta(rho2, th2, w, h)
            if pts is None: continue
            L = int(np.hypot(pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]))
            if L < int(HOUGH_LINE_MINLEN_FULL): continue
            cont = continuity_fraction(gmag_full, pts, thr_pct=CONTINUITY_THR_PCT)
            if cont < float(CONTINUITY_MIN_FRAC): continue
            width = normal_profile_width(gmag_full, pts, th,
                                         rel_thr=float(WIDTH_REL_THR),
                                         half=int(WIDTH_PROFILE_HALF),
                                         n_samples=int(WIDTH_SAMPLES_ALONG))
            if not (float(WIDTH_MIN_PX) <= width <= float(WIDTH_MAX_PX)): continue
            cand.append((sc, rho2, th2, pts))
        if cand:
            H, angles, dists = H2, angles2, dists2

    if not cand and HOUGH_EMERGENCY_ENABLE:
        keep_r, keep_t, keep_pts = emergency_candidates(img, theta_full)
    else:
        if not cand:
            keep_r, keep_t, keep_pts = [], [], []
        else:
            scores = np.array([c[0] for c in cand], dtype=np.float32)
            thr = np.percentile(scores, float(HOUGH_KEEP_SCORE_PCT))
            cand = [c for c in cand if c[0] >= thr]
            cand = sorted(cand, key=lambda x: x[0], reverse=True)[:int(HOUGH_KEEP_TOPK)]
            keep_r = [c[1] for c in cand]; keep_t = [c[2] for c in cand]; keep_pts = [c[3] for c in cand]

    # outputs
    out_acc = FULL_HOUGH_ACC_DIR / f"{stem}_accumulator.png"
    ensure_dir(out_acc)
    acc_points = accumulator_points_image(H, angles, dists, thetas, rhos)
    cv2.imwrite(str(out_acc), acc_points)

    out_inv = FULL_HOUGH_INV_DIR / f"{stem}_inversed.png"
    ensure_dir(out_inv)
    inverse = draw_inverse_lines(img, keep_r, keep_t, thickness=HOUGH_INV_LINE_THICKNESS)
    cv2.imwrite(str(out_inv), inverse)

    out_roi = ROI_IMG_DIR / f"{stem}_roi.png"
    ensure_dir(out_roi)
    roi_mask = roi_mask_from_lines(img.shape, list(zip(keep_r, keep_t)), band_half_px=int(ROI_BAND_PX))
    cv2.imwrite(str(out_roi), roi_mask)

    ensure_dir(ROI_VIS_DIR)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(roi_mask, 50, 150)
    vis[edges>0] = (0,255,0)
    cv2.imwrite(str(ROI_VIS_DIR / f"{stem}_roi.png"), vis)

    return {"full_id": stem, "variant": variant, "thetas": list(keep_t), "roi_mask": roi_mask}

# ============================== ROI CSV utils ================================
def enumerate_patches(h, w, tile):
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0 = r*tile; x0 = c*tile
            y1 = y0 + tile; x1 = x0 + tile
            if y1 <= h and x1 <= w:
                yield r, c, y0, y1, x0, x1

def write_roi_csv_header(csv_path: Path):
    ensure_dir(csv_path)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["full_id","row","col","raw_path","cover_pixels","cover_ratio"])

def append_roi_rows(csv_path: Path, full_id: str, roi_mask: np.ndarray):
    h, w = roi_mask.shape
    area = PATCH_SIZE * PATCH_SIZE
    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        for r, c, y0, y1, x0, x1 in enumerate_patches(h, w, PATCH_SIZE):
            sub = roi_mask[y0:y1, x0:x1]
            pix = int((sub > 0).sum())
            ratio = pix / float(area)
            if (pix >= int(ROI_PATCH_MIN_PIXELS)) or (ratio >= float(ROI_PATCH_MIN_RATIO)):
                raw_rel = str((RAW_STD_DIR / full_id / f"{full_id}_{r}_{c}_{SFX_R}.png").relative_to(ROOT)).replace("\\","/")
                wcsv.writerow([full_id, r, c, raw_rel, pix, f"{ratio:.6f}"])

# ============================== patch hough ==================================
def map_std_hough_out(raw_patch_path: Path) -> Path:
    fid = raw_patch_path.parent.name
    name = raw_patch_path.name.replace(f"_{SFX_R}.png", f"_{SFX_H}.png")
    return HOUGH_STD_DIR / fid / name

def map_aug_hough_out(raw_patch_path: Path) -> Path:
    fid = raw_patch_path.parent.name
    name = raw_patch_path.name.replace(f"_{SFX_R}_", f"_{SFX_H}_")
    return HOUGH_AUG_DIR / fid / name

def patch_hough_heatmap(img_patch: np.ndarray, variant: dict, theta_centers, delta_deg: float):
    base = apply_clahe(remove_background(img_patch), enable=bool(variant.get("clahe", True)))
    bw, _ = (edges_from_percentiles(base) if variant.get("edge")=="canny"
             else (cv2.threshold(cv2.normalize(grad_mag(base), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                 0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1], None))
    bw = morph_open_close(bw, variant.get("open",0), variant.get("close",0))
    thetas = restrict_theta(theta_centers, delta_deg)
    H, angles, dists = hough_accumulator(bw.astype(bool), thetas)
    Hn = H - H.min()
    Hn = (Hn / (Hn.max() + 1e-9) * 255.0).astype(np.uint8)
    return cv2.resize(Hn, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

def process_patches_for_full(full_id: str, theta_centers, variant: dict):
    std_dir = RAW_STD_DIR / full_id
    if std_dir.is_dir():
        for rp in sorted(std_dir.glob("*.png")):
            outp = map_std_hough_out(rp)
            if outp.exists(): continue
            img = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            hmap = patch_hough_heatmap(img, variant, theta_centers, PATCH_THETA_DELTA_DEG)
            ensure_dir(outp); cv2.imwrite(str(outp), hmap)

    aug_dir = RAW_AUG_DIR / full_id
    if aug_dir.is_dir():
        for rp in sorted(aug_dir.glob("*.png")):
            outp = map_aug_hough_out(rp)
            if outp.exists(): continue
            img = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            try: rot_deg = int(rp.stem.split("_")[-1])
            except Exception: rot_deg = 0
            centers_rot = rotate_angles(theta_centers, rot_deg)
            hmap = patch_hough_heatmap(img, variant, centers_rot, PATCH_THETA_DELTA_DEG)
            ensure_dir(outp); cv2.imwrite(str(outp), hmap)

# ================================= main ======================================
def main():
    files = sorted(glob.glob(str(FULL_RAW_DIR / HOUGH_ACC_INPUT_GLOB)))
    if not files:
        raise FileNotFoundError(f"No inputs in {FULL_RAW_DIR} matching {HOUGH_ACC_INPUT_GLOB}")

    write_roi_csv_header(ROI_OUT_CSV)

    for fp in files:
        info = process_full_image(Path(fp))
        if info is None: continue
        fid, variant, thetas, roi_mask = info["full_id"], info["variant"], info["thetas"], info["roi_mask"]
        append_roi_rows(ROI_OUT_CSV, fid, roi_mask)
        process_patches_for_full(fid, thetas, variant)
        print(f"[OK] {fid}: lines={len(thetas)}")

    print(f"[DONE] ROI masks : {ROI_IMG_DIR}")
    print(f"[DONE] ROI CSV   : {ROI_OUT_CSV}")
    print(f"[DONE] Patch Hough at {HOUGH_STD_DIR} / {HOUGH_AUG_DIR}")

if __name__ == "__main__":
    main()
