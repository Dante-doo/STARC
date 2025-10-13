#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Patch-wise Hough with original heuristic:
# - Global orientation context from the full image processed around 2048 scale
#   (percentile stretch + CLAHE + median background + structure-tensor coherence).
# - Per-patch HoughP + support-band validation + trimming.
# - Save per-patch inverse (lines only, white on black) and (ρ,θ) accumulator points.
# - If a given patch has trails, perform on-the-fly AUGMENTATION by rotating ONLY
#   THAT PATCH (not the full): recompute edges/coherence/orientation and run the
#   same pipeline, saving augmented inverse/accumulator and mosaics.
#   This avoids rotating or reprocessing full-resolution images.

from __future__ import annotations
import argparse
import math, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from skimage.transform import hough_line, hough_line_peaks

# ---- config (current names) --------------------------------------------------
from code.config import (
    # inputs/outputs
    FULL_RAW_STD_DIR, FULL_RAW_AUG_DIR,
    FULL_HOUGH_DIR, FULL_HOUGH_INV_DIR, FULL_HOUGH_FILT_DIR, FULL_HOUGH_COH_DIR,
    PATCH_HOUGH_ACC_DIR, PATCH_HOUGH_INV_DIR,
    PATCH_SIZE,
    SFX_H, SFX_INV,
    AUGMENT_ANGLES,

    # resolution default for processing
    HO_DEFAULT_RES,

    # preproc (baseline 2048)
    HO_PERC_LOW, HO_PERC_HIGH,
    HO_CLAHE_CLIP, HO_CLAHE_TILE,
    HO_BG_MED_K, HO_CANNY_SIGMA,

    # borders / halo
    HO_BORDER_HARD_ZERO_PX, HO_BORDER_MARGIN_PX,
    HO_HALO_Q, HO_HALO_GROW, HO_ALPHA_HALO,

    # thin vertical artifact
    HO_ARTV_MAX_THICK, HO_ARTV_MIN_AREA, HO_ARTV_CONNECT_LEN,
    HO_ARTV_BORDER_XMARGIN, HO_ARTV_FULLH_TOL,

    # rejection of exact vertical lines
    HO_REJECT_EXACT_VERTICAL,

    # profiles
    PROFILES_FULL, PROFILES_PATCH,

    # HoughP scans
    HOUGHP_GLOBAL_SCAN_PATCH, HOUGHP_LOCAL_SCAN_PATCH,

    # validation (patch)
    SUPPORT_BAND_PX_PATCH, SUPPORT_MIN_DENSITY_PATCH, SUPPORT_MIN_MEAN_COH_PATCH, SUPPORT_MAX_ODISP_PATCH, ORI_TOL_VALID_PATCH, BORDER_REJECT_FRAC_PATCH,

    # connectivity/trim
    AREA_THRESHOLD_PATCH, MIN_COMP_SIZE_PATCH, CLOSE_K_SZ,
    PROFILE_MAX_GAP_BASE, PROFILE_MIN_LEN_PATCH,

    # accumulator (ρ,θ) per patch
    HO_ACC_NUM_THETA, HO_PEAKS_PERCENTILE, HO_PEAKS_MIN_DIST,
    HO_PEAKS_MIN_ANGLE_DEG, HO_PEAKS_MAX, HO_PEAKS_DOT_RADIUS,
)

# ========================= scaling (baseline = 2048) =========================
BASELINE_MAXSIDE = 2048.0

def ksz(x: float, s: float, minv: int = 1) -> int:
    """Return odd kernel/window length scaled by s."""
    v = max(minv, int(round(x * s)))
    return v | 1

def klen(x: float, s: float, minv: int = 1) -> int:
    """Return integer length scaled by s."""
    return max(minv, int(round(x * s)))

def karea(x: float, s: float) -> int:
    """Return area-like parameter scaled ~ s^2."""
    return max(1, int(round(x * (s ** 2))))

# ========================= I/O and grid ======================================
def list_full_inputs() -> List[Path]:
    roots = [FULL_RAW_STD_DIR, FULL_RAW_AUG_DIR]
    files: List[Path] = []
    exts = ("png","PNG","jpg","JPG","jpeg","JPEG","tif","TIF","tiff","TIFF")
    for r in roots:
        if not r or not r.exists(): continue
        for e in exts:
            files.extend(sorted(r.glob(f"full*.{e}")))
    def num_key(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem); return int(m.group(1)) if m else 0
    return sorted(files, key=num_key)

def crop_to_grid(img: np.ndarray, P: int) -> tuple[np.ndarray,int,int]:
    """Crop image to R×C multiple of P and return (cropped, R, C)."""
    H,W = img.shape[:2]
    R, C = H//P, W//P
    Hc, Wc = R*P, C*P
    return img[:Hc,:Wc].copy(), R, C

def save_mosaic(tiles: List[np.ndarray], grid_hw: tuple[int,int], out_path: Path) -> None:
    """Assemble tiles into one big mosaic and save."""
    if not tiles: return
    R, C = grid_hw
    h, w = tiles[0].shape[:2]
    canvas = np.zeros((R*h, C*w), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, C)
        y, x = r*h, c*w
        canvas[y:y+h, x:x+w] = t
    out_path.parent.mkdir(parents=True, exist_ok=True
    )
    cv2.imwrite(str(out_path), canvas)

# ========================= pre-processing ====================================
def percentile_stretch(u8: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(u8, [p_low, p_high])
    if hi <= lo: return u8
    x = np.clip((u8.astype(np.float32)-lo) * (255.0/(hi-lo)), 0, 255)
    return x.astype(np.uint8)

def apply_clahe(u8: np.ndarray, clip: float, tile_hw: tuple[int,int]) -> np.ndarray:
    if clip <= 0: return u8
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile_hw))
    return clahe.apply(u8)

def flatten_bg(u8: np.ndarray, k: int) -> np.ndarray:
    if k<=1 or k%2==0: return u8
    bg = cv2.medianBlur(u8, k)
    x  = cv2.subtract(u8, bg)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def auto_canny(u8: np.ndarray, sigma: float) -> np.ndarray:
    v = float(np.median(u8))
    lo = int(max(0,(1.0-sigma)*v)); hi = int(min(255,(1.0+sigma)*v))
    if hi<=lo: hi=lo+1
    return cv2.Canny(u8, lo, hi, L2gradient=True)

def border_mask(shape: tuple[int,int], margin: int) -> np.ndarray:
    H,W = shape
    bm = np.zeros((H,W), np.uint8)
    m = int(max(1, margin))
    bm[:m,:] = 255; bm[-m:,:] = 255; bm[:,:m] = 255; bm[:,-m:] = 255
    return bm

def bright_halo_mask(u8: np.ndarray, q=99.87, grow=21) -> np.ndarray:
    th = np.percentile(u8, q)
    bin_ = cv2.threshold(u8, th, 255, cv2.THRESH_BINARY)[1]
    if grow>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow+1,2*grow+1))
        bin_ = cv2.dilate(bin_, k, 1)
    return bin_

# ========================= coherence / orientation ===========================
def coherence_map(u8: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
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

def ang_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def find_orientation_modes(ori_deg: np.ndarray, mask: np.ndarray, k=3, sep=12.0) -> List[float]:
    th = ori_deg[mask>0].ravel()
    if th.size == 0: return []
    hist, edges = np.histogram(th, bins=np.arange(0,181,1.0))
    peaks=[]
    for _ in range(k):
        i = int(np.argmax(hist))
        if hist[i]==0: break
        center = 0.5*(edges[i]+edges[i+1]); peaks.append(float(center))
        a = int(max(0, i-int(sep))); b = int(min(len(hist), i+int(sep)+1))
        hist[a:b]=0
    return peaks

# ========================= artifact: thin vertical ===========================
def oriented_close_one(img_u8: np.ndarray, theta_deg: float, length: int) -> np.ndarray:
    ksz_ = int(length)|1; ker = np.zeros((ksz_, ksz_), np.uint8); c = ksz_//2
    cv2.line(ker, (0,c), (ksz_-1,c), 255, 1)
    M = cv2.getRotationMatrix2D((c,c), float(theta_deg), 1.0)
    ker = cv2.warpAffine(ker, M, (ksz_,ksz_), flags=cv2.INTER_NEAREST)
    ker = (ker>0).astype(np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, ker, iterations=1)

def detect_thin_vertical_artifacts(edges_u8: np.ndarray, connect_len: int,
                                   min_area: int, max_thick: int,
                                   W: int, H: int) -> np.ndarray:
    vert = oriented_close_one(edges_u8, 90.0, length=connect_len)
    vert = cv2.morphologyEx(vert, cv2.MORPH_OPEN, np.ones((3,1), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    out = np.zeros((H,W), np.uint8)
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < min_area: continue
        if w > max(1, max_thick): continue
        ys, xs = np.where(labels==i)
        if xs.size < 2: continue
        near_left  = x <= HO_ARTV_BORDER_XMARGIN
        near_right = x + w >= W-1-HO_ARTV_BORDER_XMARGIN
        if (near_left or near_right):
            full_h = (y <= HO_ARTV_FULLH_TOL) and (y + h >= H-1-HO_ARTV_FULLH_TOL)
            if not full_h: continue
        out[labels==i] = 255
    return out

# ========================= Hough helpers =====================================
def reject_exact_vertical_lines(L: np.ndarray | None) -> np.ndarray | None:
    if L is None: return None
    keep = []
    for x1,y1,x2,y2 in L:
        if x1 == x2:  # 90 deg exactly
            continue
        keep.append([x1,y1,x2,y2])
    return np.array(keep) if keep else None

def houghp_fuse(mask: np.ndarray, scans: List[dict]) -> tuple[np.ndarray, np.ndarray | None]:
    fused = np.zeros_like(mask)
    lines_all = []
    for prm in scans:
        lines = cv2.HoughLinesP(mask, **prm)
        if lines is None: continue
        L = lines[:,0,:]
        if HO_REJECT_EXACT_VERTICAL:
            L = reject_exact_vertical_lines(L)
            if L is None: continue
        lines_all.append(L)
        for x1,y1,x2,y2 in L:
            cv2.line(fused,(x1,y1),(x2,y2),255,1)
    return fused, (np.vstack(lines_all) if lines_all else None)

def farthest_endpoints(lines_xyxy: np.ndarray) -> tuple[tuple[int,int], tuple[int,int]]:
    pts = np.vstack([lines_xyxy[:, :2], lines_xyxy[:, 2:4]]).astype(np.float32)
    if len(pts) > 256:
        idx = np.linspace(0, len(pts)-1, 256).astype(int); pts = pts[idx]
    d2 = ((pts[:,None,:]-pts[None,:,:])**2).sum(-1)
    i,j = np.unravel_index(np.argmax(d2), d2.shape)
    return tuple(map(int, pts[i])), tuple(map(int, pts[j]))

def validate_line(edges_like: np.ndarray, coh: np.ndarray, ori_deg: np.ndarray,
                  rho: float, theta_rad: float,
                  band: int, min_density: float, min_mean_coh: float,
                  max_odisp: float, ori_tol: float, border_reject_frac: float,
                  halo: np.ndarray | None, border_margin_px: int) -> tuple[bool, np.ndarray | None, dict | None]:
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
        w = 1.0 - hmask * (1.0 - HO_ALPHA_HALO)
        cvals = cvals * w
    mean_coh = float(cvals.mean())
    odisp    = float(np.percentile(d[hits_mask>0], 95)) if xs.size else 180.0
    if mean_coh < min_mean_coh or odisp > max_odisp: return False, None, None

    bm = border_mask((H,W), border_margin_px)
    if cv2.countNonZero(cv2.bitwise_and(hits_mask,bm))/float(hits) >= border_reject_frac:
        return False, None, None
    return True, hits_mask, dict(density=dens, mean_coh=mean_coh)

def trim_to_support(p1: tuple[int,int], p2: tuple[int,int],
                    hits_mask: np.ndarray, max_gap: int, min_len: int) -> tuple[tuple[int,int], tuple[int,int]]:
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

# ========================= ρ–θ accumulator (points) ==========================
def hough_points_image(bw: np.ndarray) -> np.ndarray:
    img_bool = np.ascontiguousarray(bw.astype(bool))
    theta = np.linspace(-np.pi/2, np.pi/2, int(HO_ACC_NUM_THETA), endpoint=False, dtype=np.float64)
    H, angles, dists = hough_line(img_bool, theta=theta)
    thr = np.percentile(H, float(HO_PEAKS_PERCENTILE))
    step_deg = 180.0 / float(HO_ACC_NUM_THETA)
    min_angle_bins = max(1, int(round(float(HO_PEAKS_MIN_ANGLE_DEG) / step_deg)))
    min_dist_bins  = max(1, int(HO_PEAKS_MIN_DIST))
    hvals, angs, rhos = hough_line_peaks(
        H, angles, dists,
        threshold=thr, num_peaks=int(HO_PEAKS_MAX),
        min_distance=min_dist_bins, min_angle=min_angle_bins
    )
    pts = np.zeros_like(H, dtype=np.uint8)
    if len(hvals):
        ang_idx = np.searchsorted(angles, np.array(angs)); ang_idx = np.clip(ang_idx, 0, len(angles)-1)
        rho_idx = np.searchsorted(dists,  np.array(rhos));  rho_idx = np.clip(rho_idx, 0, len(dists)-1)
        r = int(HO_PEAKS_DOT_RADIUS)
        if r <= 0:
            pts[rho_idx, ang_idx] = 255
        else:
            for ri, ai in zip(rho_idx, ang_idx):
                cv2.circle(pts, (int(ai), int(ri)), r, 255, -1)
    return pts

# ========================= patch pipeline ====================================
def remove_small_components(mask: np.ndarray, min_size: int=500) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: out[labels==i]=255
    return out

def extract_segments_from_seed(seed: np.ndarray, coh: np.ndarray, ori: np.ndarray, halo: np.ndarray, s: float) -> tuple[list[tuple[tuple[int,int],tuple[int,int]]], np.ndarray, int]:
    """Local HoughP + cluster + support validation + trimming."""
    min_comp = karea(MIN_COMP_SIZE_PATCH, s)
    area_thresh = karea(AREA_THRESHOLD_PATCH, s)
    scans_local=[{**prm, "minLineLength": klen(prm["minLineLength"], s),
                         "maxLineGap":   klen(prm["maxLineGap"],   s)} for prm in HOUGHP_LOCAL_SCAN_PATCH]

    accum_valid = np.zeros_like(seed)
    segments = []
    rejected90 = 0

    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            continue
        mask = np.zeros_like(seed); cv2.drawContours(mask,[cnt],-1,255,cv2.FILLED)
        loc_mask, loc_lines = houghp_fuse(mask, scans_local)
        if loc_lines is None:
            continue
        L = loc_lines
        ang = (np.degrees(np.arctan2(L[:,3]-L[:,1], L[:,2]-L[:,0]))%180.0).reshape(-1,1)
        labels = DBSCAN(eps=5, min_samples=1).fit(ang).labels_
        uniq = np.unique(labels)
        if len(uniq) >= 5:
            continue
        for lb in uniq:
            sel = L[labels==lb]
            if len(sel)==0: continue
            cl_mask = np.zeros_like(mask)
            for x1,y1,x2,y2 in sel: cv2.line(cl_mask,(x1,y1),(x2,y2),255,1)
            cl_cnts,_ = cv2.findContours(cl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c2 in cl_cnts:
                if cv2.contourArea(c2) < area_thresh:
                    continue
                cm = np.zeros_like(mask); cv2.drawContours(cm,[c2],-1,255,cv2.FILLED)
                f = []
                for x1,y1,x2,y2 in sel:
                    if cm[y1,x1]>0 and cm[y2,x2]>0: f.append([x1,y1,x2,y2])
                if not f: continue
                f = np.array(f)
                p1,p2 = farthest_endpoints(f)
                if HO_REJECT_EXACT_VERTICAL and (p1[0] == p2[0]):
                    rejected90 += 1
                    continue
                dx = p2[0]-p1[0]; dy=p2[1]-p1[1]
                th = math.atan2(dy, dx) + np.pi/2.0
                rho = p1[0]*math.cos(th) + p1[1]*math.sin(th)

                ok,hits,stats = validate_line(
                    seed, coh, ori, rho, th,
                    band=SUPPORT_BAND_PX_PATCH,
                    min_density=SUPPORT_MIN_DENSITY_PATCH,
                    min_mean_coh=SUPPORT_MIN_MEAN_COH_PATCH,
                    max_odisp=SUPPORT_MAX_ODISP_PATCH,
                    ori_tol=ORI_TOL_VALID_PATCH,
                    border_reject_frac=BORDER_REJECT_FRAC_PATCH,
                    halo=halo, border_margin_px=HO_BORDER_MARGIN_PX
                )
                if not ok: continue
                dens = stats["density"]
                max_gap = max(10, int(round(PROFILE_MAX_GAP_BASE * (1.0 + 0.7*(0.12 - min(0.12, dens))/0.12))))
                a,b = trim_to_support(p1,p2,hits,max_gap=max_gap,min_len=PROFILE_MIN_LEN_PATCH)
                segments.append((a,b))
                accum_valid |= hits

    # light “small-components” cleanup on the seed (optional)
    seed = remove_small_components(seed, min_size=min_comp)
    return segments, accum_valid, rejected90

def process_patch(edges: np.ndarray, coh: np.ndarray, ori: np.ndarray, halo: np.ndarray,
                  dirs_full: List[float], s: float) -> tuple[list[tuple[tuple[int,int],tuple[int,int]]], dict]:
    """Global scans gated by dominant directions + local refinement."""
    scans_global=[{**prm, "minLineLength": klen(prm["minLineLength"], s),
                         "maxLineGap":   klen(prm["maxLineGap"],   s)} for prm in HOUGHP_GLOBAL_SCAN_PATCH]

    accum_all = np.zeros_like(edges)
    segments_all = []

    tried_dirs = list(dirs_full) if dirs_full else []

    for th_deg in tried_dirs:
        # profiles: do not scale thresholds (dimensionless), only closers
        for prof in PROFILES_PATCH:
            thr = max(prof["COH_THR_ABS"], float(np.percentile(coh, prof["COH_THR_PERC"])))
            delta = np.vectorize(ang_diff_deg)(ori, th_deg).astype(np.float32)
            gate  = (delta <= prof["ORI_TOL_DEG"]).astype(np.uint8)*255
            cmk   = (coh >= thr).astype(np.uint8)*255
            ed = cv2.bitwise_and(edges, cv2.bitwise_and(cmk, gate))
            if np.count_nonzero(ed)==0:
                continue
            ed = oriented_close_one(ed, th_deg, length=klen(prof["DIR_CLOSE_LEN"], s))
            hmask,_ = houghp_fuse(ed, scans_global)
            seed = cv2.bitwise_or(ed, hmask)
            seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((CLOSE_K_SZ,CLOSE_K_SZ), np.uint8))

            segs, accum_valid, _ = extract_segments_from_seed(seed, coh, ori, halo, s)
            if segs:
                segments_all.extend(segs); accum_all |= accum_valid
                break  # stop at first profile that yields lines

        if segments_all:
            break

    # soft fallback if nothing found
    if not segments_all:
        cmk0 = (coh >= max(0.06, np.percentile(coh, 98.0))).astype(np.uint8)*255
        hist_dirs = find_orientation_modes(ori, cmk0, k=2, sep=12.0)
        for th_deg in hist_dirs:
            delta = np.vectorize(ang_diff_deg)(ori, th_deg).astype(np.float32)
            gate  = (delta <= 22.0).astype(np.uint8)*255
            ed = cv2.bitwise_and(edges, cmk0 & gate)
            if np.count_nonzero(ed)==0: continue
            ed = oriented_close_one(ed, th_deg, length=klen(45, s))
            seed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE, np.ones((CLOSE_K_SZ,CLOSE_K_SZ), np.uint8))
            segs, accum_valid, _ = extract_segments_from_seed(seed, coh, ori, halo, s)
            if segs:
                segments_all.extend(segs); accum_all |= accum_valid
                break

    debug_maps = dict(accum=accum_all)
    return segments_all, debug_maps

# ========================= per-full pipeline =================================
def process_full(path: Path, target_max_side: int | None) -> None:
    name = path.stem
    print(f"[+] {name}")

    # --- read full
    img0 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        print("   (skip) read fail"); return
    H0, W0 = img0.shape

    # --- scale image for processing (<= full; default from config HO_DEFAULT_RES)
    if target_max_side is None:
        scale = 1.0
        img = img0.copy()
    else:
        scale = min(1.0, float(target_max_side) / float(max(H0, W0)))
        img = cv2.resize(img0, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_AREA) if scale<1.0 else img0.copy()

    # scaling factor vs baseline 2048
    proc_maxside = max(img.shape[:2])
    s = max(1.0, proc_maxside / BASELINE_MAXSIDE)

    # --- preprocess (context, cheap) -----------------------------------------
    base_det = percentile_stretch(img, HO_PERC_LOW, HO_PERC_HIGH)
    base_det = apply_clahe(base_det, HO_CLAHE_CLIP, (klen(HO_CLAHE_TILE[0], s), klen(HO_CLAHE_TILE[1], s)))
    base_det = flatten_bg(base_det, ksz(HO_BG_MED_K, s))
    edges_det= auto_canny(base_det, HO_CANNY_SIGMA)

    # thin vertical artifact (optional, cheap)
    am = detect_thin_vertical_artifacts(edges_det,
                                        connect_len=klen(HO_ARTV_CONNECT_LEN, s),
                                        min_area=karea(HO_ARTV_MIN_AREA, s),
                                        max_thick=klen(HO_ARTV_MAX_THICK, s),
                                        W=img.shape[1], H=img.shape[0])
    if np.count_nonzero(am) > 0:
        img = cv2.inpaint(img, am, 3, cv2.INPAINT_TELEA)

    # --- definitive preprocess for full-context maps --------------------------
    base_full = percentile_stretch(img, HO_PERC_LOW, HO_PERC_HIGH)
    base_full = apply_clahe(base_full, HO_CLAHE_CLIP, (klen(HO_CLAHE_TILE[0], s), klen(HO_CLAHE_TILE[1], s)))
    base_full = flatten_bg(base_full, ksz(HO_BG_MED_K, s))
    edges_full= auto_canny(base_full, HO_CANNY_SIGMA)

    bm_hard_full = border_mask(base_full.shape, klen(HO_BORDER_HARD_ZERO_PX, s))
    halo_full    = bright_halo_mask(base_full, q=HO_HALO_Q, grow=klen(HO_HALO_GROW, s))

    # --- coherence/orientation + dominant directions -------------------------
    p0 = PROFILES_FULL[0]
    coh_win = ksz(p0["COH_WIN"], max(1.0, s**0.5))
    coh_full, ori_full = coherence_map(base_full, coh_win)
    thr_full = max(p0["COH_THR_ABS"], float(np.percentile(coh_full, p0["COH_THR_PERC"])))
    cmk_full = (coh_full >= thr_full).astype(np.uint8)*255
    cmk_full[bm_hard_full>0] = 0; cmk_full[halo_full>0] = 0

    dirs_full = find_orientation_modes(ori_full, cmk_full, k=3, sep=12.0)
    if not dirs_full:
        # fallback using edges
        Hh, ang_, rho_ = hough_line((edges_full>0))
        _, ths, _ = hough_line_peaks(Hh, ang_, rho_, num_peaks=3, min_distance=10, min_angle=4)
        dirs_full = [float((np.degrees(t)%180.0)) for t in ths]
    print(f"   dirs_full={['%.1f' % d for d in dirs_full]}")

    # --- crop to grid and slice maps -----------------------------------------
    base_c, R, C = crop_to_grid(base_full, PATCH_SIZE)
    edges_c,_,_  = crop_to_grid(edges_full, PATCH_SIZE)
    halo_c,_,_   = crop_to_grid(halo_full, PATCH_SIZE)
    coh_c,_,_    = crop_to_grid(coh_full, PATCH_SIZE)
    ori_c,_,_    = crop_to_grid(ori_full, PATCH_SIZE)

    # --- output dirs per full -------------------------------------------------
    full_id_m = re.match(r"(full\d+)", name)
    full_id = full_id_m.group(1) if full_id_m else name
    out_acc_dir = PATCH_HOUGH_ACC_DIR / full_id      # base patches
    out_inv_dir = PATCH_HOUGH_INV_DIR / full_id
    out_acc_dir.mkdir(parents=True, exist_ok=True)
    out_inv_dir.mkdir(parents=True, exist_ok=True)

    # augmented subdirs (one per angle)
    aug_dirs_acc = {ang: (PATCH_HOUGH_ACC_DIR / f"{full_id}_rot{ang:03d}") for ang in AUGMENT_ANGLES}
    aug_dirs_inv = {ang: (PATCH_HOUGH_INV_DIR / f"{full_id}_rot{ang:03d}") for ang in AUGMENT_ANGLES}
    for ang in AUGMENT_ANGLES:
        aug_dirs_acc[ang].mkdir(parents=True, exist_ok=True)
        aug_dirs_inv[ang].mkdir(parents=True, exist_ok=True)

    # mosaics (collect tiles)
    inv_tiles_base: List[np.ndarray] = []
    inv_tiles_aug: dict[int, List[np.ndarray]] = {ang: [] for ang in AUGMENT_ANGLES}

    # --- per-patch processing -------------------------------------------------
    for i in range(R*C):
        r, c = divmod(i, C)
        sl = slice(r*PATCH_SIZE, (r+1)*PATCH_SIZE)
        sc = slice(c*PATCH_SIZE, (c+1)*PATCH_SIZE)

        # context-aware maps for this patch
        raw_patch  = base_c [sl, sc]         # “raw” here is the equalized+flattened base context crop
        edges      = edges_c[sl, sc]
        coh        = coh_c  [sl, sc]
        ori        = ori_c  [sl, sc]
        halo       = halo_c [sl, sc]

        # 1) main detection on the base patch
        segs, dbg = process_patch(edges, coh, ori, halo, dirs_full, s=s)

        # inverse (base patch)
        inv_tile = np.zeros_like(edges, np.uint8)
        for a,b in segs:
            cv2.line(inv_tile, a, b, 255, 2, cv2.LINE_8)
        inv_tiles_base.append(inv_tile)
        cv2.imwrite(str(out_inv_dir / f"{full_id}_{r:02d}_{c:02d}_{SFX_INV}.png"), inv_tile)

        # accumulator (base patch) – gated by coherence vs dominant dirs
        thr_acc = max(PROFILES_PATCH[0]["COH_THR_ABS"], float(np.percentile(coh, PROFILES_PATCH[0]["COH_THR_PERC"])))
        acc_seed = np.zeros_like(edges, np.uint8)
        for th_deg in dirs_full:
            delta = np.vectorize(ang_diff_deg)(ori, th_deg).astype(np.float32)
            gate  = (delta <= PROFILES_PATCH[0]["ORI_TOL_DEG"]).astype(np.uint8)*255
            cmk   = (coh >= thr_acc).astype(np.uint8)*255
            ed = cv2.bitwise_and(edges, cv2.bitwise_and(cmk, gate))
            if np.count_nonzero(ed) == 0:
                continue
            ed = oriented_close_one(ed, th_deg, length=klen(PROFILES_PATCH[0]["DIR_CLOSE_LEN"], s))
            acc_seed = cv2.bitwise_or(acc_seed, ed)
        if np.count_nonzero(acc_seed) == 0:
            acc_seed = edges  # fallback
        acc_pts = hough_points_image(acc_seed>0)
        cv2.imwrite(str(out_acc_dir / f"{full_id}_{r:02d}_{c:02d}_{SFX_H}.png"), acc_pts)

        # 2) ON-THE-FLY AUGMENTATIONS (only if trails were detected)
        if segs:
            for ang in AUGMENT_ANGLES:
                # rotate raw patch (equalized/flattened context crop)
                rot_patch = np.ascontiguousarray(np.rot90(raw_patch, k=(ang//90) % 4))
                # recompute patch-level preproc on the rotated patch (light and local)
                rot_eq  = percentile_stretch(rot_patch, HO_PERC_LOW, HO_PERC_HIGH)
                rot_eq  = apply_clahe(rot_eq, HO_CLAHE_CLIP, HO_CLAHE_TILE)
                rot_eq  = flatten_bg(rot_eq, ksz(HO_BG_MED_K, 1.0))
                rot_ed  = auto_canny(rot_eq, HO_CANNY_SIGMA)
                rot_coh, rot_ori = coherence_map(rot_eq, ksz(PROFILES_PATCH[0]["COH_WIN"], 1.0))
                rot_halo = bright_halo_mask(rot_eq, q=HO_HALO_Q, grow=HO_HALO_GROW)

                # rotate dominant directions too (mod 180)
                rot_dirs = [float((d + ang) % 180.0) for d in dirs_full]

                # run detector on rotated patch
                segs_rot, dbg_rot = process_patch(rot_ed, rot_coh, rot_ori, rot_halo, rot_dirs, s=1.0)

                # inverse (aug patch)
                inv_tile_rot = np.zeros_like(rot_ed, np.uint8)
                for a,b in segs_rot:
                    cv2.line(inv_tile_rot, a, b, 255, 2, cv2.LINE_8)
                inv_tiles_aug[ang].append(inv_tile_rot)
                cv2.imwrite(str(aug_dirs_inv[ang] / f"{full_id}_{r:02d}_{c:02d}_{SFX_INV}.png"), inv_tile_rot)

                # accumulator (aug patch)
                thr_acc_r = max(PROFILES_PATCH[0]["COH_THR_ABS"], float(np.percentile(rot_coh, PROFILES_PATCH[0]["COH_THR_PERC"])))
                acc_seed_r = np.zeros_like(rot_ed, np.uint8)
                for th_deg in rot_dirs:
                    delta = np.vectorize(ang_diff_deg)(rot_ori, th_deg).astype(np.float32)
                    gate  = (delta <= PROFILES_PATCH[0]["ORI_TOL_DEG"]).astype(np.uint8)*255
                    cmk   = (rot_coh >= thr_acc_r).astype(np.uint8)*255
                    edr = cv2.bitwise_and(rot_ed, cv2.bitwise_and(cmk, gate))
                    if np.count_nonzero(edr) == 0:
                        continue
                    edr = oriented_close_one(edr, th_deg, length=PROFILES_PATCH[0]["DIR_CLOSE_LEN"])
                    acc_seed_r = cv2.bitwise_or(acc_seed_r, edr)
                if np.count_nonzero(acc_seed_r) == 0:
                    acc_seed_r = rot_ed
                acc_pts_r = hough_points_image(acc_seed_r>0)
                cv2.imwrite(str(aug_dirs_acc[ang] / f"{full_id}_{r:02d}_{c:02d}_{SFX_H}.png"), acc_pts_r)
        else:
            # keep mosaic slot consistent for aug paths (append zeros) if desired
            for ang in AUGMENT_ANGLES:
                inv_tiles_aug[ang].append(np.zeros_like(inv_tile, np.uint8))

    # --- save full-level products (mosaics) -----------------------------------
    FULL_HOUGH_FILT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_HOUGH_COH_DIR.mkdir(parents=True, exist_ok=True)
    FULL_HOUGH_INV_DIR.mkdir(parents=True, exist_ok=True)

    # filtered/coherence mosaics are the cropped context maps
    cv2.imwrite(str(FULL_HOUGH_FILT_DIR / f"{name}.png"), base_c)
    cv2.imwrite(str(FULL_HOUGH_COH_DIR  / f"{name}.png"), (255*np.clip(coh_c,0,1)).astype(np.uint8))

    # inverse mosaics (base + per-rotation)
    save_mosaic(inv_tiles_base, (R, C), FULL_HOUGH_INV_DIR / f"{name}.png")
    for ang in AUGMENT_ANGLES:
        save_mosaic(inv_tiles_aug[ang], (R, C), FULL_HOUGH_INV_DIR / f"{name}_rot{ang:03d}.png")

    print(f"   grid={R}x{C} patches ({R*C}) | saved full mosaics and per-patch outputs.")

# ========================= main =============================================
def main():
    ap = argparse.ArgumentParser(description="Patch-wise Hough with on-the-fly rotation aug for detected patches.")
    ap.add_argument("--res", type=int, default=None,
                    help="Max processing side. Omit to use config HO_DEFAULT_RES. Eg.: --res 2048")
    args = ap.parse_args()

    # Use CLI override if provided; else use HO_DEFAULT_RES from config (0 → native).
    if args.res is not None:
        target_max_side = args.res if args.res > 0 else None
    else:
        target_max_side = HO_DEFAULT_RES if (HO_DEFAULT_RES and HO_DEFAULT_RES > 0) else None

    inputs = list_full_inputs()
    if not inputs:
        raise FileNotFoundError(f"No full found in {FULL_RAW_STD_DIR} or {FULL_RAW_AUG_DIR}")
    print(f"[INFO] Hough patches for {len(inputs)} images (res={'full' if target_max_side is None else target_max_side})")

    for p in inputs:
        process_full(p, target_max_side)

    print("[DONE]")

if __name__ == "__main__":
    main()
