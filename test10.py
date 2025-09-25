#!/usr/bin/env python3
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
for sub in ("overlay", "lines", "panel", "debug"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

def list_inputs():
    files = []
    for ext in ("png","PNG","jpg","JPG","jpeg","JPEG","tif","TIF","tiff","TIFF"):
        files.extend(RAW_DIR.glob(f"full*.{ext}"))
    def num_key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    return sorted(files, key=num_key)

# =========================
# TUNABLES (↑ = more sensitive / ↓ = more conservative)
# =========================
PROC_MAX_SIDE = 2048

# Base preprocessing
PERC_LOW, PERC_HIGH = 0.5, 99.9
CLAHE_CLIP, CLAHE_TILE = 2.7, (16, 16)      # ↑ clip → more local contrast
BG_MED_K = 31                                # odd; ↑ removes large-scale glow harder
CANNY_SIGMA = 0.33                           # ↑ → wider hysteresis (more edges)

# Global suppressions
BORDER_MARGIN_PX = 12                        # ↑ stronger border penalty
HALO_Q = 99.87                               # ↑ tighter halo mask
HALO_GROW = 21                               # dilation (pixels)
ALPHA_HALO = 0.5                             # coherence down-weight inside halo

# Vertical 1px artifacts (columns)
VERT_TOL = 2
VERT_MIN_RUN_R = 0.10
VERT_MAX_WIDTH = 1

# Coherence profiles (used as RAMP levels from permissive→restrictive)
# Keep 3 levels max for speed.
PROFILES = [
    # Permissive (for very faint/fragmented)
    dict(LABEL="perm", COH_WIN=11, COH_THR_ABS=0.08, COH_THR_PERC=98.2, ORI_TOL_DEG=16.0, DIR_CLOSE_LEN=75),
    # Medium
    dict(LABEL="med",  COH_WIN=9,  COH_THR_ABS=0.18, COH_THR_PERC=99.2, ORI_TOL_DEG=12.0, DIR_CLOSE_LEN=55),
    # Restrictive (cleanest)
    dict(LABEL="rest", COH_WIN=9,  COH_THR_ABS=0.28, COH_THR_PERC=99.5, ORI_TOL_DEG=10.0, DIR_CLOSE_LEN=61),
]

# Multi-level HoughP scans (minLineLength, maxLineGap) – fused
# Scan from "connect more" → "stricter"
HOUGHP_GLOBAL_SCAN = [
    dict(rho=1, theta=np.pi/180, threshold=45, minLineLength=70,  maxLineGap=260),
    dict(rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=220),
]

HOUGHP_LOCAL_SCAN = [
    dict(rho=1, theta=np.pi/180, threshold=40, minLineLength=50, maxLineGap=14),
    dict(rho=1, theta=np.pi/180, threshold=50, minLineLength=60, maxLineGap=12),
]

# Validation inside a support band around the line
SUPPORT_BAND_PX      = 5                     # ↑ wider band (more hits, more tolerant)
SUPPORT_MIN_DENSITY  = 0.028                 # ↑ to reduce FPs; ↓ to keep faint tracks
SUPPORT_MIN_MEAN_COH = 0.26                  # ↑ to reduce FPs
SUPPORT_MAX_ODISP    = 18.0                  # max angular dispersion p95 (deg)
ORI_TOL_VALID        = 14.0                  # ↑ tolerant to orientation noise
BORDER_REJECT_FRAC   = 0.88                  # ↑ stronger border rejection

# Area/connectivity
AREA_THRESHOLD = 3000
MIN_COMP_SIZE  = 500
CLOSE_K_SZ     = 3

# =========================
# Utils
# =========================
def percentile_stretch(u8, p_low, p_high):
    lo, hi = np.percentile(u8, [p_low, p_high])
    if hi <= lo: return u8
    x = np.clip((u8.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return x.astype(np.uint8)

def apply_clahe(u8, clip, tile):
    if clip <= 0: return u8
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile)).apply(u8)

def flatten_bg(u8, k):
    if k <= 1 or k % 2 == 0: return u8
    bg = cv2.medianBlur(u8, k)
    x = cv2.subtract(u8, bg)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def auto_canny(u8, sigma):
    v = float(np.median(u8))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    if hi <= lo: hi = lo + 1
    return cv2.Canny(u8, lo, hi, L2gradient=True)

def detect_vertical_artifacts_strict(u8, tol=2, min_run_r=0.07, max_width=1):
    H, W = u8.shape
    min_run = max(16, int(min_run_r * H))
    white = (u8 >= (255 - tol)).astype(np.uint8)
    black = (u8 <= tol).astype(np.uint8)
    def width1(b):
        left = np.zeros_like(b);  left[:, 1:]  = b[:, :-1]
        right= np.zeros_like(b);  right[:, :-1]= b[:, 1:]
        return (b & (~left) & (~right)).astype(np.uint8)
    w1 = width1(white); b1 = width1(black)
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_run))
    w_long = cv2.morphologyEx(w1 * 255, cv2.MORPH_OPEN, vker)
    b_long = cv2.morphologyEx(b1 * 255, cv2.MORPH_OPEN, vker)
    mask = cv2.bitwise_or(w_long, b_long)
    if max_width > 1:
        hker = cv2.getStructuringElement(cv2.MORPH_RECT, (max_width + 1, 1))
        too_wide = cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, hker)
        mask[too_wide > 0] = 0
    return mask

def border_mask(shape, margin):
    H, W = shape
    bm = np.zeros((H, W), np.uint8)
    m = int(max(1, margin))
    bm[:m, :] = 255; bm[-m:, :] = 255; bm[:, :m] = 255; bm[:, -m:] = 255
    return bm

def bright_halo_mask(u8, q=99.87, grow=21):
    th = np.percentile(u8, q)
    bin_ = cv2.threshold(u8, th, 255, cv2.THRESH_BINARY)[1]
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        bin_ = cv2.dilate(bin_, k, 1)
    return bin_

# --- Structure tensor ---
def coherence_map(u8, win=9):
    f = u8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    k = int(win) | 1
    Axx = cv2.GaussianBlur(gx*gx, (k,k), 0)
    Axy = cv2.GaussianBlur(gx*gy, (k,k), 0)
    Ayy = cv2.GaussianBlur(gy*gy, (k,k), 0)
    tmp = (Axx - Ayy)**2 + 4.0*(Axy**2)
    tmp = np.sqrt(np.maximum(tmp, 0))
    l1 = 0.5*((Axx + Ayy) + tmp)
    l2 = 0.5*((Axx + Ayy) - tmp)
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    ori = 0.5*np.degrees(np.arctan2(2*Axy, (Axx - Ayy))) % 180.0
    return np.clip(coh, 0, 1).astype(np.float32), ori.astype(np.float32)

def ang_diff_deg(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def find_orientation_modes(ori_deg, mask, k=3, sep=12.0):
    th = ori_deg[mask > 0].ravel()
    if th.size == 0: return []
    hist, edges = np.histogram(th, bins=np.arange(0, 181, 1.0))
    peaks = []
    for _ in range(k):
        i = int(np.argmax(hist))
        if hist[i] == 0: break
        center = 0.5*(edges[i] + edges[i+1]); peaks.append(center)
        a = int(max(0, i - int(sep))); b = int(min(len(hist), i + int(sep) + 1))
        hist[a:b] = 0
    return peaks

def oriented_close_one(img_u8, theta_deg, length=61):
    ksz = int(length) | 1
    ker = np.zeros((ksz, ksz), np.uint8); c = ksz // 2
    cv2.line(ker, (0, c), (ksz - 1, c), 255, 1)
    M = cv2.getRotationMatrix2D((c, c), float(theta_deg), 1.0)
    ker = cv2.warpAffine(ker, M, (ksz, ksz), flags=cv2.INTER_NEAREST)
    ker = (ker > 0).astype(np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, ker, iterations=1)

def remove_small_components(mask, min_size=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size: out[labels == i] = 255
    return out

def farthest_endpoints(lines_xyxy):
    pts = np.vstack([lines_xyxy[:, :2], lines_xyxy[:, 2:4]]).astype(np.float32)
    if len(pts) > 256:
        idx = np.linspace(0, len(pts) - 1, 256).astype(int); pts = pts[idx]
    d2 = ((pts[:, None, :] - pts[None, :, :])**2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return tuple(map(int, pts[i])), tuple(map(int, pts[j]))

def validate_line(edges_like, coh, ori_deg, rho, theta_rad,
                  halo=None, band=5, min_density=0.03, min_mean_coh=0.30,
                  max_odisp=16.0, ori_tol=12.0, border_reject_frac=0.88, border_margin=12):
    """Validate a candidate infinite line using support within a band.
       edges_like should encode the seed/edge support (uint8 0/255)."""
    H, W = edges_like.shape
    diag = int(np.hypot(H, W))
    a, b = math.cos(theta_rad), math.sin(theta_rad)
    x0, y0 = a * rho, b * rho
    p1 = (int(x0 + diag * (-b)), int(y0 + diag * (a)))
    p2 = (int(x0 - diag * (-b)), int(y0 - diag * (a)))

    band_mask = np.zeros((H, W), np.uint8)
    cv2.line(band_mask, p1, p2, 255, 2 * band + 1, cv2.LINE_8)

    # Orientation gating relative to line angle
    d = np.vectorize(ang_diff_deg)(ori_deg, np.degrees(theta_rad)).astype(np.float32)
    ori_gate = (d <= ori_tol).astype(np.uint8) * 255

    hits_mask = cv2.bitwise_and(band_mask, cv2.bitwise_and(edges_like, ori_gate))
    hits = cv2.countNonZero(hits_mask); total = cv2.countNonZero(band_mask)
    if hits == 0 or total == 0: return False, None
    dens = hits / float(total)
    if dens < min_density: return False, None

    ys, xs = np.where(hits_mask > 0)
    if xs.size == 0: return False, None
    # Coherence mean with halo down-weight
    cvals = coh[ys, xs]
    if halo is not None:
        hmask = (halo[ys, xs] > 0).astype(np.float32)
        w = 1.0 - hmask * (1.0 - ALPHA_HALO)
        cvals = cvals * w
    mean_coh = float(cvals.mean())
    odisp    = float(np.percentile(d[hits_mask > 0], 95))
    if mean_coh < min_mean_coh or odisp > max_odisp: return False, None

    bm = border_mask((H, W), border_margin)
    if cv2.countNonZero(cv2.bitwise_and(hits_mask, bm)) / float(hits) >= border_reject_frac:
        return False, None
    return True, hits_mask

def draw_panel(img_w, accum, overlay, lines_only, name):
    DARK_GREEN = (22, 40, 22)
    def tile(img, title, W=900, H=800, title_h=36):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        h, w = vis.shape[:2]; s = min(W / w, H / h)
        vis = cv2.resize(vis, (int(w * s), int(h * s)), cv2.INTER_AREA)
        box = np.zeros((title_h + H, W, 3), np.uint8); box[:] = DARK_GREEN
        cv2.rectangle(box, (0, 0), (W - 1, title_h - 1), (30, 30, 30), -1)
        cv2.putText(box, title, (12, title_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_8)
        y0 = title_h + (H - vis.shape[0]) // 2; x0 = (W - vis.shape[1]) // 2
        box[y0:y0 + vis.shape[0], x0:x0 + vis.shape[1]] = vis
        cv2.rectangle(box, (0, 0), (W - 1, title_h + H - 1), DARK_GREEN, 2)
        return box
    A = tile(img_w,   "Preprocessed (stretch+CLAHE+flatten)")
    B = tile(accum,   "Seed/HoughP mask (accum)")
    C = tile(overlay, "Overlay (white segments)")
    D = tile(lines_only, "Lines only (white)")
    h, w = A.shape[:2]; sep = 8; margin = 10
    panel = np.zeros((2*h + 3*margin + sep, 2*w + 3*margin + sep, 3), np.uint8); panel[:] = DARK_GREEN
    panel[margin:margin+h, margin:margin+w] = A
    panel[margin:margin+h, margin+w+sep:margin+w+sep+w] = B
    panel[margin+h+sep:margin+h+sep+h, margin:margin+w] = C
    panel[margin+h+sep:margin+h+sep+h, margin+w+sep:margin+w+sep+w] = D
    cv2.imwrite(str(OUT_DIR / "panel" / f"{name}_panel.png"), panel)

# =========================
# Core helpers (ramp + extraction)
# =========================
def houghp_fuse(mask, scans):
    """Run multiple HoughLinesP scans and fuse results onto a single 1px mask."""
    fused = np.zeros_like(mask)
    lines_all = []
    for prm in scans:
        lines = cv2.HoughLinesP(mask, **prm)
        if lines is None: continue
        L = lines[:, 0, :]
        lines_all.append(L)
        for x1, y1, x2, y2 in L:
            cv2.line(fused, (x1, y1), (x2, y2), 255, 1)
    return fused, (np.vstack(lines_all) if lines_all else None)

def extract_segments_from_seed(seed, coh, ori, halo, scale,
                               area_thresh=AREA_THRESHOLD,
                               hough_local_scans=HOUGHP_LOCAL_SCAN):
    """Contour → local HoughP → DBSCAN by angle → farthest endpoints → validate."""
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accum_valid = np.zeros_like(seed)
    segments = []

    for cnt in contours:
        if cv2.contourArea(cnt) < (area_thresh * scale * scale if scale < 1.0 else area_thresh):
            continue
        mask = np.zeros_like(seed); cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

        # Run local HoughP scans and fuse per contour
        loc_mask, loc_lines = houghp_fuse(mask, hough_local_scans)
        if loc_lines is None: 
            continue
        L = loc_lines

        # 0–180 angles
        ang = (np.degrees(np.arctan2(L[:,3]-L[:,1], L[:,2]-L[:,0])) % 180.0).reshape(-1, 1)
        labels = DBSCAN(eps=5, min_samples=1).fit(ang).labels_
        uniq = np.unique(labels)
        if len(uniq) >= 5:   # too many orientations → probably noise
            continue

        for lb in uniq:
            sel = L[labels == lb]
            if len(sel) == 0: continue

            # cluster mask → break into connected groups to avoid mixing separated fragments
            cl_mask = np.zeros_like(mask)
            for x1, y1, x2, y2 in sel:
                cv2.line(cl_mask, (x1,y1), (x2,y2), 255, 1)
            cl_cnts, _ = cv2.findContours(cl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c2 in cl_cnts:
                if cv2.contourArea(c2) < (area_thresh * scale * scale if scale < 1.0 else area_thresh):
                    continue
                cm = np.zeros_like(mask); cv2.drawContours(cm, [c2], -1, 255, cv2.FILLED)

                f = []
                for x1, y1, x2, y2 in sel:
                    if cm[y1, x1] > 0 and cm[y2, x2] > 0:
                        f.append([x1, y1, x2, y2])
                if not f: 
                    continue
                f = np.array(f)

                p1, p2 = farthest_endpoints(f)
                # Convert segment to (rho, theta)
                dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
                th = math.atan2(dy, dx) + np.pi / 2.0   # normal
                rho = p1[0] * math.cos(th) + p1[1] * math.sin(th)

                ok, hits = validate_line(seed, coh, ori, rho, th,
                                         halo=halo,
                                         band=SUPPORT_BAND_PX,
                                         min_density=SUPPORT_MIN_DENSITY,
                                         min_mean_coh=SUPPORT_MIN_MEAN_COH,
                                         max_odisp=SUPPORT_MAX_ODISP,
                                         ori_tol=ORI_TOL_VALID,
                                         border_reject_frac=BORDER_REJECT_FRAC,
                                         border_margin=BORDER_MARGIN_PX)
                if not ok:
                    continue
                segments.append((p1, p2))
                accum_valid |= hits
    return segments, accum_valid

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for img_path in list_inputs():
        name = img_path.stem
        print(f"[+] {img_path}")

        # --- Load & downscale for speed
        img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            print("   (skip)")
            continue
        H0, W0 = img0.shape
        scale = min(1.0, PROC_MAX_SIDE / float(max(H0, W0)))
        img = cv2.resize(img0, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img0.copy()

        # --- Remove strict 1px vertical artifacts
        vmask = detect_vertical_artifacts_strict(img, tol=VERT_TOL, min_run_r=VERT_MIN_RUN_R, max_width=VERT_MAX_WIDTH)
        if np.count_nonzero(vmask) > 0:
            img = cv2.inpaint(img, vmask, 3, cv2.INPAINT_TELEA)

        # --- Base preprocessing
        base = percentile_stretch(img, PERC_LOW, PERC_HIGH)
        base = apply_clahe(base, CLAHE_CLIP, CLAHE_TILE)
        base = flatten_bg(base, BG_MED_K)
        edges0 = auto_canny(base, CANNY_SIGMA)

        # --- Global suppressions
        bm = border_mask(base.shape, int(round(BORDER_MARGIN_PX * scale)) if scale < 1.0 else BORDER_MARGIN_PX)
        halo = bright_halo_mask(base, q=HALO_Q, grow=int(round(HALO_GROW * scale)) if scale < 1.0 else HALO_GROW)

        # --- Coherence maps cache by window size (avoid recompute)
        coh_cache = {}
        ori_cache = {}

        # --- Find initial dominant orientations on a permissive mask
        # Use the most permissive profile to propose modes
        p_perm = PROFILES[0]
        if p_perm["COH_WIN"] not in coh_cache:
            coh_cache[p_perm["COH_WIN"]], ori_cache[p_perm["COH_WIN"]] = coherence_map(base, p_perm["COH_WIN"])
        coh0, ori0 = coh_cache[p_perm["COH_WIN"]], ori_cache[p_perm["COH_WIN"]]
        thr0 = max(p_perm["COH_THR_ABS"], np.percentile(coh0, p_perm["COH_THR_PERC"]))
        cmk0 = (coh0 >= thr0).astype(np.uint8) * 255
        cmk0[bm > 0] = 0; cmk0[halo > 0] = 0

        dirs = find_orientation_modes(ori0, cmk0, k=3, sep=12.0)
        if not dirs:
            # Fallback to standard Hough on edges
            h_, ang_, rho_ = hough_line(edges0 > 0)
            _, ths, _ = hough_line_peaks(h_, ang_, rho_, num_peaks=6, min_distance=10, min_angle=4)
            dirs = [float((np.degrees(t) % 180.0)) for t in ths]

        # --- Adaptive ramp per direction
        segments_all = []
        accum_all = np.zeros_like(edges0)
        ramp_log = []

        for th_deg in dirs:
            found_this_dir = False
            for level_idx, prof in enumerate(PROFILES):
                # Lazily compute coherence/orientation for this window size
                if prof["COH_WIN"] not in coh_cache:
                    coh_cache[prof["COH_WIN"]], ori_cache[prof["COH_WIN"]] = coherence_map(base, prof["COH_WIN"])
                coh = coh_cache[prof["COH_WIN"]]; ori = ori_cache[prof["COH_WIN"]]

                # Threshold from image statistics (percentile) clamped by absolute minimum
                thr = max(prof["COH_THR_ABS"], np.percentile(coh, prof["COH_THR_PERC"]))
                # Orientation gate around current dominant direction
                delta = np.vectorize(ang_diff_deg)(ori, th_deg).astype(np.float32)
                gate = (delta <= prof["ORI_TOL_DEG"]).astype(np.uint8) * 255

                cmk = (coh >= thr).astype(np.uint8) * 255
                cmk[bm > 0] = 0; cmk[halo > 0] = 0

                # Seed = Canny edges intersected with (coherence & orientation)
                ed = cv2.bitwise_and(edges0, cv2.bitwise_and(cmk, gate))
                if np.count_nonzero(ed) == 0:
                    continue

                # Directional closing to connect small gaps
                ed = oriented_close_one(ed, th_deg, length=prof["DIR_CLOSE_LEN"])

                # Global HoughP (multi-level) over current seed and fuse
                hmask, _ = houghp_fuse(ed, HOUGHP_GLOBAL_SCAN)

                seed = cv2.bitwise_or(ed, hmask)
                seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((CLOSE_K_SZ, CLOSE_K_SZ), np.uint8))
                seed = remove_small_components(seed, min_size=int(round(MIN_COMP_SIZE * scale)) if scale < 1.0 else MIN_COMP_SIZE)

                # Extract validated segments inside this direction mask
                segs, accum_valid = extract_segments_from_seed(seed, coh, ori, halo, scale)
                if segs:
                    segments_all.extend(segs)
                    accum_all |= accum_valid
                    ramp_log.append(f"{th_deg:.1f}°→{prof['LABEL']}")
                    found_this_dir = True
                    break  # stop ramp for this direction (success)
            if not found_this_dir:
                ramp_log.append(f"{th_deg:.1f}°→none")

        # If nothing found, do a last global try on the permissive seed (cheap bailout)
        if not segments_all and np.count_nonzero(cmk0) > 0:
            # very permissive global pass to try not to miss tiny short tracks
            ed_global = cv2.bitwise_and(edges0, cmk0)
            ed_global = cv2.morphologyEx(ed_global, cv2.MORPH_CLOSE, np.ones((CLOSE_K_SZ, CLOSE_K_SZ), np.uint8))
            hmask_g, _ = houghp_fuse(ed_global, HOUGHP_GLOBAL_SCAN)
            seed_g = cv2.bitwise_or(ed_global, hmask_g)
            seed_g = remove_small_components(seed_g, min_size=int(round(MIN_COMP_SIZE * scale)) if scale < 1.0 else MIN_COMP_SIZE)

            segs_g, accum_valid_g = extract_segments_from_seed(seed_g, coh0, ori0, halo, scale)
            if segs_g:
                segments_all.extend(segs_g); accum_all |= accum_valid_g
                ramp_log.append("global-bailout")

        # --- Upsample & draw outputs
        def up(pt): 
            return (int(round(pt[0] / scale)), int(round(pt[1] / scale))) if scale < 1.0 else pt

        overlay = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        lines_only = np.zeros_like(img0, np.uint8)
        for a, b in [(up(a), up(b)) for (a, b) in segments_all]:
            cv2.line(overlay, a, b, (255, 255, 255), 3, cv2.LINE_8)
            cv2.line(lines_only, a, b, 255, 3, cv2.LINE_8)

        # Debug: show preprocessed + accumulated seed used for panel
        base_show = percentile_stretch(img, 0.5, 99.9)
        acc_show  = cv2.resize(accum_all, (W0, H0), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else accum_all
        draw_panel(base_show, acc_show, overlay, lines_only, name)

        # Save main outputs
        cv2.imwrite(str(OUT_DIR / "overlay" / f"{name}_overlay.png"), overlay)
        cv2.imwrite(str(OUT_DIR / "lines"   / f"{name}_lines.png"),   lines_only)

        # Optional debug snapshots
        cv2.imwrite(str(OUT_DIR / "debug"   / f"{name}_coherence.png"),
                    (255 * np.clip(coh0, 0, 1)).astype(np.uint8))
        seed_dbg = cv2.bitwise_or(cmk0, edges0)
        cv2.imwrite(str(OUT_DIR / "debug"   / f"{name}_seed.png"), seed_dbg)

        # Log
        print(f"   segments: {len(segments_all)} | ramp: {', '.join(ramp_log) if ramp_log else 'n/a'}")
