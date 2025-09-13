#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# ============================== core paths ===================================
ROOT            = Path(__file__).resolve().parents[1]
IMAGES_DIR      = ROOT / "images"
FULL_DIR        = IMAGES_DIR / "full"
PATCHES_DIR     = IMAGES_DIR / "patches"
LABELS_DIR      = ROOT / "labels"
MODELS_DIR      = ROOT / "models"
RESULTS_DIR     = ROOT / "results"
RESULTS_VIS_DIR = RESULTS_DIR / "vis"

# ============================== full images ==================================
FULL_RAW_DIR    = FULL_DIR / "raw"
FULL_MASKS_DIR  = FULL_DIR / "full_asta_masks"
FULL_HOUGH_DIR     = FULL_DIR / "hough"
FULL_HOUGH_ACC_DIR = FULL_HOUGH_DIR / "accumulator"
FULL_HOUGH_INV_DIR = FULL_HOUGH_DIR / "inversed"

# ============================== patches & naming ==============================
RAW_STD_DIR    = PATCHES_DIR / "raw"   / "standart"   # note: 'standart' per project
RAW_AUG_DIR    = PATCHES_DIR / "raw"   / "augment"
HOUGH_STD_DIR  = PATCHES_DIR / "hough" / "standart"
HOUGH_AUG_DIR  = PATCHES_DIR / "hough" / "augment"
MASK_STD_DIR   = PATCHES_DIR / "mask"  / "standart"

SFX_R = "r"  # raw     (fullX_row_col_r.png)
SFX_H = "h"  # hough   (fullX_row_col_h.png)
SFX_M = "m"  # mask    (fullX_row_col_m.png)

PATCH_SIZE = 224
GRID_ROWS  = 47
GRID_COLS  = 47

# ============================== slice ========================================
SLICE_IN_RAW_DIR   = FULL_RAW_DIR
SLICE_IN_MASK_DIR  = FULL_MASKS_DIR
SLICE_OUT_RAW_DIR  = RAW_STD_DIR
SLICE_OUT_MASK_DIR = MASK_STD_DIR
SLICE_TILE         = PATCH_SIZE
SLICE_GRID_R       = GRID_ROWS
SLICE_GRID_C       = GRID_COLS

# ============================== label/augment/split ==========================
LABELS_ALL_CSV = LABELS_DIR / "all_labels.csv"
AUGMENT_ANGLES = [90, 180, 270]

SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
SPLIT_TEST  = 0.15
SPLIT_SEED  = 42
SPLIT_GROUP_FROM = "parent"
SPLIT_STRATIFY   = True

TRAIN_CSV = LABELS_DIR / "train.csv"
VAL_CSV   = LABELS_DIR / "val.csv"
TEST_CSV  = LABELS_DIR / "test.csv"

# ============================== train/test ===================================
TRAIN_ARCH            = "resnet18"       # "resnet18" | "cnn"(NI)
TRAIN_MODALITY        = "raw"            # "raw" | "hough" | "combined"
TRAIN_EPOCHS          = 12
TRAIN_BATCH_SIZE      = 128
TRAIN_LR              = 1e-3
TRAIN_WEIGHT_DECAY    = 1e-4
TRAIN_WORKERS         = 8
TRAIN_PREFETCH        = 2
TRAIN_AUG_FLIPS       = True
TRAIN_USE_IMAGENET    = True
TRAIN_FREEZE_BACKBONE = False
TRAIN_BALANCE_SAMPLER = False
TRAIN_POS_WEIGHT_CLIP = (0.5, 20.0)
TRAIN_LOG_INTERVAL    = 100
TRAIN_VAL_EVERY       = 1
TRAIN_AMP             = True
TRAIN_TF32            = True
TRAIN_CLIP_NORM       = 0.0

DDP_ENABLED  = True
DDP_BACKEND  = "nccl"

CKPT_DIR        = MODELS_DIR / "resnet"
CKPT_NAME       = "last.pt"
PRED_OUT_DIR    = RESULTS_DIR / "predictions"
METRICS_OUT_DIR = RESULTS_DIR / "metrics"
TEST_BATCH_SIZE = 256
RESULTS_TABLE_CSV = RESULTS_DIR / "summary.csv"

# ============================== Hough (full→ROI / patches) ===================
HOUGH_ACC_INPUT_GLOB  = "full*.png"
HOUGH_ACC_NUM_THETA   = 360
HOUGH_FULL_DOWNSCALE  = 1

# Preproc variants (best chosen by unsupervised score)
HOUGH_AUTO_VARIANTS = [
    {"edge": "sobel_otsu", "open": 0, "close": 0, "clahe": True},
    {"edge": "sobel_otsu", "open": 3, "close": 0, "clahe": True},
    {"edge": "sobel_otsu", "open": 3, "close": 3, "clahe": True},
    {"edge": "canny",      "open": 3, "close": 0, "clahe": True},
]

# Canny thresholds from |grad| percentiles (more permissive)
CANNY_PERC_LOW           = 80.0
CANNY_PERC_HIGH          = 99.0
CANNY_PERC_LOW_FALLBACK  = 75.0
CANNY_PERC_HIGH_FALLBACK = 98.0

# Heatmap score S = w_peak*peakiness(topK) + w_anis*anisotropy
HOUGH_SCORE_TOPK   = 64
HOUGH_SCORE_W_PEAK = 0.7
HOUGH_SCORE_W_ANIS = 0.3

# Peak selection (auto-percentile target)
HPOINTS_K_MIN         = 2
HPOINTS_K_MAX         = 20
HPOINTS_PMIN          = 99.30
HPOINTS_PMAX          = 99.99
HPOINTS_BIN_STEPS     = 10
HPOINTS_MIN_DIST_RHO  = 80      # rho-bin distance (not px), a bit looser
HPOINTS_MIN_ANGLE_DEG = 2.0
HPOINTS_MAX_PEAKS     = 48
HPOINTS_DOT_RADIUS    = 2
HPOINTS_BG_ALPHA      = 0.12

# Refinement
HOUGH_REFINE_PEAK_WINDOW      = 7
HOUGH_REFINE_LOCAL_DRHO_PX    = 6
HOUGH_REFINE_LOCAL_DTHETA_DEG = 0.4
HOUGH_REFINE_LOCAL_STEP_THETA = 0.1

# Artifact handling (col/row defects)
ARTIF_KMAD        = 6.0
ARTIF_MIN_RUN     = 8
ANGLE_VETO_DEG    = 2.0
ANGLE_VETO_WEIGHT = 0.2

# Line validation / pruning (looser)
HOUGH_INV_LINE_THICKNESS = 2
HOUGH_LINE_MINLEN_FULL   = 1500
HOUGH_ONOFF_OFFSET_PX    = 3
HOUGH_KEEP_TOPK          = 8
HOUGH_KEEP_SCORE_PCT     = 90.0     # keep more candidates

# Continuity/width gates (looser)
CONTINUITY_THR_PCT  = 85.0
CONTINUITY_MIN_FRAC = 0.40
WIDTH_REL_THR       = 0.5
WIDTH_MIN_PX        = 0.5
WIDTH_MAX_PX        = 12.0
WIDTH_PROFILE_HALF  = 8
WIDTH_SAMPLES_ALONG = 30

# Emergency fallback (if still 0 lines after pruning)
HOUGH_EMERGENCY_ENABLE = True
HOUGH_EMERG_MINLEN     = 1200
HOUGH_EMERG_TOPK       = 2   # pick top-N by on-off score

# ROI outputs
ROI_IMG_DIR            = LABELS_DIR / "roi"
ROI_OUT_CSV            = LABELS_DIR / "roi_candidates.csv"
ROI_BAND_PX            = 8
ROI_PATCH_MIN_PIXELS   = 300
ROI_PATCH_MIN_RATIO    = 0.005
ROI_VIS_DIR            = RESULTS_VIS_DIR / "roi_full"

# Patch Hough (input to ResNet)
PATCH_THETA_DELTA_DEG  = 6.0

# Parallelism
HOUGH_FULL_JOBS  = 1
HOUGH_PATCH_JOBS = 1

# ============================== combined =====================================
COMBINED_ALPHA = 0.5
