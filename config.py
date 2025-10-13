#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Single source of truth for paths and knobs used by slice/label/hough/combined/train.

from pathlib import Path
import os
import numpy as np

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
# New layout: raw/standart and raw/augment
FULL_RAW_BASE_DIR = FULL_DIR / "raw"
FULL_RAW_STD_DIR  = FULL_RAW_BASE_DIR / "standart"
FULL_RAW_AUG_DIR  = FULL_RAW_BASE_DIR / "augment"

# Backward-compat alias used by slicing stage
FULL_RAW_DIR    = FULL_RAW_STD_DIR

FULL_MASKS_DIR = FULL_DIR / "full_asta_masks"

# Hough (full)
FULL_HOUGH_DIR        = FULL_DIR / "hough"
FULL_HOUGH_INV_DIR    = FULL_HOUGH_DIR / "inverted"
FULL_HOUGH_COH_DIR    = FULL_HOUGH_DIR / "coherence"
FULL_HOUGH_FILT_DIR   = FULL_HOUGH_DIR / "filtered"
FULL_HOUGH_FILTER_DIR = FULL_HOUGH_FILT_DIR  # alias consistent

# ============================== patches & naming ==============================
RAW_STD_DIR    = PATCHES_DIR / "raw"   / "standart"
RAW_AUG_DIR    = PATCHES_DIR / "raw"   / "augment"
# hough outputs under:
PATCH_HOUGH_ACC_DIR = PATCHES_DIR / "hough" / "accumulator"
PATCH_HOUGH_INV_DIR = PATCHES_DIR / "hough" / "inverted"

MASK_STD_DIR   = PATCHES_DIR / "mask"  / "standart"

# File suffixes
SFX_R   = "r"     # raw     -> fullX_row_col_r.png
SFX_H   = "h"     # hough   -> points acc (per patch)
SFX_M   = "m"     # mask    -> fullX_row_col_m.png
SFX_INV = "inv"   # inverse -> fullX_row_col_inv.png

# Patch geometry (S L I C E) — 224@9x9 on 2048 downscale (original full 10560)
PATCH_SIZE = 224
GRID_ROWS  = 9
GRID_COLS  = 9

# ============================== slice ========================================
SLICE_IN_RAW_DIR   = FULL_RAW_STD_DIR
SLICE_IN_MASK_DIR  = FULL_DIR / "full_asta_masks"
SLICE_OUT_RAW_DIR  = RAW_STD_DIR
SLICE_OUT_MASK_DIR = MASK_STD_DIR
SLICE_TILE         = PATCH_SIZE
SLICE_GRID_R       = GRID_ROWS
SLICE_GRID_C       = GRID_COLS

# Downscale do full-frame antes de fatiar:
#  - 0   => não redimensiona (usa resolução nativa)
#  - > 0 => limita o maior lado a este valor, preservando aspecto
SLICE_DOWNSCALE_MAXSIDE = 2048  # deixe alinhado ao HO_DEFAULT_RES

# Interpolação (usa strings; o slice.py converte p/ PIL):
# valores aceitos: "nearest" | "bilinear" | "bicubic" | "lanczos"
SLICE_RESAMPLE_RAW  = "lanczos"   # qualidade melhor para imagem contínua
SLICE_RESAMPLE_MASK = "nearest"   # preserva rótulos da máscara
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

# Try to keep base splits close to global pos_rate and target sizes
SPLIT_ENFORCE_GLOBAL_POS_RATE = True
SPLIT_TRIES = 1000
SPLIT_ABS_TOL = 0.0025
SPLIT_VERBOSE = True
SPLIT_SIZE_TOL_FRAC = 0.01
SPLIT_BALANCE_ITERS = 5000
SPLIT_SCORE_SIZE_WEIGHT = 0.2

# ============================== training =====================================
TRAIN_ARCH            = "resnet18"       # currently only resnet18 implemented
TRAIN_MODALITY        = "raw"            # "raw" | "hough" | "combined"
TRAIN_EPOCHS          = 12
TRAIN_BATCH_SIZE      = 32
TRAIN_LR              = 1e-3
TRAIN_WORKERS         = 4
TRAIN_PREFETCH        = 2
TRAIN_PRETRAINED      = True             # ImageNet weights
TRAIN_FREEZE_BACKBONE = False

# Logging / validation cadence (needed by train.py)
TRAIN_VAL_EVERY       = 1                # validate every N epochs
TRAIN_LOG_INTERVAL    = 100              # steps between train logs

# Loss (default focal for robustness to imbalance)
TRAIN_LOSS            = "focal"          # "focal" | "bce"
FOCAL_ALPHA           = 0.25
FOCAL_GAMMA           = 2.0

# DDP / perf
TRAIN_DDP             = True
TRAIN_CHANNELS_LAST   = False
TRAIN_MIXED_PRECISION = False            # AMP
TRAIN_TF32            = True
TRAIN_COMPILE         = False

# Weighted sampler / BCE pos_weight (used only if TRAIN_LOSS="bce")
TRAIN_USE_SAMPLER     = False
TRAIN_USE_POS_WEIGHT  = False

# Inference threshold & sweep for validation
TRAIN_THRESHOLD       = 0.50
TRAIN_SELECT_METRIC   = "balacc"   # {'balacc','f1','f2'}
TRAIN_SWEEP_MIN_T     = 0.10
TRAIN_SWEEP_MAX_T     = 0.90
TRAIN_SWEEP_STEPS     = 41         # step 0.02

TRAIN_MIXED_PRECISION = True
TRAIN_CHANNELS_LAST   = True   # também ajuda um pouco

# Augment policy for the TRAIN SET rows (independent of TF flips in Dataset)
# - "none": drop augmented rows, only base
# - "rebalance": keep a subset of aug to match base pos_rate (plus cap per base)
# - "all": keep all rows as-is
TRAIN_AUG_POLICY      = "rebalance"
TRAIN_MATCH_BASE_RATE = True
TRAIN_TARGET_POS_RATE = None
TRAIN_AUG_MAX_PER_BASE= 1

# Checkpoints / results
CKPT_DIR        = MODELS_DIR / "resnet"  # composed dynamically in train.py
CKPT_NAME       = "last.pt"
EVAL_BATCH_SIZE  = 32
RESULTS_PREDS_DIR   = RESULTS_DIR / "preds"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_SUMMARY_DIR = RESULTS_DIR / "summary"
EVAL_WORKERS    = 8
EVAL_PREFETCH   = 2

# ============================== Hough (full + patches) =======================
# Inputs: search both standart and augment (full context)
HO_INPUT_GLOB   = "full*.png"
HO_INPUT_DIRS   = [FULL_RAW_STD_DIR, FULL_RAW_AUG_DIR]

# Full resolution control: process context around 2048 (downscale from 10560)
HO_DEFAULT_RES  = 2048             # 0 = native; >0 = max side target
HO_BASE_RES     = 2048             # scaling baseline for pixel knobs

# Preprocessing (baseline 2048)
HO_PERC_LOW, HO_PERC_HIGH = 0.5, 99.9
HO_CLAHE_CLIP, HO_CLAHE_TILE = 2.7, (16,16)
HO_BG_MED_K      = 31
HO_CANNY_SIGMA   = 0.33

# Borders & halo
HO_BORDER_HARD_ZERO_PX = 8
HO_BORDER_MARGIN_PX    = 12
HO_HALO_Q, HO_HALO_GROW, HO_ALPHA_HALO = 99.87, 21, 0.5

# Very thin vertical artifact
HO_ARTV_MAX_THICK      = 2
HO_ARTV_MIN_AREA       = 25
HO_ARTV_CONNECT_LEN    = 61
HO_ARTV_BORDER_XMARGIN = 3
HO_ARTV_FULLH_TOL      = 3

# Exact 90° rejection
HO_REJECT_EXACT_VERTICAL = True

# Full profiles (baseline 2048)
PROFILES_FULL = [
    dict(LABEL="perm", COH_WIN=11, COH_THR_ABS=0.08, COH_THR_PERC=98.2, ORI_TOL_DEG=18.0, DIR_CLOSE_LEN=80),
    dict(LABEL="med",  COH_WIN=9,  COH_THR_ABS=0.18, COH_THR_PERC=99.2, ORI_TOL_DEG=12.0, DIR_CLOSE_LEN=55),
    dict(LABEL="rest", COH_WIN=9,  COH_THR_ABS=0.28, COH_THR_PERC=99.5, ORI_TOL_DEG=10.0, DIR_CLOSE_LEN=61),
]
# Patch profiles (dimensionless thresholds; close length scaled relative to patch)
PROFILES_PATCH = [
    dict(LABEL="perm", COH_WIN=11, COH_THR_ABS=0.06, COH_THR_PERC=98.0, ORI_TOL_DEG=22.0, DIR_CLOSE_LEN=60),
    dict(LABEL="med",  COH_WIN=9,  COH_THR_ABS=0.14, COH_THR_PERC=99.0, ORI_TOL_DEG=16.0, DIR_CLOSE_LEN=45),
    dict(LABEL="rest", COH_WIN=9,  COH_THR_ABS=0.22, COH_THR_PERC=99.4, ORI_TOL_DEG=14.0, DIR_CLOSE_LEN=45),
]

# HoughP scans (patch)
HOUGHP_GLOBAL_SCAN_PATCH = [
    dict(rho=1, theta=np.pi/180, threshold=22, minLineLength=36, maxLineGap=110),
    dict(rho=1, theta=np.pi/180, threshold=27, minLineLength=48, maxLineGap=80),
]
HOUGHP_LOCAL_SCAN_PATCH = [
    dict(rho=1, theta=np.pi/180, threshold=20, minLineLength=28, maxLineGap=8),
    dict(rho=1, theta=np.pi/180, threshold=24, minLineLength=36, maxLineGap=6),
]

# Validation (patch)
SUPPORT_BAND_PX_PATCH      = 7
SUPPORT_MIN_DENSITY_PATCH  = 0.012
SUPPORT_MIN_MEAN_COH_PATCH = 0.16
SUPPORT_MAX_ODISP_PATCH    = 18.0
ORI_TOL_VALID_PATCH        = 20.0
BORDER_REJECT_FRAC_PATCH   = 1.10

# Connectivity
AREA_THRESHOLD_PATCH = 400
MIN_COMP_SIZE_PATCH  = 120
CLOSE_K_SZ           = 3

# Trim
PROFILE_MAX_GAP_BASE = 22
PROFILE_MIN_LEN_PATCH= 24

# Accumulator (ρ,θ) per patch (points)
HO_ACC_NUM_THETA        = 360
HO_PEAKS_PERCENTILE     = 98.5
HO_PEAKS_MIN_DIST       = 8
HO_PEAKS_MIN_ANGLE_DEG  = 2.0
HO_PEAKS_MAX            = 24
HO_PEAKS_DOT_RADIUS     = 2

# Parallelism
HO_JOBS = max(1, (os.cpu_count() or 4) // 2)

# ============================== combined =====================================
COMBINED_ALPHA = 0.5  # (unused for stacking; kept for potential blends)
COMBINED_STD_DIR = PATCHES_DIR / "combined" / "standart"
COMBINED_AUG_DIR = PATCHES_DIR / "combined" / "augment"
SFX_C = "cmb"
COMBINED_JOBS = 8
