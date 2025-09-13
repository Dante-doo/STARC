#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Grouped 70/15/15 split with refinement to better match global positive rate.
# - Start with GroupShuffleSplit (by full_id)
# - Locally move/swap groups to reduce deviation from global pos_rate
# - Enforce size near targets within a tolerance window
# - Augments go to train only, tied to their base patch key

from pathlib import Path
import re, random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from code.config import (
    LABELS_DIR, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_SEED, SPLIT_TRIES,
    SPLIT_SIZE_TOL_FRAC, SPLIT_BALANCE_ITERS, SPLIT_SCORE_SIZE_WEIGHT
)

RE_FULL = re.compile(r"(full\d+)")
RE_KEY  = re.compile(r"(full\d+_\d+_\d+)")

def is_aug(path: str) -> bool:
    return "/augment/" in path.replace("\\", "/")

def full_id_of(path: str) -> str:
    m = RE_FULL.search(path);
    if not m: raise ValueError(f"Cannot extract full_id from: {path}")
    return m.group(1)

def base_key_of(path: str) -> str:
    m = RE_KEY.search(path)
    if not m: raise ValueError(f"Cannot extract base key from: {path}")
    return m.group(1)

def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["path", "label"]].to_csv(out_path, index=False)

# ---- helpers for refinement --------------------------------------------------
def group_stats(base: pd.DataFrame) -> pd.DataFrame:
    """Return per-full stats: full_id, n, pos."""
    g = base.groupby("full_id")["label"]
    out = pd.DataFrame({"n": g.size(), "pos": g.sum().astype(int)})
    out["neg"] = out["n"] - out["pos"]
    out["rate"] = out["pos"] / out["n"].clip(lower=1)
    out = out.reset_index()
    return out

def split_objective(assign: dict[str,set], stats: pd.DataFrame, targets: dict[str,int], global_pos: float) -> float:
    """Lower is better. Sum absolute pos_rate deviations + size penalty."""
    score = 0.0
    size_pen = 0.0
    for name in ("train","val","test"):
        idx = stats["full_id"].isin(assign[name])
        n   = int(stats.loc[idx, "n"].sum())
        p   = int(stats.loc[idx, "pos"].sum())
        pr  = (p / n) if n else 0.0
        score += abs(pr - global_pos)
        size_pen += abs(n - targets[name])
    return score + SPLIT_SCORE_SIZE_WEIGHT * (size_pen / max(1, stats["n"].sum()))

def can_move(assign_counts: dict[str,int], g_n: int, src: str, dst: str, mins: dict[str,int], maxs: dict[str,int]) -> bool:
    """Check if moving group of size g_n from src->dst keeps sizes within [min,max]."""
    return (assign_counts[src] - g_n >= mins[src]) and (assign_counts[dst] + g_n <= maxs[dst])

def apply_move(assign: dict[str,set], counts: dict[str,int], gid: str, g_n: int, src: str, dst: str):
    assign[src].remove(gid); assign[dst].add(gid)
    counts[src] -= g_n; counts[dst] += g_n

# ---- main --------------------------------------------------------------------
def main():
    labels_csv = LABELS_DIR / "all_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing {labels_csv}. Run: python -m code.label")

    df = pd.read_csv(labels_csv)
    df["is_aug"] = df["path"].apply(is_aug)
    base = df[~df["is_aug"]].copy()  # standard only
    aug  = df[df["is_aug"]].copy()   # augment only

    # add group column
    base["full_id"] = base["path"].apply(full_id_of)
    global_pos = float(base["label"].mean())

    # per-group stats
    gs = group_stats(base)
    total_n = int(gs["n"].sum())

    # size targets and tolerance windows
    t_train = int(round(total_n * SPLIT_TRAIN))
    t_val   = int(round(total_n * SPLIT_VAL))
    t_test  = int(round(total_n * SPLIT_TEST))
    tol     = int(round(total_n * float(SPLIT_SIZE_TOL_FRAC)))
    mins = {"train": t_train - tol, "val": t_val - tol, "test": t_test - tol}
    maxs = {"train": t_train + tol, "val": t_val + tol, "test": t_test + tol}
    targets = {"train": t_train, "val": t_val, "test": t_test}

    # ---- initial grouped split (like before) ----
    best = None
    base_y = base["label"].to_numpy()
    base_g = base["full_id"].to_numpy()
    for t in range(SPLIT_TRIES):
        rs = SPLIT_SEED + t
        gss1 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - SPLIT_TRAIN), random_state=rs)
        tr_idx, rest_idx = next(gss1.split(base, base_y, base_g))
        rest = base.iloc[rest_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=rs)
        va_rel, te_rel = next(gss2.split(rest, rest["label"].to_numpy(), rest["full_id"].to_numpy()))
        va_idx, te_idx = rest.index[va_rel], rest.index[te_rel]

        S = {
            "train": set(base.loc[base.index[tr_idx], "full_id"]),
            "val":   set(base.loc[va_idx, "full_id"]),
            "test":  set(base.loc[te_idx, "full_id"]),
        }
        score = split_objective(S, gs, targets, global_pos)
        if best is None or score < best[0]:
            best = (score, S)
    assign = {k: set(v) for k, v in best[1].items()}

    # current sizes per split (in samples, not groups)
    counts = {name: int(gs.loc[gs["full_id"].isin(gset), "n"].sum()) for name, gset in assign.items()}

    # ---- local refinement (single-group moves, occasional swaps) ----
    gid2n = dict(zip(gs["full_id"], gs["n"]))
    random.seed(SPLIT_SEED)

    best_score = split_objective(assign, gs, targets, global_pos)
    for it in range(int(SPLIT_BALANCE_ITERS)):
        # try a move
        gid = random.choice(gs["full_id"].tolist())
        # find its source split
        src = "train" if gid in assign["train"] else ("val" if gid in assign["val"] else "test")
        dst = random.choice([x for x in ("train","val","test") if x != src])
        g_n = gid2n[gid]

        if not can_move(counts, g_n, src, dst, mins, maxs):
            continue

        # tentatively move
        apply_move(assign, counts, gid, g_n, src, dst)
        s = split_objective(assign, gs, targets, global_pos)
        if s <= best_score:  # accept improvement (<= keeps equal-score moves)
            best_score = s
        else:
            # revert if worse
            apply_move(assign, counts, gid, g_n, dst, src)

        # occasional swap attempt (optional)
        if (it % 50) == 0:
            # pick two different splits
            a, b = random.sample(["train","val","test"], 2)
            # pick groups
            if assign[a] and assign[b]:
                ga = random.choice(list(assign[a])); gb = random.choice(list(assign[b]))
                na, nb = gid2n[ga], gid2n[gb]
                # size check after swap
                if (counts[a] - na + nb >= mins[a] and counts[a] - na + nb <= maxs[a] and
                    counts[b] - nb + na >= mins[b] and counts[b] - nb + na <= maxs[b]):
                    # apply swap
                    assign[a].remove(ga); assign[b].add(ga); counts[a] = counts[a] - na + nb
                    assign[b].remove(gb); assign[a].add(gb); counts[b] = counts[b] - nb + na
                    s2 = split_objective(assign, gs, targets, global_pos)
                    if s2 <= best_score:
                        best_score = s2
                    else:
                        # revert
                        assign[a].remove(gb); assign[b].add(gb); counts[a] = counts[a] - nb + na
                        assign[b].remove(ga); assign[a].add(ga); counts[b] = counts[b] - na + nb

    # ---- materialize rows for each split ----
    def rows_for(gset: set[str]) -> pd.DataFrame:
        return base[base["full_id"].isin(gset)][["path","label"]].copy()

    tr_base = rows_for(assign["train"])
    va_base = rows_for(assign["val"])
    te_base = rows_for(assign["test"])

    # add augments only if their base key is in train base
    tr_keys = set(tr_base["path"].apply(base_key_of).tolist())
    if not aug.empty:
        aug["base_key"] = aug["path"].apply(base_key_of)
        aug_tr = aug[aug["base_key"].isin(tr_keys)][["path","label"]]
        train = pd.concat([tr_base, aug_tr], ignore_index=True)
    else:
        train = tr_base

    # save
    write_csv(train, LABELS_DIR / "train.csv")
    write_csv(va_base, LABELS_DIR / "val.csv")
    write_csv(te_base, LABELS_DIR / "test.csv")

    # report
    def stats(name, d):
        n = len(d); p = int(d["label"].sum()); neg = n - p
        pr = (p / n) if n else 0.0
        print(f"{name:>5}: n={n}  pos={p}  neg={neg}  pos_rate={pr:.6f}")
    print("\n=== split summary (base counts only for rates) ===")
    base_train = tr_base; base_val = va_base; base_test = te_base
    gpr = float(base["label"].mean())
    print(f"global pos_rate (base): {gpr:.6f}")
    stats("train", base_train)
    stats("val",   base_val)
    stats("test",  base_test)
    print(f"\nsaved: {LABELS_DIR/'train.csv'}, {LABELS_DIR/'val.csv'}, {LABELS_DIR/'test.csv'}")

if __name__ == "__main__":
    main()
