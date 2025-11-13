#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CT Patch Quality Diagnostics – now detects padded/constant patches.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def safe_nonzero_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    mask = arr != 0
    if mask.any():
        nz = arr[mask]
        return float(nz.mean()), float(nz.max()), float(mask.mean())
    return 0.0, 1.0, 0.0


def normalize_casewise(volume: np.ndarray) -> Tuple[np.ndarray, Dict]:
    mu_nz, max_nz, frac_nz = safe_nonzero_stats(volume)
    scale = max_nz - mu_nz if max_nz - mu_nz > 1e-6 else 1.0
    norm = (volume - mu_nz) / (scale + 1e-6)
    norm = np.clip(norm, -1.0, 1.0)
    return norm, {"mu_nz": mu_nz, "max_nz": max_nz, "frac_nz": frac_nz}


def same_value_ratio(patch: np.ndarray) -> float:
    """Fraction of voxels that equal the most common value (mode)."""
    flat = patch.ravel()
    if flat.size == 0:
        return 1.0
    values, counts = np.unique(flat, return_counts=True)
    return float(counts.max() / flat.size)

def rejected_summary(self, out_dir: Path, thresholds: Dict):
    from collections import Counter
    import pandas as pd

    # reuse the masks from save_rejected_patches
    s = np.array(self.stats["same_value_ratio"]) > thresholds["max_same_value_ratio"]
    f = np.array(self.stats["fg_ratio"]) < thresholds["min_fg_ratio"]
    m = np.array(self.stats["mean"]) < thresholds["min_mean"]
    d = np.array(self.stats["std"]) < thresholds["min_std"]
    v = np.array(self.stats["very_low_ratio"]) > thresholds["max_very_low_ratio"]
    any_rej = s | f | m | d | v

    df = pd.DataFrame({
        "case_id": self.stats["case_id"],
        "rejected": any_rej.astype(int)
    })
    summary = df.groupby("case_id").agg(
        rejected_patches=("rejected", "sum"),
        total_patches=("rejected", "count")
    ).reset_index()
    summary["reject_rate_%"] = summary["rejected_patches"] / summary["total_patches"] * 100

    out_path = out_dir / "rejected_by_case.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Per-case summary → {out_path}")
# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class PatchQualityAnalyzer:
    def __init__(
        self,
        data_pairs: List[Dict],
        patch_size: Tuple[int, int] = (128, 192),
        patch_depth: int = 15,
        body_threshold: float = -500.0,
        num_samples: int = 1000,
        uniformity_thr: float = 0.90,      # <--- NEW
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.patch_depth = patch_depth
        self.body_threshold = body_threshold
        self.num_samples = min(num_samples, len(data_pairs) * 200)
        self.uniformity_thr = uniformity_thr

        self.stats = {
            "same_value_ratio": [],   # NEW
            "fg_ratio": [],
            "mean": [],
            "std": [],
            "very_low_ratio": [],
            "dark_ratio": [],         # legacy
            "case_id": [],
            "position": [],           # (pair_idx, z, y, x)
        }

    # ------------------------------------------------------------------- #
    def _extract_patch(self, norm_vol, raw_vol, z, y, x):
        half = self.patch_depth // 2
        z0 = max(0, z - half)
        z1 = min(norm_vol.shape[0], z + half + 1)

        p_n = norm_vol[z0:z1, y : y + self.patch_size[0], x : x + self.patch_size[1]]
        p_r = raw_vol[z0:z1, y : y + self.patch_size[0], x : x + self.patch_size[1]]

        if p_n.shape[0] < self.patch_depth:
            pad_before = (self.patch_depth - p_n.shape[0]) // 2
            pad_after = self.patch_depth - p_n.shape[0] - pad_before
            pad = ((pad_before, pad_after), (0, 0), (0, 0))
            p_n = np.pad(p_n, pad, mode="edge")
            p_r = np.pad(p_r, pad, mode="edge")
        return p_n, p_r

    # ------------------------------------------------------------------- #
    def _patch_stats(self, patch_n, patch_r):
        flat_n, flat_r = patch_n.ravel(), patch_r.ravel()

        fg = (flat_r > self.body_threshold).mean() if self.body_threshold is not None else (flat_r != 0).mean()
        mean = float(flat_n.mean())
        std = float(flat_n.std())
        very_low = (flat_n < -0.8).mean()
        dark = (flat_n < -0.9).mean()
        same = same_value_ratio(patch_r)          # <-- RAW patch (preserves padding value)

        return {
            "same_value_ratio": same,
            "fg_ratio": fg,
            "mean": mean,
            "std": std,
            "very_low_ratio": very_low,
            "dark_ratio": dark,
        }

    # ------------------------------------------------------------------- #
    def analyze(self):
        log.info(f"Sampling up to {self.num_samples} patches …")
        np.random.seed(42)
        sampled = 0
        pbar = tqdm(total=self.num_samples, desc="Patches")

        while sampled < self.num_samples:
            idx = np.random.randint(len(self.data_pairs))
            pair = self.data_pairs[idx]

            try:
                vol = nib.load(pair["source_path"]).get_fdata()
                raw = np.transpose(vol, (2, 1, 0))          # (D,H,W)
                D, H, W = raw.shape

                if D < self.patch_depth + 2 or H < self.patch_size[0] or W < self.patch_size[1]:
                    continue

                norm, _ = normalize_casewise(raw)

                pad = self.patch_depth // 2
                z = np.random.randint(pad + self.patch_depth, D - pad - self.patch_depth)
                y = np.random.randint(0, H - self.patch_size[0] + 1)
                x = np.random.randint(0, W - self.patch_size[1] + 1)

                p_n, p_r = self._extract_patch(norm, raw, z, y, x)
                st = self._patch_stats(p_n, p_r)

                # store
                for k in self.stats:
                    if k == "case_id":
                        self.stats[k].append(pair.get("case_id", "unknown"))
                    elif k == "position":
                        self.stats[k].append((idx, z, y, x))
                    else:
                        self.stats[k].append(st[k])

                sampled += 1
                pbar.update(1)

            except Exception as e:
                log.debug(f"Skip pair {idx}: {e}")
                continue

        pbar.close()
        log.info(f"Collected {sampled} patches")

    # ------------------------------------------------------------------- #
    def percentiles(self):
        out = {}
        for key in self.stats:
            if key in ("case_id", "position"):
                continue
            vals = np.array(self.stats[key])
            out[key] = {f"p{p}": float(np.percentile(vals, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
        return out

    # ------------------------------------------------------------------- #
    def recommend(self, reject_rate: float = 0.05):
        p = reject_rate * 100
        s = np.array(self.stats["same_value_ratio"])
        f = np.array(self.stats["fg_ratio"])
        m = np.array(self.stats["mean"])
        d = np.array(self.stats["std"])
        v = np.array(self.stats["very_low_ratio"])

        return {
            "max_same_value_ratio": float(np.percentile(s, 100 - p)),
            "min_fg_ratio": float(np.percentile(f, p)),
            "min_mean": float(np.percentile(m, p)),
            "min_std": float(np.percentile(d, p)),
            "max_very_low_ratio": float(np.percentile(v, 100 - p)),
        }

    # ------------------------------------------------------------------- #
    def plot_distributions(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        metrics = [
            ("same_value_ratio", "tab:red"),
            ("fg_ratio", "tab:orange"),
            ("std", "tab:purple"),
        ]

        for i, (name, col) in enumerate(metrics):
            data = np.array(self.stats[name])
            # histogram
            ax = fig.add_subplot(gs[i, 0])
            ax.hist(data, bins=50, edgecolor="black", alpha=0.7, color=col)
            ax.axvline(np.median(data), color="black", ls="--", label=f"median {np.median(data):.3f}")
            ax.set_xlabel(name.replace("_", " ").title())
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # box
            ax = fig.add_subplot(gs[i, 1])
            ax.boxplot([data])
            ax.set_ylabel(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

            # CDF
            ax = fig.add_subplot(gs[i, 2])
            sorted_ = np.sort(data)
            ax.plot(np.linspace(0, 100, len(sorted_)), sorted_, color=col)
            ax.set_xlabel("Percentile")
            ax.set_ylabel(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Patch statistics (n={len(self.stats['mean'])})", fontsize=16)
        plt.savefig(out_dir / "distributions.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Distributions saved to {out_dir}")

    def save_rejected_patches(self, out_dir: Path, thresholds: Dict):
        """Write a CSV with every patch that fails any threshold."""
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "rejected_patches.csv"

        # --- compute boolean masks once ---
        s = np.array(self.stats["same_value_ratio"]) > thresholds["max_same_value_ratio"]
        f = np.array(self.stats["fg_ratio"]) < thresholds["min_fg_ratio"]
        m = np.array(self.stats["mean"]) < thresholds["min_mean"]
        d = np.array(self.stats["std"]) < thresholds["min_std"]
        v = np.array(self.stats["very_low_ratio"]) > thresholds["max_very_low_ratio"]
        any_rej = s | f | m | d | v

        rows = []
        for i in np.where(any_rej)[0]:
            case = self.stats["case_id"][i]
            idx, z, y, x = self.stats["position"][i]
            rows.append({
                "case_id": case,
                "pair_idx": idx,
                "z": z,
                "y": y,
                "x": x,
                "same_value_ratio": self.stats["same_value_ratio"][i],
                "fg_ratio": self.stats["fg_ratio"][i],
                "mean": self.stats["mean"][i],
                "std": self.stats["std"][i],
                "very_low_ratio": self.stats["very_low_ratio"][i],
                "reject_same": int(s[i]),
                "reject_fg": int(f[i]),
                "reject_mean": int(m[i]),
                "reject_std": int(d[i]),
                "reject_very_low": int(v[i]),
            })

        if not rows:
            log.info("No rejected patches → empty CSV")
            return

        import csv
        keys = rows[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        log.info(f"Saved {len(rows)} rejected patches → {csv_path}")

    # ------------------------------------------------------------------- #
    def report(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        perc = self.percentiles()
        rec5 = self.recommend(0.05)

        # ---- current thresholds (feel free to edit) ----
        cur = {
            "max_same_value_ratio": 0.90,   # <--- NEW
            "min_fg_ratio": 0.05,
            "min_mean": -0.60,
            "min_std": 0.03,
            "max_very_low_ratio": 0.60,
        }

        # ---- rejection counts ----
        s = np.array(self.stats["same_value_ratio"]) > cur["max_same_value_ratio"]
        f = np.array(self.stats["fg_ratio"]) < cur["min_fg_ratio"]
        m = np.array(self.stats["mean"]) < cur["min_mean"]
        d = np.array(self.stats["std"]) < cur["min_std"]
        v = np.array(self.stats["very_low_ratio"]) > cur["max_very_low_ratio"]
        any_rej = s | f | m | d | v
        total = len(any_rej)

        rej = {
            "same": int(s.sum()),
            "fg": int(f.sum()),
            "mean": int(m.sum()),
            "std": int(d.sum()),
            "very_low": int(v.sum()),
            "any": int(any_rej.sum()),
        }

        txt = f"""
{'='*70}
CT PATCH QUALITY DIAGNOSTIC REPORT
{'='*70}
Total patches analysed : {total:,}

PERCENTILES
-----------
Same-value ratio   p5: {perc['same_value_ratio']['p5']:.4f}   p50: {perc['same_value_ratio']['p50']:.4f}   p95: {perc['same_value_ratio']['p95']:.4f}
Foreground ratio   p5: {perc['fg_ratio']['p5']:.4f}   p50: {perc['fg_ratio']['p50']:.4f}   p95: {perc['fg_ratio']['p95']:.4f}
Mean (norm)        p5: {perc['mean']['p5']:.4f}   p50: {perc['mean']['p50']:.4f}   p95: {perc['mean']['p95']:.4f}
Std dev            p5: {perc['std']['p5']:.4f}   p50: {perc['std']['p50']:.4f}   p95: {perc['std']['p95']:.4f}
Very-low ratio     p5: {perc['very_low_ratio']['p5']:.4f}   p50: {perc['very_low_ratio']['p50']:.4f}   p95: {perc['very_low_ratio']['p95']:.4f}

CURRENT THRESHOLDS
------------------
max_same_value_ratio : {cur['max_same_value_ratio']}
min_fg_ratio         : {cur['min_fg_ratio']}
min_mean             : {cur['min_mean']}
min_std              : {cur['min_std']}
max_very_low_ratio   : {cur['max_very_low_ratio']}

REJECTIONS
----------
Same-value > max   : {rej['same']:,} ({rej['same']/total*100:5.1f}%)
FG < min           : {rej['fg']:,} ({rej['fg']/total*100:5.1f}%)
Mean < min         : {rej['mean']:,} ({rej['mean']/total*100:5.1f}%)
Std < min          : {rej['std']:,} ({rej['std']/total*100:5.1f}%)
Very-low > max     : {rej['very_low']:,} ({rej['very_low']/total*100:5.1f}%)
ANY REJECT         : {rej['any']:,} ({rej['any']/total*100:5.1f}%)

RECOMMENDED (reject worst 5%)
------------------------------
max_same_value_ratio : {rec5['max_same_value_ratio']:.4f}
min_fg_ratio         : {rec5['min_fg_ratio']:.4f}
min_mean             : {rec5['min_mean']:.4f}
min_std              : {rec5['min_std']:.4f}
max_very_low_ratio   : {rec5['max_very_low_ratio']:.4f}
"""
        rate = rej['any'] / total
        if rate > 0.20:
            txt += "\nWARNING HIGH REJECTION (>20%)\nCheck data padding / registration.\n"
        elif rate > 0.10:
            txt += "\nWARNING Moderate rejection (10-20%)\nInspect a few rejected patches.\n"
        else:
            txt += "\nOK Reasonable rejection (<10%)\nThresholds look good.\n"

        (out_dir / "report.txt").write_text(txt, encoding="utf-8")
        log.info(f"Report → {out_dir / 'report.txt'}")



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(config_path: str = None):
    if config_path:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    else:
        cfg = {
            "data_dir": "../ncct_cect/vindr_ds/deformable_registered_bspline",
            "labels_csv": "../ncct_cect/vindr_ds/labels.csv",
            "patch_size": [96, 96],
            "patch_depth": 15,
            "body_threshold": -500.0,
            "num_samples": 300000,
        }

    # ---- your own dataloader ------------------------------------------------
    import sys
    sys.path.append("/mnt/project")
    from dataloader_train import create_data_pairs, load_phase_mapping

    phase_map = load_phase_mapping(cfg["labels_csv"]) if Path(cfg["labels_csv"]).exists() else None
    splits = create_data_pairs(cfg["data_dir"], phase_mapping=phase_map)
    pairs = splits["train"] + splits["test"] + splits["val"]
    # === ADD THIS BEFORE analyzer = PatchQualityAnalyzer(...) ===
    log.info(f"Test split has {len(pairs)} volumes")
    for i, p in enumerate(pairs[:10]):  # show first 10
        try:
            vol = nib.load(p["source_path"]).get_fdata()
            raw = np.transpose(vol, (2,1,0))
            log.info(f"  [{i}] {p.get('case_id','?')} → shape {raw.shape}")
        except Exception as e:
            log.warning(f"  [{i}] FAILED to load: {e}")
    # ---- run ---------------------------------------------------------------
    ana = PatchQualityAnalyzer(
        data_pairs=pairs,
        patch_size=tuple(cfg["patch_size"]),
        patch_depth=cfg["patch_depth"],
        body_threshold=cfg["body_threshold"],
        num_samples=cfg["num_samples"],
    )
    ana.analyze()
    out = Path("diagnostics")
    ana.plot_distributions(out)
    ana.report(out)

    # ---- NEW: save rejected list ----
    thresholds = {
        "max_same_value_ratio": 0.90,
        "min_fg_ratio": 0.05,
        "min_mean": -0.60,
        "min_std": 0.03,
        "max_very_low_ratio": 0.60,
    }
    ana.save_rejected_patches(out, thresholds)

    print("\n=== DONE ===\nCheck folder:", out.resolve())

    # ana = PatchQualityAnalyzer(
    #     data_pairs=pairs,
    #     patch_size=tuple(cfg["patch_size"]),
    #     patch_depth=cfg["patch_depth"],
    #     body_threshold=cfg["body_threshold"],
    #     num_samples=cfg["num_samples"],
    #     uniformity_thr=0.90,               # <-- change if you want stricter/looser
    # )
    # ana.analyze()
    # out = Path("diagnostics")
    # ana.plot_distributions(out)
    # ana.report(out)

    print("\n=== DONE ===\nCheck folder:", out.resolve())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="JSON config")
    args = p.parse_args()
    main(args.config)