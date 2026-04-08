"""Standalone top-K evaluator for sampled_*.csv files.

Loads every `sampled_*.csv` in an experiment directory, adds a `topk` boolean
column (True if the predicted reactant set canonically matches the ground
truth), writes `evaluated_*.csv` next to each input, and prints top-K
accuracy for K in [1, 3, 5, 10, 20, 50, 100] (overall and per ground-truth
class).

Replicates `multiguide.dataset.helpers.compare_reactant_smiles` /
`clear_atom_map` inline so the script has zero dependency on the multiguide
package — only rdkit + pandas.

Usage:

    python scripts/evaluate_samples.py \\
        --experiment-dir experiments/failures_analysis/retroprime_uspto190_*

If --experiment-dir contains a glob, the script picks the most recently
modified match. Pass an explicit dir to be unambiguous.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def clear_atom_map(smi: str) -> str:
    """Canonicalize SMILES with atom map numbers stripped.

    Mirrors multiguide.dataset.helpers.clear_atom_map.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def compare_reactant_smiles(s1: Optional[str], s2: Optional[str]) -> bool:
    """True if the two `.`-joined SMILES strings represent the same set of
    canonicalized reactants. Mirrors multiguide.dataset.helpers.compare_reactant_smiles.
    """
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return False
    try:
        set1 = {clear_atom_map(s) for s in s1.split(".")}
        set2 = {clear_atom_map(s) for s in s2.split(".")}
    except Exception:
        return False
    return set1 == set2


def resolve_experiment_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    matches = sorted(glob.glob(arg), key=lambda m: Path(m).stat().st_mtime, reverse=True)
    matches = [m for m in matches if Path(m).is_dir()]
    if not matches:
        raise FileNotFoundError(f"No directory matched {arg!r}")
    return Path(matches[0])


def load_sampled_csvs(exp_dir: Path) -> list[tuple[Path, pd.DataFrame]]:
    files = sorted(exp_dir.glob("sampled_start*_end*.csv"))
    if not files:
        raise FileNotFoundError(f"No sampled_*.csv files in {exp_dir}")
    return [(f, pd.read_csv(f)) for f in files]


def evaluate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean `topk` column. Each row is one (product, beam_rank) pair.
    `topk` is True if that single rank's prediction matches the ground truth.
    """
    if "reactant_predictions" not in df.columns or "true_reactants" not in df.columns:
        raise ValueError(
            "sampled CSV missing reactant_predictions / true_reactants columns"
        )
    df = df.copy()
    df["topk"] = df.apply(
        lambda row: compare_reactant_smiles(row["reactant_predictions"], row["true_reactants"]),
        axis=1,
    )
    return df


def per_product_min_rank(df: pd.DataFrame) -> pd.DataFrame:
    """For each (sample/test row) product, find the minimum rank where the
    prediction matched. Returns one row per product with `min_rank`
    (-1 if never matched) plus `ground_truth_class` and `target_class`.
    """
    rows = []
    # Group by absolute index in the test set if available, otherwise by product_smi.
    group_col = "batch_index" if "batch_index" in df.columns else "product_smi"
    for key, group in df.groupby(group_col):
        matches = group[group["topk"]]
        if len(matches) == 0:
            min_rank = -1
        else:
            min_rank = int(matches["rank"].min())
        rows.append(
            {
                group_col: key,
                "product_smi": group["product_smi"].iloc[0],
                "ground_truth_class": group["ground_truth_class"].iloc[0],
                "target_class": group["target_class"].iloc[0],
                "min_rank": min_rank,
            }
        )
    return pd.DataFrame(rows)


def topk_accuracy(min_ranks: pd.Series, k: int) -> float:
    """Fraction of products whose ground-truth was hit at rank < k."""
    n = len(min_ranks)
    if n == 0:
        return 0.0
    hits = ((min_ranks >= 0) & (min_ranks < k)).sum()
    return float(hits) / n


def report(
    per_product: pd.DataFrame,
    ks: list[int],
) -> dict:
    """Compute overall + per-class top-K accuracy."""
    out = {"n_products": int(len(per_product)), "overall": {}, "per_class": {}}
    for k in ks:
        out["overall"][f"top_{k}"] = topk_accuracy(per_product["min_rank"], k)
    out["overall"]["unsolved"] = int((per_product["min_rank"] == -1).sum())
    for cls, group in per_product.groupby("ground_truth_class"):
        out["per_class"][int(cls)] = {
            "n": int(len(group)),
            **{f"top_{k}": topk_accuracy(group["min_rank"], k) for k in ks},
            "unsolved": int((group["min_rank"] == -1).sum()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory holding sampled_start*_end*.csv files (glob allowed)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50, 100],
    )
    parser.add_argument(
        "--no-write-evaluated",
        action="store_true",
        help="Skip writing evaluated_*.csv files",
    )
    args = parser.parse_args()

    exp_dir = resolve_experiment_dir(args.experiment_dir)
    print(f"==> evaluating {exp_dir}")

    pieces = load_sampled_csvs(exp_dir)
    print(f"==> loaded {len(pieces)} sampled CSVs")

    all_dfs = []
    for path, df in pieces:
        ev = evaluate_df(df)
        all_dfs.append(ev)
        if not args.no_write_evaluated:
            out_name = path.name.replace("sampled_", "evaluated_", 1)
            out_path = path.with_name(out_name)
            ev.to_csv(out_path, index=False)
        n_total = len(ev)
        n_hit = int(ev["topk"].sum())
        print(f"  {path.name}: {n_total} rows, {n_hit} matches")

    full = pd.concat(all_dfs, ignore_index=True)
    per_product = per_product_min_rank(full)

    metrics = report(per_product, args.ks)

    print()
    print(f"==> {metrics['n_products']} products evaluated")
    print(f"==> overall:")
    for k in args.ks:
        print(f"     top-{k:<3d} = {metrics['overall'][f'top_{k}']*100:6.2f}%")
    print(f"     unsolved = {metrics['overall']['unsolved']}")

    if metrics["per_class"]:
        print()
        print("==> per ground_truth_class:")
        header = "  class    n   " + "   ".join(f"top-{k:<3d}" for k in args.ks) + "   unsolved"
        print(header)
        for cls in sorted(metrics["per_class"].keys()):
            stats = metrics["per_class"][cls]
            row = f"  {cls:<5d}  {stats['n']:<4d} " + "   ".join(
                f"{stats[f'top_{k}']*100:6.2f}" for k in args.ks
            ) + f"   {stats['unsolved']}"
            print(row)

    metrics_path = exp_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print()
    print(f"==> metrics written to {metrics_path}")
    if not args.no_write_evaluated:
        print(f"==> evaluated_*.csv written next to each sampled_*.csv")


if __name__ == "__main__":
    main()
