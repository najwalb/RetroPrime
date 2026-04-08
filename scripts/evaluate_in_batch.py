"""Slim Hydra entrypoint for RetroPrime self-contained inference.

Mirrors multiguide/scripts/evaluate_single_step_model_in_batch.py but strips
out classifier_guidance / search.steered / property_predictor /
round_trip_model / forward_model branches. Only plain backward inference is
performed; the resulting CSV matches the schema multiguide's failures-analysis
tooling expects.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from syntheseus.interface.molecule import Molecule  # noqa: E402

from syntheseus_inference.dataset import get_batch, parse_batch_to_reaction_data  # noqa: E402
from syntheseus_inference.output import get_rows_per_batch, save_df  # noqa: E402
from syntheseus_inference.wrapper import RetroPrimeBackwardModel  # noqa: E402


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def sample_in_batch(config):
    _set_seed(int(config.general.seed))
    print(f"======== start_idx: {config.single_step_evaluation.start_idx}")
    print(f"======== end_idx:   {config.single_step_evaluation.end_idx}")

    batch_df = get_batch(config)
    reactions = parse_batch_to_reaction_data(
        batch_df, int(config.single_step_evaluation.start_idx)
    )
    print(f"======== loaded {len(reactions)} reactions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RetroPrimeBackwardModel(
        model_dir=config.single_step_model.model_dir,
        device=device,
        default_num_results=int(config.single_step_model.default_num_results),
        use_cache=bool(config.single_step_model.use_cache),
        beam_size=int(config.single_step_model.beam_size),
        repo_root=REPO_ROOT,
        cores=1,
        gpu=0,
    )

    # Batched inference: RetroPrime's wrapper subprocesses run_example.sh,
    # which loads two large OpenNMT models from scratch every call (~60s of
    # torch startup tax per call). Calling the wrapper once with all
    # products amortizes that cost across the whole batch — for a 50-product
    # slurm task this is the difference between ~50 minutes and ~5 minutes.
    mols = [Molecule(r.product, canonicalize=False) for r in reactions]
    print(f"======== running batched inference on {len(mols)} products")
    t0 = time.time()
    with torch.no_grad():
        results = model(
            mols,
            num_results=int(config.single_step_model.default_num_results),
        )
    total_time = time.time() - t0
    print(f"======== batched inference done in {total_time:.1f}s "
          f"({total_time / max(len(mols), 1):.2f}s/product)")

    rows = get_rows_per_batch(reactions, results)
    # The wrapper batches into a single subprocess call, so per-product time
    # is not directly measured. Record the amortized average so downstream
    # analysis sees a sensible value.
    per_row_time = total_time / max(len(mols), 1)
    for row in rows:
        row["sampling_time_s"] = per_row_time
        row["model_type"] = config.single_step_model.model_type
    all_rows = rows
    del results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    out_name = (
        f"sampled_start{config.single_step_evaluation.start_idx}"
        f"_end{config.single_step_evaluation.end_idx}.csv"
    )
    out_path = save_df(df, config, out_name)
    print(f"======== Sampled df saved to {out_path}")


if __name__ == "__main__":
    sample_in_batch()
