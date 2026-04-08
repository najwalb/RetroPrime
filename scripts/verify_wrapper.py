"""Standalone smoke test for the RetroPrime syntheseus wrapper.

Run from the repo root inside the `retroprime` conda env:

    python scripts/verify_wrapper.py --model-dir checkpoints

This:
  1. Imports the wrapper.
  2. Boots the model with a dummy product (no inference).
  3. If --run is passed, runs one full inference call on aspirin and prints
     the top-3 reactant predictions.

The boot step requires the staged checkpoints to exist; --run additionally
requires a working RetroPrime install + a GPU (or `--cpu`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Staged checkpoint dir")
    parser.add_argument("--run", action="store_true", help="Also run a 1-mol inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is present")
    args = parser.parse_args()

    from syntheseus.interface.molecule import Molecule

    from syntheseus_inference.wrapper import RetroPrimeBackwardModel

    import torch

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify] device={device} model_dir={args.model_dir}")

    model = RetroPrimeBackwardModel(
        model_dir=args.model_dir,
        device=device,
        default_num_results=5,
        use_cache=False,
        beam_size=10,
        repo_root=REPO_ROOT,
    )
    print("[verify] wrapper instantiated successfully")

    if args.run:
        aspirin = Molecule("CC(=O)Oc1ccccc1C(=O)O")
        results = model([aspirin], num_results=3)
        print(f"[verify] inference returned {len(results)} batches, "
              f"first batch has {len(results[0])} reactions")
        for i, rxn in enumerate(results[0]):
            reactants = ".".join(m.smiles for m in rxn.reactants)
            print(f"  rank {i}: {reactants}")


if __name__ == "__main__":
    main()
