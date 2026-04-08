"""Syntheseus wrapper for RetroPrime.

RetroPrime ships its inference pipeline as a 5-step shell script
(`run_example.sh`) — tokenize → P2S → synthon-rerank → S2R → mix-and-rerank.
Lifting the OpenNMT-py 0.4.1 fork's `Translator` into Python is brittle (its
public API is path-based, not in-memory), so this wrapper invokes
`run_example.sh` as a subprocess **inside the same conda env** and parses the
final `reactants_predicted_mix.txt`.

Output schema of `reactants_predicted_mix.txt`: one SMILES per line, in input
order, with `beam_size` consecutive lines per input. See
`retroprime/transformer_model/script/mix_c2c_top3_after_rerank.py` for the
construction.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from rdkit import Chem
from syntheseus.interface.models import InputType, ReactionType
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    process_raw_smiles_outputs_backwards,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol, canonical=True)


class RetroPrimeBackwardModel(ExternalBackwardReactionModel):
    """In-env subprocess wrapper around RetroPrime/run_example.sh.

    Expected `model_dir` layout (mirrors what `run_example.sh` already
    expects):

        {model_dir}/USPTO-50K_pos_pred/USPTO-50K_pos_pred_model_step_90000.pt
        {model_dir}/USPTO-50K_S2R/USPTO-50K_S2R_model_step_100000.pt

    The model_dir is *staged* — `run_example.sh` looks under
    `retroprime/transformer_model/experiments/checkpoints/` by default. To
    avoid editing the upstream shell script, this wrapper symlinks the staged
    `model_dir` into that path on first use if the symlink doesn't already
    exist.
    """

    def __init__(
        self,
        *args,
        beam_size: int = 10,
        repo_root: Optional[Path] = None,
        run_script: str = "run_example.sh",
        cores: int = 1,
        gpu: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beam_size = int(beam_size)
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.run_script = run_script
        self.cores = int(cores)
        self.gpu = int(gpu)

        # Validate the staged model_dir.
        model_dir = Path(self.model_dir)
        p2s = model_dir / "USPTO-50K_pos_pred" / "USPTO-50K_pos_pred_model_step_90000.pt"
        s2r = model_dir / "USPTO-50K_S2R" / "USPTO-50K_S2R_model_step_100000.pt"
        for required in (p2s, s2r):
            if not required.exists():
                raise FileNotFoundError(
                    f"RetroPrime checkpoint missing: {required}. "
                    "Download the Drive archive linked in RetroPrime/README.md "
                    "and stage it under your --model-dir."
                )

        # run_example.sh hard-codes its checkpoint path under
        # retroprime/transformer_model/experiments/checkpoints. Symlink the
        # staged model_dir there if it isn't already in place.
        expected = (
            self.repo_root
            / "retroprime"
            / "transformer_model"
            / "experiments"
            / "checkpoints"
        )
        if expected.resolve() != model_dir.resolve():
            expected.parent.mkdir(parents=True, exist_ok=True)
            if expected.exists() or expected.is_symlink():
                if expected.is_symlink() and expected.resolve() == model_dir.resolve():
                    pass  # already correct
                else:
                    raise RuntimeError(
                        f"{expected} already exists and points elsewhere. "
                        "Remove it manually or pass model_dir matching that path."
                    )
            else:
                expected.symlink_to(model_dir.resolve(), target_is_directory=True)

    def get_parameters(self):  # not used; here for API parity with LocalRetro
        return []

    # ----- syntheseus dispatch / cache -----
    def __call__(
        self,
        inputs: list[InputType],
        num_results: Optional[int] = None,
        reaction_types=None,
    ) -> list[Sequence[ReactionType]]:
        num_results = num_results or self.default_num_results
        inputs_not_in_cache = list(
            {inp for inp in inputs if (inp, num_results) not in self._cache}
        )
        if len(inputs_not_in_cache) > 0:
            new_rxns = self._get_reactions(
                inputs=inputs_not_in_cache,
                num_results=num_results,
                reaction_types=reaction_types,
            )
            assert len(new_rxns) == len(inputs_not_in_cache)
            for inp, rxns in zip(inputs_not_in_cache, new_rxns):
                self._cache[(inp, num_results)] = self.filter_reactions(rxns)

        output = [self._cache[(inp, num_results)] for inp in inputs]
        if not self._use_cache:
            self._cache.clear()

        self._num_cache_misses += len(inputs_not_in_cache)
        self._num_cache_hits += len(inputs) - len(inputs_not_in_cache)

        return output

    # ----- core inference -----
    def _get_reactions(
        self,
        inputs: List[Molecule],
        num_results: int,
        reaction_types=None,
    ) -> List[Sequence[SingleProductReaction]]:
        beam_size = max(int(num_results), self.beam_size)
        n_inputs = len(inputs)
        smis = [_canonicalize(m.smiles) for m in inputs]

        with tempfile.TemporaryDirectory(prefix="retroprime_", dir=str(self.repo_root)) as tmp:
            tmp_path = Path(tmp)
            in_file = tmp_path / "input.txt"
            in_file.write_text("\n".join(smis) + "\n")

            run_script_path = self.repo_root / self.run_script
            if not run_script_path.exists():
                raise FileNotFoundError(
                    f"RetroPrime run script not found at {run_script_path}"
                )

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            # The shell script hard-codes core=8; we override via env so the
            # subprocess uses --core matching `self.cores` if the script is
            # ever extended. For the upstream script we just rely on it.
            cmd = ["bash", str(run_script_path), str(in_file), str(tmp_path), str(beam_size)]
            subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                env=env,
                check=True,
            )

            mix_file = tmp_path / "reactants_predicted_mix.txt"
            if not mix_file.exists():
                raise RuntimeError(
                    f"RetroPrime did not produce {mix_file}. "
                    "Check the subprocess stderr above."
                )
            lines = [ln.strip() for ln in mix_file.read_text().splitlines()]

        # mix_c2c_top3_after_rerank writes beam_size lines per input, in
        # input order. Group them.
        if len(lines) != n_inputs * beam_size:
            # Pad / truncate to be defensive against deduped lines.
            grouped = [
                lines[i * beam_size : (i + 1) * beam_size] for i in range(n_inputs)
            ]
        else:
            grouped = [
                lines[i * beam_size : (i + 1) * beam_size] for i in range(n_inputs)
            ]

        out: List[Sequence[SingleProductReaction]] = []
        for mol, group in zip(inputs, grouped):
            raw_outputs = []
            metadata_list = []
            for rank, smi in enumerate(group[:num_results]):
                if not smi:
                    continue
                raw_outputs.append(smi)
                metadata_list.append({"probability": None, "rank": rank})
            out.append(
                process_raw_smiles_outputs_backwards(
                    input=mol,
                    output_list=raw_outputs,
                    metadata_list=metadata_list,
                )
            )
        return out
