"""Slim port of multiguide.evaluation.helpers.{get_rows_per_batch, save_df}.

The row schema mirrors what the multiguide failures-analysis pipeline ingests
today (see multiguide/multiguide/evaluation/helpers.py:126-173):

    product_smi, true_reactants, ground_truth_class, target_class,
    true_most_similar_reactants_similarity, true_least_similar_reactants_similarity,
    true_most_similar_reactants, true_least_similar_reactants,
    true_similarity_to_target, conditional_starting_material, conditional_target,
    original_target, original_starting_material, route_most_starting_material,
    immediate_most_starting_material, reactant_predictions, product_idx, sample_index

`reactant_predictions` is a single dot-joined SMILES string per row (one row
per (product, beam_rank) pair), matching multiguide's format.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import pandas as pd


def _mols_to_str_smiles(mols) -> str:
    return ".".join(m.smiles for m in mols)


def get_rows_per_batch(reaction_batch, results: Sequence[Sequence]) -> List[dict]:
    """Build one row per (product, beam_rank) pair.

    `results` is the raw `List[Sequence[SingleProductReaction]]` returned by a
    syntheseus BackwardReactionModel.
    """
    rows = []
    for product_idx, product_results in enumerate(results):
        reaction = reaction_batch[product_idx]
        for sample_idx, single_rxn in enumerate(product_results):
            reactant_str = _mols_to_str_smiles(single_rxn.reactants)
            rows.append(
                {
                    "product_smi": reaction.product,
                    "true_reactants": reaction.reactants,
                    "ground_truth_class": reaction.ground_truth_class,
                    "target_class": reaction.target_class,
                    "true_most_similar_reactants_similarity": reaction.most_similar_reactants_similarity,
                    "true_least_similar_reactants_similarity": reaction.least_similar_reactants_similarity,
                    "true_most_similar_reactants": reaction.most_similar_reactants,
                    "true_least_similar_reactants": reaction.least_similar_reactants,
                    "true_similarity_to_target": reaction.similarity_to_target,
                    "conditional_starting_material": reaction.conditional_starting_material,
                    "conditional_target": reaction.conditional_target,
                    "original_target": reaction.original_target,
                    "original_starting_material": reaction.original_starting_material,
                    "route_most_starting_material": reaction.route_starting_material,
                    "immediate_most_starting_material": reaction.immediate_starting_material,
                    "reactant_predictions": reactant_str,
                    "product_idx": product_idx,
                    "sample_index": sample_idx,
                    "batch_index": reaction.batch_index,
                    "score": (single_rxn.metadata or {}).get("probability"),
                    "rank": (single_rxn.metadata or {}).get("rank", sample_idx),
                }
            )
    return rows


def save_df(df: pd.DataFrame, config, out_file_name: str) -> str:
    repo_root = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
    out_dir = (
        repo_root
        / "experiments"
        / config.general.experiment_group
        / config.general.experiment_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_file_name
    df.to_csv(out_path, index=False)
    return str(out_path)
