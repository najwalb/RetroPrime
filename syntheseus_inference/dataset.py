"""Slim port of multiguide.dataset.helpers used by the RetroPrime self-contained
inference script. Drops rxn_insight, rxnmapper, textdistance, scipy, sklearn,
matplotlib, and the multiguide.property/desp dependencies.

Reads uspto190/processed/test.csv (or compatible) and yields ReactionData rows
that mirror the schema multiguide's get_rows_per_batch + downstream
failures-analysis tooling expects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class ReactionData:
    """Container for one row of test.csv after slim parsing.

    Field names mirror multiguide.dataset.helpers.ReactionData so that the
    failures-analysis tooling reads the resulting CSVs without changes.
    """

    reactants: str
    product: str
    target_class: int
    ground_truth_class: int
    original_starting_material: Optional[str]
    most_similar_reactants: Optional[str]
    least_similar_reactants: Optional[str]
    most_similar_reactants_similarity: Optional[float]
    least_similar_reactants_similarity: Optional[float]
    similarity_to_target: Optional[float]
    conditional_starting_material: Optional[str]
    original_target: Optional[str]
    conditional_target: Optional[str]
    route_starting_material: Optional[str] = None
    immediate_starting_material: Optional[str] = None
    batch_index: int = 0


def get_batch(config) -> pd.DataFrame:
    """Load test.csv and slice [start_idx:end_idx]."""
    repo_root = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
    csv_path = repo_root / config.single_step_evaluation.data_dir / config.single_step_evaluation.subset
    df = pd.read_csv(csv_path)
    start = int(config.single_step_evaluation.start_idx)
    end = int(config.single_step_evaluation.end_idx)
    return df.iloc[start:end]


def parse_batch_to_reaction_data(batch_df: pd.DataFrame, start_idx: int) -> List[ReactionData]:
    """Convert a slice of test.csv into ReactionData objects.

    Only the columns that uspto190/processed/test.csv actually carries are
    consumed; everything else (route metrics, similarities, etc.) is preserved
    where present and set to None otherwise.
    """
    out: List[ReactionData] = []
    for offset, row in enumerate(batch_df.itertuples(index=False)):
        row_d = row._asdict()
        out.append(
            ReactionData(
                reactants=row_d.get("sorted_cano_reactants"),
                product=row_d.get("sorted_cano_products"),
                target_class=int(row_d.get("target_class", -1)),
                ground_truth_class=int(row_d.get("ground_truth_class", -1)),
                original_starting_material=row_d.get("most_sm"),
                most_similar_reactants=row_d.get("immediate_most_similar_reactants"),
                least_similar_reactants=row_d.get("immediate_least_similar_reactants"),
                most_similar_reactants_similarity=row_d.get("immediate_most_similar_reactants_similarity"),
                least_similar_reactants_similarity=row_d.get("immediate_least_similar_reactants_similarity"),
                similarity_to_target=-1.0,
                conditional_starting_material=None,
                original_target=row_d.get("main_target") or row_d.get("sorted_cano_products"),
                conditional_target=None,
                route_starting_material=row_d.get("most_sm"),
                immediate_starting_material=row_d.get("immediate_most_similar_reactants"),
                batch_index=start_idx + offset,
            )
        )
    return out
