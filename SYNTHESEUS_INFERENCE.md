# RetroPrime self-contained inference

This directory layout adds a syntheseus-compatible inference pipeline on top
of the upstream RetroPrime repo, so it can be run end-to-end (and submitted
as a slurm array job) without depending on the `multiguide` package.

## Layout

```
RetroPrime/
├── (upstream model code, untouched)
├── syntheseus_inference/         # Python package added by this work
│   ├── __init__.py
│   ├── wrapper.py                # syntheseus BackwardReactionModel subclass
│   ├── dataset.py                # slim get_batch + parse_batch_to_reaction_data
│   └── output.py                 # get_rows_per_batch + save_df
├── scripts/
│   └── evaluate_in_batch.py      # Hydra entrypoint
├── configs/
│   ├── config.yaml
│   ├── single_step_model/retroprime.yaml
│   └── single_step_evaluation/uspto190.yaml
├── slurm/
│   ├── slurm_utils.py            # copied verbatim from multiguide
│   └── submit_inference.py       # array-job submitter
├── data/uspto190/processed/test.csv  # copied from retrosynthesis-dataset
├── checkpoints/                  # download from Drive (see below)
└── install_env.sh                # one-shot conda env build
```

## Setup

```
bash install_env.sh           # default: torch 1.5.0+cu101
# bash install_env.sh --cpu   # CPU-only (useful for dev machines)
conda activate retroprime
```

Then download the pretrained checkpoints from
<https://drive.google.com/file/d/1-715B8jU0rRC3YaY4p6URQcgjcRG2OlV/view>
and extract them under:

- `checkpoints/USPTO-50K_pos_pred/USPTO-50K_pos_pred_model_step_90000.pt`
- `checkpoints/USPTO-50K_S2R/USPTO-50K_S2R_model_step_100000.pt`

## Run inference locally

```
conda activate retroprime
python scripts/evaluate_in_batch.py \
    single_step_evaluation.start_idx=0 \
    single_step_evaluation.end_idx=2 \
    general.experiment_group=smoke \
    general.experiment_name=local_smoke
```

Output lands in `experiments/smoke/local_smoke/sampled_start0_end2.csv`.

## Submit a slurm array job (Puhti / Mahti)

Edit the parameters at the top of `slurm/submit_inference.py`
(`start_array_job`, `end_array_job`, `targets_per_job`), then:

```
python slurm/submit_inference.py --platform puhti
```

The submitter uses `slurm_utils.get_platform_info()` to auto-pick
account/partition/modules. One slurm task processes `targets_per_job` rows of
`data/uspto190/processed/test.csv`.

## Output schema

`sampled_start{X}_end{Y}.csv` mirrors the schema multiguide's existing
single-step models produce, so the failures-analysis tooling reads them
without changes. Key columns: `product_smi`, `true_reactants`,
`ground_truth_class`, `target_class`, `reactant_predictions` (one
dot-joined SMILES per row), `product_idx`, `sample_index`,
`sampling_time_s`, `model_type`.

## Architecture notes

- The wrapper (`syntheseus_inference/wrapper.py`) shells out to
  `run_example.sh` from inside the same conda env. Lifting OpenNMT-py 0.4.1's
  `Translator` into Python proved fragile (path-based API), so the file-based
  pipeline is used as-is and the wrapper parses
  `reactants_predicted_mix.txt`.
- `run_example.sh` hard-codes its checkpoint path under
  `retroprime/transformer_model/experiments/checkpoints/`. The wrapper
  symlinks the staged `model_dir` into that location on first init.
- Python is bumped from the upstream README's 3.6 to **3.8** (the syntheseus
  minimum). torch is installed via pip from the official PyTorch wheel CDN
  because torch 1.5.0 has no Py 3.8 conda-forge build.
