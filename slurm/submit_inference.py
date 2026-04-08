"""Slurm array-job submitter for RetroPrime inference on uspto190/test.csv.

Slim copy of multiguide/slurm/slurm_evaluate_single_step_model_in_batch.py.
Strips every Hydra override that referenced classifier_guidance / search /
forward_model — RetroPrime only does plain backward inference.

Usage from inside RetroPrime/:

    python slurm/submit_inference.py --platform puhti

Adjust `start_array_job` / `end_array_job` / `targets_per_job` for the slice
of test.csv you want to run; one slurm task processes
`targets_per_job` rows.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Use the in-repo slurm_utils.py copy.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from slurm_utils import create_and_submit_batch_job, get_platform_info  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent  # = RetroPrime/

# ----------------------- inference parameters -----------------------
data_dir = "data/uspto190/processed"
subset = "test.csv"
default_num_results = 10
beam_size = 10
targets_per_job = 50
start_array_job = 0
end_array_job = 0  # bump to ceil(len(test.csv)/targets_per_job)-1 for full run
offset = 0
use_cache = "true"
seed = 42
experiment_group = "failures_analysis"
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"retroprime_uspto190_{time_stamp}"

# Stage your downloaded checkpoints under this dir before submitting:
#   {model_dir}/USPTO-50K_pos_pred/USPTO-50K_pos_pred_model_step_90000.pt
#   {model_dir}/USPTO-50K_S2R/USPTO-50K_S2R_model_step_100000.pt
model_dir = str(PROJECT_ROOT / "checkpoints")

# ----------------------- slurm config -----------------------
slurm_args = get_platform_info(use_gpu=True)
slurm_args.update(
    {
        "use_srun": True,
        "job_dir": str(PROJECT_ROOT / "slurm" / "jobs"),
        "job_ids_file": "job_ids.txt",
        "output_dir": str(PROJECT_ROOT / "slurm" / "output"),
        "time": "03:00:00",
        "nodes": 1,
        "ntasks-per-node": 1,
        "cpus-per-task": 1,
        "gpus-per-node": 1,
        "mem": "100G",
        "start_array_job": start_array_job,
        "end_array_job": end_array_job,
        # ----- puhti-specific env overrides -----
        # These override what `get_platform_info()` set so the slurm job
        # uses our retroprime env (Python 3.9 + torch 1.13) rather than
        # multiguide's pytorch/2.4 venv.
        "project": "project_2015608",
        "puhti_module": "pytorch/1.13",
        "venv_path": "/projappl/project_2015608/retroprime",
    }
)

script_args = {
    "script_dir": "scripts",
    "use_torchrun": "false",
    "args": {
        "general.seed": seed,
        "general.experiment_group": experiment_group,
        "general.experiment_name": experiment_name,
        "single_step_evaluation.data_dir": data_dir,
        "single_step_evaluation.subset": subset,
        "single_step_evaluation.start_idx": (
            "$start_idx"
            if not slurm_args["interactive"]
            else offset + (start_array_job * targets_per_job)
        ),
        "single_step_evaluation.end_idx": (
            "$end_idx"
            if not slurm_args["interactive"]
            else offset + (start_array_job * targets_per_job) + targets_per_job
        ),
        "single_step_model.model_dir": model_dir,
        "single_step_model.default_num_results": default_num_results,
        "single_step_model.beam_size": beam_size,
        "single_step_model.use_cache": use_cache,
    },
    "variables": {
        "targets_per_job": targets_per_job,
        "offset": offset,
        "start_idx": "$((offset+(SLURM_ARRAY_TASK_ID * targets_per_job)))",
        "end_idx": "$((start_idx+targets_per_job))",
    },
}

script_args["script_name"] = "evaluate_in_batch.py"
slurm_args["job_name"] = experiment_name
slurm_args["output_dir"] = os.path.join(
    slurm_args["output_dir"], experiment_group, experiment_name
)
os.makedirs(slurm_args["job_dir"], exist_ok=True)
os.makedirs(slurm_args["output_dir"], exist_ok=True)

create_and_submit_batch_job(slurm_args, script_args, interactive=slurm_args["interactive"])
