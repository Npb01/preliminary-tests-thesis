#!/bin/bash
#SBATCH --job-name=ax2-17dr
#SBATCH --partition=tue.gpu1.q
#SBATCH --gres=gpu:1g.6gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/scratch-shared/%u/thesis/logs/%x-%A_%a.out
#
# Server settings only. Configs live in run_configs/bpic17dr_axiom2_sweep.py --
# run it with no arguments to list them and see the tier boundaries.
#
# The default --array=0-11%6 is tiers 1-2 (lambda 1.0 and 0.5, all three detach
# modes, both seeds): ~12 runs, ~8-10 h on six slices. Override on the command
# line to run a different slice of the grid:
#
#     sbatch --array=0-5%6   run_configs/run_bpic17dr_axiom2_sweep.sh   # tier 1 only, ~4 h
#     sbatch --array=12-23%6 run_configs/run_bpic17dr_axiom2_sweep.sh   # tiers 3-4 later
#
# --time is PER TASK (one training run), 4x the ~4 h estimate. A job killed at
# the walltime loses all its compute -- train_eval has no resume path -- while an
# over-long request costs only marginally worse backfill placement, and
# fair-share bills consumed time rather than requested time.
#
# logs/ must already exist: Slurm opens --output before this script runs, and
# logs/ is gitignored so a fresh clone will not have it.
#
#     mkdir -p /scratch-shared/$USER/thesis/logs

set -euo pipefail
cd "/scratch-shared/${USER}/thesis"
source .venv/bin/activate

# Provenance: which code and which GPU produced this run (HANDOFF §5.9).
# --untracked-files=no so accumulating result directories are not miscounted as
# uncommitted code changes.
echo "commit:    $(git rev-parse HEAD)"
echo "uncommitted files: $(git status --porcelain --untracked-files=no | wc -l)"
echo "gpu:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "slurm_job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "started:   $(date -Is)"

python run_configs/bpic17dr_axiom2_sweep.py "${SLURM_ARRAY_TASK_ID}"
