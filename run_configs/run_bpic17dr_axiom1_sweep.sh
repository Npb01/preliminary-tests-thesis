#!/bin/bash
#SBATCH --job-name=ax1-17dr
#SBATCH --partition=tue.gpu1.q
#SBATCH --gres=gpu:1g.6gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --array=0-38%6
#SBATCH --output=/scratch-shared/%u/thesis/logs/%x-%A_%a.out
#
# Server settings only. The experiment configs live in
# run_configs/bpic17dr_axiom1_sweep.py -- run it with no arguments to list them.
# The --array range must match that file's config count (it prints the right
# directive; a mismatch makes the extra tasks exit with a clear message).
#
# --time is PER ARRAY TASK (one training run), not for the whole sweep. The
# estimate is ~4 h/run, from a single measured epoch (5.62 min) on an OTHERWISE
# IDLE MIG node. With %6 concurrency, six tasks share one physical A30's host
# memory bandwidth, PCIe and 32 CPU threads, so the real figure will be higher by
# an unmeasured amount. 16 h is deliberate over-provisioning: a job killed at the
# walltime loses all its compute (train_eval has no resume path), whereas an
# over-long request costs only marginally worse backfill placement -- negligible
# on a partition with 7 free slices. Fair-share bills consumed time, not
# requested time. Tighten this once `sacct` shows the real Elapsed.
#
#   mkdir -p /scratch-shared/$USER/thesis/logs   # Slurm opens --output before
#                                                # this script runs, and logs/
#                                                # is gitignored
#   sbatch run_configs/run_bpic17dr_axiom1_sweep.sh

set -euo pipefail
cd "/scratch-shared/${USER}/thesis"
source .venv/bin/activate

# Provenance: which code and which GPU produced this run (HANDOFF §5.9).
echo "commit:    $(git rev-parse HEAD)"
echo "uncommitted files: $(git status --porcelain | wc -l)"
echo "gpu:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "slurm_job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "started:   $(date -Is)"

python run_configs/bpic17dr_axiom1_sweep.py "${SLURM_ARRAY_TASK_ID}"
