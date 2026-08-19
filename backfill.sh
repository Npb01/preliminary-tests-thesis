#!/bin/bash
#SBATCH --job-name=backfill-17dr
#SBATCH --partition=tue.gpu1.q
#SBATCH --gres=gpu:1g.6gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/%u/thesis/logs/%x-%j.out

set -euo pipefail
cd /scratch-shared/$USER/thesis
source .venv/bin/activate

{
  echo "commit:    $(git rev-parse HEAD)"
  echo "uncommitted files: $(git status --porcelain | wc -l)"
  echo "gpu:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "slurm_job: ${SLURM_JOB_ID}  partition: ${SLURM_JOB_PARTITION}"
} | tee "logs/provenance-${SLURM_JOB_ID}.txt"

time python backfill_diagnostics.py --log_name BPIC_17_DR
