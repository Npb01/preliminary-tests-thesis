#!/bin/bash
#SBATCH --job-name=sutran-smoke-mig
#SBATCH --partition=tue.gpu1.q
#SBATCH --gres=gpu:1g.6gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/%u/thesis/logs/%x-%j.out

set -euo pipefail
cd /scratch-shared/$USER/thesis
source .venv/bin/activate

# --- provenance: which code, which hardware, produced this run ---
{
  echo "commit:    $(git rev-parse HEAD)"
  echo "uncommitted files: $(git status --porcelain | wc -l)"
  echo "host:      $(hostname)"
  echo "gpu:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "slurm_job: ${SLURM_JOB_ID}  partition: ${SLURM_JOB_PARTITION}"
  echo "started:   $(date -Is)"
} | tee "logs/provenance-${SLURM_JOB_ID}.txt"

python -c "import torch;print('cuda available:',torch.cuda.is_available())"

# No GPU-utilisation sampling here: nvidia-smi cannot report per-instance
# utilisation for MIG slices. That measurement comes from the L4 run.

time python -m TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment \
    --log_name BPIC_17_DR --MTO_technique equal_weighting --seed 98 \
    --subset_fraction 0.5 --val_subset_fraction 0.5 \
    --num_epochs 2 --patience 8
