"""Axiom-1 lambda sweep on BPIC_17_DR.

Same idea as run_phase3.py, but the configs are run in PARALLEL as a Slurm job
array rather than serially in one process: 26 runs x ~4 h is 4.3 days serially,
or ~17 h across six MIG slices. `run_bpic17dr_axiom1_sweep.sh` calls this once
per array task with the task index.

    python run_configs/bpic17dr_axiom1_sweep.py       # list every config
    python run_configs/bpic17dr_axiom1_sweep.py 7     # run config 7

WHY THIS SWEEP
--------------
The six existing BPIC_17_DR runs only cover detach_mode="ttne" at lambda in
{0.5, 1.0}, chosen from BPIC_19's operating point -- an assumption HANDOFF §2.4
later showed to be wrong: lambda does not transfer between logs, the gap responds
3-4x more strongly here, and lambda=0.5 is already past the optimum. HANDOFF §8.1
and §8.2 both ask for this sweep, and §8.2's detach_mode="rrt" condition -- the
direct test of whether the wrong head was constrained -- has never been run here.

WHY FRESH SEEDS (101, 102 rather than 3, 17)
--------------------------------------------
1. Re-running seeds 3/17 would resolve to run directories that already exist and
   silently overwrite them (HANDOFF pitfall §5.2).
2. Those runs were trained on a Quadro P1000; these run on A30 MIG slices, and a
   comparison family must not mix GPU architectures when the effects are 1-2%
   (HANDOFF §9.3). Re-running the baseline on the cluster is the price of a
   clean, self-contained sweep.
"""

import subprocess
import sys
from pathlib import Path

LOG_NAME = "BPIC_17_DR"
NUM_EPOCHS = 40
PATIENCE = 8
SUBSET_FRACTION = 0.5
VAL_SUBSET_FRACTION = 0.5

# Grid shifted DOWN from BPIC_19's: §2.4 predicts the useful range here is
# ~0.2-0.3. lambda=1.0 is kept as a reproduction check of the P1000 result.
SEEDS = [101, 102]
LAMBDAS = [0.1, 0.25, 0.5, 1.0]
DETACH_MODES = ["none", "ttne", "rrt"]   # both free / only RRT moves / only Sttne moves

# (lambda, detach_mode, seed) -- one baseline per seed, then the full grid.
CONFIGS = []
for _seed in SEEDS:
    CONFIGS.append((0.0, "none", _seed))
    for _lam in LAMBDAS:
        for _det in DETACH_MODES:
            CONFIGS.append((_lam, _det, _seed))


def run_dir(lam, detach, seed):
    """The directory train_eval will create. Mirrors the model_string logic in
    TRAIN_EVAL_EQUAL_WEIGHTING.py, including float formatting (0.25 -> '0.25')."""
    name = "SUTRAN_DA_results_subset_{}".format(SUBSET_FRACTION)
    if lam > 0.0:
        name += "_ltn_{}".format(lam)
        if detach != "none":
            name += "_detach_{}".format(detach)
    return Path(LOG_NAME) / "{}_multiclass_outcome_seed_{}".format(name, seed)


assert len({run_dir(*c) for c in CONFIGS}) == len(CONFIGS), "duplicate run directories"


def run(lam, detach, seed):
    target = run_dir(lam, detach, seed)
    if target.exists():
        sys.exit("REFUSING: {} already exists -- would overwrite (pitfall §5.2).".format(target))
    cmd = [
        sys.executable, "-u", "-m", "TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment",
        "--log_name", LOG_NAME,
        "--MTO_technique", "equal_weighting",
        "--seed", str(seed),
        "--subset_fraction", str(SUBSET_FRACTION),
        "--val_subset_fraction", str(VAL_SUBSET_FRACTION),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
        "--lambda_ltn", str(lam),
        "--detach_mode", detach,
    ]
    print("run dir: {}\ncmd: {}\n".format(target, " ".join(cmd)), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for i, (lam, detach, seed) in enumerate(CONFIGS):
            exists = " ALREADY EXISTS" if run_dir(lam, detach, seed).exists() else ""
            print("{:>3}  lambda={:<5} detach={:<5} seed={}  {}{}".format(
                i, lam, detach, seed, run_dir(lam, detach, seed).name, exists))
        print("\n{} configs -> sbatch --array=0-{}%6   (~{:.0f} GPU-h, ~{:.0f} h on 6 slices)".format(
            len(CONFIGS), len(CONFIGS) - 1, len(CONFIGS) * 4.0, len(CONFIGS) * 4.0 / 6))
        sys.exit(0)

    idx = int(sys.argv[1])
    if not 0 <= idx < len(CONFIGS):
        sys.exit("index {} out of range 0..{} -- --array and CONFIGS have diverged.".format(
            idx, len(CONFIGS) - 1))
    sys.exit(run(*CONFIGS[idx]))
