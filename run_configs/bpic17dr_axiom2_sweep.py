"""Axiom-2 (outcome consistency) sweep on BPIC_17_DR.

Same shape as bpic17dr_axiom1_sweep.py: a Slurm job array, one config per task.

    python run_configs/bpic17dr_axiom2_sweep.py       # list configs + tiers
    python run_configs/bpic17dr_axiom2_sweep.py 7     # run config 7

ORDERED BY VALUE, so a prefix of the array is a coherent experiment on its own:

    tier 1  idx  0-5    lambda=1.0   all three detach modes, both seeds
    tier 2  idx  6-11   lambda=0.5   "
    tier 3  idx 12-17   lambda=0.25  "
    tier 4  idx 18-23   lambda=2.0   "

At ~4 h/run on a MIG slice with six concurrent, each tier is ~4 h, so
`--array=0-11%6` is roughly a 10-hour overnight batch that answers both "does
anything happen" and "is there a dose response". Run `--array=12-23%6` later to
complete the grid; nothing needs re-running.

WHY THIS DESIGN
---------------
Three detach modes, not one. The axiom-1 experience (HANDOFF §2.3) was that the
constraint helps only when it pulls a head toward the BETTER-calibrated
estimator, so the direction matters more than the strength. There, one head was
clearly better and a single direction was defensible. Here the gating measurement
(§8.7) found the two routes to the outcome are comparably accurate -- 0.7862 head
vs 0.7700 suffix, level on macro-F1 -- while disagreeing on 9.9% of instances.
Neither dominates, so no direction can be justified in advance and both-free is
genuinely motivated.

    none      both heads move toward each other
    act       q detached -> only the OUTCOME head moves
    outcome   y detached -> only the ACTIVITY suffix head moves

Expect `none` and `act` to look similar: softmax saturation attenuates the
gradient reaching a confident 28-way activity head by orders of magnitude
(demonstrated in test_ltn_outcome_consistency.py). If they are indistinguishable,
that is the explanation, and it is a finding rather than a bug.

LAMBDA
------
Calibrated, not guessed. A 2-epoch run at lambda=1.0 put the axiom term at ~10%
of the total objective and drove satisfaction to 0.78, so 1.0 is a sensible
centre and the grid brackets it by 4x each way. Both axioms produce a term of
`1 - sat` under the same aggregator, so lambda is on a comparable scale to
axiom 1's -- though the optimum need not transfer (§2.4).

AXIOM 1 IS OFF throughout (lambda_ltn=0.0): the standalone effect of axiom 2 is
characterised first, per the decision recorded in §8.7. Combining them would
confound both.

BASELINES are NOT re-run. bpic17dr_axiom1_sweep.py already trains
`..._subset_0.5_multiclass_outcome_seed_{101,102}` with the identical protocol on
the same hardware, and those are the correct comparison for axiom 2 as well.
This script checks they exist and warns if they do not.
"""

import subprocess
import sys
from pathlib import Path

LOG_NAME = "BPIC_17_DR"
NUM_EPOCHS = 40
PATIENCE = 8
SUBSET_FRACTION = 0.5
VAL_SUBSET_FRACTION = 0.5

SEEDS = [101, 102]                       # matched to the axiom-1 sweep's baselines
DETACH_MODES = ["none", "act", "outcome"]
LAMBDA_TIERS = [1.0, 0.5, 0.25, 2.0]     # tier order = priority order

HOURS_PER_RUN = 4.0
CONCURRENT = 6

# (lambda, detach_mode, seed), ordered so each tier is a self-contained block and
# both seeds of a condition land adjacently -- a truncated run then leaves
# complete pairs rather than orphaned single seeds.
CONFIGS = []
for _lam in LAMBDA_TIERS:
    for _det in DETACH_MODES:
        for _seed in SEEDS:
            CONFIGS.append((_lam, _det, _seed))


def run_dir(lam, detach, seed):
    """Directory train_eval will create. Mirrors the model_string logic in
    TRAIN_EVAL_EQUAL_WEIGHTING.py. Axiom 2 uses its own `_ltnout_`/`_detachout_`
    tags so it can never be confused with an axiom-1 run (§5.2)."""
    name = "SUTRAN_DA_results_subset_{}".format(SUBSET_FRACTION)
    if lam > 0.0:
        name += "_ltnout_{}".format(lam)
        if detach != "none":
            name += "_detachout_{}".format(detach)
    return Path(LOG_NAME) / "{}_multiclass_outcome_seed_{}".format(name, seed)


def baseline_dir(seed):
    return Path(LOG_NAME) / "SUTRAN_DA_results_subset_{}_multiclass_outcome_seed_{}".format(
        SUBSET_FRACTION, seed)


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
        # Axiom 1 explicitly off rather than relying on the CLI default, so a
        # future change to that default cannot silently turn it on here.
        "--lambda_ltn", "0.0",
        "--detach_mode", "none",
        "--lambda_ltn_outcome", str(lam),
        "--detach_mode_outcome", detach,
    ]
    print("run dir: {}\ncmd: {}\n".format(target, " ".join(cmd)), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        tier_size = len(DETACH_MODES) * len(SEEDS)
        for i, (lam, detach, seed) in enumerate(CONFIGS):
            if i % tier_size == 0:
                hrs = (i + tier_size) * HOURS_PER_RUN / CONCURRENT
                print("\n  -- tier {}: lambda={}  (idx {}-{}, done by ~{:.0f} h) --".format(
                    i // tier_size + 1, lam, i, i + tier_size - 1, hrs))
            exists = " ALREADY EXISTS" if run_dir(lam, detach, seed).exists() else ""
            print("{:>3}  lambda={:<5} detach={:<8} seed={}  {}{}".format(
                i, lam, detach, seed, run_dir(lam, detach, seed).name, exists))

        print("\n{} configs; {:.0f} GPU-h total, ~{:.0f} h wall clock on {} slices".format(
            len(CONFIGS), len(CONFIGS) * HOURS_PER_RUN,
            len(CONFIGS) * HOURS_PER_RUN / CONCURRENT, CONCURRENT))
        print("  overnight (~10 h):  sbatch --array=0-11%{} run_configs/run_bpic17dr_axiom2_sweep.sh".format(CONCURRENT))
        print("  remainder later:    sbatch --array=12-23%{} run_configs/run_bpic17dr_axiom2_sweep.sh".format(CONCURRENT))

        print("\nBaselines (shared with the axiom-1 sweep, not re-run here):")
        for s in SEEDS:
            d = baseline_dir(s)
            done = (d / "CaLenDiR_training" / "Default_Equal_Weighting" /
                    "TEST_SET_RESULTS" / "averaged_results_CB.pkl").is_file()
            print("  {}  {}".format("present " if d.exists() else "MISSING ", d.name)
                  + ("" if done else "   (no TEST_SET_RESULTS yet -- still training?)"))
        sys.exit(0)

    idx = int(sys.argv[1])
    if not 0 <= idx < len(CONFIGS):
        sys.exit("index {} out of range 0..{} -- --array and CONFIGS have diverged.".format(
            idx, len(CONFIGS) - 1))
    sys.exit(run(*CONFIGS[idx]))
