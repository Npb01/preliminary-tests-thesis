"""
Phase 3: locate the accuracy elbow, and run the control that actually rivals
the headline finding.

Replaces the original Tier 3 (whose balanced runs were inert -- see BUG below)
and redirects the Tier 4 budget, which was aimed at the wrong lambda region.

WHY THESE TWO GROUPS
--------------------
1) lambda = 1.5  (3 seeds)
   With n=6, best-available accuracy (Sigma-ttne MAE, CB) is FLAT out to
   lambda=1.0 and only breaks at lambda=2.0:
       lambda   0      0.5     1.0     2.0
       MAE  18,615  18,579  18,619  19,327   (min; baseline seed sd ~253)
       gap  12,416  11,149  10,111   9,260   (min)
   So the elbow sits between 1.0 and 2.0, not between 0.5 and 1.0 as the
   earlier 2-seed reading suggested. lambda=1.5 asks whether the extra gap
   reduction available above 1.0 can be banked before the cost switches on.
   This sets the recommended operating point.

2) RRT-only loss upweighting  (3 seeds)  -- the control
   The safe condition (detach_mode="ttne") feeds axiom gradient to the RRT
   head ONLY. So the direct rival explanation is not "you rebalanced the time
   losses against the activity loss" (what the old `balanced` config tested)
   but simply "you pushed the RRT head harder". This control isolates that:
       --balance_losses --scale_ttne 1.0 --scale_rrt 0.5   =>  RRT loss x2,
                                                               ttne untouched
   Upweighting RRT makes it fit its OWN labels better, shrinking its
   over-prediction bias -- and since the gap is driven by that bias, it may
   close part of the gap too. If it reproduces the lambda=1.0 effect, the
   axiom adds nothing beyond reweighting. If it does not, the logical
   constraint is doing real work. Either result is worth reporting.

BUG THIS SUPERSEDES
-------------------
`train_model` used to pass balance_losses=False / scale_ttne=1.0 /
scale_rrt=1.0 as literals to `train_epoch` instead of forwarding its own
arguments. The flag still reached `train_eval`, so result folders were tagged
'_balanced' -- but the loss was never rescaled. Every balanced run ever
produced is therefore an exact duplicate of its unbalanced twin at the same
seed, and none of them are usable as controls. Fixed in
SuTraN/train_procedure.py, together with a hard failure if the rescaling is
ever a no-op again.

Because of that bug, the pre-existing '_balanced' result folders should be
EXCLUDED from analysis rather than reinterpreted -- see
`quarantine_inert_balanced.py`.

    python run_phase3.py

Every run is independent and the script is safe to interrupt between runs.
"""

import subprocess
import sys
from pathlib import Path

LOG_NAME = "BPIC_19"

NUM_EPOCHS = 40
PATIENCE = 8
SUBSET_FRACTION = 0.15
VAL_SUBSET_FRACTION = 0.5

# Seeds reused from the n=6 configs so comparisons stay paired.
SEEDS = [3, 17, 5]

# RRT-only upweight factor for the control. 0.5 => RRT loss doubled (the loss
# is divided by the scale). ttne left at 1.0 so ONLY the rrt head is pushed.
CTRL_SCALE_TTNE = 1.0
CTRL_SCALE_RRT = 0.5

LOG_DIR = Path("run_logs")
LOG_DIR.mkdir(exist_ok=True)

# Run the elbow group first: it decides the recommended operating point, so it
# is the more valuable half if the budget runs out partway.
TIERS_TO_RUN = [1, 2]

SAFE = ["--detach_mode", "ttne"]
RRT_ONLY_UPWEIGHT = [
    "--balance_losses",
    "--scale_ttne", str(CTRL_SCALE_TTNE),
    "--scale_rrt", str(CTRL_SCALE_RRT),
]

# (tier, name, seed, extra CLI args)
CONFIGS = []
for seed in SEEDS:
    CONFIGS.append((1, "safe_lambda1.5", seed, ["--lambda_ltn", "1.5"] + SAFE))
for seed in SEEDS:
    CONFIGS.append((2, "ctrl_rrt_upweight", seed, ["--lambda_ltn", "0.0"] + RRT_ONLY_UPWEIGHT))

selected = [c for c in CONFIGS if c[0] in TIERS_TO_RUN]
if not selected:
    sys.exit(f"No configs selected -- TIERS_TO_RUN={TIERS_TO_RUN} matched nothing.")

print(f"Tiers {sorted(TIERS_TO_RUN)}: {len(selected)} run(s) queued "
      f"(~{len(selected) * 2.75:.0f}h at ~2.75h/run).")
print("  Tier 1 = lambda=1.5 elbow;  Tier 2 = RRT-only upweight control")
print("NOTE: the control runs should print a '[balance_losses] active: ...' line "
      "on their first batch. If they do not, the rescaling is not taking effect "
      "-- stop and investigate rather than trusting the run.\n")

for i, (tier, name, seed, extra_args) in enumerate(selected, start=1):
    cmd = [
        sys.executable, "-u", "-m", "TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment",
        "--log_name", LOG_NAME,
        "--MTO_technique", "equal_weighting",
        "--seed", str(seed),
        "--subset_fraction", str(SUBSET_FRACTION),
        "--val_subset_fraction", str(VAL_SUBSET_FRACTION),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
    ] + extra_args
    log_path = LOG_DIR / f"phase3_t{tier}_{name}_seed{seed}.log"
    print(f"[{i}/{len(selected)}] Tier {tier} :: {name} (seed={seed})")
    print("  Command:", " ".join(cmd))
    print(f"  Logging to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  !!! FAILED (exit {result.returncode}). See {log_path}. Stopping.")
        sys.exit(result.returncode)
    print("  done.\n")

print(f"All {len(selected)} Phase 3 runs completed. Re-run aggregate_results.py / "
      f"ltn_presentation_notebook.ipynb to refresh the analysis.")
