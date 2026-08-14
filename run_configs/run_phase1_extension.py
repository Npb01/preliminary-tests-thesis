"""
Phase 1 extension: strengthens and probes the existing LTN consistency-loss
experiment with the ~32h compute budget between the initial 12-run sweep
(run_prelim_configs.py, already completed) and the analysis/design day.

Does NOT touch any already-completed config (baseline, lambda sweep,
detach reruns, balanced controls) -- those results are reused as-is.

Runs are grouped in priority tiers, front-loaded so the most
decision-relevant results land first if time runs out mid-sweep:

  Tier 1 (2 runs) -- confirm the detach-mode asymmetry with a 2nd seed.
    The single most important gap in the current results: the "Sigma-ttne
    chases RRT" finding (the core cheating-risk diagnostic) currently rests
    on one seed each for detach_ttne / detach_rrt.

  Tier 2 (4 runs) -- push the "safe" condition (detach_mode="ttne", i.e.
    only RRT responds to the axiom -- the one config that showed NO
    correlation rise over baseline) across higher lambda values. At
    lambda=0.1 it barely moved anything; this checks whether a higher dose
    of the safe mechanism can close more of the consistency gap without
    triggering the bias-overshoot / correlation-rise cheating signature.
    This is the run group most likely to produce your headline result
    either way (genuine safe improvement, or "even at high dose it doesn't
    help").

  Tier 3 (2 runs, OPTIONAL -- only run if Tier 1+2 finish with time to
    spare) -- fills in lambda=0.02/0.03 on the both-heads-free sweep, to
    pin down more precisely where Sigma-ttne's bias crosses zero (between
    the existing 0.05 and 0.1 points).

To trim Tier 3, just delete or comment out its two entries in CONFIGS below
-- each run is independent and safe to stop after.
"""

import subprocess
import sys
from pathlib import Path

LOG_NAME = "BPIC_19"
SEED_A = 3   # matches the seed used throughout the completed sweep
SEED_B = 17  # second seed, used in the completed sweep's headline configs
SEED_C = 5 # Third seed used for running this file a second time

NUM_EPOCHS = 40
PATIENCE = 8
VAL_SUBSET_FRACTION = 0.5

LOG_DIR = Path("run_logs")
LOG_DIR.mkdir(exist_ok=True)

# (name, seed, extra CLI args)
CONFIGS = [
    # --- Tier 1: confirm detach-mode asymmetry (2nd seed) ---
    ("detach_ttne_seed17", SEED_C, ["--lambda_ltn", "0.1", "--detach_mode", "ttne"]), #changed from seedb to seedc
    ("detach_rrt_seed17",  SEED_C, ["--lambda_ltn", "0.1", "--detach_mode", "rrt"]), #changed from seedb to seedc

    # --- Tier 2: push the safe (only-RRT-moves) condition harder ---
    ("safe_detach_ttne_lambda0.3", SEED_B, ["--lambda_ltn", "0.3", "--detach_mode", "ttne"]), #changed from seeda to seedb
    ("safe_detach_ttne_lambda0.5", SEED_B, ["--lambda_ltn", "0.5", "--detach_mode", "ttne"]),
    ("safe_detach_ttne_lambda1.0", SEED_B, ["--lambda_ltn", "1.0", "--detach_mode", "ttne"]),
    ("safe_detach_ttne_lambda2.0", SEED_B, ["--lambda_ltn", "2.0", "--detach_mode", "ttne"]),

    # --- Tier 3 (optional): fill in the overshoot-crossing region ---
    ("ltn_lambda0.02_seed3", SEED_B, ["--lambda_ltn", "0.02"]), #from seeda to seedb
    ("ltn_lambda0.03_seed3", SEED_A, ["--lambda_ltn", "0.03"]),
]

for name, seed, extra_args in CONFIGS:
    cmd = [
        sys.executable, "-u", "-m", "TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment",
        "--log_name", LOG_NAME,
        "--MTO_technique", "equal_weighting",
        "--seed", str(seed),
        "--subset_fraction", "0.15",
        "--val_subset_fraction", str(VAL_SUBSET_FRACTION),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
    ] + extra_args
    log_path = LOG_DIR / f"{name}_seed{seed}.log"
    print(f"\n=== Running {name} (seed={seed}) ===")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"!!! {name} FAILED (exit code {result.returncode}). "
              f"Check {log_path} for details. Stopping here.")
        sys.exit(result.returncode)
    else:
        print(f"{name} completed successfully.")

print(f"\nAll {len(CONFIGS)} Phase 1 extension configs completed. "
      f"Run aggregate_results.py / the notebook to view results.")
