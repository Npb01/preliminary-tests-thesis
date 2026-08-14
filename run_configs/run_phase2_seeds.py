"""
Phase 2: turn the BPIC_19 findings into a statistically defensible case,
using the remaining ~48-72h of compute.

PRIORITIES REVISED after the backfilled MAE metrics landed. What changed:

  * Sigma-ttne (summing the predicted suffix) is the MORE accurate route to
    total remaining time in 29/29 runs -- ~18,440 min vs the RRT head's
    ~24,347 min at baseline. The "best available" estimate is therefore always
    Sigma-ttne, and NO configuration improves it beyond seed noise (baseline
    seed sd is ~1%; the best config is 0.29% better, i.e. nothing). The
    accuracy story is a NULL RESULT and more seeds will not change that.

  * That makes lambda=0.5 (safe), not lambda=1.0, the strongest headline:
        lambda=0.5 safe : gap -14.3%, best-available -0.03%  <- free
        lambda=1.0 safe : gap -22.0%, best-available +1.21%  <- borderline cost
    lambda=0.5 buys real consistency at zero measured accuracy cost, which is a
    cleaner claim than "more consistency, possibly at 1.2% accuracy".

  * lambda=2.0 is already conclusive (+4.45% best-available, seed sd 25 min --
    the two seeds are 19,244 and 19,279). It does not need 4 seeds.

So the compute goes into: (a) baseline, since it anchors BOTH the gap and the
best-available comparison, and (b) lambda=0.5 and lambda=1.0, the two candidate
operating points. Seeds are MATCHED across configs ({3,5,17,23}, then {31,47})
so comparisons can use a paired test, which has more power at small n.

  Tier 1 (7 runs, ~19h) -- the three load-bearing configs to n=4.
    baseline, safe lambda=0.5, safe lambda=1.0 each 2 -> 4 seeds; safe
    lambda=2.0 gets ONE more seed (2 -> 3) purely to firm up the upper end of
    the cost curve, since its effect is already unambiguous.

  Tier 2 (6 runs, ~17h) -- the same three configs to n=6.
    At n=6 vs n=6 a paired test on "does lambda=0.5 cost any accuracy?" is
    genuinely powered. This is the question the headline now rests on.

  Tier 3 (4 runs, ~11h) -- the controls, currently n=1.
    balanced_only and balanced_ltn 1 -> 3 seeds. Answers "did you rule out that
    this is just loss rebalancing?" -- which n=1 cannot survive.

  Tier 4 (4 runs, OPTIONAL) -- curve refinement.
    lambda=0.3 to n=4, plus lambda=0.7 as a new point to locate the knee
    between "free" (0.5) and "costly" (1.0). Lower value than seeds on the
    existing points: a fresh n=2 point is weaker evidence than depth.

  Tier 5 (6 runs, OPTIONAL, >=20h) -- BPIC_17_DR generalization.
    Lowest priority, per the decision to make ONE dataset strong first.
    BPIC_17_DR is already fully preprocessed (unlike plain BPIC_17, which is
    missing its tensordatasets), so no reconfiguration is needed -- but see the
    warnings on it below.

Tiers 1-3 = 17 runs, roughly 47h. Edit TIERS_TO_RUN to change scope. Every run
is independent and the script is safe to interrupt between runs.

    python run_phase2_seeds.py

WARNING -- result folders do NOT encode val_subset_fraction, so re-running an
existing (config, seed) pair with a different validation setting OVERWRITES the
earlier result. Every entry below therefore uses a seed not yet used for that
config. For the same reason, baseline seed 3 (trained before val_subset_fraction
existed, i.e. on full validation) is deliberately left alone rather than
"harmonised" -- re-running it would destroy the original. Document that seed 3
used full per-epoch validation; it affects checkpoint selection only, not the
final test-set evaluation.
"""

import subprocess
import sys
from pathlib import Path

LOG_NAME = "BPIC_19"

NUM_EPOCHS = 40
PATIENCE = 8
SUBSET_FRACTION = 0.15
VAL_SUBSET_FRACTION = 0.5

# Loss-rescaling constants for the balanced configs -- unchanged from
# run_prelim_configs.py (best-epoch values from the baseline rerun).
SCALE_TTNE = 0.4014
SCALE_RRT = 0.3516

# --- BPIC_17_DR (Tier 5) --------------------------------------------------
# Different dataset => its own settings:
#   * window is 46 vs BPIC_19's 17, and inference decodes autoregressively over
#     the suffix, so expect notably SLOWER runs (measure the first one before
#     trusting the tier estimate).
#   * it has only 16,471 training cases vs BPIC_19's 108,856, so the 0.15
#     fraction used for BPIC_19 would leave ~2.5k cases and likely undertrain.
#     0.5 keeps ~8.2k cases, closer to BPIC_19's ~16.3k in absolute terms.
#   * it has a multiclass outcome head (4 tasks, not 3) -- the axiom only
#     touches ttne/rrt, but the training dynamics are not identical.
DR_LOG_NAME = "BPIC_17_DR"
DR_SUBSET_FRACTION = 0.5

LOG_DIR = Path("run_logs")
LOG_DIR.mkdir(exist_ok=True)

# Which tiers to execute, in order. Trim this list to fit the time you have.
TIERS_TO_RUN = [5]          # add 4 / 5 only if these finish with time left

SAFE = ["--detach_mode", "ttne"]  # "safe" condition: only RRT responds to the axiom
BALANCED = ["--balance_losses", "--scale_ttne", str(SCALE_TTNE), "--scale_rrt", str(SCALE_RRT)]

# (tier, name, log_name, subset_fraction, seed, extra CLI args)
CONFIGS = [
    # --- Tier 1: the three load-bearing configs to n=4 (+1 seed for lambda=2.0) ---
    (1, "baseline",        LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "0.0"]),
    (1, "baseline",        LOG_NAME, SUBSET_FRACTION, 23, ["--lambda_ltn", "0.0"]),
    (1, "safe_lambda0.5",  LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "0.5"] + SAFE),
    (1, "safe_lambda0.5",  LOG_NAME, SUBSET_FRACTION, 23, ["--lambda_ltn", "0.5"] + SAFE),
    (1, "safe_lambda1.0",  LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "1.0"] + SAFE),
    (1, "safe_lambda1.0",  LOG_NAME, SUBSET_FRACTION, 23, ["--lambda_ltn", "1.0"] + SAFE),
    # lambda=2.0's cost is already unambiguous; one extra seed, not three.
    (1, "safe_lambda2.0",  LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "2.0"] + SAFE),

    # --- Tier 2: the same three to n=6, so the "no accuracy cost" test is powered ---
    (2, "baseline",        LOG_NAME, SUBSET_FRACTION, 31, ["--lambda_ltn", "0.0"]),
    (2, "baseline",        LOG_NAME, SUBSET_FRACTION, 47, ["--lambda_ltn", "0.0"]),
    (2, "safe_lambda0.5",  LOG_NAME, SUBSET_FRACTION, 31, ["--lambda_ltn", "0.5"] + SAFE),
    (2, "safe_lambda0.5",  LOG_NAME, SUBSET_FRACTION, 47, ["--lambda_ltn", "0.5"] + SAFE),
    (2, "safe_lambda1.0",  LOG_NAME, SUBSET_FRACTION, 31, ["--lambda_ltn", "1.0"] + SAFE),
    (2, "safe_lambda1.0",  LOG_NAME, SUBSET_FRACTION, 47, ["--lambda_ltn", "1.0"] + SAFE),

    # --- Tier 3: controls out of n=1 territory ---
    (3, "balanced_only",   LOG_NAME, SUBSET_FRACTION, 17, ["--lambda_ltn", "0.0"] + BALANCED),
    (3, "balanced_only",   LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "0.0"] + BALANCED),
    (3, "balanced_ltn",    LOG_NAME, SUBSET_FRACTION, 17, ["--lambda_ltn", "0.1"] + BALANCED),
    (3, "balanced_ltn",    LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "0.1"] + BALANCED),

    # --- Tier 4 (OPTIONAL): where does the accuracy cost switch on? ---
    (4, "safe_lambda0.3",  LOG_NAME, SUBSET_FRACTION,  5, ["--lambda_ltn", "0.3"] + SAFE),
    (4, "safe_lambda0.3",  LOG_NAME, SUBSET_FRACTION, 23, ["--lambda_ltn", "0.3"] + SAFE),
    (4, "safe_lambda0.7",  LOG_NAME, SUBSET_FRACTION,  3, ["--lambda_ltn", "0.7"] + SAFE),
    (4, "safe_lambda0.7",  LOG_NAME, SUBSET_FRACTION, 17, ["--lambda_ltn", "0.7"] + SAFE),

    # --- Tier 5 (OPTIONAL): does the safe condition transfer to another log? ---
    # Two lambda values because the BPIC_19 optimum may not transfer -- testing
    # only one risks a false negative if this log's knee sits elsewhere.
    (5, "DR_baseline",     DR_LOG_NAME, DR_SUBSET_FRACTION,  3, ["--lambda_ltn", "0.0"]),
    (5, "DR_baseline",     DR_LOG_NAME, DR_SUBSET_FRACTION, 17, ["--lambda_ltn", "0.0"]),
    (5, "DR_safe_l0.5",    DR_LOG_NAME, DR_SUBSET_FRACTION,  3, ["--lambda_ltn", "0.5"] + SAFE),
    (5, "DR_safe_l0.5",    DR_LOG_NAME, DR_SUBSET_FRACTION, 17, ["--lambda_ltn", "0.5"] + SAFE),
    (5, "DR_safe_l1.0",    DR_LOG_NAME, DR_SUBSET_FRACTION,  3, ["--lambda_ltn", "1.0"] + SAFE),
    (5, "DR_safe_l1.0",    DR_LOG_NAME, DR_SUBSET_FRACTION, 17, ["--lambda_ltn", "1.0"] + SAFE),
]

selected = [c for c in CONFIGS if c[0] in TIERS_TO_RUN]
if not selected:
    sys.exit(f"No configs selected -- TIERS_TO_RUN={TIERS_TO_RUN} matched nothing.")

print(f"Tiers {sorted(TIERS_TO_RUN)}: {len(selected)} run(s) queued "
      f"(~{len(selected) * 2.75:.0f}h at ~2.75h/run; Tier 5 runs will be slower).")
for tier in sorted(TIERS_TO_RUN):
    n = sum(1 for c in selected if c[0] == tier)
    print(f"  Tier {tier}: {n} run(s)")
print()

for i, (tier, name, log_name, subset_fraction, seed, extra_args) in enumerate(selected, start=1):
    cmd = [
        sys.executable, "-u", "-m", "TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment",
        "--log_name", log_name,
        "--MTO_technique", "equal_weighting",
        "--seed", str(seed),
        "--subset_fraction", str(subset_fraction),
        "--val_subset_fraction", str(VAL_SUBSET_FRACTION),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
    ] + extra_args
    log_path = LOG_DIR / f"phase2_t{tier}_{name}_seed{seed}.log"
    print(f"[{i}/{len(selected)}] Tier {tier} :: {name} (seed={seed}, log={log_name})")
    print("  Command:", " ".join(cmd))
    print(f"  Logging to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  !!! FAILED (exit {result.returncode}). See {log_path}. Stopping.")
        sys.exit(result.returncode)
    print("  done.\n")

print(f"All {len(selected)} Phase 2 runs completed. Re-run aggregate_results.py / "
      f"ltn_presentation_notebook.ipynb to refresh the analysis.")
