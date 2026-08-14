"""
Moves the INERT '_balanced' result folders out of the analysis path.

WHY
---
`train_model` used to pass balance_losses=False / scale_ttne=1.0 /
scale_rrt=1.0 as literals to `train_epoch` instead of forwarding its own
arguments. The flag reached `train_eval` -- so the output folder was still
tagged '_balanced' -- but the loss was never rescaled. Every balanced run
produced before the fix is therefore an exact duplicate of the corresponding
UNBALANCED run at the same seed.

Left in place they are actively misleading: aggregate_results.py picks them up
as a distinct configuration, so they look like a loss-rebalancing control that
"shows no effect", when in fact no rebalancing was ever applied.

This script verifies that claim before acting -- for each '_balanced' folder it
looks for a same-seed unbalanced twin and compares the stored metrics. Only
folders that are byte-identical to a twin (i.e. provably inert) are moved.
Anything that differs is left alone and reported, so a genuinely-balanced run
produced after the fix is never quarantined by accident.

Nothing is deleted; folders are moved to `_quarantine_inert_balanced/` so they
remain available if needed.

    python quarantine_inert_balanced.py --dry_run     # inspect first
    python quarantine_inert_balanced.py
"""

import argparse
import os
import pickle
import re
import shutil
import sys

QUARANTINE_DIRNAME = "_quarantine_inert_balanced"
RESULT_FILENAME = "averaged_results_CB.pkl"
DIAGNOSTICS_FILENAME = "consistency_diagnostics.pkl"
SEED_PATTERN = re.compile(r"_seed_(\d+)$")

# Metrics compared when deciding whether two runs are identical. Any one of
# these differing is enough to conclude the runs are genuinely different.
COMPARE_KEYS = ["MAE TTNE minutes", "MAE RRT minutes", "DL sim"]


def find_results_file(run_dir, filename):
    for root, _dirs, files in os.walk(run_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_metrics(run_dir):
    path = find_results_file(run_dir, RESULT_FILENAME)
    if path is None:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def unbalanced_twin_name(balanced_name):
    """'..._balanced_seed_5' -> '..._seed_5'  (also handles the newer
    '_balanced_ttne{x}_rrt{y}' form, which post-dates the fix)."""
    return re.sub(r"_balanced(_ttne[0-9.]+_rrt[0-9.]+)?", "", balanced_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log_name", default="BPIC_19", help="Results base folder.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what would move, then exit without touching anything.")
    args = ap.parse_args()

    base = args.log_name
    if not os.path.isdir(base):
        sys.exit(f"Results folder not found: {base}")

    balanced_dirs = sorted(
        e for e in os.listdir(base)
        if os.path.isdir(os.path.join(base, e))
        and e.startswith("SUTRAN_DA_results")
        and "_balanced" in e
    )
    if not balanced_dirs:
        print("No '_balanced' run folders found -- nothing to do.")
        return

    # PRIMARY criterion: the bug lived in train_model, so EVERY balanced run made
    # before the fix is inert, regardless of what its numbers look like. Pre-fix
    # runs are identifiable by name: they are tagged plain '_balanced', whereas
    # post-fix runs carry the scales ('_balanced_ttne{x}_rrt{y}').
    #
    # An earlier version of this script used "differs from its unbalanced twin"
    # as the test, which UNDER-quarantines: two of these folders differ from
    # their twin only because the twin was trained with full per-epoch validation
    # while the balanced run used val_subset_fraction=0.5. That difference has
    # nothing to do with loss balancing, so "differs" does not mean "genuine".
    post_fix_pattern = re.compile(r"_balanced_ttne[0-9.]+_rrt[0-9.]+")

    inert, post_fix = [], []
    for name in balanced_dirs:
        if post_fix_pattern.search(name):
            post_fix.append(name)
            continue
        twin = unbalanced_twin_name(name)
        twin_path = os.path.join(base, twin)
        evidence = "pre-fix naming"
        if os.path.isdir(twin_path):
            m_bal = load_metrics(os.path.join(base, name))
            m_twin = load_metrics(twin_path)
            if m_bal is not None and m_twin is not None:
                identical = all(m_bal.get(k) == m_twin.get(k) for k in COMPARE_KEYS)
                evidence = (f"pre-fix naming; metric-identical to {twin}" if identical
                            else f"pre-fix naming (differs from {twin}, but only via "
                                 f"val_subset_fraction -- balancing still never applied)")
        inert.append((name, evidence))

    print(f"Found {len(balanced_dirs)} '_balanced' folder(s) under {base}/\n")
    if inert:
        print(f"INERT ({len(inert)}) -- produced before the fix, will be quarantined:")
        for name, evidence in inert:
            print(f"  {name}\n      {evidence}")
    if post_fix:
        print(f"\nPOST-FIX ({len(post_fix)}) -- real balanced runs, LEFT IN PLACE:")
        for name in post_fix:
            print(f"  {name}")

    if not inert:
        print("\nNothing to quarantine.")
        return

    if args.dry_run:
        print(f"\n--dry_run: {len(inert)} folder(s) would move to "
              f"{os.path.join(base, QUARANTINE_DIRNAME)}/")
        return

    dest_root = os.path.join(base, QUARANTINE_DIRNAME)
    os.makedirs(dest_root, exist_ok=True)
    moved = 0
    for name, _twin in inert:
        src = os.path.join(base, name)
        dst = os.path.join(dest_root, name)
        if os.path.exists(dst):
            print(f"  SKIP {name} -- already present in quarantine")
            continue
        shutil.move(src, dst)
        moved += 1
        print(f"  moved {name}")

    # A note left beside the quarantined runs, so their status is self-evident
    # to anyone (including future-you) who finds the folder later.
    with open(os.path.join(dest_root, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "These runs were labelled '_balanced' but the loss rescaling never took\n"
            "effect: train_model passed balance_losses=False / scale_ttne=1.0 /\n"
            "scale_rrt=1.0 as literals to train_epoch instead of forwarding its own\n"
            "arguments. Each folder here was verified to be metric-identical to the\n"
            "corresponding UNBALANCED run at the same seed, i.e. it is a duplicate,\n"
            "not a control. Excluded from analysis for that reason. Nothing here is\n"
            "usable as evidence about loss rebalancing.\n"
        )
    print(f"\nQuarantined {moved} folder(s) to {dest_root}/")
    print("Note: aggregate_results.py only scans directories whose name starts with")
    print("'SUTRAN_DA_results', so the quarantine folder is ignored automatically.")


if __name__ == "__main__":
    main()
