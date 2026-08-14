"""Multiple-comparison correction over the declared family of reported tests.

The λ sweep in `HANDOFF.md` §2.1 reports raw (nominal) p-values: each comparison
was judged against α=0.05 as though it were the only test in the study. Across a
family of 28 tests, roughly 1.4 false positives are expected by chance alone, so
the raw p-values cannot support a claim on their own.

This script declares the family explicitly, recomputes every test from the
per-seed results, and applies three corrections:

* **Bonferroni** -- multiply each p by the family size. Controls the family-wise
  error rate (FWER), assumes independence, and is the harshest option.
* **Holm-Bonferroni** -- same FWER guarantee, uniformly more powerful, no extra
  assumptions. There is no statistical reason to prefer plain Bonferroni over it;
  it is reported alongside only because the earlier write-up quoted Bonferroni.
* **Benjamini-Hochberg** -- controls the false discovery rate (the expected
  proportion of rejections that are false) rather than the chance of any false
  rejection. The more sensible target for an exploratory sweep, and far less
  conservative when the tests are correlated -- which these are, since the same
  six baseline runs are the comparison group for every condition, and the four
  metrics are functions of the same predictions.

Two-test-type note
------------------
Seeds are matched across conditions, so a paired t-test is valid and is the
appropriate default. It is not uniformly stronger than Welch, though: pairing
removes the shared seed effect but costs degrees of freedom (5 rather than ~10 at
n=6). It wins clearly for RRT MAE, whose across-seed variance is large and
shared; it loses for the consistency gap, whose pooled spread is already small.
Both are therefore reported, each corrected within its own family.

Usage
-----
    python multiple_comparisons.py [--log BPIC_19] [--out results.csv]
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

import aggregate_results as ar

# Metrics stored in seconds by the diagnostics; the reported metrics are minutes.
SECONDS_METRIC_STEMS = ["mean_abs_gap", "mean_signed_gap", "signed_bias_rrt",
                        "signed_bias_ttne_sum", "mae_ttne_sum", "mae_rrt"]

# --- the declared family -----------------------------------------------------
# Every condition in HANDOFF Table 2.1 that has n>=2 and a stated effect, crossed
# with the four metrics that table reports. 7 x 4 = 28, which is where the
# "~28 tests" figure in the original write-up came from.
#
# Deliberately EXCLUDED, and the exclusions must be stated wherever these numbers
# are quoted:
#   - both-free and detach_rrt configs (n=1-2, reported as directional only)
#   - the BPIC_17_DR results (n=2; reported as seed agreement, no p-values)
#   - the per-head bias / correlation diagnostics (read jointly per §6, not tested)
CONDITIONS = [
    ("safe λ=0.1", lambda x: np.isclose(x.lam, 0.1) & (x.det == "ttne") & ~x.bal),
    ("safe λ=0.3", lambda x: np.isclose(x.lam, 0.3) & (x.det == "ttne") & ~x.bal),
    ("safe λ=0.5", lambda x: np.isclose(x.lam, 0.5) & (x.det == "ttne") & ~x.bal),
    ("safe λ=1.0", lambda x: np.isclose(x.lam, 1.0) & (x.det == "ttne") & ~x.bal),
    ("safe λ=1.5", lambda x: np.isclose(x.lam, 1.5) & (x.det == "ttne") & ~x.bal),
    ("safe λ=2.0", lambda x: np.isclose(x.lam, 2.0) & (x.det == "ttne") & ~x.bal),
    ("ctrl RRT×2", lambda x: x.bal & (x.lam == 0.0) & (x.s_ttne == 1.0) & (x.s_rrt != 1.0)),
]

METRICS = [("MAE RRT minutes", "RRT MAE"),
           ("MAE TTNE minutes", "TTNE/step"),
           ("DL sim", "DL sim"),
           ("mean_abs_gap_CB", "gap")]


def parse_config_name(name: str) -> dict:
    """Mirrors the parser in `ltn_presentation_notebook.ipynb`."""
    scales = re.search(r"_balanced_ttne([0-9.]+)_rrt([0-9.]+)", name)
    lam = re.search(r"_ltn_([0-9.]+)", name)
    det = re.search(r"_detach_(ttne|rrt)", name)
    return {"lam": float(lam.group(1)) if lam else 0.0,
            "det": det.group(1) if det else "none",
            "bal": "_balanced" in name,
            "s_ttne": float(scales.group(1)) if scales else np.nan,
            "s_rrt": float(scales.group(2)) if scales else np.nan}


def holm(p):
    """Holm-Bonferroni adjusted p-values (step-down, monotone-enforced)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values (step-up, monotone-enforced)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)[::-1]
    adjusted = np.empty_like(p)
    running = 1.0
    for i, idx in enumerate(order):
        rank = n - i                     # 1-based ascending rank
        running = min(running, n / rank * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def run(log_name: str) -> pd.DataFrame:
    per_seed, _ = ar.aggregate(log_name, None)
    per_seed = per_seed.reset_index(drop=True)
    parsed = per_seed["config"].apply(parse_config_name).apply(pd.Series)
    d = pd.concat([per_seed, parsed], axis=1)
    for col in [c for c in d.columns
                if any(c == s or c.startswith(s + "_") for s in SECONDS_METRIC_STEMS)]:
        d[col] = d[col] / 60.0

    base = d[(d.lam == 0) & (~d.bal)].set_index("seed").sort_index()
    if base.empty:
        raise RuntimeError(f"No baseline runs found for {log_name}.")

    rows = []
    for cname, selector in CONDITIONS:
        cond = d[selector(d)].set_index("seed").sort_index()
        if cond.empty:
            print(f"  (skipping '{cname}': no runs found)")
            continue
        shared = sorted(set(base.index) & set(cond.index))
        for col, mname in METRICS:
            b, c = base[col].values, cond[col].values
            _, p_welch = st.ttest_ind(c, b, equal_var=False)
            _, p_paired = st.ttest_rel(cond.loc[shared, col].values,
                                       base.loc[shared, col].values)
            rows.append({"condition": cname, "metric": mname,
                         "n": len(cond), "n_paired": len(shared),
                         "pct_change": 100 * (c.mean() - b.mean()) / abs(b.mean()),
                         "p_welch": p_welch, "p_paired": p_paired})

    res = pd.DataFrame(rows)
    for tag in ("welch", "paired"):
        p = res[f"p_{tag}"].values
        res[f"bonferroni_{tag}"] = np.minimum(p * len(res), 1.0)
        res[f"holm_{tag}"] = holm(p)
        res[f"bh_{tag}"] = benjamini_hochberg(p)
    return res


def report(res: pd.DataFrame) -> None:
    n = len(res)
    print(f"\nDeclared family size: {n} tests "
          f"({len(CONDITIONS)} conditions x {len(METRICS)} metrics)")
    print(f"Bonferroni threshold: 0.05 / {n} = {0.05 / n:.6f}\n")
    for tag in ("paired", "welch"):
        print(f"===== {tag.upper()} t-test =====")
        cols = ["condition", "metric", "n", "pct_change", f"p_{tag}",
                f"bonferroni_{tag}", f"holm_{tag}", f"bh_{tag}"]
        show = res[cols].sort_values(f"p_{tag}").head(10)
        print(show.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print(f"  raw p<0.05:            {(res[f'p_{tag}'] < 0.05).sum():2d} / {n}")
        print(f"  survives Bonferroni:   {(res[f'bonferroni_{tag}'] < 0.05).sum():2d} / {n}")
        print(f"  survives Holm:         {(res[f'holm_{tag}'] < 0.05).sum():2d} / {n}")
        print(f"  survives BH (FDR 5%):  {(res[f'bh_{tag}'] < 0.05).sum():2d} / {n}\n")
    # Conditions with n<=3 give a paired test 1-2 degrees of freedom; the p-value
    # is technically valid but carries almost no information. Flag rather than hide.
    thin = res[res["n_paired"] <= 3]["condition"].unique()
    if len(thin):
        print(f"NOTE: paired tests for {list(thin)} have <=2 df -- treat as directional only.")


def main() -> None:
    # Condition labels contain 'λ', which the default Windows console codepage
    # (cp1252) cannot encode -- without this the script dies on printing, not on
    # computing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default="BPIC_19", help="event log name")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()

    res = run(args.log)
    report(res)
    out = args.out or f"multiple_comparisons_{args.log}.csv"
    res.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
