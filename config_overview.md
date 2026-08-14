# Config overview

LTN cross-task consistency — preliminary study on BPIC_19.

Fixed across all runs: CaLenDiR training, `subset_fraction=0.15`, 40 epochs,
patience 8, `MTO_technique=equal_weighting`.

Δ% vs baseline (n=6). RRT MAE / TTNE-step / Gap: **negative = better**.
DL sim: **positive = better**. *n then* = seeds available when the next decision
was made; *n now* = current total. Chronology anchored to `run_logs/` timestamps
and result-folder mtimes.

| Wave | Config | λ | detach / scales | n then | n now | RRT MAE | TTNE/step | DL sim | Gap |
|---|---|---|---|---|---|---|---|---|---|
| **0** 07-22 | Baseline | 0 | — | 1 | **6** | 23,999 min | 12,548 min | 0.8541 | 12,416 min |
| **1** 07-23 | Both heads free | 0.1 | none | 1 | 2 | +1.25% | +0.35% | +0.11% | −6.8% |
| **1** | Only RRT moves | 0.1 | ttne | 1 | 3 | −0.32% (p=.79) | −0.10% (p=.88) | +0.19% | −3.3% (p=.18) |
| **1** | Only Σttne moves | 0.1 | rrt | 1 | 3 | −0.06% | +0.18% | +0.18% | −9.1% |
| **1** | Loss-balanced | 0 | 0.4014/0.3516 | 1 | 3 | *inert — quarantined* | | | |
| **1** | Loss-balanced + LTN | 0.1 | 0.4014/0.3516 | 1 | 3 | *inert — quarantined* | | | |
| **2** 07-24 | Both free | 0.01 | none | 2 | 2 | +1.44% | +0.40% | +0.12% | −1.4% |
| **2** | Both free | 0.05 | none | 2 | 2 | +1.01% | +0.23% | +0.14% | −3.2% |
| **2** | Both free | 0.5 | none | 2 | 2 | −0.41% | +0.79% | +0.27% | **−29.3%** |
| **3** 07-25 | Both free (fill-in) | 0.02 | none | 2 | 2 | +2.07% | +0.69% | +0.07% | +0.9% |
| **3** | Both free (fill-in) | 0.03 | none | 1 | 1 | +1.86% | +0.31% | +0.12% | −3.5% |
| **3** | Safe | 0.3 | ttne | 2 | 2 | −0.05% (p=.95) | +0.21% | +0.16% | −7.6% (p=.07) |
| **3** | Safe | 0.5 | ttne | 2 | **6** | **−1.45% (p=.03)** | +0.35% (p=.54) | +0.09% | **−10.2% (p<.001)** |
| **3** | **Safe** ★ | **1.0** | ttne | 2 | **6** | **−2.18% (p=.006)** | +0.39% (p=.54) | +0.17% | **−18.6% (p<.001)** |
| **3** | Safe | 2.0 | ttne | 2 | 3 | +0.14% (p=.84) | +0.60% (p=.26) | +0.26% | **−25.4% (p=.004)** |
| **4** 07-27 | *(no new configs — seeds to n=6)* | | | | | | | | |
| **5** 07-29 | Safe | 1.5 | ttne | 3 | 3 | −1.32% (p=.08) | +1.19% (p=.09) | +0.32% | **−24.8% (p<.001)** |
| **5** | **Control: RRT loss ×2** ★ | 0 | ttne 1.0 / rrt 0.5 | 3 | 3 | **−0.13% (p=.90)** | +0.80% (p=.14) | +0.19% | **−2.1% (p=.05)** |
| **6** 07-30 | BPIC_17_DR | 0 / 0.5 / 1.0 | ttne | — | 2 each | *running* | | | |

★ = the two rows the headline claim rests on.

---

**Wave 0 — Baseline.** Run twice at seed 3 (once with a validation-subsetting bug,
once fixed); `summary_csv` confirms it was the only config with results on 07-23.
It kept gaining seeds through Wave 4 because it is the denominator of every
comparison.

**Wave 1 — Original 6-config design.** The symmetric reading of `rt = Σ Δt` (both
heads respond) plus detach diagnostics to isolate which head moves and balanced
controls to rule out loss rebalancing. The gap shrank, but RRT accuracy got
*worse*, which later motivated constraining the axiom to one head. Four of the six
were silently invalid — `run_mto_experiment.py` hardcoded `detach_mode` and
`balance_losses`, so they ran as duplicates of the both-free config.

**Wave 2 — λ sweep.** One λ can't separate "the mechanism works" from "one lucky
setting," so the sweep tested dose-dependence, alongside corrected detach/balanced
reruns. The gap falls monotonically with λ — the mechanism is real. All n=2, so
direction only.

**Wave 3 — Safe condition + replication.** `detach_ttne` was the only condition
showing no error-correlation rise, so it was dosed harder to see whether one-head
constraint avoids the accuracy penalty seen in Wave 2 — it does, with RRT improving
through λ=1.0. Third seeds on both detach modes **killed the cheating-asymmetry
finding**: at n=3 the conditions overlapped baseline and per-seed biases disagreed
on sign.

**Wave 4 — Statistical power.** No new configs; baseline, λ=0.5 and λ=1.0 taken to
n=6 with matched seeds so claims could be tested rather than eyeballed. Baseline
itself moved (RRT MAE 24,347 → 23,999) and λ=1.0's apparent accuracy cost vanished.
The balanced controls returned byte-identical to their twins, exposing a second
hardcoding bug in `train_model` → `train_epoch`.

**Wave 5 — Elbow + real control.** With accuracy flat through λ=1.0 and breaking
only at λ=2.0, the elbow sat *above* λ=1.0, so the planned λ=0.3/0.7 points were
redirected to λ=1.5. The control upweights the RRT loss alone — the matched rival,
since the safe condition also feeds gradient to RRT alone — and reproduces almost
none of the effect (−2.1% gap vs −18.6%).

**Wave 6 — Generalization.** BPIC_17_DR is the only fully preprocessed second log,
so it needs no reconfiguration. Two λ values because λ=1.0 was tuned on BPIC_19 and
testing one risks a false negative if this log's optimum differs.

---

## Notes

- Remaining time is measured by the **RRT head's own output** (`MAE RRT minutes`),
  not by summing the ttne suffix, because that is the head called in deployment.
  Time-till-next-event is measured **per step**. This follows the SuTraN+ paper's
  metric convention.
- All gap/bias figures are case-based (CB) and converted from seconds to minutes.
- p-values are Welch t-tests vs the n=6 baseline. Configs at n=1–2 are
  underpowered and are reported for direction only.
- The two `_balanced` configs from Wave 1 are in
  `BPIC_19/_quarantine_inert_balanced/` — the loss rescaling never took effect, so
  they are duplicate baselines rather than controls.
