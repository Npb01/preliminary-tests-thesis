# HANDOFF — LTN Cross-Task Consistency, preliminary study

Supersedes the earlier handoff (which described the original 6-config design,
before any of it had been validated). Everything below reflects the state after
~55 training runs across two event logs.

**Read §1, §5 and §8 first.** §5 (pitfalls) is where the time gets lost.

**Formal spec of both axioms** — groundings, predicates, quantifiers, gradient
routing, and the rejected alternative formulation — lives in `axioms.md`. This
file holds results and process; that one holds the maths.

---

## 1. What this study is

Masters thesis groundwork: Multi-Task Learning with Neuro-Symbolic AI (Logic
Tensor Networks / LTNtorch) in Predictive Process Monitoring, on top of SuTraN+
(https://github.com/BrechtWts/SuTraN_Plus).

**The axiom.** SuTraN+ has a time-till-next-event (ttne) suffix head and a
remaining-runtime (rrt) head. They are trained independently, yet the quantities
are logically linked: the sum of the ttne suffix equals the remaining runtime.
This holds *exactly* in the ground truth — verified per-instance across all
250,545 BPIC_19 test instances, max discrepancy 1.0 second against a mean target
of ~4.3M seconds (float32 rounding).

**The question.** Does injecting that identity as a soft LTN constraint improve
accuracy, cross-task consistency, or both — beyond what loss reweighting achieves?

**The headline answer (BPIC_19).** Yes for consistency, conditionally for
accuracy. At λ=1.0 with `detach_mode="ttne"`: consistency gap **−18.6%
(p<0.001)**, RRT MAE **−2.18% (p=0.006)**, no detectable harm to the other tasks.
A matched control that simply doubles the RRT loss reproduces **neither**.

---

## 2. Current results

### 2.1 BPIC_19 (primary; `subset_fraction=0.15`, 40 epochs, patience 8)

Baseline n=6, seeds {3,5,17,23,31,47}. Seed noise: RRT ±1.25%, TTNE/step ±0.90%,
DL sim ±0.34%, gap ±2.08%.

| Config | n | RRT MAE | TTNE/step | DL sim | Gap |
|---|---|---|---|---|---|
| Baseline | 6 | 23,999 min | 12,548 min | 0.8541 | 12,416 min |
| Safe λ=0.1 | 3 | −0.32% (p=.79) | −0.10% | +0.19% | −3.3% (p=.18) |
| Safe λ=0.3 | 2 | −0.05% (p=.95) | +0.21% | +0.16% | −7.6% (p=.07) |
| Safe λ=0.5 | 6 | −1.45% (p=.03) | +0.35% | +0.09% | −10.2% (p<.001) |
| **Safe λ=1.0** | 6 | **−2.18% (p=.006)** | +0.39% | +0.17% | **−18.6% (p<.001)** |
| Safe λ=1.5 | 3 | −1.32% (p=.08) | +1.19% | +0.32% | −24.8% (p<.001) |
| Safe λ=2.0 | 3 | +0.14% (p=.84) | +0.60% | +0.26% | −25.4% (p=.004) |
| **Control: RRT loss ×2** | 3 | **−0.13% (p=.90)** | +0.80% | +0.19% | **−2.1% (p=.05)** |
| Both-free λ=0.01…0.5 | 1–2 | mostly WORSE (+1 to +2%) | — | — | −1% to −29% |
| detach_rrt λ=0.1 | 3 | −0.06% | +0.18% | +0.18% | −9.1% |

"Safe" = `detach_mode="ttne"`, i.e. only the RRT head receives axiom gradient.

**λ=1.0 is the operating point.** RRT benefit peaks there and is gone by λ=2.0
even as the gap keeps shrinking.

### 2.2 BPIC_17_DR (generalization; `subset_fraction=0.5`, n=2, seeds {3,17})

| Metric | Baseline | λ=0.5 | λ=1.0 |
|---|---|---|---|
| Gap (CB) | 2,701 min | 1,555 (**−42.4%**) | 1,195 (**−55.8%**) |
| RRT MAE | 6,944 ± 4 | 6,977 (**+0.48%**) | 7,053 (**+1.58%**) |
| TTNE/step MAE | 959 ± 6 | 954 (−0.54%) | 951 (−0.78%) |
| DL sim | 0.7742 | 0.7757 | 0.7742 |
| RRT bias | **−182 min** | −1,334 | −2,056 |
| Σttne bias | −2,692 min | −2,719 | −2,982 |
| Error corr. | 0.959 | 0.983 | 0.987 |

Gap reduction replicates and is *stronger*. **RRT accuracy reverses** — it gets
worse, cleanly (both λ=1.0 seeds sit above both baseline seeds).

Three things the full metric sweep added later (notebook §9a, every metric at
once rather than the curated subset above):

- **RRT bias moves −1,874 min while its MAE moves only +110 min.** The damage is
  overwhelmingly to calibration, not to accuracy.
- **Negative per-step ttne predictions go 0.254 → 0.408** (+59%, both seeds
  agree). Part of the consistency is being bought below zero, where the official
  metric path clamps — so the clamped per-step MAE hides it. Not yet checked on
  BPIC_19; it should be.
- **TTNE/step, DL-sim and frac-negative-RRT all fail the both-seeds-agree check**
  at λ=1.0. With n=2 those "changes" are noise, not small effects.

### 2.3 The unifying explanation — a scope condition, not a failure

The axiom does exactly one thing: **it pulls RRT toward Σttne.** That helps only
when Σttne is the better-calibrated estimator.

| | BPIC_19 | BPIC_17_DR |
|---|---|---|
| RRT bias / its MAE | +10,407 / 23,999 = **43%** | −182 / 6,944 = **2.6%** |
| Σttne bias / its MAE | −1,003 / 18,615 = 5.4% | −2,692 / 7,245 = **37%** |
| Σttne bias | −1,003 min | −2,692 min |
| Better calibrated | **Σttne** | **RRT** |
| Lower MAE | Σttne (18,615 vs 23,999) | **RRT** (6,944 vs 7,245) |
| Axiom effect on RRT MAE | improves (−2.18%) | degrades (+1.58%) |

So: measure which head is better calibrated *before* applying the constraint.
This is the most defensible framing of the whole study and the strongest thesis
angle — *predicting when cross-task constraints help*.

**Corollary — the wrong head was probably constrained on BPIC_17_DR.**
`detach_mode="ttne"` freezes Σttne and lets only RRT move. On this log RRT is the
better head on *both* criteria, so the constraint was applied to the side that was
already right. Falsifiable version: `detach_mode="rrt"` at comparable λ should
shrink the gap by moving Σttne (the miscalibrated side) and leave RRT MAE alone.
That condition has n=3 on BPIC_19 and has **never been run here** — see §8.2.

Figure `12_which_head_should_move.png` is this argument in one slide.

### 2.4 λ does not transfer between logs

Gap reduction per unit λ is **~3–4× stronger** on BPIC_17_DR (−42% at λ=0.5 vs
−10% on BPIC_19). λ=0.5 there ≈ λ≈1.75 on BPIC_19 — already past BPIC_19's
optimum. **The useful range on BPIC_17_DR (≈λ 0.2–0.3) was never tested.**
Concrete falsifiable prediction: λ≈0.25 should give a smaller gap reduction with
less RRT degradation. 2–3 runs.

λ does not transfer between *axioms* either. Axiom 2 needs its own sweep from
scratch; nothing from §2.1 carries over.

---

## 3. Code state

### 3.1 Modified
| File | Change |
|---|---|
| `SuTraN/inference_procedure.py` | Consistency diagnostics block (~line 646). Computes gap / per-head signed bias / MAE-of-sum / MAE-rrt / error correlation, **each in clamped and `_raw` variants**, plus `frac_neg_ttne_step_preds`, `frac_neg_rrt_preds`, `num_ttne_steps_evaluated`. Writes `consistency_diagnostics.pkl`. |
| `SuTraN/train_procedure.py` | **Bug fix** (~line 795): `train_model` now forwards `balance_losses`/`scale_ttne`/`scale_rrt` to `train_epoch` instead of passing literals. Plus a hard failure (~line 187) if the rescaling is ever a no-op, and a `[balance_losses] active:` log line. LTN hook at ~line 217. |
| `TRAIN_EVAL_FUNCTIONALITY/run_mto_experiment.py` | **Bug fix**: `base_kwargs` forwards `detach_mode`/`balance_losses`/scales instead of hardcoding. Added `val_subset_fraction`. **Only `equal_weighting` accepts these — see §5.8.** |
| `TRAIN_EVAL_FUNCTIONALITY/TRAIN_EVAL_EQUAL_WEIGHTING.py` | `val_subset_fraction` param + CLI (subsets per-epoch validation only; test set untouched). `model_string` now encodes balance scales as `_balanced_ttne{x}_rrt{y}`. |
| `aggregate_results.py` | `METRIC_KEYS` extended: `MAE TTNE minutes`, `mae_ttne_sum_*`, `mae_rrt_*`, the `_raw` variants, negative-prediction fractions. |

### 3.2 Created
| File | Purpose |
|---|---|
| `axioms.md` | **Formal spec of both axioms** — grounding / predicate / quantifier for each, gradient routing, why formulation B was rejected, and a caution list |
| `run_prelim_configs.py` | Wave-2 λ sweep + corrected reruns (12 runs) |
| `run_phase1_extension.py` | Wave-3 safe-condition dose sweep + detach replication |
| `run_phase2_seeds.py` | Wave-4 tiered seed expansion; tier 5 = BPIC_17_DR |
| `run_phase3.py` | Wave-5 λ=1.5 + RRT-only upweight control |
| `backfill_diagnostics.py` | Re-runs **inference only** from saved checkpoints to add new metrics to old runs. Resumable, `--dry_run`/`--filter`/`--force`. Verifies recomputed RRT MAE matches the stored value before trusting output. ~17 min/run. |
| `outcome_consistency_metrics.py` | Axiom-2 metrics as pure tensor functions (no model/loop state, so unit-testable without a GPU): implied outcome from a decoded suffix, head-vs-suffix accuracy / macro-F1 / disagreement, and name→id resolution. Called from `inference_procedure`'s diagnostics block |
| `test_outcome_consistency_metrics.py` | Synthetic cases (incl. accept-then-cancel, and a determining act after the END token) plus a ground-truth check that reproduces the exact 1.000000 of §8.7 |
| `run_configs/bpic17dr_axiom1_sweep.py` | Config list for the BPIC_17_DR λ × detach sweep; run with no args to list all 26 and check for name clashes, with an index to run one. `run_bpic17dr_axiom1_sweep.sh` is a thin Slurm-array wrapper holding only the `#SBATCH` settings |
| `multiple_comparisons.py` | Declares the family of reported tests explicitly, recomputes each from per-seed results, applies Bonferroni / Holm / Benjamini-Hochberg under both paired and Welch tests. Writes `multiple_comparisons_<log>.csv`. See §8.6 |
| `quarantine_inert_balanced.py` | Moves pre-fix `_balanced` folders (inert duplicates) to `BPIC_19/_quarantine_inert_balanced/` |
| `ltn_presentation_notebook.ipynb` | The analysis. Parses λ/detach/scales from folder names, so new configs need no edits. Writes `figures/*.png`. **§9 loads BPIC_17_DR into its own frame via `load_log()`** — `tidy_df` stays BPIC_19, so both logs are readable side by side without re-running anything. |
| `config_overview.md` | Chronological config history with rationale per wave |

### 3.3 Figures (`figures/`, regenerated by the notebook)
`00` headline summary · `01` dose-response · `02` bias trajectory · `03` detach
diagnostic · `04` safe-condition sweep · `05` head-to-head MAE · `06`
best-available · `07` do-no-harm audit · **`08` control comparison (key slide)** ·
**`09` operating-point frontier** · `10` head-to-head bias ·
`11` BPIC_17_DR every metric (2×4 small multiples) ·
**`12` which head should move — BPIC_19 vs BPIC_17_DR (key slide for §2.3)**

---

## 4. Evaluation conventions (decided; don't silently change)

- **Each head is judged on its own metric.** Remaining time = `MAE RRT minutes`
  (the RRT head's output), because that is what you call in deployment — one
  forward pass, not an autoregressive suffix decode. TTNE = per-step MAE.
  This follows the SuTraN+ paper's convention.
- **Σttne-as-total-time is a secondary observation**, not the evaluation
  criterion. Interesting finding in its own right (it beat RRT by 24% on
  BPIC_19, and lost on BPIC_17_DR), but it is not how the model is used.
- **CB (case-based) is primary**, IB secondary — matching the paper.
- **Clamped variants are primary.** The `_raw` variants exist only to quantify
  how much the clamping choice matters.
- **All reported numbers are test-set.** `aggregate_results.py` reads
  `averaged_results_CB.pkl` from each run's `TEST_SET_RESULTS/` folder, and
  `consistency_diagnostics.pkl` is written by the same test-set inference pass.
  Validation is used only for early stopping.
- **Normalised quantities are presentation devices, not metrics.** Figure `12`
  plots Σttne MAE as a % of RRT MAE, and |bias| as a % of that head's own MAE,
  purely so two logs on different time scales can be compared within-log. The
  raw-unit table sits in the cell directly above it; quote that one.
- **Statistical tests: paired two-sided t-test** across matched seeds, with
  Holm correction over the declared family of 28 (§8.6). Welch is reported
  alongside because the earlier waves used it.
  Paired is *not* uniformly stronger, despite what an earlier version of this
  file claimed — it removes the shared seed effect but costs df (5 vs ~10 at
  n=6), so it pays only where the across-seed variance is large and shared:
  | | Welch p | Paired p |
  |---|---|---|
  | λ=1.0 RRT MAE | 0.00593 | **0.00104** (pairing wins) |
  | λ=0.5 gap | **0.00027** | 0.00176 (Welch wins) |
  | λ=1.5 gap | **0.00034** | 0.00625 (Welch wins) |
  Rule of thumb: paired for the accuracy metrics, Welch for the gap. State which
  was used wherever a p-value is quoted.
- **Paired tests on the n=2 and n=3 conditions have 1–2 df.** Technically valid,
  almost no information. Treat those as directional only.
- **Wilcoxon signed-rank is unusable here**: at n=6 its smallest attainable
  p-value is 2/2⁶ = 0.031, so it can never clear a corrected threshold.
- With n=2 (BPIC_17_DR) report **whether both seeds agree on the sign**, not a
  p-value or a std. The notebook's §9a table has an `agree` column for this.

---

## 5. Pitfalls — read before running anything

1. **Dropped-parameter bugs happened TWICE**, in different files, both silent.
   A flag reached `train_eval` (so the result folder was *named* correctly) but
   never reached the loss. Symptom: byte-identical result rows across configs.
   **When adding a parameter, trace it to its use site and assert it took
   effect** — don't verify via the folder name.
2. **`val_subset_fraction` is NOT in the folder name.** Re-running an existing
   (config, seed) with a different validation setting **silently overwrites**
   the old result. Always use a fresh seed.
3. **Baseline seed 3 was trained with full per-epoch validation** (pre-dates
   `val_subset_fraction`); every other run uses 0.5. Affects checkpoint
   selection only, not test evaluation. Documented, not "fixed" — re-running it
   would destroy the original.
4. **Units.** Everything in `consistency_diagnostics.pkl` is in **seconds**;
   `MAE RRT/TTNE minutes` are in **minutes**. The notebook converts by stem
   match. Sanity anchor: |mean signed error| can never exceed MAE.
5. **Clamping.** The official metric path clamps negative times to 0
   (`inference_environment.convert_to_seconds`); the diagnostics originally did
   not. 17–25% of per-step predictions are negative on BPIC_19, 25–40% on
   BPIC_17_DR. Both variants are now recorded.
6. **pandas index alignment.** In the notebook, `per_seed_df.reset_index()`
   must happen *before* deriving the parsed-parameter frame, or
   `pd.concat(axis=1)` pairs configs with the wrong parameters — silently, no
   error. This bit once. `load_log()` replicates the guard.
7. **Error correlation is not a cheating detector on its own.** Baseline is
   already ~0.83 (BPIC_19) / ~0.96 (BPIC_17_DR) because both heads see the same
   cases. Only read it jointly with bias and MAE — see §6.
8. **`run_mto_experiment.py` currently only works for `equal_weighting`.** It
   puts `lambda_ltn`, `detach_mode`, `balance_losses`, `scale_ttne`, `scale_rrt`,
   `val_subset_fraction`, `num_epochs` and `patience` into `base_kwargs` for
   *every* technique, but only `TRAIN_EVAL_EQUAL_WEIGHTING.train_eval` accepts
   them — `gradnorm`, `uw`, `uw_plus` and `pcgrad` are each missing all eight and
   raise `TypeError` on entry. It fails loudly, so nothing is silently corrupted,
   but the dispatcher is single-technique today. Must be fixed before §8.9.
9. **Parallel jobs make pitfall 2 far more dangerous.** Two array tasks that
   resolve to the same `model_string` will write to the same output directory
   *simultaneously* and corrupt each other, with no error. Before the first array
   submission, make the run directory collision-proof (assert it does not already
   exist, or include the Slurm job id). Related: Slurm snapshots the batch script
   at `sbatch` time but reads **data files at launch**, so editing code or data
   while an array is still pending silently mixes versions across one batch —
   which is exactly what the provenance block (commit hash + dirty count) written
   by the job script exists to expose.
10. **Smoke-test and debug runs silently join a real config's seed group.**
    Another variant of pitfall 2, and the easiest one to walk into. A throwaway
    run gets the same `model_string` as a real config and differs only by seed —
    and `aggregate_results.py` groups by config, treating seed as a repeat. The
    junk run is therefore averaged into the real group with no error anywhere.

    This nearly happened: the two 2-epoch cluster smoke tests wrote to
    `SUTRAN_DA_results_subset_0.5_multiclass_outcome_seed_98` and `_seed_99` —
    the *exact* BPIC_17_DR baseline config string. Left in place they would have
    folded two barely-trained models into the baseline group, corrupting the
    baseline mean and every percentage change computed against it, and would also
    have added two spurious runs to `backfill_diagnostics.py`'s todo list.

    Two habits that prevent it:
    - Delete smoke-test output directories as soon as the timings are read. The
      numbers belong in §9.3; the 2-epoch checkpoints are worthless.
    - Better, give throwaway runs a `subset_fraction` that no real config uses
      (e.g. 0.49). The config string then differs and grouping is impossible.

    To detect it: run the sweep's config script with no arguments — it flags
    ` ALREADY EXISTS` — or check the per-config run counts in the notebook's
    coverage table (§6 there), where an unexpected seed is the giveaway.

---

## 6. How to read the diagnostics

`gap = err_ts − err_rrt` exactly, so
`Var(gap) = σ²_ts + σ²_rrt − 2ρσ_ts σ_rrt`. The gap can shrink three ways:
smaller spreads, means moving together, or **higher ρ** — the last leaves both
heads equally wrong. Hence the three-way rule:

| ρ | bias | MAE | Reading | Observed |
|---|---|---|---|---|
| ↑ | ↓ | ↓ | genuine correction | λ=0.5, **λ=1.0** (BPIC_19) |
| ↑ | ↓ | flat (σ↑) | over-applied: real correction, offsetting scatter | **λ=2.0** |
| ↑ | ↑ | ↑ | wrong head constrained | **BPIC_17_DR** |
| ↑ | flat | flat | true cheating | *never observed* |

Note `detach_mode` does **not** fully isolate the heads — they share the
encoder/decoder, so at λ=2.0 the untargeted Σttne head's bias degrades
(−1,003 → −2,320 min) despite its axiom gradient being frozen.

---

## 7. Findings that did NOT survive

Report these; they justify the seed budget.

- **The "cheating asymmetry."** At n=1 the detach diagnostics appeared to show
  Σttne chasing RRT. At n=3 the conditions overlapped each other *and* baseline,
  and per-seed biases didn't agree on sign. Withdrawn.
- **λ=1.0's accuracy cost.** Looked like +1.21% at n=2; **+0.02% at n=6**.
- **DL-sim gains.** Looked like 2.7× noise at n=2; baseline noise turned out to
  be 0.34% not 0.06%, so within noise at n=6.
- **All pre-fix `_balanced` runs.** Inert; quarantined.

---

## 8. Open work, roughly by value

1. **λ≈0.2–0.3 on BPIC_17_DR** (2–3 runs). Tests the §2.4 prediction directly.
   Highest value per GPU-hour.

2. **`detach_mode="rrt"` on BPIC_17_DR** (2–3 runs). Tests the §2.3 corollary:
   if the wrong head was constrained, moving Σttne instead should buy the gap
   reduction without the RRT degradation. Cheap, and it converts a post-hoc
   explanation into a prediction that can fail.

3. **More BPIC_17_DR seeds** — currently n=2, so nothing there is powered.

4. **Three metrics that need one shared backfill** (~8h, re-inference only):
   - per-**position** TTNE bias (bias at suffix step 1, 2, 3…) — the current
     summed bias hides position-dependent distortion, and per-step bias under
     the naive definition is just summed bias ÷ mean suffix length
   - **error SD** (`err_rt.std()`, `err_ts.std()`) — would replace the currently
     *inferred* bias/variance decomposition with a measured one
   - ~~**outcome-head metrics** in `METRIC_KEYS`~~ — **DONE, and the premise was
     wrong.** The outcome head was never unmeasured: `Multi-Class Accuracy`,
     `Macro-F1`, `Weighted-F1`, `Macro-Precision`, `Macro-Recall` and `CE` have
     been written into every outcome-log run's `averaged_results_*.pkl` all along.
     They were simply absent from `METRIC_KEYS`, and `load_metrics` uses
     `.get(k, None)`, so `aggregate_results.py` dropped them silently. Adding the
     keys cost no GPU time. Result: **axiom 1 does no harm to the outcome head** —
     accuracy 0.7862 / 0.7870 / 0.7867 at λ = 0 / 0.5 / 1.0, with the seeds not
     even agreeing on sign at λ=1.0. A reminder that "unmeasured" should be
     verified by opening the pickle, not inferred from the aggregation table.

5. **Switch the n=6 comparisons to paired t-tests** (seeds are matched; ~5×
   more power, same conclusions).

6. **Multiple comparisons — DONE, and the headline survives.** Run
   `python multiple_comparisons.py`; writes `multiple_comparisons_BPIC_19.csv`.

   "Uncorrected" in the old version of this note meant the p-values in §2.1 are
   *raw*: each was judged against 0.05 as though it were the only test in the
   study. Across 28 tests, ~1.4 false positives are expected by chance and there
   is a ~76% chance of at least one, so raw p-values cannot carry a claim.

   **The declared family is 7 conditions × 4 metrics = 28** — the λ sweep plus
   the RRT×2 control, crossed with RRT MAE / TTNE-per-step / DL-sim / gap.
   Excluded, and this must be restated wherever the numbers are quoted: the
   both-free and detach_rrt configs (n=1–2, directional only), all of BPIC_17_DR
   (n=2, reported as seed agreement), and the bias/correlation diagnostics (read
   jointly per §6, never tested).

   **Under paired tests + Holm** (the defensible default, §4):

   | Test | Raw p | Holm | BH | Verdict |
   |---|---|---|---|---|
   | λ=1.0 gap | 0.00006 | **0.0017** | 0.0017 | survives everything |
   | **λ=1.0 RRT MAE** | 0.00104 | **0.0282** | 0.0146 | **survives everything** |
   | λ=0.5 gap | 0.00176 | **0.0457** | 0.0164 | survives everything |
   | λ=0.5 RRT MAE | 0.00400 | 0.0999 | **0.0280** | FDR only |
   | λ=1.5 gap | 0.00625 | 0.1501 | **0.0350** | FDR only |
   | λ=2.0 gap | 0.00851 | 0.1957 | **0.0397** | FDR only |
   | **ctrl RRT×2, all 4** | ≥0.089 | 1.0 | ≥0.250 | **nothing survives** ✓ |

   6 of 28 raw < 0.05; 3 survive Holm; 6 survive BH at FDR 5%.

   **So both headline claims survive full correction** — the λ=1.0 gap *and* the
   λ=1.0 RRT MAE, the latter at Holm-adjusted p=0.028. That is stronger than the
   previous note claimed. Under Welch instead, the three gap results survive Holm
   but RRT MAE does not (adj. 0.142) — which is exactly why the test choice in §4
   has to be stated explicitly rather than left implicit.

   And the control passes its own test by failing: not one of its four metrics
   survives any correction.

   Prefer **Holm** over plain Bonferroni (same FWER guarantee, uniformly more
   powerful, no extra assumptions) and consider **BH** as the primary for an
   exploratory sweep — Bonferroni assumes independence, while these 28 tests share
   the same six baseline runs and four mutually-dependent metrics.

7. **Axiom 2 — outcome consistency.** No longer deferred: the label check is done
   and the formulation is decided. Maths in `axioms.md`.

   **Label check (done, no GPU needed).** On BPIC_17_DR the outcome is *defined*
   as the last Offer event in `{O_Accepted, O_Cancelled, O_Refused}`
   (`create_BPIC17_DR_data_multiclass.py:41`). On the non-leaky subset
   (`instance_mask_out == False`; 90.2% train / 88.9% test):

   | Outcome | Determining act | n (test) | Last-occurrence | Any-occurrence | ≥2 det. acts | Det. act step | Suffix len |
   |---|---|---|---|---|---|---|---|
   | Accepted | `O_Accepted` | 73,898 | **1.000000** | 0.701 | 2.8% | 11.0 | 14.0 |
   | Canceled | `O_Cancelled` | 74,615 | **1.000000** | 0.965 | **43.9%** | 11.1 | 12.6 |
   | Refused | `O_Refused` | 23,832 | **1.000000** | 1.000 | 2.9% | 11.0 | 13.0 |
   | All | — | 172,345 | **1.000000** | — | 20.6% | 11.0 | 13.3 |

   Train agrees throughout (n=346,415, all rates identical to 4 dp). Coverage is
   100% of the non-leaky subset and no suffix is truncated by the window, so the
   axiom's domain of validity and the outcome head's training domain coincide
   exactly.

   **The intuitive form of the axiom is false.** "`O_Accepted` in the suffix ⟹
   Accepted" holds only 70.1% of the time — an accepted offer can be cancelled
   later. That is almost entirely a Canceled phenomenon (43.9% of Canceled cases
   contain ≥2 determining activities, vs ~3% of the others). Only the **last**
   occurrence counts.

   **Decisions taken.** Formulation A (soft last-occurrence, bidirectional; see
   `axioms.md`). Axiom 1 **off** — standalone effect first. BPIC_17_DR only for
   now; BPIC_17 (OG) later, and note its tensors don't exist yet (`BPIC_17/` holds
   only mapping pkls, so `create_BPIC17_OG_data_multiclass.py` must run first).
   Start with **three arms** — both-free, detach-act, detach-outcome — rather than
   a single detach direction, because unlike axiom 1 we do not yet know which head
   is better calibrated. New `detach_mode` values must not reuse `"ttne"`/`"rrt"`,
   or folder names become ambiguous (§5.2).

   **Gating measurement — DONE (2026-08-15). The gate is passed.**
   Backfilled all six BPIC_17_DR runs (~1.6 h on one MIG slice) to compare the two
   available routes to the case outcome: the outcome head, and the outcome implied
   by the *decoded* activity suffix. Test set, CB, seeds {3, 17}, n=2 per row:

   | λ | Head acc | Suffix acc | Disagreement | Head macro-F1 | Suffix macro-F1 | No det. act |
   |---|---|---|---|---|---|---|
   | **0.0** | **0.7862** | **0.7700** | **0.0993** | 0.6553 | 0.6541 | 0.0012 |
   | 0.5 | 0.7870 | 0.7801 | 0.0575 | 0.6535 | 0.6557 | 0.0010 |
   | 1.0 | 0.7867 | 0.7729 | 0.0986 | 0.6543 | 0.6566 | 0.0008 |

   (The two macro-F1 columns are IB and comparable to each other, but **not** to
   the `Macro-F1` = 0.6906 stored in `averaged_results_CB.pkl`, which is CB.)

   **The two routes are comparably accurate but fail on different cases.** The
   suffix route is only 1.6 pp behind the head (0.7700 vs 0.7862) and level on
   macro-F1 — yet the two **disagree on 9.9%** of instances. If the errors were
   nested, disagreement could not exceed ~1.6 pp. Decomposing the baseline row,
   the suffix route is right where the head is wrong on **up to 4.2%** of cases,
   putting an oracle combination at **≤ 0.828**, roughly 4 pp above either alone.
   That is real, exploitable headroom, and it is the most favourable of the four
   outcomes that were written down before the measurement.

   **A prediction that failed, and why it matters.** It was predicted beforehand
   that the suffix route would be *substantially* worse, on the reasoning that the
   implied outcome is a step function of activity order — one misplaced
   `O_Cancelled` flips it, with none of the error cancellation that protects
   axiom 1's sum. That mechanism is real but empirically small:
   `frac_suffix_no_determining_act ≈ 0.001` shows the decoder nearly always emits
   a determining activity, and usually the right one, despite DL-sim of only
   0.774. **Do not assume a decoded-suffix quantity is unusable just because the
   suffix metric looks mediocre** — DL-sim penalises every position, whereas this
   axiom depends on one.

   **Consequences for the design.**
   - Proceed. Enough headroom, neither route degenerate, domain fully populated.
   - **The three-arm design is now positively justified**, not just cautious. For
     axiom 1 the §2.3 scope condition clearly favoured one direction because one
     head was better calibrated; here neither route dominates, so there is no
     a priori reason to prefer `detach="act"` over `detach="outcome"`, and
     both-free is genuinely motivated. A single-direction start would probably
     have picked wrong.
   - **Formulation A's residual-mass risk is largely retired.** With
     `frac_suffix_no_determining_act ≈ 0.001`, the soft `q(o)` carries almost no
     mass on "no determining event". Still worth logging per epoch.

   **Caveats.** n=2, so the λ column is not interpretable: the disagreement dip at
   λ=0.5 (0.0575) and return at λ=1.0 (0.0986) is non-monotonic, and there is no
   mechanism by which a *time*-consistency axiom would systematically change
   outcome/suffix agreement. Treat as noise unless both seeds agree on sign.
   Head accuracy is flat across λ (0.7862 → 0.7870 → 0.7867), which independently
   confirms axiom 1 does no harm to the outcome head.

   **Biggest known risk** (full list at the end of `axioms.md`): both axioms are
   enforced under teacher forcing but evaluated on greedy decoding, and axiom 2's
   implied outcome is a step function of activity *order* — one drift event flips
   it, with no error cancellation of the kind that protects axiom 1's sum. Log
   teacher-forced satisfaction and free-running consistency **separately**; if
   they diverge, that divergence is the result.

8. **Why Σttne beat RRT on BPIC_19 but not BPIC_17_DR.** Working hypothesis:
   window 17 vs 46, so far more per-step error accumulates in the longer sum.
   A SuTraN+ architecture finding, independent of LTN.

9. **Port the axiom to UW / UW+** (not started; do it *after* axiom 2 works under
   equal weighting, or a null result has two possible causes).
   `UW_train_procedure.train_epoch` is structurally identical to the
   equal-weighting one — the LTN block drops in unchanged between lines 269 and
   272. Roughly half a day, but §5.8 must be fixed first and §5.1 is exactly the
   failure mode to guard against.
   **Design decision: put the axiom outside the UW weighting**
   (`loss = uw_loss + λ·ltn_term`), so λ stays an independent variable you can
   sweep — noting λ will not be comparable to the equal-weighting values, since UW
   rescales the task losses. Making the axiom a 5th "task" with its own
   `log_sigma` is tempting but destroys the dose-response experiment (the model
   can shed the constraint by growing sigma) and, under UW+, renormalises every
   other task weight so the UW+ baseline stops being comparable.

---

## 9. Environment

### 9.1 Software (both machines)

`uv` for deps (`uv sync` — `pyproject.toml` pins `torch==2.5.1+cu118` to a
dedicated index, so do **not** install torch separately). `ltntorch==1.0.2`.
Entry point is `python -m TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment`;
`main.py` is an unused cookiecutter stub.

Local machine is Windows; the cluster is Linux, so use **Git Bash** locally for
anything involving `ssh`/`scp`/heredocs, and keep `git config core.autocrlf
input` set — a CRLF shell script fails on the cluster with `bad interpreter`.

### 9.2 TU/e Umbrella cluster (added 2026-08-15)

Slurm on Rocky Linux 8, access via `hpc.tue.nl` (VPN required off campus).
No per-user job limits worth planning around: `MaxJobs` unset,
`MaxSubmitJobs=10000`, **`MaxTime = 5 days` on every partition** — so no
checkpoint-resume machinery is needed. `DefaultTime` is only 2 h, so
**always pass `--time` explicitly** or the job is killed at two hours.

| Partition | Hardware | Notes |
|---|---|---|
| `tue.gpu1.q` | 8× A30 MIG `1g.6gb` | **Production target** — small slices, but the least contended |
| `tue.gpu2.q` | 4× L4 (24 GB) | 2.5× a MIG slice |
| `tue.gpu3.q` | 8× L4 (24 GB) | same |
| `tue.cpu1–3.q` | 64–96-core EPYC, 384–773 GB | preprocessing, analysis, BPIC_17 OG tensor generation |

Storage: `$HOME` 200 GB / 1M inodes; `/scratch-shared/$USER` 8 TB / 3M inodes.
Code and datasets live on scratch; **results are <1 MB per run and are versioned
in git**, which is the only off-site copy. Checkpoints (116 MB/run) stay on
scratch and are the one thing genuinely lost to a scratch purge — losing them
costs the `backfill_diagnostics.py` capability, not the results.

Billing weights are `CPU=1.0, Mem=0.25/GB, GPU=8.0`, so padding CPU/memory
requests burns fair-share disproportionately. Measured `MaxRSS` is ~4.0 GB, so
`--mem=8G` and `--cpus-per-task=4` are appropriate.

### 9.3 Measured run costs

`subset_fraction=0.5`, `val_subset_fraction=0.5`, BPIC_17_DR (window 46).

| | Full epoch | Training loop | Remainder (val + ckpt) |
|---|---|---|---|
| Local Quadro P1000 (4 GB) | 14.5 min | not measured | not measured |
| A30 MIG `1g.6gb` | **5.62 min** | 1.68 min (13.33 it/s) | 3.93 min (**70%**) |
| L4 | **2.27 min** | 0.64 min (35.09 it/s) | 1.63 min (**72%**) |

**The single most useful fact for planning: ~70% of an epoch is validation,
not training.** Both scale with the GPU at about the same rate.

Derived per-run and campaign costs (see §9.4 for how far to trust these):

| | per 40-epoch run | 27 runs ÷ 7 MIG slices |
|---|---|---|
| MIG, subset 0.5 | ~4.0 h | ~15 h |
| L4, subset 0.5 | ~1.7 h | — (contended) |
| MIG, full data, ~100 epochs | ~12.5 h | ~2 days |

**Scale by concurrency, not by hardware.** An L4 has roughly 20× a MIG slice's
nominal FP32 throughput and delivers 2.5×, so the model is not compute-bound —
a 240k-parameter model at batch 128 cannot saturate a modern GPU. Seven free MIG
slices (1.75 runs/h) beat the L4 partitions, which were 12/12 busy both times
they were checked; matching MIG throughput would need 3 concurrent L4s.
**Chasing department-specific nodes is therefore not worth the effort.**

Checkpoints are retained for every epoch — **2.88 MB each, 116 MB per 40-epoch
BPIC_17_DR run** — which is what makes `backfill_diagnostics.py` possible.

Keep every run in a comparison family on **one partition**: different GPU
architectures can produce different floating-point results, and the effects
under study are 1–2%.

### 9.4 Limitations of the measurements in §9.3

Stated plainly so nobody over-trusts them later:

- **The P1000 figure is not a controlled measurement.** 14.5 min/epoch was
  derived from checkpoint file mtimes across one local run — it therefore
  includes any pauses, thermal throttling, and whatever else the laptop was
  doing. Its train/validation split was never measured, so the P1000 column
  above is deliberately blank rather than estimated. **Any laptop-vs-cluster
  speedup ratio is approximate.** An earlier draft of this file quoted an "8.6×
  training-loop speedup", which was circular — it assumed the whole laptop epoch
  was training. Withdrawn.
- **Cluster epoch times are n=1**, from the mtime gap between `model_epoch_0.pt`
  and `model_epoch_1.pt` of a single 2-epoch job per GPU type. No repeats, no
  variance estimate. Co-tenants on the same node share memory bandwidth and I/O,
  so run-to-run variation is expected and unquantified.
- **The train/validation split is a subtraction**, not a direct timing. "Remainder"
  = full epoch − training loop, and lumps validation, checkpoint writing and
  per-epoch overhead together.
- **Test-inference cost is also a subtraction** (total − 2 epochs), and includes
  data loading.
- **40-epoch and full-data projections are linear extrapolations from 2 epochs.**
  The full-data figure additionally assumes `subset_fraction` scales training
  only, with validation fixed by `val_subset_fraction` — plausible from the
  parameter names but **not verified**. Confirm on the first full-scale run.
- **GPU spec ratios (~20×) come from published datasheets, not from this
  hardware.** The empirical 2.5× MIG→L4 ratio is measured; the "20× nominal" it
  is compared against is not.
- **`mean SM util: 63.0%` (L4) does not mean 63% of the GPU's capacity.**
  `nvidia-smi dmon`'s `sm` column is the fraction of *time* at least one kernel
  was resident, not the fraction of SMs used. A model launching many small
  kernels can show high values while using a few percent of the device. Treat it
  as "the GPU was idle 37% of the time", nothing more. (Note the first attempt
  reported 0.0% because `dmon -o T` prepends a timestamp column and the parsing
  read the GPU-index column instead — use `$3`, not `$2`.)
- **The contention snapshot (7/8 MIG free, 0/12 L4 free) was taken twice, both
  around 01:00 local time.** That is not a representative sample of cluster load;
  daytime availability may differ substantially. Re-check with
  `sinfo -O partition,gres,gresused` before sizing a campaign.
- **Queue wait is unmeasured.** The L4 job pended on `(Resources)` for an
  unrecorded period. Campaign wall-clock estimates above count compute only.
