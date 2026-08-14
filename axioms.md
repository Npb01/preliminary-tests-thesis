# Axioms — formal specification

The two cross-task logical constraints used in this study, as implemented (Axiom 1) and as
designed (Axiom 2, formulation A). Companion to `HANDOFF.md`, which holds results and process
notes; this file holds only the maths.

Both axioms follow the same three-part pattern, which is worth naming once:

1. a **grounding** — ordinary differentiable PyTorch that turns raw model outputs into the
   term the logic talks about (a sum, a soft "last occurrence");
2. a **predicate** — a smooth function into $[0,1]$ giving the truth degree of the statement;
3. a **quantifier** — an LTN aggregator turning per-instance truth degrees into one batch-level
   satisfaction, which becomes the loss.

Only steps 2 and 3 are LTN operations. Step 1 is just tensor code, and that is normal: in LTN
terms it is the grounding $\mathcal{G}$ of the terms.

---

## Shared notation

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $W$ | window size (max suffix length; 46 for BPIC_17_DR, 17 for BPIC_19) |
| $L_i$ | true suffix length of instance $i$ (position of the END token) |
| $m_{i,t}$ | suffix mask, $1$ if $t < L_i$ else $0$ |
| $\lambda$ | axiom weight (`lambda_ltn`) |
| $\mathrm{sat}$ | batch satisfaction $\in [0,1]$ |

Both axioms enter training the same way, as an additive term on the composite task loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \big(1 - \mathrm{sat}\big)$$

Both use the same quantifier, LTN's **p-mean error** aggregator for $\forall$ with $p=2$:

$$
\mathrm{sat} \;=\; \mathrm{A}_{\text{pME}}\big(x_1,\dots,x_n\big)
\;=\; 1 - \left( \frac{1}{n}\sum_{i=1}^{n} (1 - x_i)^{p} \right)^{1/p}
$$

This is a soft *minimum*: raising $p$ makes the aggregate increasingly dominated by the
worst-satisfied instances. At $p=1$ it is the plain mean; at $p\to\infty$ it is the minimum.
$p=2$ is a mild bias toward fixing the worst violations.

---

## Axiom 1 — time consistency (implemented)

### The statement

SuTraN+ predicts the time till each next event across the suffix (the *ttne* head) and, from a
single forward pass, the total remaining runtime (the *rrt* head). The two are trained
independently, yet the quantities are logically linked:

$$\sum_{t=0}^{L_i - 1} \Delta t_{i,t} \;=\; R_i$$

**This holds exactly in the ground truth** — verified per-instance across all 250,545 BPIC_19
test instances, maximum discrepancy 1.0 second against a mean target of ~4.3M seconds
(float32 rounding).

### Grounding

Both heads emit standardized values, so they are first returned to seconds using the training
set's stored moments $(\mu_{\Delta t}, \sigma_{\Delta t})$ and $(\mu_R, \sigma_R)$, and the
padded tail is zeroed:

$$
S_i \;=\; \sum_{t=0}^{W-1} m_{i,t}\,\big(\hat{z}^{\Delta t}_{i,t}\,\sigma_{\Delta t} + \mu_{\Delta t}\big)
\qquad\qquad
\hat{R}_i \;=\; \hat{z}^{R}_{i}\,\sigma_{R} + \mu_{R}
$$

$S_i$ is the summed ttne suffix; $\hat{R}_i$ is the rrt head's single prediction, read from
decoder position 0.

### Predicate

A smooth equality whose truth decays exponentially in the absolute discrepancy:

$$\mathrm{Eq}(S_i, \hat{R}_i) \;=\; \exp\!\left(-\frac{\lvert S_i - \hat{R}_i\rvert}{c}\right)
\;\in\; (0, 1]$$

with scale $c = \sigma_R$ by default. The scale sets what counts as "close": a discrepancy of
one target standard deviation gives truth $e^{-1} \approx 0.37$.

### Axiom

$$\mathrm{sat} \;=\; \forall i \;\; \mathrm{Eq}(S_i, \hat{R}_i)
\qquad\text{aggregated with } \mathrm{A}_{\text{pME}},\, p = 2$$

### Gradient routing (`detach_mode`)

Because the constraint is symmetric but the two heads are not equally trustworthy, the
implementation can freeze either side:

| `detach_mode` | Effect | Reading |
|---|---|---|
| `"none"` | both terms carry gradient | both heads move toward each other |
| `"ttne"` | $S_i \leftarrow \mathrm{sg}[S_i]$ | **only the rrt head moves**, toward $\Sigma\Delta t$ |
| `"rrt"` | $\hat{R}_i \leftarrow \mathrm{sg}[\hat{R}_i]$ | only the ttne head moves, toward rrt |

($\mathrm{sg}[\cdot]$ = stop-gradient.) Note that detaching a *term* does not isolate a *head*:
the encoder and decoder stack is shared, so the frozen side still receives gradient indirectly.

### Scope condition

The axiom does exactly one thing: it pulls one head toward the other. It therefore helps only
when the target is the better-calibrated estimator. Measured at baseline:

| | BPIC_19 | BPIC_17_DR |
|---|---|---|
| $\lvert$rrt bias$\rvert$ / rrt MAE | **43%** | 2.6% |
| $\lvert\Sigma\Delta t$ bias$\rvert$ / $\Sigma\Delta t$ MAE | 5.4% | **37%** |
| Better estimator | $\Sigma\Delta t$ | **rrt** |
| Effect of `detach_mode="ttne"` on rrt MAE | **−2.18%** | **+1.58%** |

Implementation: [`ltn_consistency.py`](ltn_consistency.py), hooked in at
[`SuTraN/train_procedure.py:217`](SuTraN/train_procedure.py).

---

## Axiom 2 — outcome consistency (designed, formulation A)

### The statement

On BPIC_17_DR the case outcome is *defined* as the last Offer event of the case, restricted to
three determining activities. Write

$$\mathcal{D} = \{\texttt{O\_Accepted},\ \texttt{O\_Cancelled},\ \texttt{O\_Refused}\}$$

$$\mathcal{O} = \{\texttt{Accepted},\ \texttt{Canceled},\ \texttt{Refused}\}$$

with the bijection $a : \mathcal{O} \to \mathcal{D}$ mapping each outcome to the activity that
produces it. The identity is:

$$
y_i \;=\; a^{-1}\!\Big(\,\alpha_{i,\,\tau_i}\,\Big),
\qquad
\tau_i \;=\; \max\{\, t < L_i \;:\; \alpha_{i,t} \in \mathcal{D} \,\}
$$

in words: **the last determining activity in the suffix fixes the outcome.** $\alpha_{i,t}$ is
the true activity at suffix step $t$.

This holds **exactly** — agreement $1.000000$, zero exceptions across all 346,415 train and
172,345 test instances with `instance_mask_out == False`, and separately exact for each of the
three classes.

Two caveats that shape the formulation:

- **Occurrence is not enough.** `O_Accepted` $\in$ suffix implies Accepted only 70.1% of the
  time — an accepted offer can be cancelled afterwards. 20.6% of instances contain $\ge 2$
  determining activities (43.9% of Canceled ones). Only the *last* one counts.
- **Applicability.** The axiom is valid on the non-leaky subset only, which is exactly the
  subset the outcome loss already trains on. Every such instance has at least one determining
  activity in its suffix, so coverage is 100% of the relevant domain.

### Grounding — soft "last determining activity"

Let $p_{i,t}(\cdot) = \mathrm{softmax}\big(\text{act\_logits}_{i,t}\big)$ over the activity
vocabulary. Define the probability that step $t$ is *some* determining event:

$$d_{i,t} \;=\; m_{i,t} \sum_{a \in \mathcal{D}} p_{i,t}(a)$$

the probability that **nothing determining happens after** $t$:

$$N_{i,t} \;=\; \prod_{s=t+1}^{W-1} \big(1 - d_{i,s}\big)$$

and hence the soft distribution over the outcome the suffix *implies*:

$$q_i(o) \;=\; \sum_{t=0}^{W-1} m_{i,t}\; p_{i,t}\big(a(o)\big)\; N_{i,t}
\qquad \text{for } o \in \mathcal{O}$$

$$\tilde q_i(o) \;=\; \frac{q_i(o)}{\sum_{o' \in \mathcal{O}} q_i(o')}$$

**The product is not an ad-hoc device — it is fuzzy logic.** The event "$a(o)$ occurs at $t$
and is the last determining event" is the conjunction

$$\mathrm{Occurs}(t, a(o)) \;\wedge\; \bigwedge_{s>t} \neg\,\mathrm{Det}(s)$$

and under the **product t-norm** (LTN's `AndProd`) fuzzy $\wedge$ is multiplication while
$\neg x = 1-x$. So $p_{i,t}(a(o)) \cdot \prod_{s>t}(1 - d_{i,s})$ *is* that formula evaluated,
and $q_i(o)$ is the fuzzy $\exists_t$ over it under the sum-based existential.

The residual mass

$$r_i \;=\; 1 - \sum_{o \in \mathcal{O}} q_i(o)$$

is the model's belief that *no* determining event occurs at all. In the ground truth $r_i = 0$
always, so a large $r_i$ during training means the axiom has nothing to bite on and should be
monitored.

### Predicate

Let $\hat{y}_i = \mathrm{softmax}\big(\text{outcome\_logits}_i\big) \in \Delta^2$ be the outcome
head's distribution, read from decoder position 0. Define agreement as the probability that two
independent draws from the two distributions coincide:

$$\mathrm{Agree}(\tilde q_i, \hat y_i) \;=\; \sum_{o \in \mathcal{O}} \tilde q_i(o)\,\hat y_i(o)
\;\in\; [0, 1]$$

This equals $1$ iff both distributions are the same one-hot, and is smooth everywhere. (A
Bhattacharyya coefficient $\sum_o \sqrt{\tilde q_i(o)\hat y_i(o)}$ is a defensible alternative
with softer gradients near the corners; the inner product is the simpler default.)

### Axiom

$$
\mathrm{sat} \;=\; \forall\, i \in \mathcal{V} \;\; \mathrm{Agree}(\tilde q_i, \hat y_i),
\qquad \mathcal{V} = \{\, i : \texttt{instance\_mask\_out}_i = \text{False} \,\}
$$

aggregated with $\mathrm{A}_{\text{pME}},\, p=2$, exactly as in Axiom 1. Restricting to
$\mathcal{V}$ matters: outside it the prefix already reveals the outcome, the outcome head is
not trained, and the identity is not guaranteed.

### Gradient routing

| `detach_mode` | Effect | Reading |
|---|---|---|
| `"none"` | both terms carry gradient | both heads move |
| `"act"` | $\tilde q_i \leftarrow \mathrm{sg}[\tilde q_i]$ | only the outcome head moves |
| `"outcome"` | $\hat y_i \leftarrow \mathrm{sg}[\hat y_i]$ | only the activity head moves |

These names must **not** reuse Axiom 1's `"ttne"` / `"rrt"`, or result-folder names become
ambiguous and runs can silently overwrite each other.

### Contrast with formulation B (rejected as a primary)

The weaker one-directional form $\;\forall i\,\forall o:\ \mathrm{Outcome}(i,o) \rightarrow
\exists t\, \mathrm{Occurs}(i,t,a(o))\;$ is also exactly true in the labels, and is far simpler
to implement. It is rejected as the headline formulation because it has a **degenerate
satisfying solution**: predicting all three determining activities in every suffix satisfies it
for any outcome at zero cost — and since 20.6% of real cases genuinely contain $\ge 2$, the
model barely has to distort anything to reach it. Formulation A has no such loophole, because
the product term forces a commitment about which activity is *last*. B is worth keeping only as
a numerical sanity floor if A returns a null result.

---

## Cautions / open items

- **Teacher forcing vs. free-running.** Both axioms are enforced on teacher-forced decoder
  outputs but evaluated on greedy decoding. Worse for Axiom 2: the implied outcome is a step
  function of activity *order*, so one drift event flips it, with no error cancellation.
  Log teacher-forced satisfaction and test-set consistency separately.
- **Softmax saturation.** $\partial p / \partial \text{logit} = p(1-p) \approx 0.02$ for a
  confident activity head, vs. a much less saturated 3-way outcome softmax. Expect Axiom 2 to
  move the outcome head far more than the activity head even under `detach_mode="none"`.
- **Masking of the product.** Positions past END must contribute a factor of exactly $1$.
  An unmasked $N_{i,t}$ silently runs over arbitrary predictions. Unit-test $q$ on synthetic
  one-hot suffixes (including accept-then-cancel) before spending a GPU-hour.
- **Numerics.** With ~1.2 determining events per suffix, $\prod(1-d_s) \approx 0.30$ — direct
  computation in float32 is fine. Log-space is a contingency; log the minimum product and switch
  only if it drops below ~1e-6. The `log1p` clamp introduces a silent gradient cliff.
- **Residual mass $r_i$.** If it grows, the axiom is toothless. Monitor it per epoch.
- **Detach does not isolate.** Shared encoder/decoder means the "frozen" head still moves
  (observed for Axiom 1 at $\lambda=2.0$).
- **Length masks come from labels.** $m_{i,t}$ is derived from the ground-truth suffix length,
  a mild train-time leak. Consistent with Axiom 1's existing behaviour; state it, don't hide it.
- **Clamping.** Axiom 1 at $\lambda=1.0$ raises the fraction of negative per-step ttne
  predictions from 0.254 to 0.408 on BPIC_17_DR. Some consistency is bought below zero, where
  the official metric clamps. Check the analogous effect for Axiom 2.
- **$\lambda$ does not transfer** — not between logs (3–4× difference for Axiom 1), and
  certainly not between axioms. Every new setting needs its own sweep.
- **Matched control required.** A result only counts if upweighting the corresponding task loss
  by a comparable amount fails to reproduce it. Budget those runs from the start.
- **Do not combine the two axioms** until each is characterised alone; the interaction would
  confound both.
