# Atlanta Housing ABM — Project Summary

A quick-orientation document: what this model is, what's been done to it, and
where things currently stand. For full technical detail on any item below,
see `docs/methodology.md` (the complete defect-by-defect and
finding-by-finding narrative this summary is drawn from).

## What this is

A Mesa-based agent-based model of the Atlanta, GA housing market, adapted
from Baptista et al. 2016 (Bank of England Staff Working Paper 619, whose
reference Java implementation is at
[INET-Complexity/housing-model](https://github.com/INET-Complexity/housing-model)).
Renters, first-time buyers, repeat buyers, small landlords, and institutional
investors trade in a single-tract market. The model evaluates six
institutional-investor restriction policies (waiting period, soft/hard
ownership caps, purchase tax, vacancy tax, progressive portfolio tax) against
homeownership rate, rental vacancy, price appreciation, and first-time-buyer
purchase share. The accompanying paper is at `paper/main.tex`.

**Repo note:** this session pushes to `anirudhvt/housing-abm-expansion` only,
not `anirudhvt/housing-abm` — the two may diverge; treat `-expansion` as the
current one.

## Status at a glance

- **7/7 validation targets met** at the paired-comparison spin-up/window
  (down from an original 2/5), after the bug-fixing and stationarity work
  below.
- **Two of the paper's original headline conclusions reverse** under the
  corrected model (waiting period, hard ownership cap — see below).
- **The price-formation mechanism was rewritten mid-session** (EMA +
  reversion port from the Java reference) to fix a runaway bubble at
  N=300 households; the delivered policy results in `results/` predate
  that fix and have **not yet been re-run** on the corrected model — the
  single highest-priority open item.
- **Extensive real-data grounding work** (HMDA, Zillow, Fannie Mae loan
  performance, SCF) has been layered on since, documented in
  `docs/methodology.md` Sections 9-12.

## Phase 1 — Fixing the original wide-confidence-interval problem

The starting complaint: paired policy effects had confidence intervals wide
enough to cross zero on almost every outcome. Diagnosed as three compounding
problems, in order of impact:

1. **Model bugs that severed the channels the policies act through** — seven
   found, three of which made specific policies incapable of any measurable
   effect: resale listings never actually reaching the sale market
   (`queue_listing` bug, 79% of repeat buyers stuck), an investor buy/sell
   logistic saturated at P=1.0 from an annual/monthly unit mismatch (EQ9/12),
   and a paired-experiment design that didn't actually pair (two independent
   models per seed instead of a shared spin-up forked into arms — arm
   correlation was ~0, sometimes negative). Also: purchase tax mis-charged,
   vacancy tax grace period off by a unit (months vs. days), portfolio tax
   brackets set far above the realized portfolio distribution, and the
   waiting period restricting twice as many agent classes as the other five
   policies.
2. **An experiment design that discarded information** — rebuilt around a
   shared spin-up per seed forked into baseline/policy arms, split RNG
   substreams for exogenous processes, window-averaged outcomes instead of a
   single-month snapshot, Holm-Bonferroni correction across the full
   policy x outcome grid, bootstrap CIs alongside normal-theory ones, and
   minimum-detectable-effect reporting.
3. **A baseline that hadn't reached demographic/population stationarity** —
   wrong initial age distribution (uniform instead of the mortality-implied
   steady state), an uncompensated population sink from rent-burden exits,
   no capital outflow for institutional investors (added a REIT-style payout
   ratio), and a zero rental-vacancy floor from instant re-letting (added a
   1-month frictional void).

Also added: an income-qualification gate on the renter→first-time-buyer
transition (previously savings-only) and listing withdrawal after 6 months
on market (nothing previously took a stale listing off the market, which had
deadlocked ~20% of repeat buyers and run inventory to ~108 months).

**Result:** median CI width 7.4x tighter, mean arm correlation 0.22 → 0.71,
tests surviving Holm correction 0 → 10 (of 30), validation targets 2/5 → 7/7.
Two conclusions reversed: the **waiting period** (the paper's headline
recommendation, originally +3.3% homeownership) now measures **-2.1%** —
excluded from fresh listings, investors buy discounted stale inventory
instead, where the same capital buys more homes. The **hard ownership cap**
does suppress institutional purchases and is the only policy that raises
homeownership (+0.82%), but the diverted purchases go mostly to **small
landlords**, not first-time buyers. The null result on all three financial
penalties (purchase/vacancy/portfolio tax) is now a real, informative bound
rather than an artifact of a broken mechanism.

## Phase 2 — Price-formation port (mid-session rewrite)

Implementing the paper's mechanisms as closely as possible surfaced a new
problem: at N=300 households, price ran to ~29x median income within 15
years (vs. ~7.6x at N=900) — a reflexive bubble specific to thin markets.
Root cause: the tract's price was a raw overwrite from each month's sales
median, and next month's appreciation signal (EQ4) was computed from that
same series, so noise compounded through the demand-appreciation feedback
loop.

**Fix:** ported the reference Java model's two-stage price formation
(`HousingMarketStats.postClearingRecord()`) — an EMA blend of realized
sales into the smoothed price, plus a reversion term pulling back toward a
calibrated reference level (the 2019 Atlanta ZHVI/ZORI mean, which happened
to already be sitting unused in the repo and matches the paper's own cited
figures almost exactly). A further bug specific to this model's single
continuous-quality tract (no quality bins to pool sales across, unlike the
reference) required also EMA-smoothing the house-price-index itself, not
just the price level — caught via a runaway-price test before shipping.
12 new tests pin the two-stage arithmetic.

**Not yet done:** the full 60-seed policy study has not been re-run since
this fix. The results in `results/` predate it.

## Phase 3 — Grounding the model in real data ("the calibration work")

Triggered by: "I want to ground the model and refine it... how do we
calibrate it as close as possible to how the reference calibrated theirs?"
The reference's own pattern (visible in its
`src/main/resources/calibration-code/*.py`): pull real microdata → build a
weighted empirical distribution → feed it into the model, for nearly every
parameter. A full tiered roadmap of US data-source equivalents was built
first (published as an artifact), then worked through in order:

**Already-downloaded data, just not fully used (fixed):**
- `pull_zillow_data.py` was mislabeled — claimed to pull ZHVI (home prices)
  but its URL/output was actually ZORI (rents); now pulls and labels both
  correctly.
- `use_external_appreciation_data` was a config flag read nowhere — the full
  2015-2026 ZHVI/ZORI history sat unused beyond a single 2019 mean. Now
  actually wires the real series into the model's appreciation signal when
  enabled (off by default — a real modeling choice, not a bug fix).
- `pyproject.toml` was missing `requests` as a dependency despite
  `pull_hmda_data.py` needing it — surfaced when re-running that pull
  locally.
- `pull_hmda_data.py`'s `pull_year()` silently dropped any requested field
  not present in the API response (including `applicant_age` on first
  attempt, and a `combined_loan_to_value_ratio` that had *never* existed in
  the live API) — now warns loudly instead of failing silently.

**Down-payment calibration (EQ17), in two passes:**
- First pass: fit against real Atlanta HMDA data using loan type (FHA vs.
  conventional) as a proxy for first-time vs. repeat buyer, since HMDA
  carries no direct first-time-buyer flag.
- Second pass: replaced the proxy with **Fannie Mae's Single-Family Loan
  Performance Data**, which has a genuine `First Time Home Buyer Indicator`.
  The real flag *reversed* the proxy's main claim — only 33.5% of actual
  first-time buyers sit at the down-payment floor (the proxy said 85%,
  *above* the model's own 55% assumption, not below).
- A rigorous paired before/after comparison (60 seeds, same CRN design as
  the policy study) showed adopting either fit moves modeled leverage
  (LTV/LTI) but not homeownership or purchase-share outcomes — **not
  adopted into config**, left as a documented finding.

**Age-conditioned LTV/LTI (attempted, then deliberately not wired in):**
- Fit a real regression (`LTV ~ log(income) + age`) from HMDA's
  `applicant_age` field: mean LTV declines monotonically from 92.6% (<25)
  to 83.6% (75+), a genuine effect net of income.
- Investigated how the reference model actually uses its equivalent
  (`decideLTV()`) before designing how to wire this in — and found it's
  **dead code**: the only call site is commented out in the reference's own
  `Household.java`, under the original authors' own TODO ("decision needed
  between this and previous specification"). The paper's actual results
  never depended on age at all. Decision: leave this as a documented
  finding rather than build a competing LTV-cap mechanism into the model.

**Survey of Consumer Finances (SCF), the US analogue of the reference's
single richest source (UK's Wealth and Assets Survey):**
- Fit age distribution, income-given-age, EQ1's wealth-given-income
  relationship, and investor propensity by income decile, all from one
  2022 SCF extract.
- Investor propensity was the standout: a real, monotonic ~20x gradient
  (3% of the bottom income decile owns a second residential property,
  60% of the top decile) that the model had zero equivalent for. Checked
  precedent first (learned from the decideLTV surprise) — the reference's
  equivalent (`BTLProbability.getBinAt(incomePercentile)`) *is* genuinely
  live in every household's constructor, not commented out.
- **This one was wired in**: small-landlord selection now draws incomes
  from the same household distribution as everyone else, then selects
  without replacement weighted by the real SCF curve — replacing a
  disconnected, unexplained "landlords skew higher-income" lognormal.
  Verified with 8 new unit tests, a smoke-run income comparison, and a
  before/after `validate_against_paper.py` run (via `git stash`) to isolate
  this change's effect from everything else — 5/7 targets both times,
  identical pre-existing failures both times, investor-share calibration
  undisturbed.

## Stage 2 — Convert-not-append investor fix

Problem 4 from the gap analysis: SmallLandlords were appended to the
population as extra agents, inflating the household count used by
construction targeting. Raising the investor dose raised the housing
supply, making prices fall instead of rise.

- SmallLandlords are now selected FROM the N households at init, weighted
  by the SCF income-decile propensity curve (without replacement). New
  births get the same probabilistic assignment. Population stays at exactly
  N for any investor fraction.
- InstitutionalInvestors remain separate corporate entities, excluded from
  construction targeting.
- SmallLandlord replenishment removed; the demographic birth process
  maintains the fraction in expectation.
- Standardized at N=600 across all scripts and config.
- Validation: 7/7 targets met at N=600, 12 seeds, 120mo spinup+window.
- 60-seed policy study re-run with the fix (results/ directory updated).

## What's still open

Roughly in priority order:

1. **Rewrite `paper/main.tex` Sections 5-7** to match the corrected results
   — still arguing the original (reversed) conclusions.
2. **A residual +1.7%/yr appreciation drift**, population-invariant, left as
   an open judgment call (strengthen the price-reversion constant, or
   revisit EQ3's own beta calibration) — not yet decided.
3. **Remaining calibration roadmap steps** (see `docs/methodology.md`
   Section 9's tiered plan): SSA/CDC life tables + American Housing Survey
   tenure data (step 3), checking what county GIS/assessor portals expose
   before deciding on ZTRAX (step 4), and the two structural SCF findings
   not yet wired in — age distribution and income-given-age both replacing
   an analytic/flat approximation with a real empirical target, and EQ1's
   wealth-given-income dispersion finding once its income-unit mismatch
   (SCF annual vs. the model's monthly) is reconciled.

## Where to look

- `docs/methodology.md` — the full technical narrative, section by section,
  including every defect, every calibration finding, and the measurement
  behind each one.
- `scripts/calibrate_*.py` — the calibration scripts (HMDA proxy, Fannie Mae
  ground truth, HMDA age/LTV, SCF), each runnable against a real data file.
- `scripts/compare_downpayment_calibration.py`, `scripts/run_all_policies.py`
  — the paired-CRN comparison harness, reused across every "does adopting
  this finding actually change the results" check in Phase 3.
- `results/` — raw CSVs and JSON summaries from every comparison run.
- `config/baseline_params.yaml` — every parameter, with provenance comments
  added anywhere this session touched a value.
- `tests/` — 108 tests, including regression tests for the price-formation
  fix and the investor-propensity wiring.
