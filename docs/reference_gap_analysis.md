# Gap analysis: our model vs. Baptista et al. (2016)

A full audit of the Atlanta model against the reference paper and its Java
implementation, the behaviour problems that audit turned up, and a plan to
close them around a single focused result.

Every number below came from an experiment run against the current model;
the run that produced it is named so it can be re-run.

---

## Part 1 — What the reference actually does

### 1.1 The scientific claim

The reference's contribution is not "we built a housing model." It is:

> **House price boom-bust cycles emerge endogenously from the interaction of
> agents, and we can prove which mechanism causes them.**

They prove it with one experiment (their Figure 2): run the model normally,
then re-run it with the house-price-growth expectation `g` forced to zero.
The cycles disappear. That single ablation turns a descriptive model into a
causal claim, and it is the template for everything below.

Their second claim is a comparative static: raise the buy-to-let share of
households from 4% to 16%, and price cycles amplify (sd of quarterly price
growth 1.2% → 2.3%), house price levels rise, and the owner-occupier share
falls by ~25%. Then one policy — an LTI limit of 3.5 with a 15% allowance —
is shown to dampen the cycle (sd 1.21 → 1.09).

Note the shape: **one mechanism, one dose-response, one policy.** Not six
policies against five outcomes.

### 1.2 Their behavioural rules, and what each is for

| # | Rule | Purpose in the system |
|---|---|---|
| EQ1 | `ln(w) = α + β ln(y) + ε` desired bank balance | Sets wealth distribution → down payments |
| EQ2 | `C = max(α(b−w), 0)` consumption | Relaxes wealth toward target |
| EQ3 | `p_desired = αy·exp(ε)/(1−βg)` | **Appreciation enters demand** (feedback loop #1) |
| EQ4 | `g = α(h₋₁+h₋₂+h₋₃)/(h₋₁₃+h₋₁₄+h₋₁₅) − 1` | The expectations channel — the cycle driver |
| EQ5 | `P(buy) = σ(β[rQ(1+τ) − 12(m−pg)])` | Buy vs. rent margin; **couples rental market to sale market** |
| EQ6 | `P(sell)` with stock and interest-rate terms | Turnover → transaction volume |
| EQ7 | `ln p_s = α + ln(p̄) − β ln(ζ(1+f̄)) + ε` | Asking price from comparables |
| EQ8 | 5.5% chance of price cut | Clears stale listings |
| EQ9/12 | `Ω = (p/d)(δ(g+κ) + (1−δ)r̄) − m/d` | **Investor demand responds to both capital gain and rental yield** |
| EQ10/13 | `σ(βΩ)^(1/12)` | Investor buy/sell probability |
| EQ14 | `q = min(bχ/(1−χ), yψ, y_d ν·annuity)` | LTV / LTI / affordability limits |
| EQ15 | ICR constraint for BTL | Investor credit limit |
| EQ16 | `i_spread += α(M − T)` | Credit supply → interest rate |
| A.3 | `p'_q = D·p_q + (1−D)·h·pr(q)` | Stabilises price *distribution across qualities* at small N |
| A.4 | Multi-round double auction with bid-up | Price discovery |

The two feedback loops that make the model interesting:

- **Amplifying:** prices rise → `g` rises → EQ3 raises desired expenditure and
  EQ9 raises investor expected yield → more demand → prices rise further.
- **Stabilising:** investors buy → rental supply rises → **rents fall** → EQ5
  makes renting relatively more attractive → fewer buyers → price pressure
  falls.

The paper's whole BTL analysis is about the balance between these two. **Both
loops require rents to move.**

### 1.3 Their validation strategy — five distinct tests

This is the part worth copying most directly:

1. **Level matching.** Model averages vs. the FPC's nine core housing
   indicators. (What we currently do — and only this.)
2. **Distribution matching.** Model LTI and LTV *distributions* vs. loan-level
   PSD data. Explicitly: *"the target is to match distributions of data rather
   than aggregates or averages."*
3. **Emergent relationships.** A relationship they never coded appears anyway:
   credit growth ↔ house price growth, ≈0.13 in the model, compared against
   Favara & Imbs' published 0.2. Validation by "the model reproduces something
   nobody put in it."
4. **Sign tests.** Shock income +10%, credit target +30%, housing supply +7%;
   check every indicator moves in the direction theory predicts, by a
   plausible magnitude.
5. **Mechanism ablation.** Set `g = 0` → cycles vanish. Both a validation and
   the causal claim.

### 1.4 Scale

10,000 households, ~200-year spin-up, ≈40–50 transactions/month. This matters
more than it looks — see Problem 2.

---

## Part 2 — Audit of our model

### 2.1 What matches the reference well

These are genuinely faithful and should not be touched:

- **Market clearing (A.4).** Multi-round double auction, households matched to
  best affordable *quality*, investors to best expected *rental yield*,
  geometric bid-up on contested listings. Close port.
- **EQ6 selling.** Same functional form, α=4.0 / β=5.0 — the reference's exact
  values.
- **EQ7 asking price.** α=0.04, β=0.011, ζ≈1/31, ε=N(0,0.5) — exact match.
- **EQ10/13.** β=50, monthly `^(1/12)` exponent — matches (after this
  session's units fix).
- **Step order.** Demographics → construction → decisions → ownership market →
  rental market → interest rate. Matches.
- **EQ1/EQ2, EQ16, bankruptcy injection, inheritance, social housing state.**
  All present and structurally faithful.

### 2.2 Parameter divergences from the reference

| Parameter | Reference | Ours | Effect of the difference |
|---|---|---|---|
| EQ4 `α` (expectation strength) | 0.5 | **1.0** | Doubles the amplifying loop's gain |
| EQ3 `β` (appreciation → demand) | 0.08 | **0.3** | ~4x the appreciation feedback into demand |
| EQ3 `ε` sd | 0.5 | **0.15** | Much less heterogeneity in willingness to pay |
| Investor `δ` (capital-gain weight) | 0.5 / 0.9 | **0.3 / 0.6** | Our investors are *less* trend-following than even the reference's fundamentalists |
| EQ6 tenure | 11 years | 8 years | More turnover (helps volume — fine) |
| Households | 10,000 | **300–600** | See Problem 2 |
| Spin-up | ~200 years | **10 years** | Far less time to reach a settled state |

The first two stack: our amplifying loop is nominally ~8x the reference's
gain, while our investors — the agents meant to *carry* that loop — weight
capital gains less than the reference's most conservative type. These were
presumably tuned independently; together they don't describe a coherent
system.

### 2.3 Structural differences

- **Single price-per-quality.** The reference maintains a price for each
  quality band, and A.3 relaxes the *shape of the price distribution across
  qualities* (their words: "the shape, but not the level"). We collapsed this
  to one number with price linear in quality. We lose any "which market
  segment is hot" dynamics, and our version of A.3 acts on the level instead.
- **No income-by-age profile.** The reference fixes a household's income
  *percentile* for life and maps (percentile, age) → income, so income follows
  a life-cycle hump. We draw one lognormal at birth and hold it flat. (Already
  quantified in §11 of `methodology.md`: real median income swings ~3x over
  the life cycle.)
- **No income tax / National Insurance.** Minor.
- **Investor types.** The reference splits investors into fundamentalists
  (δ=0.5) and trend-followers (δ=0.9) at 50/50 and makes that split the
  subject of its headline experiment. We split by *institution size* instead,
  which is the right adaptation for the Atlanta question — but we never gave
  either type a trend-following weight comparable to the reference's.

---

## Part 3 — Diagnosed behaviour problems

Ranked by how much they block the science we want. The first three are fatal
to the story; nothing else matters until they're fixed.

### Problem 1 (FATAL) — Rents have no price discovery

`tract.rent_per_quality` is set once from config and **never updated by the
rental market**. Measured directly: across 240 simulated months it took
exactly one value, `1307.55`, forever. Meanwhile actual unit-level rents in
the same run ranged 740–7,846 with a mean of **2,324** — the "market rent"
every agent reasons against is 44% below the rent actually being charged.

There is no `record_rental` / rent-index update anywhere in
`markets/rental_market.py` (confirmed by grep — the sale market has
`record_sale` → `update_hpi_history`; the rental side has no equivalent).

That frozen number is read as the market signal by:

- `model.market_rent_for_quality()` → **EQ5's buy-vs-rent decision**
- `tract.gross_rental_yield()` → **r̄ in EQ9/12 investor decisions**
- `small_landlord.py` → r̄ for **EQ11 rent setting** (new rents anchor to a
  frozen number)
- `institutional_investor.py` → the base for its premium
- `ownership_market.py` → the yield ranking investors buy on

**Consequence:** the reference's stabilising loop (investors buy → rental
supply up → rents down → renting more attractive) **cannot operate at all**,
and investor yield expectations cannot respond to the rental market. This one
bug disables half the mechanism the paper is about.

### Problem 2 (FATAL) — The market is too thin for price discovery

Measured: **2.1 sales/month at N=300, 2.6 at N=600, 8.4 at N=2400.** The
reference at 10,000 households sees roughly 40–50. Our monthly price index is
formed from about two transactions.

The consequence is measurable and severe:

| | sd of quarterly house-price growth |
|---|---|
| **Real Atlanta (Case-Shiller ATXRSA, 2015–2026)** | **1.70%** |
| Reference model, benchmark | 1.2% |
| **Our model (N=300)** | **5.14%** |

We are **3x more volatile than the real Atlanta market**, and the volatility
is not the interesting kind. The decisive test — the reference's own Figure 2
ablation:

| Configuration | sd(quarterly growth) | peak-to-trough range |
|---|---|---|
| Baseline | 5.14% | 64.6% |
| **Expectations channel off (`g = 0`)** | **4.92%** | 57.0% |

In the reference, this ablation *removes the cycles*. In ours it changes
almost nothing. **Our price movement is sampling noise, not a behavioural
cycle.** Volatility does fall with scale (5.04% → 3.87% → 2.99% at N = 300 →
600 → 1200), confirming the noise diagnosis.

This also explains two things already in `methodology.md`: why the paired
confidence intervals were so wide, and why the EQ4-based appreciation metric
was biased upward (§13) — ratio asymmetry bites exactly when sd(g) is large,
and ours is ~0.08.

### Problem 3 (FATAL) — The investor sector is a decaying endowment, not a behaviour

Measured over 240 months at N=600:

- Investor-held units **fell from 242 to 188 (−22%)**. Investors are net
  sellers over 20 years.
- Investor purchases are **10.7% of all transactions** (institutional 3.5%,
  small landlord 7.2%); households are 89%.
- Multiplying the institutional investor population by **8x** (0.0125 → 0.10)
  changed investor-held share of stock from **0.248 to 0.247** — no effect at
  all.

The investor share of housing stock is set by initial conditions
(`initial_rental_stock`), not by investor behaviour, and it decays from there.
**A policy that restricts investor purchases is acting on 3.5% of transactions
in a sector that is shrinking anyway** — which is the real reason the
financial-penalty policies showed nothing.

### Problem 4 — The headline comparative static runs backwards

The reference: more BTL investors → higher prices, more volatility, ~25% fewer
owner-occupiers. Ours, sweeping institutional share at N=600, 4 seeds:

| Institutional share | sd(qtr growth) | Price level | Homeownership | FTB share of purchases |
|---|---|---|---|---|
| 0.0% | 4.25% | 273,424 | 0.526 | 0.250 |
| 1.25% | 4.39% | 268,099 | 0.514 | 0.270 |
| 5.0% | 4.02% | 266,597 | 0.523 | 0.290 |
| 10.0% | 3.85% | 253,706 | **0.548** | **0.308** |

More investors → **lower** prices, **higher** homeownership, **higher**
first-time-buyer share. That is backwards from the reference, backwards from
the real-world premise of the project, and it is a direct consequence of
Problems 1–3: investors can't bid prices up (thin market + reversion anchor),
can't respond to rents (frozen), and barely transact at all.

**This is the single most important thing to fix.** The project's entire
question presupposes a crowding-out channel that the model currently does not
contain.

### Problem 5 — EQ5 discards falling prices

`equations/buy_rent.py`: `g_upside = max(g_safe, 0.0)` — "only positive
appreciation enters buying cost." The reference uses `g` signed. Truncating it
removes the entire downside half of the expectations channel: when prices
fall, owning should look *worse* and the bust should deepen. In our model
falling prices are invisible to the buy/rent decision, which further flattens
the cycle.

### Problem 6 — Scale-dependent results, and validation run at the wrong scale

`validate_against_paper.py` defaults to **N=300**; `run_all_policies.py`
defaults to **N=600**. The script prints "spin-up and window match the policy
experiments" — true, but the *population* doesn't. And the results differ
materially:

| Target | N=300 | N=600 |
|---|---|---|
| homeownership_rate | 0.402 ✗ | **0.538 ✓** |
| rental_vacancy_rate | 0.107 ✓ | **0.037 ✗** |
| institutional_share_of_rentals | 0.265 | 0.379 |

6/7 at both scales — but a *different* target fails at each. A well-specified
model shouldn't reorganise its behaviour between 300 and 600 agents. This is
Problem 2 wearing a different hat.

---

## Part 4 — The plan

### 4.1 The story to build toward

The reference's shape, adapted to Atlanta:

> **"Do institutional investors push first-time buyers out of the market —
> and can one simple rule stop it?"**

- **One independent variable:** institutional investors' share of home
  purchases (the dose).
- **One outcome:** first-time-buyer share of purchases.
- **One policy:** the ownership cap — the only policy that showed a real
  effect, and the one with the cleanest mechanism to explain to a judge
  ("investors can't own more than N homes").
- **One mechanism proof:** turn off the expectations channel; show the
  crowding-out weakens. This is the reference's Figure 2 move, and it is what
  separates "I ran a simulation" from "I found a cause."

Why this beats the current six-policies-by-five-outcomes design for judges:
it fits on one poster, the causal claim is testable, and the ablation gives a
second, independent piece of evidence for the same mechanism.

### 4.2 Sequenced fixes

**Stage 1 — make the mechanism exist (required before any result is meaningful)**

1. **Give rents price discovery.** Mirror the sale side exactly: record each
   completed letting, form a smoothed rent index per tract, and update
   `rent_per_quality` from it (EMA + reversion, same two-stage form as
   `update_hpi_history`). This single change re-enables EQ5's rental margin,
   investor yield response, and the reference's stabilising loop.
   *Validation that it worked:* rent level responds to rental vacancy;
   investor share sweep now moves rents.
2. **Make the investor sector a flow, not an endowment.** Calibrate investor
   *purchase share of transactions* against Redfin's published Atlanta
   investor-purchase share (free, metro-level, quarterly) rather than setting
   stock at t=0. Investor holdings should be roughly stationary, not −22%
   over 20 years.
3. **Restore signed `g` in EQ5.** One-line change; restores the downside
   channel.

**Stage 2 — make the measurement trustworthy**

4. **Raise scale to where price discovery works.** Target ≥20 transactions/
   month. That is roughly N≈2,400–5,000 on current turnover. Runtime measured
   at ~87s per 360-month run at N=2400, so a 30-seed two-arm experiment is a
   few hours — affordable if run once, not interactively.
   *Acceptance test:* sd(quarterly price growth) lands near the real Atlanta
   1.70%, and the `g=0` ablation visibly reduces it.
5. **Align validation and experiment scale.** Make both default to the same N.
6. **Reconcile the amplification parameters.** Set EQ4 α and EQ3 β to the
   reference's 0.5 / 0.08 unless there's an Atlanta-specific reason not to,
   and raise investor δ toward the reference's 0.5/0.9 range. Currently we
   have an over-strong feedback path driven by under-responsive investors.

**Stage 3 — the experiment**

7. **Dose-response curve.** Institutional purchase share swept 0% → 20%, with
   FTB purchase share as the outcome. Paired CRN design (already built).
8. **Policy overlay.** Ownership cap applied at each dose level — does it
   restore FTB share, and by how much?
9. **Mechanism ablation.** Repeat the dose-response with `g=0`. If
   crowding-out weakens, the expectations channel is the cause.

**Stage 4 — deferred, only if time**

Quality bands, income-by-age (SCF fit already done, §11), income tax. None of
these block the story.

---

## Part 5 — Validation strategy

Five tiers, mirroring §1.3. We currently do only Tier 1.

### Tier 1 — Level targets (have it; fix the scale mismatch)
Keep the seven current targets. Fix: run validation at the experiment's N.

### Tier 2 — Distribution matching (highest-value addition, data already in hand)
The reference's most emphasised test, and we already have the data.

| Model distribution | Real comparison | Data source | Status |
|---|---|---|---|
| Owner-occupier LTV | HMDA Atlanta LTV distribution | `atlanta_hmda_2019.csv` | **have it** |
| Owner-occupier LTI | HMDA loan/income | `atlanta_hmda_2019.csv` | **have it** |
| Down payment by buyer type | Fannie Mae FTB flag | already pulled | **have it** |
| Sale price distribution | Zillow ZHVI | in repo | **have it** |
| Rent distribution | Zillow ZORI / HUD FMR by bedroom | ZORI in repo (level only) | partial |

This is a plot of two overlaid histograms per row — cheap to produce, and
exactly the kind of evidence a judge finds convincing.

### Tier 3 — Emergent relationship (not directly modelled)
Our analogue of their credit↔price test: **investor purchase share vs.
subsequent price growth**, compared against published estimates of the
investor-share effect in US metros. Nothing in the model codes this
relationship directly, so reproducing it is real evidence.

### Tier 4 — Sign tests
Shock and check direction on every indicator:
- household income +10% → prices up, LTI down, homeownership up
- housing supply +7% → prices down, rents down, homeownership up
- mortgage rate +200bp → prices down, FTB share down, months-of-inventory up

Cheap to run, and catches sign errors like Problem 4 automatically. **Had this
tier existed, Problem 4 would have been caught immediately.**

### Tier 5 — Mechanism ablation
`g = 0` → cycle amplitude collapses; investor crowding-out weakens. This is
simultaneously the strongest validation and the centrepiece of the science
fair story.

### Headline validation targets (with the real numbers)

| Metric | Atlanta real value | Source | Model now |
|---|---|---|---|
| sd quarterly house-price growth | **1.70%** | Case-Shiller ATXRSA (in repo) | 5.14% ✗ |
| sd annual house-price growth | **6.33%** | same | — |
| Homeownership rate | 52.9% | ACS 5-yr, 7 metro counties | 0.538 ✓ (N=600) |
| Rental vacancy | 7.3% | ACS | 0.037 ✗ (N=600) |
| Institutional share of SFR | ~30% | paper's cited figure | 0.379 ✓ |
| Investor purchase share | **needed** | Redfin Data Center (free) | 10.7% |

### Data sources still needed

1. **Redfin Data Center — investor purchase share, Atlanta metro.** Free bulk
   CSV, no signup. This is the missing calibration target for Problem 3, and
   the most important single data gap. It turns "investors are 10.7% of
   purchases" from an unvalidated model output into a calibrated input.
2. **Rent distribution, not just level.** ZORI gives the level; HUD Fair
   Market Rents by bedroom count (free) would give a distribution to validate
   against once Problem 1 is fixed.
3. Everything else needed is already in the repo.

---

## Part 6 — Honest framing for judges

Two things here are worth presenting rather than hiding, because they're
genuinely good science:

- **The `g=0` ablation currently *fails* to change anything.** That is a real
  finding: it says the model's price movement is noise, and it's why the fixes
  above are needed. Showing a negative result that you then diagnosed and
  fixed is a stronger story than showing only the final clean number.
- **The measurement bug in §13 of `methodology.md`** — the appreciation metric
  reporting +3.75%/yr when actual price growth was +0.99%/yr, positive in all
  8 seeds, including two where prices fell. Finding and fixing your own
  measurement error is exactly what judges mean by scientific rigour.

The current model cannot yet support the claim the project wants to make.
Stage 1 is what makes it able to.

---

## Part 7 — Stage 1 progress

### Done: rent price discovery (Problem 1)

`Tract.record_letting` / `Tract.update_rent_history` now mirror
`record_sale` / `update_hpi_history` exactly — two-stage EMA plus reversion
to a `reference_rent_per_quality` anchor, with an EMA'd `rent_index` guarding
the same double-counting failure the price side had. `_settle_lease` records
every completed letting; `model.step()` forms the rent signal after the
rental market clears. Rent went from **exactly one value across 240 months**
to 217 distinct values.

Two further reference gaps surfaced while building it:

- **No downward force on rents.** The reference (3.4.4) cuts demanded rent 5%
  for every month a unit sits unfilled — the rental analogue of EQ8. We had
  no equivalent: bid-up pushed rents up, EQ11's noise is zero-mean, so once
  rents were endogenous the level ratcheted to **3.4x the anchor with a 22.5%
  gross yield**. Added as `rental_market.vacant_rent_reduction`.
- **EQ11 was reading the wrong `f_bar`.** `small_landlord` passed
  `avg_days_on_market()` — the *ownership* market's days — where the
  reference is explicit that EQ11's `f_bar` is days on the *rental* market.
  Now `avg_days_vacant()`.

**The mechanism is unbiased.** Sweeping rental supply, rent responds
monotonically and lands almost exactly on the calibrated ZORI anchor at a
realistic vacancy rate:

| units/household | vacancy | rent vs. anchor |
|---|---|---|
| 0.5 | 2.7% | 1.95x |
| 0.7 | 2.7% | 1.66x |
| **0.9** | **8.9%** | **0.93x** |
| 1.1 | 16.0% | 0.62x |

### Done: signed `g` in EQ5 (Problem 5)

`max(g, 0)` removed; falling prices now raise the cost of owning as the
reference intends.

### Result

**Validation went from 6/7 to 7/7 at experiment scale (N=600, 8 seeds)** —
and not by tuning toward the targets. `rental_vacancy_rate` was the failing
target at 0.037; it is now 0.069 (real Atlanta: 7.3%) purely because rents
can finally clear the market.

The expectations-channel ablation also strengthened. Forcing `g = 0`:

| | before Stage 1 | after Stage 1 |
|---|---|---|
| baseline sd(qtr growth) | 5.14% | 4.91% |
| `g = 0` sd(qtr growth) | 4.92% | 4.07% |
| **gap** | **0.22pp (4%)** | **0.84pp (17%)** |

The channel now does roughly 4x more work, because EQ5 responds on both the
rent and price sides. It is still far from the reference's "cycles almost
completely disappear," and volatility is still 4.91% against a real Atlanta
1.70% — that is Problem 2, and it is Stage 2's job.

### FIXED: Problem 4 — convert not append

Re-measured after the Stage 1 fixes, the comparative static was still wrong:

| Institutional share | Price | Rent | Homeownership | FTB share |
|---|---|---|---|---|
| 0.0% | 260,059 | 2,183 | 0.542 | 0.255 |
| 1.25% | 249,503 | 2,082 | 0.509 | 0.231 |
| 5.0% | 243,485 | 1,954 | 0.509 | 0.257 |
| 10.0% | 210,501 | 2,667 | **0.571** | **0.389** |

This was expected: Problem 4 is downstream of the append-not-convert issue.
The fix (below) makes SmallLandlords a weighted selection from the household
pool rather than extra agents, matching the reference's BTL gene mechanism.
After the fix, household count stays at exactly N regardless of investor
fraction, and the construction target is no longer contaminated.

### Root cause of Problem 4 found: the investor dose is confounded with housing supply

Calibrating against Redfin's Atlanta investor-purchase share turned up the
actual reason the comparative static runs backwards.

**Investors are added to the population, not selected from it.** `model.py`
creates `n_households` renters and *then* appends
`n_households x small_landlord_fraction` additional `SmallLandlord` agents.
`SmallLandlord` is in `construction.HOUSEHOLD_TYPES`, so those extra agents
count toward the household total that `run_construction` sizes the housing
stock against (`target_units = 1.098 x n_households`).

So raising the investor dose raises the household count, which raises the
construction target, which builds more houses:

| Investor dose (x baseline) | Households | Housing stock |
|---|---|---|
| 0.0 | 622 | 700 |
| 1.0 | 678 | 738 |
| 2.0 | 761 | 816 |
| 3.0 | 848 | **900** |

A 3x investor dose silently comes with **+36% households and +29% housing
supply**. Prices fall and homeownership rises because *supply went up*, not
because of anything investors did. The independent variable is contaminated.

**The reference does not do this.** Its BTL "gene" is assigned to a fixed
proportion of *existing* households ("a fixed proportion (8%) of households
that are above the 50th percentile of income are given a BTL 'gene'").
Population is held constant; only composition changes. That is what makes
their 4% -> 16% comparative static clean.

**Fixed.** SmallLandlords are now selected from the N households at init
(weighted by SCF income-decile propensity, without replacement) and
assigned probabilistically at demographic birth. The population stays at
exactly N, construction targets are clean, and the 7/7 validation passes
at N=600 with 12 seeds.

### Investor purchase share is now calibrated against real data

Redfin Data Center investor-purchase share, Atlanta metro
(`redfin_investors_by_metro`, 2016Q1–2026Q2, investor = any buyer
purchasing under a business name, so both mom-and-pop and institutional):

| | Investor share of purchases |
|---|---|
| **Atlanta 2019** (our calibration year) | **22.3%** |
| Atlanta full-sample mean | 21.2% |
| Atlanta range | 13.2% – 37.6% (peak 2021-22) |
| Model before Stage 1 | 10.7% |
| **Model after Stage 1** | **24.1%** |

Rent discovery alone closed most of this gap without touching an investor
parameter: with `r_bar` frozen, investors' rental-yield signal was dead, so
EQ9's expected yield could not respond to the rental market. Restoring it
roughly doubled investor purchase activity, landing close to the real
figure. Investor holdings still drift down (~-16% over 180 months), which
the convert-not-append fix above should also address.

Rent discovery in fact made Problem 3 *more* visible rather than fixing it.
Holding total housing stock fixed and varying only the rental/ownership
split barely moves vacancy (1.7% → 3.6% across a wide range), because the
rental sector bleeds supply as investors sell down (242 → 188 units over 240
months). The frozen rent was concealing a genuinely starved rental market;
now it shows up as an elevated rent level. That is the same Problem 3 seen
from a new angle, and it is the next thing to fix.
