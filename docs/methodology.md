# Why the original confidence intervals crossed zero

The first version of this study reported paired differences with 95% intervals
that spanned zero on almost every outcome. Three things were responsible, in
roughly this order of magnitude:

1. **Model bugs that severed the channels the policies act through.** Several
   policies could not have produced an effect no matter how many seeds were
   run.
2. **An experiment design that discarded most of the information each run
   produced**, and in particular a pairing scheme that did not actually pair.
3. **A baseline that was still drifting** when the outcomes were measured, so
   seeds were compared at different points on their own trajectories.

This document records what each of those was, how it was diagnosed, and what
replaced it.

---

## 1. Bugs that removed the policy transmission channels

### Resale listings never reached the market

`AtlantaHousingModel.queue_listing` recorded the seller of a resale listing in
`_resale_sellers` but never added the unit to `for_sale_units`. Since
`run_ownership_market` only ever iterates that registry, every resale listing
was flagged for sale and then never offered to a buyer.

Measured on the original code at 300 households over 300 months: **316 of 363
queued resale listings were orphaned, and 291 of 369 repeat buyers (79%) ended
the run permanently stuck holding a home they had listed and could not sell.**

This is the single most consequential defect, because the supply-release
channel is the entire mechanism by which the three financial penalties are
supposed to work: reduce effective yield -> raise the investor's probability of
selling -> release units to owner-occupiers. Step three did not exist. It also
disabled forced divestiture under the hard ownership cap, which lists excess
units through the same path.

Listing and rental-stock membership now go through single entry points
(`list_for_sale`, `add_rental_unit`, `drop_rental_unit`, `register_unit`) so
the registries cannot drift out of sync with unit state again.

### The investor decision logistic was fully saturated

EQ 9/12 compute the expected and effective yield; EQ 10/13 turn them into
monthly probabilities via `sigma(beta * Omega)^(1/12)`. Baptista et al.
calibrate `beta = 50`, and their `Omega` is monthly: `g` is a monthly
house-price growth expectation and `m/d` is a monthly mortgage payment over the
down payment.

This model computed `g`, `kappa` and `r_bar` as *annual* rates (EQ 4 compares a
three-month price average against the same average twelve months earlier) while
passing a monthly `m/d` alongside them. `Omega` came out roughly an order of
magnitude too large, and internally inconsistent.

At `beta = 50` and annual-scale `Omega`, across every market state tested:

| `g` (annual) | `Omega` | `P(buy)` | `P(buy)` with 5% purchase tax | `P(sell)` |
|---|---|---|---|---|
| -0.05 | 0.190 | 1.0000 | 0.9999 | 0.00022 |
|  0.00 | 0.250 | 1.0000 | 1.0000 | 0.00003 |
|  0.02 | 0.274 | 1.0000 | 1.0000 | 0.00002 |
|  0.05 | 0.310 | 1.0000 | 1.0000 | 0.00001 |
|  0.10 | 0.370 | 1.0000 | 1.0000 | 0.00000 |

Investors bought with probability 1.0000 every month regardless of market
conditions, essentially never sold, and **no financial penalty of any plausible
size could move them** -- a 5% purchase tax shifted `P(buy)` by less than
1e-4. Landlord portfolios could therefore only ever grow.

The paper attributed the null result on financial penalties to Atlanta's gross
rental yield providing "a cushion against tax surcharges". The actual cause was
a units mismatch that placed the logistic in a region where its derivative is
about 1e-5. On the monthly basis the equation is defined on, the same 5%
purchase tax moves `P(buy)` by ~0.007 per month.

`tests/test_investor_yield.py::test_omega_stays_inside_the_logistic_response_region`
now pins `|beta * Omega| < 2` across the realistic range and requires a visible
`P(buy)` response to the purchase tax, so this cannot regress silently.

### The purchase tax was charged to existing holdings

`compute_policy_cost` summed all three taxes and was passed to both
`expected_yield_buy` (EQ 9) and `effective_yield_sell` (EQ 12). A tax on
*acquiring* a home therefore also reduced the effective yield on homes an
investor already owned, raising their probability of selling. The scenario made
investors churn their portfolios rather than deterring acquisition, and
investor purchases went *up* under the tax.

Acquisition costs and holding costs are now separate. The purchase tax is
applied as the paper specifies it -- as an increase in the cash the investor
must put up for the same asset -- and is carried on the bid separately from the
down payment, since it buys no equity. (Folding it into the down payment would
shrink the mortgage principal, so the "tax" would have reduced the buyer's
borrowing instead of costing them anything.)

### The vacancy tax grace period was 90 months

`HousingUnit.day_vacant` is incremented once per model step, and a step is one
month, so it counts months. It was compared directly against
`vacancy_threshold_days: 90`, enforcing the 90-day grace period as 90 months --
seven and a half years. The tax essentially never bound.

### The progressive portfolio tax was a tax of zero

Brackets ran 10 / 50 / 100 / 350 units. The realised institutional portfolio
distribution is roughly 0-18 units, so nearly every holding sat in the 0%
bracket. The scenario nominally tested a progressive tax and actually tested no
tax, which is why it produced "the smallest effect ... not statistically
distinguishable from zero". Brackets now span the distribution the model
produces.

### The waiting period restricted twice as many agents as the other policies

`waiting_period.yaml` had `applies_to: [small_landlord, institutional]` while
all five other scenarios were institutional-only. The waiting period's larger
measured effect partly reflected a broader treatment rather than a better
mechanism. All six now target the same population.

### Registry and accounting defects

- `rental_units` retained units sold to owner-occupiers, so the denominator of
  the rental vacancy rate accumulated owner-occupied homes and the measured
  vacancy rate fell steadily toward zero for purely mechanical reasons.
- `for_sale_units` grew monotonically because sold units were never removed, so
  `houses_per_capita` -- the market-supply signal in EQ 6 -- measured
  cumulative construction rather than live listings.
- The initial rental stock was created with `owner = None` and never assigned.
  At the default configuration that left **~35% of the housing stock inert**:
  no agent could sell it, it never reached the sale market, and no investor
  policy could touch it. It is now distributed across small landlords and
  institutional investors to hit two calibration targets at once -- a ~30%
  institutional share of the rental stock, and 2-10 units per mom-and-pop
  landlord.
- `_liquidate_estate` demolished units a deceased agent merely rented.

---

## 2. The pairing did not pair

The original harness built two independent models per seed and ran each through
its own spin-up with the policy active from month one, then differenced them.
Passing the same seed to both does not make them comparable: the policy changes
an agent decision in the first month, every later random draw shifts, and after
a 600-month spin-up the two arms are two unrelated realisations that happen to
share a seed.

The diagnostic is the correlation between arms across seeds. Under working
common random numbers it should be close to 1. Measured on the original
results:

| policy | corr(baseline, policy) on homeownership |
|---|---|
| ownership_cap_hard | +0.11 |
| ownership_cap_soft | +0.27 |
| portfolio_tax | +0.59 |
| purchase_tax | +0.25 |
| vacancy_tax | +0.14 |
| waiting_period | **-0.06** |

With correlation near zero, `sd(difference) = sqrt(2) x sd(one arm)`:
differencing *added* variance rather than removing it. The waiting-period
figures show exactly this -- `sd(base) = 0.033`, `sd(diff) = 0.040`, against
`sqrt(2) x 0.033 = 0.046`.

### Shared spin-up, then fork

Each seed is now spun up **once**, with no policy, and the resulting model is
deep-copied into each arm. Both begin from a bit-identical microstate: same
agents, same balances, same houses, same prices. The fork was verified to
produce bit-identical trajectories when both copies are stepped without an
intervention.

This also halves the compute, since the spin-up runs once per seed instead of
once per arm.

### Split RNG streams

Even after forking, the arms desynchronise as soon as the policy changes how
many random numbers are consumed -- so the two arms of a seed would see
different interest-rate paths and different birth/death sequences, noise
unrelated to the policy that enters the paired difference in full.

The model now spawns independent substreams from one `SeedSequence`:

- `random_gen` -- market and behavioural draws
- `rng_macro` -- the exogenous interest-rate path
- `rng_demography` -- births, deaths, heir selection

The macro and demographic paths are therefore bit-identical across arms for a
given seed whatever the policy does.

Arm correlation after these changes: **+0.5 to +0.99**, with achieved variance
reduction up to 50x versus unpaired sampling. Both numbers are reported for
every effect so the reader can check the pairing is working rather than assume
it.

---

## 3. Estimating outcomes from a single month

Every outcome was read from one terminal snapshot. Over a few hundred agents,
one month's homeownership rate is close to a single binomial draw: at
`p = 0.5` and `n = 300`, its standard deviation is ~2.9 percentage points,
against policy effects of 1-3 points.

Outcomes are now averaged over the measurement window. How much this buys
depends on the autocorrelation of the series, which the design diagnostics
report as an effective sample size:

| outcome | `n_eff` of 240 months | SE reduction vs a snapshot |
|---|---|---|
| rental vacancy rate | 59 | ~7.7x |
| first-time-buyer purchase share | 32 | ~5.6x |
| annual appreciation | 4.7 | ~2.2x |
| homeownership rate | 2.7 | ~1.6x |
| institutional share of rentals | 3.2 | ~1.8x |

The stocks are highly persistent and gain little; the flows are nearly white
month to month and gain a great deal. This is worth knowing before choosing
which outcome to build an argument on.

### Measure the flow, not only the stock

`ftb_purchase_share` -- first-time buyers as a share of completed purchases --
was added because it is the quantity every one of these policies acts on most
directly: who wins the bidding on a given listing. The homeownership *stock*
only moves as fast as that flow accumulates, so it responds later and more
weakly to the same intervention.

It is computed by pooling numerator and denominator over the window, not by
averaging monthly ratios. A handful of homes change hands per month, so the
monthly share is a ratio of very small integers -- 0/2, 1/3, undefined -- and
its mean is both noisy and biased toward low-transaction months. Pooling first
raised the arm correlation on this metric from a value that decayed to **-0.53**
at long windows to a stable **+0.84 to +0.92**.

### Choosing the measurement window

Longer windows average away more within-run noise, but the two arms drift apart
as they run, so the CRN correlation decays. The standard error of the paired
effect is minimised somewhere in between rather than at the longest window.
`scripts/diagnose_design.py` sweeps window length and reports both, so the
choice is made from measurement rather than convention.

---

## 4. The baseline was not stationary

If the model is still drifting when outcomes are measured, each seed is sampled
at a different point on its own trajectory and that spread enters the
cross-seed standard deviation as noise. It also means a validation table
computed at one spin-up length says nothing about the model state a policy
experiment at a different spin-up length is measuring -- which is why the
original paper reported a validated homeownership rate of 0.52 while the policy
runs were producing ~0.24.

Four sources of drift were found and removed.

**The initial age distribution was wrong.** Ages were drawn `U(22, 65)` against
a stationary distribution -- the one implied by the model's own mortality
hazard -- with median 54 and a tail past 88. The initial population had nobody
near the mortality midpoint, so deaths stayed near zero for several decades
while births continued; the population grew, hit the hazard wall together as one
cohort, and collapsed. That transient is longer than the whole simulation, so
**no spin-up length reached a steady state.** Ages are now drawn from the
stationary distribution, which starts the model where the spin-up was supposed
to end.

Note that the mortality curve was already correctly calibrated: at
`mortality_scale = 1.0` the implied mean household lifetime is 53.3 years,
giving a death rate of 1.876%/yr, exactly the configured birth rate. Baptista
et al. enforce the same condition by scaling the mortality pdf. Only the
*initial condition* was inconsistent with it.

**Rent-burden exits were an uncompensated population sink.** Households pushed
out by rent burden left permanently, at ~0.4%/yr on top of mortality, tipping
the birth/death balance negative. Real metros replace out-migrants with
in-migrants; they are now replaced one-for-one.

**Institutional investors had no capital outflow at all.** Household agents are
drained by EQ 2 consumption and small landlords by the same rule, but every
dollar of institutional net rental income compounded into buying power forever.
Their share of the rental stock climbed monotonically -- 30% to ~60% over a long
run -- so the ownership distribution never settled. Real single-family rental
operators are largely REIT-structured and must distribute the bulk of taxable
income, so a payout ratio is both the realistic and the stabilising choice.

**Entrants were drawn richer than the initial population.** The initial cohort
came from `lognormal(8.6, 0.65)` (median ~$5,430/mo, matching the ACS metro
Atlanta median) while later entrants came from `lognormal(8.9, 0.55)` (median
~$7,330/mo). As the initial cohort was replaced, the population drifted steadily
richer and homeownership drifted up with it -- a trend in every reported outcome
that had nothing to do with any policy. One distribution now serves both.

A fifth change was needed for a different reason: vacated units were re-let in
the same month they were vacated, so turnover produced no vacancy at all and
the rental vacancy rate sat at ~0 in every run. That both mis-stated the level
against the ACS figure and **censored one of the three reported outcomes at a
floor**, where a policy can only move it in one direction. A one-month
frictional void between tenancies against an ~18-month average lease puts
turnover vacancy in the ACS range.

`scripts/diagnose_design.py` regresses each monthly series on the month index
and reports which outcomes are still drifting, so any residual trend is stated
rather than assumed away.

---

## 5. Population size, and why the matching loop had to be rewritten

Population is the most direct lever on sampling noise, and the study was pinned
at 300 households by runtime: the model scaled roughly quadratically, so 1200
households cost about 11x what 300 did.

The cost was concentrated in Phase 1 of the double auction, where every
remaining bid scanned every remaining offer -- and ran the per-unit policy
predicate for each `(bid, offer)` pair, tens of millions of times per run.
Offers are now sorted by price once per round with a running argmax of the
preference key, so a bid's preferred offer is a binary search plus one array
lookup (`housing_abm/markets/offer_book.py`). Ties in the preference key are
common, because the construction sector emits units at exactly quality 1.0, and
are broken by a per-round random jitter that reproduces the uniform random
tie-break the original did explicitly. Unit-level policy eligibility depends
only on the agent's *class*, so it is resolved once per class per round rather
than once per pair.

At 1200 households this cut a 30-step profile from 125s to 10s.

### Policy thresholds must be absolute, not per-capita

An early attempt expressed ownership caps and tax brackets per 1000 households,
on the assumption that thresholds should scale with market size. Measurement
showed the opposite: because `institutional_investor_fraction` ties the *number*
of investors to the household count, the rental stock and the investor count
grow together and the mean institutional portfolio stays near 7-12 units at
every population from 300 to 2400 households.

A per-capita cap therefore gets looser as the population grows. At 1200
households it stopped binding entirely, and the measured effect was **exactly
zero on every outcome** -- a result that would have looked like a substantive
finding rather than a specification error. Absolute unit counts are the
scale-invariant form here, and the configs say so.

---

## 6. Reporting

Every effect is reported with:

- the paired 95% t interval **and** a bootstrap interval over the same paired
  differences; material disagreement means the normal approximation is not safe
  for that outcome
- the arm-to-arm correlation and achieved variance reduction, so the reader can
  verify the pairing worked
- **Holm-Bonferroni adjusted p-values across the full grid** of six policies x
  five outcomes. Thirty tests against one baseline need family-wise error
  control before any single result is called significant; Holm is uniformly
  more powerful than Bonferroni and assumes nothing about independence, which
  matters because the arms share a spun-up state
- the **minimum detectable effect** at 80% power, so a null reads as a bound on
  the effect rather than as evidence of its absence, and the seed count that
  would be needed to resolve the observed point estimate

Purchases are also broken out by buyer class, so the incidence question -- who
actually buys the homes a policy diverts from institutional investors -- is
read directly from the data rather than inferred from the homeownership stock.


---

## 7. Robustness of the corrected findings

`beta_institutional` is the shape parameter of the EQ 10/13 logistic -- the
parameter that governs how far any policy can move investor behaviour, and the
one the saturation bug lived in. It is also the parameter the paper itself
identifies as most uncertain. Sweeping it across the paper's own range and
re-running the full paired comparison at each value (15 seeds per cell):

| policy | beta=10 | beta=25 | beta=50 | beta=75 | beta=100 |
|---|---|---|---|---|---|
| waiting period | -0.0206 * | -0.0205 * | -0.0220 * | -0.0258 * | -0.0175 * |
| ownership cap (hard) | +0.0086 * | +0.0110 * | +0.0088 * | +0.0045 | +0.0108 * |
| ownership cap (soft) | +0.0041 | +0.0000 | +0.0003 | -0.0015 | +0.0016 |
| purchase tax | +0.0015 | -0.0002 | +0.0019 | -0.0001 | +0.0044 |
| vacancy tax | +0.0040 | -0.0030 | -0.0039 | -0.0045 | +0.0010 |
| portfolio tax | +0.0005 | -0.0012 | -0.0011 | -0.0061 * | +0.0011 |

(effect on the homeownership rate; `*` = 95% interval excludes zero)

The waiting period harms homeownership at every value, significantly. The hard
cap helps at every value, significantly at four of five. The three financial
penalties stay null throughout; the single portfolio-tax cell at beta=75 is one
significant result in thirty uncorrected tests, which is what a 5% false
positive rate looks like.

The reversal of the paper's headline finding is therefore not an artefact of a
parameter choice.


---

## 8. Porting the reference Java model's price-formation mechanism

Section 5's `beta_institutional` sensitivity sweep was run against a known
limitation flagged in that section's own diagnosis but not yet fixed: the
model was pinned at 300-900 households partly because `tract.price_per_quality`
was a hard overwrite from the raw median of the last 60 sales, with no
smoothing and no reversion to any fundamental level. In a thin market that
raw median is a noisy estimator, and because next month's appreciation signal
(EQ4) is computed from that same series, its jumps compound: a noisy jump
raises `g`, which raises `desired_expenditure` (EQ3) via `1/(1-beta*g)`,
which raises the next sale price, which raises the median further. Measured
at 300 households this produced a price level running to **~29x median
household income after 15 years** (vs ~7.6x at 900 households) -- a reflexive
bubble specific to thin markets, not a calibration-band miss, and the direct
cause of the low homeownership rate and inflated appreciation reported for
N=300 in Section 5.

The reference Java model (Baptista et al. 2016;
`housing/collectors/HousingMarketStats.java`) never does a raw overwrite.
Every price series in it is formed by two-stage smoothing:

1. **EMA.** Blend this month's actual transaction average into a running
   smoothed price at a fixed monthly weight
   (`SMOOTHING_FACTOR`, derived from `CUMULATIVE_WEIGHT_BEYOND_YEAR = 0.25`
   via `1 - 0.25^(1/12) ~= 0.109`). Skipped in months with no sales.
2. **Reversion to reference.** Every month, regardless of sales, pull the
   smoothed price back toward a *fixed calibrated reference level* scaled by
   an evolving house price index: `DECAY * smoothed + (1-DECAY) * (HPI *
   reference_price)`, with `MARKET_AVERAGE_PRICE_DECAY = 0.5`.

Stage 2 is what a pure EMA lacks: nothing stops an EMA-only price from
drifting arbitrarily far from fundamentals over a long sales drought, since
it only ever updates toward whatever (possibly sparse, possibly noisy)
transactions happen to occur. This is now ported into `Tract` (see
`src/housing_abm/tract.py` and the `market_smoothing` / `tract_calibration`
sections of `config/baseline_params.yaml`), using the reference's own
calibrated constants.

### An adaptation the port required, and why

The reference model pools each month's sales across many quality bins
(`N_QUALITIES`) before computing its house price index, which averages out a
lot of single-transaction noise even when any one bin trades thinly. This
Atlanta port has one continuous-quality price series per tract, not banded
quality classes to pool across -- so a first implementation, which recomputed
`house_price_index` directly from each month's (necessarily sparse) sales,
still blew up: an outlier month's raw ratio fed straight into the reversion
term at 50% weight, effectively double-counting that same month's noise
instead of damping it (stage 1 had already absorbed ~11% of it; stage 2 then
pulled toward a reversion target contaminated by the same observation).
`house_price_index` is therefore also EMA'd here, at the same
`smoothing_factor` -- a structural adaptation for the single-tract case, not
a value taken from the reference. `tests/test_tract_price_formation.py`
pins this specifically (`test_house_price_index_is_smoothed_not_overwritten`).

### Calibration data wired in alongside the mechanism

The reference price level itself was also a hand-typed round default
(`250_000.0` / `1400.0`) rather than pulled from the calibration data
already present in the repo. `atlanta_zillow_zhvi.csv` /
`atlanta_zillow_zori.csv` were already committed but unused (their loader in
`external_data.py` is only wired to the *optional*, off-by-default exogenous
appreciation path, not to the model's initial/reference price level). Their
calendar-2019 means are:

| | computed from CSV | paper's Section III figure |
|---|---|---|
| ZHVI (price) | $248,716.72 | "$248,717" |
| ZORI (rent) | $1,307.55 | "$1,308" |

This confirms the CSVs are the paper's actual cited calibration source, just
not connected to the code. `reference_price_per_quality` and
`reference_rent_per_quality` (`config/baseline_params.yaml`
`tract_calibration`) now use these values, and the initial rental stock
(`generate_placeholder_rental_stock`, previously hard-coded to `1400.0`
regardless of the tract's own rent level) now reads the tract's calibrated
rent instead of its own separate default.

### Result

| households | price/income after 15yr, before fix | after fix |
|---|---|---|
| 300 | ~29x | ~9.5x |
| 600 | -- | ~8.1x |

Both `annual_appreciation_g` runs no longer diverge; the price level converges
rather than exploding. Re-validating across the population range:

| N | homeownership | rental vacancy | appreciation | months of inventory | targets met |
|---|---|---|---|---|---|
| 300 | 0.393 OUT (low) | 0.127 OK | 0.050 OUT (high) | 3.40 OK | 5/7 |
| 450 | 0.517 OK | 0.033 OUT (low) | 0.036 OUT (high) | 5.60 OK | 5/7 |
| 600 | 0.521 OK | 0.038 OUT (low) | 0.032 OUT (high) | 5.73 OK | 5/7 |
| 900 | 0.561 OK | 0.036 OUT (low) | 0.029 OUT (high) | 7.42 OK | 5/7 |

Homeownership is now stable and within band from N=450 up (previously it swung
with population size in a way that tracked the runaway directly). Rental
vacancy's shortfall is the pre-existing, separately-diagnosed limitation from
Section 1 (still trending down as N grows, unrelated to price formation).

**A new, distinct finding**: `annual_appreciation_g` is now *systematically*
above its target band across every population size, converging toward roughly
**+3%/year real** as N grows, rather than exploding. This is no longer a
thin-market artifact -- it is population-invariant, which means it is a real
property of the model's demand-side dynamics, not sampling noise. Construction
was checked and ruled out as the cause: the housing-stock-to-household ratio
holds steady at its 1.098 target throughout (1.093-1.108 across a 180-month
window at N=600), so supply is not lagging population growth. The remaining
candidate is EQ3's own reflexive term (`desired_expenditure` scales with
`1/(1-beta*g)`, `beta=0.3`): raising `g` raises demand, which raises the next
period's realized appreciation, which raises `g` again -- smoothing the
*measurement* of price does not remove this *behavioral* feedback, it only
stops the measurement itself from being the noise source. Whether the
reference model's `MARKET_AVERAGE_PRICE_DECAY = 0.5` is strong enough to fully
cancel this loop depends on how EQ3's demand elasticity compares to the
reference model's own (differently parameterized) equivalent -- they were not
calibrated together. Closing this gap further means either strengthening the
reversion (a port decision) or revisiting EQ3's `beta` (a recalibration of
this model's own documented equation, not a port) -- deliberately left open
rather than decided unilaterally.

---

## 9. Data sources: what's used, what exists but isn't wired, what's missing

The reference Java model draws on several UK-specific empirical inputs this
port either has an equivalent for, has-but-doesn't-use, or has no analogue
for at all. Organized by which of those three:

**Already has a working equivalent:**
- Down payment distributions: reference fits `DOWNPAYMENT_FTB_SCALE/SHAPE`
  and `DOWNPAYMENT_OO_SCALE/SHAPE` (lognormal, by income percentile) from PSD
  survey data; this model fits the equivalent from HMDA
  (`scripts/fit_downpayment_lognormal.py` -> `downpayment_lognormal_params.csv`).
- Price/rent level calibration: reference uses UK HPI/ONS reference prices per
  quality band; this model now uses Zillow ZHVI/ZORI 2019 means (Section 8
  above) as the single-tract equivalent.
- Appreciation validation target: reference validates against ONS house price
  data; this model validates against Case-Shiller ATXRSA
  (`atlanta_case_shiller.csv`, already wired in `validate_against_paper.py`).

**Exists in the repo but is not wired in -- worth connecting:**
- `atlanta_zillow_zhvi.csv` / `atlanta_zillow_zori.csv` full monthly series
  (2015-2026, not just the 2019 mean used above) could drive
  `external_g_series` / `external_rent_growth_series` for historical
  validation runs the way the reference model can be run against actual
  historical HPI, rather than only for the always-on baseline calibration
  level. Currently `external_g_series = None` unconditionally in `model.py`.
- `atlanta_hmda_2019.csv` has loan-level records beyond what
  `fit_downpayment_lognormal.py` currently extracts (e.g., applicant
  income, action taken, denial reason) that could support an FTB
  qualification-rate check analogous to the reference's LTI-by-age
  regression (below).

**No analogue exists -- would need new data to port faithfully:**
- **Empirical age-distribution histogram.** The reference model's
  `Demographics.java` steers births/deaths *every month* toward a real
  monthly-binned age histogram from ONS data, continuously self-correcting
  any drift. This port instead derives a one-time *stationary* age
  distribution analytically from the mortality hazard (Section 4) and only
  gets the *initial* condition right -- it does not have a real Atlanta
  age-distribution target to steer toward continuously. **ACS/PUMS microdata
  for the seven Atlanta counties (age x household-formation) would let this
  be ported properly** -- it's a more robust fix than the analytic
  steady-state approach already in place, since it self-corrects over time
  rather than only at initialization.
- **Empirical initial sale/rent markup distributions.** The reference draws
  `saleMarkUpPdf` / `rentMarkUpPdf` from real listing-vs-eventual-sale-price
  data (a Zoopla/HPI-derived empirical distribution, not a parametric guess).
  EQ7's `asking_price` here uses a parametric lognormal-noise term instead
  (`asking_price_eq7` config) with no empirical distribution behind it.
  **A comparable Atlanta listing-vs-sale-price dataset** (Zillow's
  transaction-level data, or a county assessor/MLS extract if accessible)
  would let this be fit the same way, rather than assumed.
- **LTI-by-age regression coefficients.** `decideLTV()` in the reference is a
  fitted linear regression of target loan-to-value on income and age,
  separately for FTBs and home-movers, calibrated against UK mortgage data.
  This model's mortgage terms (`config/mortgage_terms.yaml`) use flat
  regulatory maxima (FHA/conventional LTV/DTI ceilings) rather than an
  age-conditioned empirical curve. HMDA carries applicant age brackets in
  some vintages; **if the pulled HMDA extract includes age, a comparable
  regression could be fit** the same way the down payment distributions
  already are.
- **Income-percentile-conditioned BTL/investor propensity.** The reference's
  `BTLProbability` is a binned empirical curve (probability of carrying the
  "BTL gene" by income percentile). This model instead sets institutional
  and small-landlord *counts* directly from fixed population fractions
  (`config/baseline_params.yaml` `simulation`), with no income-percentile
  dependence on which households become investors. No US household-level
  survey-based analogue was identified for this session; the Survey of
  Consumer Finances might carry a usable investment-property-ownership rate
  by income percentile, but this was not checked.

None of the "no analogue" items block the model from running or from the
policy comparisons already delivered -- they would each incrementally improve
fidelity to the reference's *empirical* grounding, not fix a defect the way
the price-formation port did.

## 10. Closing three data-wiring gaps found while surveying data sources

Re-reading the codebase against the data-sources question above (Section 9)
surfaced three places where real, already-downloaded Atlanta data existed but
was not actually reaching the model or the calibration it was meant to
inform -- distinct from the "no analogue exists" gaps above, these had no
missing data, only missing plumbing.

- **`use_external_appreciation_data` was dead config.** The flag existed in
  `config/baseline_params.yaml`'s `simulation` block and the loader functions
  (`load_g_series`, `load_monthly_growth_series`) existed in
  `external_data.py`, but `model.py` hardcoded `external_g_series` and
  `external_rent_growth_series` to `None` unconditionally -- the flag was
  read nowhere. The full 2015-2026 `atlanta_zillow_zhvi.csv` /
  `atlanta_zillow_zori.csv` history (already in the repo) was therefore never
  used beyond the single 2019 mean wired into `tract_calibration`. Fixed:
  `model.py` now loads both CSVs when the flag is true, with a path override
  via `tract_calibration.zhvi_csv_path`/`zori_csv_path` and a `warnings.warn`
  fallback to the endogenous EQ4 computation if the files are missing.
  Verified end to end (loads real values, survives several `step()` calls,
  and falls back cleanly with a warning when a path is wrong) --see
  `tests/test_external_appreciation_wiring.py`. Left `false` by default:
  switching a study's price/rent trajectory from endogenous to
  historical-real is a modeling decision about what the counterfactual
  policies are being evaluated against, not a plumbing fix, so it wasn't
  flipped without sign-off.
- **`pull_zillow_data.py` could never regenerate `atlanta_zillow_zhvi.csv`.**
  The script's docstring, its `ZHVI_URL` constant, and its CLI all claimed to
  pull ZHVI (home values), but the URL actually pointed at Zillow's ZORI
  (rent index) endpoint, and its only output was `atlanta_zillow_zori.csv`.
  The already-checked-in `atlanta_zillow_zhvi.csv` must have been produced
  some other way; running the script as committed could not have produced
  it. Fixed by pulling both series explicitly from their own correctly
  labeled URLs into their own output files. Not re-run here (see the network
  note below) -- the existing checked-in CSVs are unchanged and still what
  the model uses.
- **`pull_hmda_data.py` never requested `applicant_age`.** HMDA's public
  loan-level API has carried a bucketed borrower age field since the 2018
  reporting rule; Section 9 flagged this as the single highest-value,
  lowest-effort gap (age is the input the reference model's own `AgeDist.py`
  calibration script is built around, and the equivalent field here was
  simply never asked for). Added it to `FIELDS`, with a comment pointing at
  where to confirm the exact field name against the live schema once this
  can be pulled with real network access.

While in the down-payment calibration code, also found and exercised
`equations/mortgage.estimate_floor_share_and_fit` -- a fit-a-lognormal-to-HMDA
helper that was written (correct docstring, correct math) but never called by
anything. Wrote `scripts/calibrate_downpayment_eq17.py` to actually run it
against `atlanta_hmda_2019.csv`, using `loan_type` (FHA vs. conventional) as a
proxy for first-time vs. repeat buyer, since HMDA carries no direct
first-time-buyer flag. Result, compared against the current
`downpayment_eq17` config:

| | `p_floor` | `lognorm_m` | `lognorm_s` |
|---|---|---|---|
| Config: first_time_buyer | 0.55 | -1.6 | 0.5 |
| **Data (FHA, n=15,523)** | **0.917** | **-2.184** | **0.528** |
| Config: repeat_buyer | 0.55 | -0.8 | 0.75 |
| **Data (conventional, n=42,635)** | **0.372** | **-1.701** | **0.510** |

The gaps are large enough to matter (`p_floor` in particular: real FHA
borrowers cluster at the minimum far more than the config assumes, real
conventional borrowers far less). This was **not** applied to
`baseline_params.yaml` -- `loan_type` is a proxy, not a ground-truth
first-time-buyer flag (a repeat buyer can take out an FHA loan; a first-time
buyer can use a low-down-payment conventional program), and this parameter
pair is the one the entire policy comparison turns on. Adopting it is a
methodology call, not a plumbing fix, and is left for review alongside the
other Section 9 findings.

**On the "check if network egress works" question**: it does not, from this
sandbox. `curl`/`WebFetch` to every external host tried (`ffiec.cfpb.gov`,
`federalreserve.gov`, `api.stlouisfed.org`, `census.gov`, even
`google.com`/`example.com`) returned a `403` at the egress proxy
(`gateway answered 403 to CONNECT`, confirmed via
`$HTTPS_PROXY/__agentproxy/status`); only GitHub and the package registries
already allowlisted for this environment are reachable. So the HMDA/Zillow
pull-script fixes above are code-only -- they could not be run to produce
fresh data from here, and the exact HMDA age field name could not be
verified against the live CFPB schema. Both need to be run from a machine
with normal internet access (e.g. the user's own) before the new field or
the corrected ZHVI pull actually reach a CSV on disk.

### 10a. What adopting the down-payment fit above would actually do

The first pass of the fit above used one blanket `floor_band=0.05` for both
buyer types, which is wrong -- the model's own floor is 3.5% for first-time
buyers (FHA) and 20% for repeat buyers (conventional), not 5% for both.
Re-fit with each buyer type's own floor
(`scripts/calibrate_downpayment_eq17.py`, `floor_band=d_minimum_pct`):

| | `p_floor` | `lognorm_m` | `lognorm_s` |
|---|---|---|---|
| Data (FHA, floor=3.5%) | 0.851 | -2.627 | 0.641 |
| Data (conventional, floor=20%) | 0.812 | -1.150 | 0.297 |

Rather than reason about the effect by hand, `scripts/compare_downpayment_calibration.py`
runs the same paired-CRN design as `run_all_policies.py` -- one shared
120-month spin-up per seed under the *current* config, forked into a
baseline arm and an arm that switches to the values above for the
120-month measurement window -- at 60 seeds x 600 households
(`results/downpayment_calibration_raw.csv` / `_summary.json`). Result:

- **Homeownership rate and both purchase-share metrics: no detectable
  effect.** FTB purchase share moves +0.85pp (p=0.067) but does not survive
  Holm correction across the 7 outcomes tested; homeownership rate and
  repeat-buyer purchase share are indistinguishable from zero (MDE ~0.4-1.6pp
  at this seed count).
- **Leverage moves, precisely.** Mean owner-occupier LTV: +0.87pp
  (95% CI [+0.66pp, +1.07pp], p_holm<0.0001). Mean owner-occupier LTI: +1.07pp
  (95% CI [+0.48pp, +1.66pp], p_holm=0.0035). Buyers who purchase carry
  measurably more debt relative to price and income -- exactly the mechanism
  predicted (smaller down payment -> larger loan for the same price) -- but it
  does not translate into more purchases at this scale.
- **Institutional share of rentals (negative control): no effect**, as
  expected -- `downpayment_eq17` only touches owner-occupier down payments,
  never investor mechanics (`downpayment_eq18`).

So adopting these values would not change the headline finding the paper's
policies are evaluated on (who ends up buying, and at what homeownership
rate); it would raise the modeled leverage of owner-occupier purchases by
about one percentage point on both LTV and LTI, with no effect on the
institutional-investor side of the model. Left un-adopted in
`baseline_params.yaml` pending review of the `loan_type`-as-buyer-type proxy
caveat above.

### 10b. Replacing the proxy with a genuine first-time-buyer flag

HMDA has no direct first-time-buyer field, so 10a's fit used loan type
(FHA vs. conventional) as a stand-in. Fannie Mae's Single-Family Loan
Performance Data does carry a genuine field for this -- `First Time Home
Buyer Indicator` (Y/N), reported by the lender at origination, not
inferred. The user pulled a real extract (2019 Q1, filtered to the Atlanta
MSA, CBSA 12060) from Fannie Mae's Data Dynamics portal and it was fit the
same way (`scripts/calibrate_downpayment_eq17_fnma.py`), split on the real
flag instead of the loan-type proxy, one row per loan (the file is a
monthly performance panel; origination terms don't change month to month,
so only the first reporting month per loan is kept), restricted to
purchase-money, owner-occupied loans:

| | n | `p_floor` | `lognorm_m` | `lognorm_s` |
|---|---|---|---|---|
| Config: first_time_buyer | -- | 0.55 | -1.6 | 0.5 |
| Proxy fit (FHA, 10a) | 15,523 | 0.851 | -2.627 | 0.641 |
| **Ground truth (real FTB flag)** | **1,718** | **0.335** | **-2.226** | **0.702** |
| Config: repeat_buyer | -- | 0.55 | -0.8 | 0.75 |
| Proxy fit (conventional, 10a) | 42,635 | 0.812 | -1.150 | 0.297 |
| **Ground truth (real FTB flag)** | **1,777** | **0.738** | **-0.992 ** | **0.362** |

The genuine flag doesn't just refine the proxy's first-time-buyer number --
it reverses it. The proxy said 85% of FHA borrowers sit at the down-payment
floor; the real flag says only 33.5% of *actual first-time buyers* do,
*below* the model's own 55% assumption, not above it. FHA-borrower and
first-time-buyer are overlapping but distinct populations (plenty of
first-time buyers use low-down-payment conventional programs instead of
FHA; not every FHA borrower is buying a first home), and the proxy was
measuring loan-program choice, not buyer-type behavior. The repeat-buyer
number moved in the same direction as the proxy suggested (more
floor-clustering than the config assumes), just to a different value.

Re-running `scripts/compare_downpayment_calibration.py` with the ground-truth
values (same 60-seed x 600-household paired design as 10a) gives an even
smaller, more thoroughly null result than the proxy-based run did:

- Homeownership rate: +0.33pp, p=0.020 unadjusted but p_holm=0.14 -- doesn't
  survive Holm correction across the 7 outcomes tested, though it's the
  closest of any outcome to significant.
- Every other outcome (FTB/repeat purchase share, mean LTV, mean LTI, median
  price, institutional share of rentals) is indistinguishable from zero,
  including the LTV/LTI leverage effects that *were* significant under the
  proxy-based values in 10a -- the real data moves average down payment less
  than the mismatched-proxy fit did (repeat buyer: 38.2% -> 25.2% of price,
  vs. the proxy's implied larger swing).

Net: with the actual ground-truth flag in hand, the case for adopting this
into `baseline_params.yaml` is *weaker* than it looked with the proxy, not
stronger -- both because the honest effect on headline metrics is smaller,
and because the first-time-buyer proxy turned out to have gotten the sign
of its main qualitative claim wrong. Still left un-adopted. The Atlanta
2019 Q1 extract and this result are single-quarter, single-vintage; a
sturdier calibration would pool several quarters before treating these
exact numbers as final.

### 10c. An age-conditioned LTV/LTI curve (grounding the reference's decideLTV)

`applicant_age` (bucketed: `<25`, `25-34`, ..., `>74`, or `8888` for
unknown) was confirmed present and populated in the live HMDA API response
once actually requested -- the field itself checked out, unlike the
`combined_loan_to_value_ratio` field also in `FIELDS`, which turned out not
to exist in the current API at all and has now been dropped (this only
surfaced because of the new missing-field warning in `pull_year()`; it had
presumably been silently absent from every past pull).

This grounds an input the model previously had zero analogue for: the
reference's `decideLTV()` is a fitted regression of target LTV on income
and age, separately for FTBs and home-movers; this model's mortgage terms
(`config/mortgage_terms.yaml`) use flat regulatory LTV/DTI ceilings with no
age term whatsoever. HMDA carries no first-time-vs-repeat-buyer flag (see
10a/10b), so `scripts/calibrate_ltv_by_age.py` fits one pooled regression
across all owner-occupied purchase loans instead of splitting by buyer
type, using each age bucket's midpoint as a continuous predictor:

| Age | n | mean LTV | mean LTI | mean income ($k) |
|---|---|---|---|---|
| <25 | 2,245 | 92.6 | 3.31 | 77.9 |
| 25-34 | 19,934 | 91.4 | 3.09 | 98.1 |
| 35-44 | 18,374 | 89.8 | 3.03 | 122.2 |
| 45-54 | 12,759 | 88.5 | 2.90 | 131.8 |
| 55-64 | 6,625 | 86.4 | 2.84 | 126.3 |
| 65-74 | 2,394 | 83.6 | 3.11 | 99.3 |
| >74 | 543 | 83.6 | 3.42 | 90.0 |

Mean LTV declines monotonically and substantially with age (92.6% -> 83.6%,
a real ~9-point spread) -- younger buyers lever up more, controlling for
nothing else. The regression confirms this is a genuine age effect, not
just an income artifact: `LTV ~ log(income) + age` gives
`age_coef = -0.148` (R2=0.115), i.e. roughly 1.5 LTV points lower per
decade of age, holding income fixed. LTI shows no comparable age
trend once income is controlled for (`age_coef = -0.0007` on
`LTI ~ log(income) + age`, R2=0.268) -- the U-shape visible in the raw
means table is explained by income varying by age, not by age itself.

This is a real, verified finding, but it is not yet wired into the model,
deliberately: doing so means changing `max_loan_owner_occupier` (EQ14) or
`chi_max_ltv` from a flat constant into an age-conditioned function, which
is a structural change to core borrowing logic, not a config edit like
10a/10b's down-payment work.

Before designing that wiring, it's worth knowing what the reference itself
actually does with the equivalent mechanism -- and the answer is: nothing.
`decideLTV()` in the reference's `HouseholdBehaviour.java` is exactly this
regression (age + income -> target LTV, fitted separately for FTBs and
home-movers, real UK coefficients baked in), but its only call site is a
commented-out line in `Household.java`:

```java
desiredPurchasePrice = behaviour.updateDesiredPurchasePrice(annualGrossEmploymentIncome);
// desiredPurchasePrice = behaviour.getAltDesiredPurchasePrice(annualGrossEmploymentIncome, behaviour.decideLTV(this));
```

The live path, `updateDesiredPurchasePrice()`, is income-only (a power law
in income times lognormal noise) -- no age term. The commented line sits
under the reference authors' own TODO: *"Decision needed between this and
previous specification."* They wrote and calibrated the age-conditioned
version and then shipped the paper's results without it. (The same pattern
shows up in `decideDownPayment()`: its first-time-buyer branch is just
`me.getBankBalance()`, with an income-percentile lognormal draw commented
out as "old ... implementation, kept for now as legacy/alternative" --
only the repeat-buyer branch uses a calibrated distribution in the live
code.)

That reframes this finding: it is not "catching up" to something the
reference validates its results with, since the reference's own headline
numbers never depended on age either. It's a genuine, real pattern in
Atlanta data that the reference happened to anticipate in code it wrote
and then didn't use. Decision: leave it as a documented empirical result
(this section) rather than build any of the wiring options above into the
live model -- there's no reference precedent forcing the choice, and it
avoids creating a second LTV-determining mechanism alongside EQ17's
down-payment draw.

## 11. Grounding from the Survey of Consumer Finances (SCF)

The reference model's single richest calibration source is the UK Wealth
and Assets Survey (WAS), which grounds four separate parameters -- age
distribution, income-given-age, wealth-given-income, and buy-to-let
propensity by income percentile -- from one panel, because one survey asks
all of those questions of the same households. The US analogue is the
Federal Reserve's Survey of Consumer Finances (SCF); `scripts/calibrate_scf.py`
fits the same four things from the 2022 wave's Summary Extract Public Data
(`SCFP2022.csv`, 4,595 households, implicate 1 of 5 used -- a standard
simplification for building a population-level distribution rather than
full multiple-imputation variance estimation).

**Age distribution.** The model currently has no empirical age target at
all -- `demographics.stationary_age_distribution` is a purely analytic
steady state derived from the mortality hazard, with no survey behind it
(see Section 4/6). Compared against the real SCF-weighted distribution:

| Age | Analytic (model) | SCF (real) |
|---|---|---|
| 18-25 | 0.9% | 4.5% |
| 25-35 | 12.2% | 15.5% |
| 35-45 | 18.6% | 17.0% |
| 45-55 | 18.5% | 16.4% |
| 55-65 | 18.2% | 18.5% |
| 65-75 | 16.9% | 16.1% |
| 75+ | 14.7% | 12.0% |

Middle and older ages track reasonably well; the analytic approximation
substantially underrepresents the youngest bracket (0.9% vs. a real 4.5%,
a 5x gap) and modestly overrepresents 75+.

**Income given age.** The model draws every household's income -- initial
cohort and every later entrant alike -- from one age-independent
`lognormal(8.6, 0.65)` (median ~$65k/year). Real median household income
by age is a clear hump: $32k (<25) -> $75k (25-34) -> peaks around $85-90k
(35-64) -> declines to $49k (75+). The model's single flat draw compresses
a roughly 3x lifecycle swing into one number.

**EQ1 (wealth-given-income).** `desired_bank_balance`'s functional form is
`ln(w) = alpha + beta*ln(income) + epsilon`. Fit against SCF `FIN` (total
financial assets -- readily liquidated wealth, unlike `NETWORTH` which
includes illiquid home/business equity a household wouldn't draw on for a
down payment):

| | Config | SCF fit |
|---|---|---|
| renter: alpha, beta, eps_std | 0.693, 1.0, 0.3 | -10.69, 1.77, 2.48 (n=1438, R2=0.34) |
| owner (FTB+repeat combined): alpha, beta, eps_std | -- | -7.70, 1.65, 2.00 (n=2082, R2=0.54) |
| small_landlord: alpha, beta, eps_std | 3.178, 1.0, 0.3 | -1.49, 1.19, 1.65 (n=1074, R2=0.65) |

`alpha` isn't directly comparable -- config's income is monthly, SCF's is
annual, and log-linear intercepts shift under that rescaling (`beta` and
`epsilon_std` don't: rescaling income by a constant only shifts alpha).
Those two are the meaningful, unit-independent findings, and both say the
same thing: **config assumes wealth is a tight, exactly-proportional
function of income (beta=1.0 for every group, eps_std=0.3) that real data
doesn't support.** Real beta is 1.19-1.77 (wealth grows *faster* than
proportionally with income -- consistent with the standard finding that
wealth-income elasticity exceeds 1), and real eps_std is 1.65-2.48, roughly
6-8x the config's assumed dispersion -- income explains much less of
wealth than the model currently assumes. SCF has no first-time-vs-repeat
flag (same limitation as HMDA in 10a), so owner-occupiers are one combined
fit here, not split like the config's two separate blocks.

**Investor propensity by income (the strongest candidate for actually
wiring in).** `HORESRE` (owns other residential real estate) is the direct
US analogue of WAS's rental-income question, and by construction can only
capture "mom and pop" landlords -- SCF surveys households, not LLCs/REITs,
consistent with the earlier finding that institutional-investor share has
no free survey analogue. The income gradient is stark and monotonic:

| Income decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| P(owns other residential RE) | 3.0% | 3.8% | 7.5% | 8.1% | 13.8% | 13.1% | 12.9% | 32.1% | 43.0% | 59.9% |

Unlike `decideLTV` (Section 10c), this one has real reference precedent:
`BTLProbability.getBinAt(incomePercentile)` runs in every household's
constructor in the reference's live code (`HouseholdBehaviour.java`), not
commented out -- the reference's actual reported results depend on this
mechanism. Atlanta's `small_landlord_fraction`/`institutional_investor_fraction`
(`config/baseline_params.yaml` `simulation`) are fixed population fractions
with zero income dependence -- currently *any* household is equally likely
to become a landlord regardless of income, which this data says is wrong
by roughly 20x (3% at the bottom decile vs. 60% at the top).

Of the four findings, only investor propensity had both real reference
precedent and no unit ambiguity to resolve first (it's a probability, not
a rescaled quantity) -- age distribution and income-given-age would each
be a genuine structural upgrade but touch initialization code paths beyond
this session's scope to modify carefully, and the EQ1 dispersion finding
needs its unit mismatch reconciled before `alpha` is usable. So this one
was actually wired in; the other three remain findings only.

## 12. Wiring investor propensity into small-landlord selection

`SmallLandlord` agents are created in two places -- the initial population
(`AtlantaHousingModel.__init__`) and a monthly top-up
(`construction.run_investor_replenishment`, which tops both small-landlord
and institutional-investor counts back up to their target share of
households every month, an existing mechanism this session hadn't seen
until now). Both previously drew each new landlord's income from its own
disconnected lognormal (`small_landlord_lognormal_mean/sigma`, 9.8/0.5,
"landlords skew higher-income than renters" -- unexplained, hand-picked).

New `equations/investor_propensity.py`: landlord incomes are now drawn from
the *same* household income distribution as everyone else, then *selected*
without replacement, weighted by the real SCF income-decile propensity
curve from Section 11 (`sample_landlord_incomes`). The realistic income
skew now falls out of who gets selected instead of being asserted as a
separate distribution. Both call sites (`model.py`'s initial population,
`construction.py`'s replenishment) now share this one function instead of
duplicating the draw. `small_landlord_lognormal_mean/sigma` is removed from
`config/baseline_params.yaml` as dead config.

Institutional investors are untouched -- SCF surveys households, not
LLCs/funds, so it has nothing to say about institutional counts, which
keep their own existing calibration (Section 9's tier-5 finding: no free
survey analogue exists for institutional-ownership data).

**Verification, not just "it runs":**
- 8 new unit tests (`tests/test_investor_propensity.py`) pin the selection
  weighting (low/high income map to the correct decile, a flat curve gives
  uniform weight) and the sampling mechanics (exact count returned, handles
  a pool smaller than requested, a real propensity curve measurably skews
  the selected sample's mean income above flat/uniform selection).
- A smoke run (N=600, seed=1) confirms the new mechanism actually changes
  the outcome in the expected direction without breaking anything: selected
  landlords' median income is $127,749/yr vs. the general population's
  $65,568/yr (landlords still skew higher-income, as expected) but well
  below the old hand-picked distribution's $216,410/yr median -- less
  extreme because it's now a realistic mixture across deciles instead of
  an assumption. Replenishment was confirmed still working (35 -> 37
  landlords over 24 months as the household base grew).
- Ran `validate_against_paper.py` before and after (`git stash`/`pop`) to
  separate this change's effect from pre-existing issues: 5/7 targets met
  both times, identical two failures both times
  (`homeownership_rate`, `annual_appreciation_g` -- both already-documented
  open issues, unrelated to this change, from the price-formation port's
  residual appreciation drift and the never-yet-rerun post-fix policy
  study). `institutional_share_of_rentals` stayed comfortably in range
  (0.254 -> 0.265), confirming the already-validated investor-share
  calibration wasn't disturbed by changing *who* gets selected as a
  landlord.

## 13. The appreciation target was failing on a biased estimator, not on model behaviour

`annual_appreciation_g` sat outside its validation band (+3.9%/yr against a
target of [-0.01, 0.02]) from the price-formation port onward, and was
carried as an open "residual appreciation drift" issue. Two candidate
fixes were assumed: strengthen the price-reversion constant
(`market_smoothing.market_average_price_decay`) or weaken EQ3's demand-side
appreciation feedback (`expenditure_eq3.beta`). Both were swept before
being adopted, and neither worked:

| variant | appreciation | homeownership |
|---|---|---|
| baseline (decay 0.5, beta 0.3) | 0.039 | 0.391 |
| decay 0.3 | 0.040 | 0.399 |
| decay 0.2 | 0.033 | 0.390 |
| beta 0.15 | **0.049** | 0.432 |
| beta 0.10 | **0.045** | 0.422 |
| decay 0.3 + beta 0.15 | 0.042 | 0.386 |

Stronger reversion barely moved it; weakening EQ3's feedback made it
*worse*, the opposite of the predicted direction. That ruled out both
stated hypotheses and prompted instrumenting an actual run instead.

**What the instrumented run showed.** Over 180 months at 300 households:
housing stock never changed (construction correctly idle -- houses per
household stayed above its 1.098 build trigger the whole time), mean
household income was flat (7,530 -> 7,527, so the earlier income-drift fix
is holding), and the price level oscillated between ~373k and ~516k with
little net trend -- while the metric still reported ~4%/yr. Those two facts
are inconsistent, which located the problem in the measurement.

**The bias, quantified.** Across 8 seeds, the mean of monthly EQ4 g came to
**+3.75%/yr against an actual price CAGR of +0.99%/yr** -- overstating by
**+2.76%/yr, positive in every single seed** (range +2.04 to +4.17pp). In
two seeds the price level *fell* over the window (-0.54%/yr, -0.72%/yr)
while the metric still read ~+3%/yr.

The cause is ratio asymmetry. EQ4's g is a growth *ratio*, and its monthly
mean is taken over a volatile mean-reverting series: a rise from A to B
reads as +(B-A)/A, while the matching fall back to A reads as -(B-A)/B,
smaller in magnitude. Averaged over up-and-down cycles that yields a
positive number at zero net trend, and the bias scales with volatility --
sd(g) is ~0.08-0.09 in this thin market, exactly the regime where it bites.

**Fix.** Added `housing_abm.metrics.price_trend_cagr`: the annualised
log-OLS slope of the tract's smoothed price series (log-OLS rather than
endpoint-to-endpoint CAGR, so it uses the whole window instead of two noisy
endpoints), surfaced as `price_cagr`. `validate_against_paper.py` now
checks the trend band against that. **EQ4 itself is unchanged** -- its g is
the behavioural signal households actually perceive and act on, which is
its real job, and it is still reported. What changed is only that a trend
target is now validated against a trend estimator. 7 tests pin the
estimator (recovers known growth and decline rates, returns None on too
short a window, and does not manufacture a trend from mean-reverting
noise).

**Result: 6/7 validation targets met**, with `price_cagr` at 0.015
[0.002, 0.027] against [-0.01, 0.02]. `homeownership_rate` (0.402 against
[0.48, 0.60]) is now the only remaining gap.
