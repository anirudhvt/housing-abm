# housing-abm

## Project Overview
Agent based model of the Atlanta, Georgia housing market, built with Mesa. Institutional investors (large, corporate organizations that pool capital to purchase primarily single-family homes) have caused increased competition in the US homeownership market, raising ownership prices nationally. Because of this impact, policies have been proposed to curb institutional investors' stake in the real estate market, but long term effects of these policies remain unclear. The model evaluates six institutional investor restriction policies: waiting periods, ownership caps, purchase taxes, vacancy taxes, and progressive portfolio taxes, and tracks their effects on first-time homeownership rate, rental vacancy rate, and annual price appreciation across Monte Carlo simulations. 

The accompanying paper is at paper/main.tex

## Setup

First, clone the repo with
```bash
git clone https://github.com/anirudhvt/housing-abm.git
```

and enter the folder you just created with
```bash
cd housing-abm
```

You must create a virtual environment with the necessary packages:
```bash
python3 -m venv myenv
source myenv/bin/activate 
pip install -e .
```

## Reproducing Results

Follow the following steps in order. Steps 1-2 need API keys to access data, but if you want to skip data pulling and use the existing CSVs committed in the repo, start from Step 3.

### Step 1 - Pull raw data

**Census API key**: Sign up to get a free API key at https://api.census.gov/data/key_signup.html and set it as an environment variable with 
 
```bash
export CENSUS_API_KEY="your_key_here"
```

Then pull all data sources: 
```bash
python scripts/pull_acs_data.py          # ACS B25024, B25003, B25004, B19001
python scripts/pull_hmda_data.py         # HMDA 2019 Georgia purchase mortgages
python scripts/pull_zillow_data.py       # ZHVI and ZORI for Atlanta metro
python scripts/pull_case_shiller_data.py # Case-Shiller ATXRSA via FRED
```

Each script writes CSVs to the project root.

### Step 2 - Fit downpayment distribution
 
```bash
python scripts/fit_downpayment_lognormal.py
```

This reads `atlanta_hmda_2019.csv` and writes `downpayment_lognormal_params.csv`, which the model uses to initialize FTB and investor down payment distributions. 

### Step 3 - Check the experiment design

```bash
python3 scripts/diagnose_design.py --households 600 --spinup 120 --months 240 --seeds 12
```

Reports three things that determine how tight the final intervals can be:

- **Stationarity.** Regresses each monthly outcome on the month index. A
  non-zero slope means seeds are being measured at different points on their
  own trajectories, and that spread enters the cross-seed standard deviation as
  noise unrelated to any policy.
- **Effective sample size.** How many *independent* months a window average is
  worth, given autocorrelation. Rental vacancy and the purchase shares are
  nearly white month to month, so averaging the window is worth 6-13x on their
  standard errors; the slow-moving stocks gain much less.
- **Measurement window.** Longer windows average away more within-run noise but
  let the two arms of a paired comparison drift apart, so the standard error of
  the effect is minimised at an intermediate length rather than the longest one.

### Step 4 - Validate the baseline

```bash
python3 scripts/validate_against_paper.py --households 600 --spinup 120 --months 120 --seeds 12
```

Runs under the *same* spin-up and window as the policy experiments and reports
an interval across seeds, so the validation table describes the model state the
policy effects are actually measured in.

### Step 5 - Run the policy comparison

All six scenarios against one shared baseline:

```bash
python3 scripts/run_all_policies.py \
    --households 600 --spinup 120 --months 120 --seeds 60 \
    --outdir results
```

Or one policy at a time, with more diagnostic output:

```bash
python3 scripts/run_policy_comparison.py \
    --policy config/policy_scenarios/waiting_period.yaml \
    --households 600 --spinup 120 --months 120 --seeds 60 \
    --output results/waiting_period.csv
```

### Step 6 - Sensitivity analysis

```bash
python3 scripts/run_sensitivity.py --param all --seeds 20
```

Sweeps `beta_institutional`, `beta_small_landlord`, `alpha_consumption` and
`delta_institutional`, re-running the full paired comparison at each value, and
asks whether the *ranking* of policies survives parameter uncertainty.

`beta_institutional` matters most: it is the shape parameter of the EQ 10/13
logistic, so how much any financial penalty can move investor behaviour depends
on where on that logistic the market sits.

### Step 7 - Figures and paper

```bash
mkdir -p figures
python3 scripts/generate_figures.py --results results/ --output figures/
python3 scripts/generate_heatmap.py \
    --input results/sensitivity_beta_institutional.csv \
    --output figures/figure6_sensitivity_heatmap.png

cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

If `pdflatex` is missing: `sudo apt install texlive-latex-base`.

---

## Experiment design

The policy effects are small relative to the model's stochastic variation, so
the design does most of the work in making them identifiable. Three choices
matter, and `docs/methodology.md` explains each in full.

**Shared spin-up, then fork.** Each seed is spun up once with no policy; the
resulting model is deep-copied into a baseline arm and a policy arm. Both begin
from a bit-identical microstate -- same agents, balances, houses and prices --
so their difference isolates the intervention. Running two independent models
per seed does *not* achieve this: the policy changes agent decisions in the
first month, every later draw shifts, and the two arms end up as unrelated
realisations that merely share a seed.

**Window-averaged outcomes.** Outcomes are averaged over the measurement window
rather than read off the final month. A single month's homeownership rate over
a few hundred agents is close to one binomial draw.

**Split RNG streams.** The exogenous interest-rate path and the demographic
process draw from dedicated substreams, so those are identical across the two
arms of a seed no matter what the policy does.

Reported alongside every effect: the arm-to-arm correlation (the diagnostic for
whether pairing is working), the achieved variance reduction, a bootstrap
interval next to the normal-theory one, Holm-adjusted p-values across the full
grid of policies x outcomes, and the minimum detectable effect -- so a null
reads as a bound on the effect rather than as its absence.
