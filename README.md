# housing-abm
Agent based model of the Atlanta, Georgia housing market, adapted from Baptista et al. (2016), built with Mesa. The model evaluates six institutional investor restriction policies: waiting periods, ownership caps, purchase taxes, vacancy taxes, and progressive portfolio taxes, and tracks their effects on first-time homeownership rate, rental vacancy rate, and annual price appreciation across Monte Carlo simulations. 

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

### Step 3 - Validate the baseline model
```bash
python scripts/validate_against_paper.py
```
 
Runs one seed and checks the model's output against plausibility targets from ACS, HMDA, and Case-Shiller. Expected output: homeownership rate ~0.52, rental vacancy ~0.20, LTV ~0.84, LTI ~2.19. Annual appreciation will typically be at the floor (-0.030) due to simplified repricing assumptions noted in the paper.

### Step 4 - Run policy comparisons

```bash
mkdir -p results
 
for policy in waiting_period ownership_cap_soft ownership_cap_hard \
              purchase_tax vacancy_tax portfolio_tax; do
    echo "Running $policy..."
    python scripts/run_policy_comparison.py \
        --policy config/policy_scenarios/${policy}.yaml \
        --seeds 40 --months 150 --spinup 600 \
        --output results/${policy}.csv
done
```
 
Each run takes approximately 20 minutes. All six together take roughly 2 hours, so take a walk or set it overnight. Results are written to `results/` as CSV files with one row per seed per situation (baseline and policy).

### Step 5 — Run sensitivity analysis
 
```bash
python scripts/run_sensitivity.py \
    --param all \
    --seeds 20 --months 150 --spinup 60 \
    --output results/sensitivity.csv
```
 
This sweeps `beta_institutional`, `beta_small_landlord`, and `alpha_consumption` across five values each and re-runs the full policy comparison at each value. Writes three CSVs to `results/`.

### Step 6 — Generate figures
 
```bash
mkdir -p figures
 
# Figures 4, 4b, 5 — policy comparison bar charts and scatter
python scripts/generate_figures.py --results results/ --output figures/
 
# Figure 6 — sensitivity heatmap
python scripts/generate_heatmap.py \
    --input results/sensitivity_beta_institutional.csv \
    --output figures/figure6_sensitivity_heatmap.png
```
 
Figures are written to `figures/` as both `.pdf` and `.png`.

### Step 7 — Compile the paper
 
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
 
---
If you get an error that command pdflatex is not found, install with 
'''bash
sudo apt install texlive-latex-base
'''
