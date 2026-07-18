# housing-abm
Agent based model of the Atlanta, Georgia housing market, adapted from Baptista et al. (2016), built with Mesa. The model tracks households moving between renting, first-time homeownership, and repeat-buying/small landlords/ institutional investors, month by month.

## Reproducibility

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
python -m venv myenv
source myenv/bin/activate 
pip install -e .
pip install matplotlib
```

Now, to reproduce the baseline, run
```bash
python scripts/run_baseline.py
```
This creates a model with `N_HOUSEHOLDS = 100` households and a fixed `SEED = 42`, steps it forward over `N_MONTHS = 36` months, and writes `baseline_run.png` (renting/vacancy rate/owning over time) to the project root. Edit constants at the top of the script to change household count, seed, or run length in months. This may take a minute, depending on the number of households and time in months, so take a stretch break if needed. 

