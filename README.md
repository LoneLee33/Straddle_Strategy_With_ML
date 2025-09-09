# Straddle Strategy with Machine Learning

This repository implements machine‑learning methods to **time entries** for ATM equity/ETF **straddles**. Rather than trading every day, the pipeline generates signals from volatility, momentum, and liquidity features and enters only when forecasted edge exceeds calibrated thresholds. The workflow includes **rolling walk‑forward training**, **validation‑time threshold calibration**, and a **backtest** that converts scores into trades.

## Overview
Daily option quotes and underlying prices are assembled into **ATM straddles** within a target DTE window (typically ~30 days). Trailing features are engineered; models are trained with **month‑by‑month walk‑forward** splits; and entry thresholds are tuned on each validation month to maximize per‑trade Sharpe. The resulting rules are then applied to the next held‑out month to generate out‑of‑sample trades and P&L.

Two notebooks demonstrate the pipeline end‑to‑end:
- **`Straddle_With_ML.ipynb`** — regression pipeline with walk‑forward training and threshold tuning.  
- **`Test on 23.ipynb`** — the most up‑to‑date notebook. It runs **two trainings** (details below) and contains the current trade simulator and reporting.

2023 backtest (summary): Long straddle +12%, short straddle +3%, combined portfolio +7%.

> Paths inside notebooks may be local; switch them to relative paths such as `Data/...` when running in a fresh environment.

## Repository layout
```
Data/        # raw or prepared CSV/Parquet (not tracked)
Notebooks/   # notebooks (or keep them at repo root)
Results/     # saved tables/plots
README.md
LICENSE
```

## Data requirements
OptionMetrics‑style (or similar) fields are expected.

**Options (per leg)**  
`ticker`, `date` (trade date), `exdate` (expiration), `cp_flag` ∈ {C,P}, `strike_price`, quote fields (`best_bid`, `best_offer` or adjusted variants). Corporate‑action adjusters such as `cfacpr`/`cfadj` are used if adjusted series are present.

**Underlying (daily)**  
`ticker`, `date`, adjusted close (e.g., `adj_prc`). Realized volatility and momentum may be precomputed or derived in‑notebook.

Example filenames: `options_*.csv`, `opt_full_list_mm.csv`, `openprice_and_adjprc_cleaned_ret.csv`, `stock_data_2018_2023_*.csv`, and intermediates such as `Straddle_entries_with_features_*.csv`, `weighted_trades_ret_*.csv`.

If column names differ, edit the early cells that normalize keys, choose ATM strikes, and compute mid quotes.

## Feature set
- **Volatility & ranks**: `straddle_iv`, `iv_rank` (e.g., trailing 60‑day percentile, current day excluded), `realized_vol_10/30`, `realized_vol_30_rank`, `rv_change = rv30 − rv10`  
- **Momentum**: `mom_5`, `mom_10` on the underlying  
- **Skew & premium**: `iv_skew = IV_call − IV_put`, `log_straddle_premium`  
- **Liquidity / flow**: per‑leg bid–ask spreads, `total_volume`, `total_oi`, `oi_change_*`  
- **Micro / meta**: `DTE_norm`, `day_of_week`, `net_delta ≈ delta_call + delta_put`  
- **Filters**: DTE window (commonly **28–35 days**); ATM via adjusted strikes and mid quotes

**Targets (forward returns)**  
Set a horizon `H` (e.g., `H = min(7, DTE − 5)` with `H ≥ 1`). For each entry, build a future `exit_date` and the exit straddle mid for the same `(ticker, exdate, strike)`. Returns: `r_long = (exit − entry)/entry`, `r_short = −r_long`.

## Training & validation

### Regression baseline (`Straddle_With_ML.ipynb` and reused in `Test on 23.ipynb`)
- **Model**: `HistGradientBoostingRegressor` (scikit‑learn)  
- **Split**: rolling **walk‑forward** — train on months 1..k, **validate on month k+1**  
- **Threshold calibration** (on the validation month): grid‑search `delta_long` / `delta_short` to maximize per‑trade Sharpe  
  - go **long** if `pred ≥ delta_long`  
  - go **short** if `pred ≤ −delta_short`

## Two trainings in `Test on 23.ipynb` (most recent design)
`Test on 23.ipynb` intentionally executes **two independent training passes** to compare scoring paradigms under identical data plumbing.

**Training A — Regression pass (HGBRegressor)**  
Replicates the baseline with the latest cleaning utilities (e.g., normalized keys and quotes, explicit leg tables). Produces validation‑tuned thresholds and fold‑level summaries.

**Training B — Classification pass (HGBClassifier)**  
Trains a **classifier** on the same features to obtain probability‑like scores. Key implementation details:
- Entry/exit legs are built explicitly and merged by strict keys (e.g., `ticker`, `exdate`, `strike_key`).  
- Profit is tracked **per $1 of premium**, enabling consistent aggregation across contracts.  
- Thresholds are tuned **on executed (signaled) rows** within the validation month.  
- Reports include per‑trade Sharpe (with annualized variant), cumulative profit per $1, equity curve, and drawdown table.

Running both passes in sequence enables a clean comparison between **regression‑score thresholding** and **probability‑score thresholding**; the out‑of‑sample Sharpe and stability determine the preferred approach.

## Backtest methodology
- Apply the tuned thresholds from each validation month to the next **held‑out month** only.  
- Entry at straddle mid on `date`; exit at straddle mid on `date + H` for the same option tuple.  
- Metrics: per‑trade return vector, win rate, total return, per‑trade Sharpe; plus a daily equity curve with running max and drawdown.  
- Transaction costs, borrow, and slippage are not modeled by default.

## Install & quickstart

**Environment**
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy matplotlib scikit-learn jupyterlab
```

**Optional `requirements.txt`**
```
pandas
numpy
matplotlib
scikit-learn
jupyterlab
```

**Data**  
Place inputs under `Data/` (or update notebook paths).

**Execute**
```
# interactive
jupyter lab

# headless
jupyter nbconvert --to notebook --execute Straddle_With_ML.ipynb --output Results/run_Straddle_With_ML.ipynb
jupyter nbconvert --to notebook --execute "Test on 23.ipynb" --output Results/run_Test_on_23.ipynb
```

## Reproducibility & anti–look‑ahead controls
- Trailing transforms only: rolling stats and percentile ranks are computed with groupwise transforms that exclude the current row.  
- Exit construction: future calendar dates (`date + H`) are merged using the same `(ticker, exdate, strike)` as the entry.  
- Walk‑forward splits: validation/test months are never included in the fit that produces their predictions.  
- Determinism: seeds are set for NumPy and estimators where applicable.

## Outputs
Typical artifacts include:
- Tables: per‑trade returns, threshold sets, fold summaries (in `Results/` if configured)  
- Plots: equity curve and drawdown; optional feature importance charts  
- Console metrics: total return, win rate, Sharpe (per‑trade and annualized variants where reported)

Because results depend on the input universe and filters, specific numbers are not fixed here.

