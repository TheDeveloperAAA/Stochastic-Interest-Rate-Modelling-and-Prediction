# Stochastic Interest-Rate Modelling & Prediction, the CIR Model

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/out--of--sample%20R²-0.89-success.svg)
![Gate](https://img.shields.io/badge/evaluation%20gate%20%3E0.85-PASS-brightgreen.svg)

Implementing, calibrating, and stress-testing the **Cox–Ingersoll–Ross (CIR)** short-rate model on real
yield-curve data, then reconstructing an entire yield curve **from a single input — the 3-Month rate** — and
scoring it strictly out-of-sample.

> *Launched by Finance Club, IIT Roorkee as Open Projects 2026.* The official competition deliverable is the Colab notebook;
> this repository is the version-controlled showcase. **The notebook is the single source of truth**;
> [`src/cir.py`](src/cir.py) is the same validated logic packaged for reuse.

---

## TL;DR — Results

| Model | Out-of-sample pooled R² | RMSE | Verdict |
|---|---|---|---|
| **Base CIR** (cross-sectional calibration) | **0.8929** | 22.0 bps |  **clears the 0.85 gate** |
| CIR++ (frozen deterministic shift) | 0.8355 |overfits — base CIR generalises better |

**Per-maturity (base CIR, out-of-sample):**

| Maturity | 6M | 9M | 1Y | 2Y |
|---|---|---|---|---|
| R² | 0.994 | 0.968 | 0.910 | **0.389** |

Calibrated parameters: **κ = 0.166, θ = 2.44%, σ = 0.50%** — the **Feller condition holds** (rates stay strictly
positive), implying a rate-shock half-life of ≈ 4.2 years. Train R² (0.961) vs test R² (0.893) shows a small,
healthy gap — no severe overfitting.

---

## The one decision that shapes everything

The CIR short rate is *usually* calibrated from the **time series** of the short rate. We deliberately **do not**.
A naive time-series fit on this data gives a **negative** mean-reversion speed (κ ≈ −0.18), because the 2016–2024
short rate **trended** from ~0% to ~5% rather than reverting to a fixed mean — a single-regime estimator misreads
that drift as *anti*-reversion.

The task never asks us to *forecast* the short rate; it **hands us** the 3M rate each day and asks us to build the
curve **off** it. So we calibrate (κ, θ, σ) **cross-sectionally** — to the geometry of the observed yield curves —
which is both the honest reading of the challenge and numerically stable.

---

## Key findings (the honest version)

1. **The 3M alone reconstructs the short-maturity curve well** — pooled out-of-sample R² ≈ 0.89, clearing the
   0.85 gate.
2. **The 2Y is the hard maturity (R² ≈ 0.39), and we prove why.** The empirical 3M→2Y slope **drifts from ~0.79
   in training to ~0.50 in test** (a rate-cutting cycle in which the 2Y leads the 3M down). No 3M-only model can
   anticipate that — the affine *ceiling* for the 2Y is itself only ~0.55. The pooled score clears 0.85 because
   the high-variance short maturities dominate.
3. **The extension overfits, and that is the most instructive result.** A frozen CIR++ shift encodes a
   training-period bias that the regime shift makes counterproductive out-of-sample. Under real regime change,
   the parsimonious model is the more reliable one — a direct, evidence-based answer to the brief's question
   *"does your extension improve out-of-sample, or overfit?"*
4. **The long end is structurally outside a single factor's reach** (20Y > 30Y on 73% of training days — a hump
   one factor cannot produce), which is why it is excluded from scoring and discussed as a limitation.

---

## Repository structure

```
cir-yield-curve-modelling/
├── README.md                          # you are here
├── requirements.txt                   # pinned dependencies
├── LICENSE                            # MIT
├── .gitignore
├── CIR_Yield_Curve_Modelling.ipynb    # MAIN DELIVERABLE — narrative report + runnable pipeline
├── src/
│   └── cir.py                         # reusable, importable core (data → model → calibration → scoring)
└── data/
    ├── README.md                      # data dictionary + provenance note
    ├── train_data.csv                 # 9 maturities, 2016–2024  (calibration)
    ├── test_data.csv                  # 5 maturities, 2024–2026  (held-out actuals)
    └── test_data_3M.csv               # 3M only                  (the single legal input)
```

---

## How to run

### Option A — Google Colab (the competition deliverable)
1. Open `CIR_Yield_Curve_Modelling.ipynb` in Colab.
2. Upload the three CSVs into the session (or set a Drive path — the loader searches common locations).
3. **Runtime → Run all.** The notebook runs top-to-bottom, calibrates, reconstructs the curve, scores it, and
   renders the diagnostics.

### Option B — Locally
```bash
git clone https://github.com/<your-username>/cir-yield-curve-modelling.git
cd cir-yield-curve-modelling
pip install -r requirements.txt

# Reproduce the headline result from the reusable module:
python src/cir.py

# …or run the full notebook:
jupyter notebook CIR_Yield_Curve_Modelling.ipynb
```

Expected console output from `python src/cir.py`:
```
Calibrated CIRModel(kappa=0.1657, theta=0.0244, sigma=0.0050)  |  Feller holds
BASE CIR  out-of-sample pooled R^2 = 0.8929  (PASS > 0.85)  RMSE 22.0 bps
   ZC050YR: R^2 = 0.9944
   ZC075YR: R^2 = 0.9675
   ZC100YR: R^2 = 0.9101
   ZC200YR: R^2 = 0.3889
CIR++     out-of-sample pooled R^2 = 0.8355  (overfits — base CIR generalises better)
```

---

## Methodology

**The model.** CIR (1985) models the instantaneous short rate as a mean-reverting square-root diffusion,
`dr = κ(θ − r)dt + σ√r dW`, which keeps rates positive (strictly so under Feller, `2κθ ≥ σ²`). It admits a
closed-form zero-coupon bond `P(τ) = A(τ)·exp(−B(τ)·r)`, giving a yield `y(τ, r) = (B(τ)·r − ln A(τ)) / τ` that is
**affine in r** — one parameter triple fixes the slope and intercept at *every* maturity at once.

**Calibration.** We minimise the mean squared error between model and observed yields across the **full training
panel (all days × all 9 maturities)** with a bounded global optimiser (`differential_evolution`, fixed seed).
Fitting all maturities — not just the scored ones — pins (κ, θ, σ) to economically sensible values and avoids a
degenerate `κ→0, θ→∞` optimum.

**Prediction (leakage-free).** For each test day the model ingests **only that day's 3M rate** as `r_t` and
evaluates the closed-form curve. A runtime `assert` (the *leakage firewall*) guarantees the held-out test columns
are opened only by the final scoring step.

**Extension.** CIR++ (Brigo–Mercurio) adds a deterministic shift to fit the term structure. Because the 3M-only
constraint forbids observing the test curve, the shift is estimated on training data and **frozen** — which makes
the project a clean test of whether that correction generalises (it does not).

---

## Limitations (and risk-management implications)

- **Single factor ⇒ one curve shape.** Cannot reproduce a hump or a persistently inverted short end → do not
  price curve-shape-sensitive or long-dated instruments off this model.
- **Constant parameters ⇒ no regime awareness.** One (κ, θ, σ) cannot span the 0%→5% journey; the 2Y failure is
  the visible symptom → recalibrate frequently; treat long maturities as indicative only.
- **Frozen shift ⇒ stale bias.** An in-sample improvement can silently become an out-of-sample loss — exactly the
  overfitting trap quantified here.
- **Cross-sectional calibration fits geometry, not dynamics.** The parameters describe curve shape, not the short
  rate's (non-stationary) time-series law; do not reuse them for VaR or scenario simulation without re-estimation.

---

## References

- Cox, J., Ingersoll, J., & Ross, S. (1985). *A Theory of the Term Structure of Interest Rates.* Econometrica.
- Brigo, D., & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice* (CIR++).
- Longstaff, F., & Schwartz, E. (1992). *Interest Rate Volatility and the Term Structure: A Two-Factor General
  Equilibrium Model.*

## License

MIT — see [`LICENSE`](LICENSE). Replace `<Your Name>` in the licence and the clone URL before publishing.
