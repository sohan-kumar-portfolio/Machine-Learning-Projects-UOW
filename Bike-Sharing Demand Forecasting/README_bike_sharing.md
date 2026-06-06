# 🚲 Bike-Sharing Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=flat)
![Course](https://img.shields.io/badge/UOW-CSCI933%20ML-103a75?style=flat)

> End-to-end machine learning pipeline comparing statistical regressors and deep learning models for hourly bike rental demand prediction — built to production standards of reproducibility, evaluation rigour, and temporal validation.

---

## 📌 Overview

Predicting bike-sharing demand is a real-world regression problem with practical implications for urban mobility planning. This project benchmarks **four statistical models** against **two custom PyTorch neural networks** on the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset), examining how regularisation, feature engineering, and data-split strategy affect generalisation.

**Key question explored:** Does deep learning outperform well-tuned statistical baselines on tabular time-series data — and does it hold up under temporal generalisation?

---

## 📂 Project Structure

```
bike-sharing-forecast/
├── data/
│   └── hour.csv                  # UCI Bike Sharing dataset (17,379 records)
├── notebooks/
│   └── bike_sharing_forecast.ipynb
├── outputs/
│   ├── residual_plots.png
│   ├── lasso_shrinkage_paths.png
│   └── model_comparison_dashboard.png
├── requirements.txt
└── README.md
```

---

## 🧠 Models & Approach

### Statistical Regressors
| Model | Regularisation | Hyperparameter Search |
|---|---|---|
| Linear Regression | None | — |
| Ridge Regression | L2 | RidgeCV (automated) |
| Lasso Regression | L1 | LassoCV (automated) |
| Elastic Net | L1 + L2 | ElasticNetCV |

- **5-fold cross-validation** with `RidgeCV`/`LassoCV` for automated, data-driven hyperparameter selection
- **Polynomial feature engineering**: degree-2 interaction terms expanding 7 → 35 features
- **Conditioning number analysis** to assess numerical stability trade-offs from feature expansion

### Neural Networks (PyTorch)
| Model | Architecture | Regularisation |
|---|---|---|
| Linear NN | Input → Output (no hidden layer) | Weight decay (L2) |
| Hidden NN | Input → ReLU hidden layer → Output | Weight decay + Dropout |

- 3 independent random seeds per model to report **mean ± std RMSE** — ensuring stochastic reproducibility
- Systematic grid search over weight decay values and dropout rates

---

## 📊 Evaluation Strategy

Two data-split regimes were used to probe different aspects of model quality:

| Split | Train | Test | Purpose |
|---|---|---|---|
| Random 80/20 | Random 80% | Random 20% | Overall predictive performance |
| Temporal | Pre-Oct 2012 | Oct–Dec 2012 | Real-world temporal generalisation |

**Primary metrics:** RMSE, R²

---

## 🏆 Results Summary

- **Best statistical model:** Ridge Regression (lowest RMSE among statistical regressors)
- **Best overall (random split):** Hidden NN with optimal weight decay — highest R²
- **Temporal split insight:** Neural networks showed more sensitivity to distributional shift across time vs. well-regularised linear models — highlighting the importance of split strategy in model selection

---

## 📈 Visualisations

The pipeline produces three reproducible figures saved to `outputs/`:

- **Residual plots** — per-model error distribution analysis
- **Lasso shrinkage paths** — coefficient behaviour across regularisation strengths
- **Model comparison dashboard** — side-by-side RMSE and R² across all models and splits

---

## 🛠 Setup & Usage

```bash
# Clone the repo
git clone https://github.com/sohan-kumar/bike-sharing-forecast.git
cd bike-sharing-forecast

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/bike_sharing_forecast.ipynb
```

**Requirements:**
```
torch
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

---

## 📚 Dataset

**UCI Bike Sharing Dataset**
- 17,379 hourly records (2011–2012)
- Features: season, hour, weather, temperature, humidity, windspeed
- Target: `cnt` — total hourly bike rentals

[Download from UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

---

## 🎓 Academic Context

Developed as part of **CSCI933 Machine Learning Algorithms and Applications**
University of Wollongong, 2026

---

## 👤 Author

**Sohan Kumar**
[LinkedIn](https://linkedin.com/in/sohan-kumar-006599220) · [GitHub](https://github.com/sohan-kumar)
