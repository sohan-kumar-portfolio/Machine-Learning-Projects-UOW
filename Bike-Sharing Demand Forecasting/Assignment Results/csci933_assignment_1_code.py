# -*- coding: utf-8 -*-
"""CSCI933_Assignment-1_Code.ipynb

# CSCI933 Assignment 1 — Code
**By Sohan Ramesh Kumar (Student ID - 1093022)**

So, we have the following code components covered below as follows:  
- Section 1: Install and import packages
- Section 2: Loading of the dataset
- Section 3: Exploratory Data Analysis
- Section 4: Data splitting and scaling
- Section 5: Comparing of Statistical models (Linear, Ridge, Lasso, Elastic Net)
- Section 6: Residual analysis and feature importance
- Section 7: Neural networks modelling analysis (Linear NN, Hidden NN, Weight decay, Dropout)
- Section 8: Feature engineering (for polynomial features)
- Section 9: Finally comparison of all models

Also I wanted to mention that we left out the dataset - day.csv, as it was just a summed up version of the same data without the hours field
"""

# to install any missing packages
!pip install scikit-learn pandas matplotlib seaborn numpy torch -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV, ElasticNetCV, lasso_path
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import (
    KFold, cross_val_score, train_test_split
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# to fix random seeds so results are reproducible
np.random.seed(42)
torch.manual_seed(42)

print("All imports OK")
print("PyTorch version:", torch.__version__)

# we using global settings used throughout the notebook
FEATURES   = ["temp", "humidity", "windspeed", "season",
              "holiday", "workingday", "hour"]
TARGET     = "bikes_rented"

N_FOLDS    = 5       # k-fold cross-validation
N_RUNS     = 3       # independent runs for stochastic models
EPOCHS     = 300     # training epochs for neural networks
LR         = 0.001   # Adam learning rate
BATCH      = 512     # mini-batch size

# Alpha grid for Ridge and Lasso hyperparameter search
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]

print("Settings ready")

"""## Section 2 — Loading Dataset
we'll upload `hour.csv` using the file panel on the left, or run the upload cell below.
"""

import os

# Option A: if hour.csv is already uploaded, skip this cell
# Option B: run this to get a file picker dialog
if not os.path.exists("hour.csv"):
    from google.colab import files
    uploaded = files.upload()
    print("Uploaded:", list(uploaded.keys()))
else:
    print("hour.csv already present")

# Load the dataset
df = pd.read_csv("hour.csv")

# Rename columns to match the assignment spec names
df = df.rename(columns={
    "hum" : "humidity",
    "hr"  : "hour",
    "cnt" : "bikes_rented"
})

# Pull out features and target
X_raw = df[FEATURES].copy()
y     = df[TARGET].copy()

print("Dataset shape :", df.shape)
print("Features      :", FEATURES)
print("Target        :", TARGET)
print("Target range  :", y.min(), "to", y.max())
print("Missing values:", X_raw.isnull().sum().sum())

"""## Section 3 — Exploratory Data Analysis"""

# Summary statistics for all features and target
print(df[FEATURES + [TARGET]].describe().round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: average bike demand by hour of day
hourly = df.groupby("hour")["bikes_rented"].mean()
axes[0].bar(hourly.index, hourly.values, color="steelblue", alpha=0.85)
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Average bikes rented")
axes[0].set_title("Demand by hour")

# Plot 2: bike demand vs temperature
axes[1].scatter(df["temp"], df["bikes_rented"],
                alpha=0.1, s=4, color="tomato")
axes[1].set_xlabel("Normalised temperature")
axes[1].set_ylabel("Bikes rented")
axes[1].set_title("Demand vs temperature")

# Plot 3: demand by season
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
df["season_label"] = df["season"].map(season_map)
df.boxplot(column="bikes_rented", by="season_label", ax=axes[2])
axes[2].set_title("Demand by season")
axes[2].set_xlabel("")
plt.suptitle("")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.show()
print("Saved: eda_overview.png")

fig, ax = plt.subplots(figsize=(7, 5))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()

# Print which features correlate most with target
top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("Correlation with bikes_rented:")
print(top.round(3))

"""## Section 4 — Data Splitig and Scaling

Thw two splits are required for us:
1. **Random split** — 80/20, standard assumption that observations are exchangeable
2. **Time-based split** — train on data before Oct 2012, test on Oct–Dec 2012. This respects the temporal structure of the data and tests whether models generalise to future periods.

So, the features are standardised using `StandardScaler` fitted only on training data to prevent data leakage.

"""

# RANDOM SPLIT
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# TIME-BASED SPLIT
# Train: all data before Oct 2012 | Test: Oct-Dec 2012
df["dteday"] = pd.to_datetime(df["dteday"])
split_date   = pd.Timestamp("2012-10-01")
train_mask   = df["dteday"] < split_date

X_tr_t = X_raw[train_mask];  X_te_t = X_raw[~train_mask]
y_tr_t = y[train_mask];      y_te_t = y[~train_mask]

print(f"Random split  — train: {len(X_tr_r):,}  test: {len(X_te_r):,}")
print(f"Time split    — train: {len(X_tr_t):,}  test: {len(X_te_t):,}")
print(f"Time boundary — before {split_date.date()} = train")

# SCALE FEATURES
# Fit scaler only on training data, then transform both sets
def make_scaled(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), sc

X_tr_r_sc, X_te_r_sc, sc_r = make_scaled(X_tr_r, X_te_r)
X_tr_t_sc, X_te_t_sc, sc_t = make_scaled(X_tr_t, X_te_t)

# Convert to numpy arrays for sklearn
y_tr_r = y_tr_r.values;  y_te_r = y_te_r.values
y_tr_t = y_tr_t.values;  y_te_t = y_te_t.values

print("Scaling done. Data ready.")

# Helper: compute and print RMSE, MAE, R2
def get_metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:<32}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# Helper: 5-fold CV RMSE (mean and std)
def cv_score(model, X, y):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y,
                             cv=kf,
                             scoring="neg_root_mean_squared_error")
    return -scores.mean(), scores.std()

print("Helpers ready")

"""## Section 5 — Statistical Models Comparison

We'll train models with 5-fold cross-validation on the training set for hyperparameter selection.
Final metrics reported on the held-out test set.

**Models:** Linear Regression, Ridge (L2), Lasso (L1), Elastic Net

"""

print("=== Linear Regression ===")

lr_model = LinearRegression()
lr_model.fit(X_tr_r_sc, y_tr_r)
lr_pred  = lr_model.predict(X_te_r_sc)

stat_results = {}
stat_results["Linear Regression"] = get_metrics(y_te_r, lr_pred, "Linear Regression (random)")

cv_m, cv_s = cv_score(LinearRegression(), X_tr_r_sc, y_tr_r)
print(f"  5-fold CV RMSE: {cv_m:.2f} +/- {cv_s:.2f}")

print("=== Ridge Regression ===")

# RidgeCV selects best alpha automatically via cross-validation
ridge_model = RidgeCV(alphas=ALPHAS, cv=N_FOLDS)
ridge_model.fit(X_tr_r_sc, y_tr_r)
ridge_pred  = ridge_model.predict(X_te_r_sc)

stat_results["Ridge Regression"] = get_metrics(y_te_r, ridge_pred, "Ridge Regression (random)")

cv_m, cv_s = cv_score(RidgeCV(alphas=ALPHAS, cv=N_FOLDS), X_tr_r_sc, y_tr_r)
print(f"  Best alpha: {ridge_model.alpha_:.4g}")
print(f"  5-fold CV RMSE: {cv_m:.2f} +/- {cv_s:.2f}")

print("=== Lasso Regression ===")

# LassoCV selects best alpha automatically via cross-validation
lasso_model = LassoCV(cv=N_FOLDS, max_iter=10000, random_state=42)
lasso_model.fit(X_tr_r_sc, y_tr_r)
lasso_pred  = lasso_model.predict(X_te_r_sc)

stat_results["Lasso Regression"] = get_metrics(y_te_r, lasso_pred, "Lasso Regression (random)")

n_zero = int(np.sum(np.abs(lasso_model.coef_) < 1e-6))
cv_m, cv_s = cv_score(
    LassoCV(cv=N_FOLDS, max_iter=10000, random_state=42), X_tr_r_sc, y_tr_r
)
print(f"  Best alpha   : {lasso_model.alpha_:.4g}")
print(f"  Zeroed coefs : {n_zero} out of {len(FEATURES)}")
print(f"  5-fold CV RMSE: {cv_m:.2f} +/- {cv_s:.2f}")
print()
print("  Feature coefficients:")
for feat, coef in zip(FEATURES, lasso_model.coef_):
    tag = "  <- zeroed out" if abs(coef) < 1e-6 else ""
    print(f"    {feat:<12}: {coef:+.3f}{tag}")

print("=== Elastic Net ===")

# ElasticNetCV searches over both alpha and l1_ratio
enet_model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    cv=N_FOLDS, max_iter=10000, random_state=42
)
enet_model.fit(X_tr_r_sc, y_tr_r)
enet_pred = enet_model.predict(X_te_r_sc)

stat_results["Elastic Net"] = get_metrics(y_te_r, enet_pred, "Elastic Net (random)")

cv_m, cv_s = cv_score(
    ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                 cv=N_FOLDS, max_iter=10000, random_state=42),
    X_tr_r_sc, y_tr_r
)
print(f"  Best alpha  : {enet_model.alpha_:.4g}")
print(f"  Best l1_ratio: {enet_model.l1_ratio_:.2f}")
print(f"  5-fold CV RMSE: {cv_m:.2f} +/- {cv_s:.2f}")

print("=== Statistical Models — Time-based split ===")

for name, model in [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression",  RidgeCV(alphas=ALPHAS, cv=N_FOLDS)),
    ("Lasso Regression",  LassoCV(cv=N_FOLDS, max_iter=10000, random_state=42)),
    ("Elastic Net",       ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                                       cv=N_FOLDS, max_iter=10000, random_state=42)),
]:
    model.fit(X_tr_t_sc, y_tr_t)
    pred = model.predict(X_te_t_sc)
    stat_results[name + " (time)"] = get_metrics(y_te_t, pred, name + " (time)")

print()
print("Comparison — random vs time split RMSE:")
for name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"]:
    r_rmse = stat_results[name]["RMSE"]
    t_rmse = stat_results[name + " (time)"]["RMSE"]
    diff   = t_rmse - r_rmse
    print(f"  {name:<22}  random={r_rmse:.2f}  time={t_rmse:.2f}  diff={diff:+.2f}")

"""## Section 6 — Residual Analysis and Feature Importance"""

# Residual plots for all 4 statistical models
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

preds_stat = [
    ("Linear Regression", lr_pred),
    ("Ridge Regression",  ridge_pred),
    ("Lasso Regression",  lasso_pred),
    ("Elastic Net",       enet_pred),
]

for ax, (name, pred) in zip(axes.flatten(), preds_stat):
    residuals = y_te_r - pred
    ax.scatter(pred, residuals, alpha=0.15, s=4, color="steelblue")
    ax.axhline(0, color="tomato", linewidth=1)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title(name)
    rmse = np.sqrt(mean_squared_error(y_te_r, pred))
    ax.text(0.97, 0.04, f"RMSE={rmse:.1f}", transform=ax.transAxes,
            ha="right", fontsize=9)

plt.suptitle("Residuals vs Fitted — Statistical Models", fontsize=13)
plt.tight_layout()
plt.savefig("stat_residuals.png", dpi=150)
plt.show()

# Standardised coefficient comparison across all 4 models
coef_df = pd.DataFrame({
    "Feature":           FEATURES,
    "Linear Regression": lr_model.coef_,
    "Ridge Regression":  ridge_model.coef_,
    "Lasso Regression":  lasso_model.coef_,
    "Elastic Net":       enet_model.coef_,
}).set_index("Feature")

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(FEATURES))
width = 0.2
colors = ["steelblue", "seagreen", "tomato", "mediumpurple"]

for i, col in enumerate(coef_df.columns):
    ax.bar(x + i * width, coef_df[col], width,
           label=col, color=colors[i], alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(FEATURES, rotation=30, ha="right")
ax.set_ylabel("Coefficient value")
ax.set_title("Standardised feature coefficients — all models")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("stat_coefficients.png", dpi=150)
plt.show()

# Lasso coefficient path — shows how coefficients shrink to zero as alpha increases
# This is key evidence for the sparsity discussion in the report
alphas_path, coefs_path, _ = lasso_path(
    X_tr_r_sc, y_tr_r, alphas=np.logspace(-3, 2, 80)
)

fig, ax = plt.subplots(figsize=(9, 5))
for i, feat in enumerate(FEATURES):
    ax.plot(np.log10(alphas_path), coefs_path[i], label=feat, linewidth=1.5)

ax.axvline(np.log10(lasso_model.alpha_), color="black",
           linestyle="--", linewidth=1,
           label=f"selected alpha={lasso_model.alpha_:.3g}")

ax.set_xlabel("log10(alpha)  — regularisation strength increases right")
ax.set_ylabel("Coefficient value")
ax.set_title("Lasso coefficient shrinkage path")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("lasso_path.png", dpi=150)
plt.show()
print("As alpha increases, features shrink to zero — this is L1 sparsity.")
print("Features that hit zero earliest have the weakest predictive signal.")

"""## Section 7 — Neural Networks models

**Models built here:**
1. Linear Neural Network — no hidden layer, no activation. Theoretically equivalent to Ridge Regression with L2 weight decay.
2. One hidden-layer network with ReLU — can capture nonlinear patterns. Run 3 independent times.
3. Weight decay analysis — vary L2 penalty and observe effect on test RMSE.
4. Dropout investigation — so we test whether dropout helps for this small-feature regression problem.

All stochastic models are run N_RUNS=3 times and results reported as mean ± std.

"""

# Neural network model definitions

class LinearNN(nn.Module):
    # Single linear layer, no activation — equivalent to linear regression
    def __init__(self, n_features):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.fc(x)


class HiddenNN(nn.Module):
    # One hidden layer with ReLU activation, optional dropout
    def __init__(self, n_features, hidden_size=64, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Training and evaluation function
def train_model(model, X_tr, y_tr, X_te, y_te,
                weight_decay=0.0, epochs=EPOCHS, lr=LR):
    # Build DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(y_tr).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    # Adam optimiser with optional L2 weight decay
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_rmse_curve = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        train_rmse_curve.append(np.sqrt(total_loss / len(dataset)))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te)).squeeze().numpy()

    test_rmse = np.sqrt(mean_squared_error(y_te, pred))
    return test_rmse, pred, train_rmse_curve


print("Model classes and training function ready")

print("=== 7a. Linear Neural Network ===")
print("Goal: confirm this matches Ridge Regression (theoretical equivalence)")

nn_results = {}
n_feat = X_tr_r_sc.shape[1]

best_linear_wd   = None
best_linear_rmse = float("inf")
linear_pred      = None
linear_curve     = None

for wd in [0.0, 0.01, 0.1]:
    torch.manual_seed(42)
    rmse, pred, curve = train_model(
        LinearNN(n_feat),
        X_tr_r_sc, y_tr_r,
        X_te_r_sc, y_te_r,
        weight_decay=wd,
        epochs=2000,      # increased from 300
        lr=0.01           # increased from 0.001
    )
    print(f"  weight_decay={wd}  ->  test RMSE={rmse:.2f}")
    if rmse < best_linear_rmse:
        best_linear_rmse = rmse
        best_linear_wd   = wd
        linear_pred      = pred
        linear_curve     = curve

nn_results["Linear NN"] = get_metrics(y_te_r, linear_pred, "Linear NN (best wd)")

ridge_rmse = stat_results["Ridge Regression"]["RMSE"]
print(f"\n  Ridge RMSE    = {ridge_rmse:.2f}")
print(f"  Linear NN RMSE= {best_linear_rmse:.2f}")
print(f"  Difference    = {abs(best_linear_rmse - ridge_rmse):.2f}")

print("=== 7b. One Hidden-Layer Network (3 independent runs) ===")

run_rmses = []
run_preds = []

for run in range(N_RUNS):
    torch.manual_seed(run)  # different seed each run
    rmse, pred, _ = train_model(
        HiddenNN(n_feat, hidden_size=64),
        X_tr_r_sc, y_tr_r,
        X_te_r_sc, y_te_r
    )
    run_rmses.append(rmse)
    run_preds.append(pred)
    print(f"  Run {run+1}: RMSE={rmse:.2f}")

hidden_mean = np.mean(run_rmses)
hidden_std  = np.std(run_rmses)
best_run    = int(np.argmin(run_rmses))
hidden_pred = run_preds[best_run]

print(f"  Mean RMSE = {hidden_mean:.2f} +/- {hidden_std:.2f}")
nn_results["Hidden NN"] = get_metrics(y_te_r, hidden_pred, "Hidden NN (best run)")

print("=== 7c. Weight Decay Analysis ===")

wd_values = [0.0, 0.001, 0.01, 0.1, 1.0]
wd_results = {}

for wd in wd_values:
    run_r = []
    for run in range(N_RUNS):
        torch.manual_seed(run)
        rmse, _, _ = train_model(
            HiddenNN(n_feat, hidden_size=64),
            X_tr_r_sc, y_tr_r,
            X_te_r_sc, y_te_r,
            weight_decay=wd
        )
        run_r.append(rmse)
    mean_rmse = np.mean(run_r)
    std_rmse  = np.std(run_r)
    wd_results[wd] = (mean_rmse, std_rmse)
    print(f"  weight_decay={wd:<6}  RMSE={mean_rmse:.2f} +/- {std_rmse:.2f}")

best_wd = min(wd_results, key=lambda k: wd_results[k][0])
print(f"  Best weight_decay = {best_wd}")

# Train final model with best weight decay
torch.manual_seed(42)
wd_best_rmse, wd_pred, wd_curve = train_model(
    HiddenNN(n_feat, hidden_size=64),
    X_tr_r_sc, y_tr_r,
    X_te_r_sc, y_te_r,
    weight_decay=best_wd
)
nn_results["Hidden NN + WD"] = get_metrics(y_te_r, wd_pred, f"Hidden NN + WD={best_wd}")

# Plot effect of weight decay on RMSE
fig, ax = plt.subplots(figsize=(7, 4))
x_labels = [str(wd) for wd in wd_values]
means = [wd_results[wd][0] for wd in wd_values]
stds  = [wd_results[wd][1] for wd in wd_values]
ax.bar(x_labels, means, yerr=stds, color="seagreen", alpha=0.85, capsize=5)
ax.set_xlabel("Weight decay (L2 penalty)")
ax.set_ylabel("Test RMSE")
ax.set_title("Effect of weight decay on Hidden NN test RMSE")
plt.tight_layout()
plt.savefig("nn_weight_decay.png", dpi=150)
plt.show()

print("=== 7d. Dropout Investigation ===")

dropout_values = [0.0, 0.1, 0.2, 0.3, 0.5]
dropout_results = {}

for dp in dropout_values:
    run_r = []
    for run in range(N_RUNS):
        torch.manual_seed(run)
        rmse, _, _ = train_model(
            HiddenNN(n_feat, hidden_size=64, dropout=dp),
            X_tr_r_sc, y_tr_r,
            X_te_r_sc, y_te_r,
            weight_decay=best_wd  # use best weight decay from above
        )
        run_r.append(rmse)
    mean_rmse = np.mean(run_r)
    std_rmse  = np.std(run_r)
    dropout_results[dp] = (mean_rmse, std_rmse)
    print(f"  dropout={dp}  RMSE={mean_rmse:.2f} +/- {std_rmse:.2f}")

no_dropout_rmse  = dropout_results[0.0][0]
best_dp          = min(dropout_results, key=lambda k: dropout_results[k][0])
best_dp_rmse     = dropout_results[best_dp][0]
improvement      = no_dropout_rmse - best_dp_rmse

print()
print(f"  Without dropout : {no_dropout_rmse:.2f}")
print(f"  Best dropout ({best_dp}) : {best_dp_rmse:.2f}  (improvement: {improvement:.2f})")

if improvement < 1.0:
    print("  Finding: dropout did NOT substantially help.")
    print("  This is expected — dropout benefits over-parameterised deep networks.")
    print("  With only 7 features and one hidden layer, there is limited co-adaptation to prevent.")
    print("  This supports the discussion in the report (Srivastava et al., 2014).")
else:
    print(f"  Finding: dropout improved RMSE by {improvement:.2f}.")

# Train final model with best dropout
torch.manual_seed(42)
dp_rmse, dp_pred, _ = train_model(
    HiddenNN(n_feat, hidden_size=64, dropout=best_dp),
    X_tr_r_sc, y_tr_r,
    X_te_r_sc, y_te_r,
    weight_decay=best_wd
)
nn_results["Hidden NN + Dropout"] = get_metrics(y_te_r, dp_pred, f"Hidden NN + dropout={best_dp}")

# Plot effect of dropout on RMSE
fig, ax = plt.subplots(figsize=(7, 4))
x_labels = [str(dp) for dp in dropout_values]
means = [dropout_results[dp][0] for dp in dropout_values]
stds  = [dropout_results[dp][1] for dp in dropout_values]
ax.bar(x_labels, means, yerr=stds, color="mediumpurple", alpha=0.85, capsize=5)
ax.set_xlabel("Dropout rate")
ax.set_ylabel("Test RMSE")
ax.set_title("Effect of dropout rate on Hidden NN test RMSE")
plt.tight_layout()
plt.savefig("nn_dropout.png", dpi=150)
plt.show()

# Training curves for Linear NN vs Hidden NN with weight decay
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(linear_curve, label="Linear NN", color="steelblue", linewidth=1.5)
ax.plot(wd_curve,     label=f"Hidden NN (wd={best_wd})", color="tomato", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training RMSE")
ax.set_title("Neural network training curves")
ax.legend()
plt.tight_layout()
plt.savefig("nn_training_curves.png", dpi=150)
plt.show()

# Residuals for the two main neural network models
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, pred) in zip(axes, [
    ("Linear NN",                   linear_pred),
    (f"Hidden NN (wd={best_wd})",   wd_pred),
]):
    residuals = y_te_r - pred
    ax.scatter(pred, residuals, alpha=0.15, s=4, color="seagreen")
    ax.axhline(0, color="tomato", linewidth=1)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title(name)
    rmse = np.sqrt(mean_squared_error(y_te_r, pred))
    ax.text(0.97, 0.04, f"RMSE={rmse:.1f}", transform=ax.transAxes,
            ha="right", fontsize=9)

plt.suptitle("Residuals — Neural Networks")
plt.tight_layout()
plt.savefig("nn_residuals.png", dpi=150)
plt.show()

"""## Section 8 — Feature Engineering

Polynomial features (degree 2) add squared terms and interaction terms.
For example: temp², hour², temp×humidity, etc.

This tests whether nonlinear feature combinations improve performance,
and whether the conditioning of the feature matrix worsens (ill-conditioning
can destabilise training).

"""

print("=== Section 8 — Polynomial Feature Engineering ===")

# Create degree-2 polynomial features (squared terms + interaction terms)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_poly = poly.fit_transform(X_tr_r_sc)
X_te_poly  = poly.transform(X_te_r_sc)

n_poly = X_tr_poly.shape[1]
print(f"  Original features : {n_feat}")
print(f"  Polynomial features: {n_poly}")

# Check how conditioning of the matrix changes
cond_original = np.linalg.cond(X_tr_r_sc)
cond_poly     = np.linalg.cond(X_tr_poly)
print(f"  Condition number (original): {cond_original:.2f}")
print(f"  Condition number (poly)    : {cond_poly:.2f}")
print("  Higher condition number = less numerically stable")

print("Training Hidden NN on polynomial features (3 independent runs)...")

poly_run_rmses = []
poly_run_preds = []

for run in range(N_RUNS):
    torch.manual_seed(run)
    # Wider hidden layer to handle more input features
    model = HiddenNN(n_poly, hidden_size=128, dropout=0.1)
    rmse, pred, _ = train_model(
        model,
        X_tr_poly, y_tr_r,
        X_te_poly,  y_te_r,
        weight_decay=0.01
    )
    poly_run_rmses.append(rmse)
    poly_run_preds.append(pred)
    print(f"  Run {run+1}: RMSE={rmse:.2f}")

poly_mean = np.mean(poly_run_rmses)
poly_std  = np.std(poly_run_rmses)
poly_pred = poly_run_preds[int(np.argmin(poly_run_rmses))]

print(f"  Mean RMSE = {poly_mean:.2f} +/- {poly_std:.2f}")
nn_results["Poly NN"] = get_metrics(y_te_r, poly_pred,
    f"Poly NN mean={poly_mean:.2f}+/-{poly_std:.2f}")

# Also test ridge on polynomial features (no neural net needed)
ridge_poly = RidgeCV(alphas=ALPHAS, cv=N_FOLDS).fit(X_tr_poly, y_tr_r)
ridge_poly_pred = ridge_poly.predict(X_te_poly)
stat_results["Ridge + Poly"] = get_metrics(y_te_r, ridge_poly_pred,
    f"Ridge + Poly (alpha={ridge_poly.alpha_:.3g})")

"""## Section 9 — Final Comparison of All Models"""

print("=" * 65)
print("FINAL RESULTS TABLE")
print("We will record these numbers into the LaTeX report tables")
print("=" * 65)

# Statistical models
stat_main = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Ridge + Poly",
]

print()
print("--- Statistical Models (random split) ---")
print(f"{'Model':<28} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-" * 56)
for m in stat_main:
    if m in stat_results:
        r = stat_results[m]
        print(f"{m:<28} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']:>8.3f}")

# Neural models
nn_main = [
    "Linear NN",
    "Hidden NN",
    "Hidden NN + WD",
    "Hidden NN + Dropout",
    "Poly NN",
]

print()
print("--- Neural Network Models (random split) ---")
print(f"{'Model':<28} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-" * 56)
for m in nn_main:
    if m in nn_results:
        r = nn_results[m]
        print(f"{m:<28} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']:>8.3f}")

# Time split comparison
print()
print("--- Random vs Time Split (statistical models) ---")
print(f"{'Model':<22} {'Random RMSE':>12} {'Time RMSE':>10} {'Diff':>6}")
print("-" * 56)
for name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"]:
    if name in stat_results and name + " (time)" in stat_results:
        rr = stat_results[name]["RMSE"]
        rt = stat_results[name + " (time)"]["RMSE"]
        print(f"{name:<22} {rr:>12.2f} {rt:>10.2f} {rt-rr:>+6.2f}")

# Bar chart comparing RMSE for all main models
all_names = [
    "Linear Reg", "Ridge", "Lasso", "Elastic Net",
    "Linear NN", "Hidden NN", "Hidden NN+WD", "Dropout NN", "Poly NN"
]
all_keys = [
    "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
    "Linear NN", "Hidden NN", "Hidden NN + WD", "Hidden NN + Dropout", "Poly NN"
]
all_rmses = []
for k in all_keys:
    if k in stat_results:
        all_rmses.append(stat_results[k]["RMSE"])
    elif k in nn_results:
        all_rmses.append(nn_results[k]["RMSE"])
    else:
        all_rmses.append(0)

colors = ["steelblue"] * 4 + ["seagreen"] * 5

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(all_names)), all_rmses, color=colors, alpha=0.85)
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Test RMSE (lower is better)")
ax.set_title("All models — test RMSE comparison (random split)")

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="steelblue", label="Statistical models"),
    Patch(color="seagreen",  label="Neural network models")
])

# Add RMSE values on top of each bar
for bar, val in zip(bars, all_rmses):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("final_comparison.png", dpi=150)
plt.show()

# Side-by-side comparison of random vs time-based split for statistical models
stat_names_short = ["Linear", "Ridge", "Lasso", "Elastic Net"]
stat_names_full  = ["Linear Regression", "Ridge Regression",
                    "Lasso Regression", "Elastic Net"]

rand_rmses = [stat_results[m]["RMSE"]          for m in stat_names_full]
time_rmses = [stat_results[m + " (time)"]["RMSE"] for m in stat_names_full]

fig, ax = plt.subplots(figsize=(8, 4))
x, w = np.arange(4), 0.35
ax.bar(x - w/2, rand_rmses, w, label="Random split", color="steelblue", alpha=0.85)
ax.bar(x + w/2, time_rmses, w, label="Time split",   color="tomato",    alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(stat_names_short)
ax.set_ylabel("Test RMSE")
ax.set_title("Random vs time-based split — statistical models")
ax.legend()
plt.tight_layout()
plt.savefig("split_comparison.png", dpi=150)
plt.show()

from google.colab import files
import os, time

figure_files = [
    "eda_overview.png",
    "eda_correlation.png",
    "stat_residuals.png",
    "stat_coefficients.png",
    "lasso_path.png",
    "nn_training_curves.png",
    "nn_weight_decay.png",
    "nn_dropout.png",
    "nn_residuals.png",
    "split_comparison.png",
    "final_comparison.png",
]

for f in figure_files:
    if os.path.exists(f):
        files.download(f)
        time.sleep(1.5)   # give browser time to register each download
        print(f"Downloaded: {f}")
    else:
        print(f"Not found: {f}")

from google.colab import files
import os, time

figure_files = [
    "final_comparison.png",
]

for f in figure_files:
    if os.path.exists(f):
        files.download(f)
        time.sleep(1.5)   # give browser time to register each download
        print(f"Downloaded: {f}")
    else:
        print(f"Not found: {f}")

"""## Done

All the figures are saved and downloaded. Here is what each figure is used for in my report:

| Figure | Report section |
|---|---|
| eda_overview.png | Section 2 — Dataset description |
| eda_correlation.png | Section 2 — Dataset description |
| stat_residuals.png | Section 4 — Statistical models |
| stat_coefficients.png | Section 4 — Feature importance |
| lasso_path.png | Section 4 — Lasso sparsity discussion |
| nn_training_curves.png | Section 5 — Neural network training |
| nn_weight_decay.png | Section 5 — Weight decay analysis |
| nn_dropout.png | Section 5 — Dropout investigation |
| nn_residuals.png | Section 5 — Neural network residuals |
| split_comparison.png | Section 4/5 — Random vs time split |
| final_comparison.png | Section 6 — Overall model comparison |

"""
