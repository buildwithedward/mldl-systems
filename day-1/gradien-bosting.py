# %%
# Build Gradient Boosting by Hand
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Synthetic clinical dataset: predict LOS (length of stay) in days
np.random.seed(42)
n = 300
age = np.random.randint(30, 85, n)
comorbidities = np.random.randint(0, 6, n)
severity_score = np.random.uniform(1, 10, n)

# True LOS (non-linear relationship)
y = (
    0.05 * age
    + 0.8 * comorbidities
    + 0.4 * severity_score
    + 0.3 * (age * comorbidities) / 40
    + np.random.normal(0, 0.8, n)
)
print(f"y: {y}")

X = np.column_stack([age, comorbidities, severity_score])
print(f"X: {X}")

# Manual gradient boosting
N_TREES = 50
LR = 0.1
MAX_DEPTH = 3

# Initialise: predict the mean (best constant model)
F = np.full(n, y.mean())
# print(f"F: {F}")
trees = []

for m in range(N_TREES):
    # Residuals = negative gradient of MSE loss = (y - F)
    residuals = y - F
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=m)
    tree.fit(X, residuals)
    update = tree.predict(X)
    F += LR * update
    trees.append(tree)

mse_manual = np.mean((y - F) ** 2)
print(
    f"Manual GBM — {N_TREES} trees | MSE: {mse_manual:.4f} | RMSE: {np.sqrt(mse_manual):.4f}"
)
print(f"Baseline (predict mean) — MSE: {np.mean((y - y.mean()) ** 2):.4f}")


# %%
# Watch the Ensemble Learn — Residuals Shrinking Over Trees
# Re-run and track residuals at each step
F_track = np.full(n, y.mean())
rmse_by_tree = []

for m in range(N_TREES):
    residuals = y - F_track
    rmse_by_tree.append(np.sqrt(np.mean(residuals**2)))
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=m)
    tree.fit(X, residuals)
    F_track += LR * tree.predict(X)

plt.figure(figsize=(10, 4))
plt.plot(range(1, N_TREES + 1), rmse_by_tree, color="steelblue", linewidth=2)
plt.axhline(
    np.sqrt(np.mean((y - y.mean()) ** 2)),
    color="red",
    linestyle="--",
    label="Baseline (mean)",
)
plt.xlabel("Number of Trees")
plt.ylabel("RMSE (on training data)")
plt.title("Manual GBM — Residuals Shrinking With Each Tree")
plt.legend()
plt.tight_layout()
plt.savefig("gbm_learning_curve.png", dpi=120)
plt.show()
print("Notice: each tree reduces the remaining error — that's boosting.")
# %%
# XGBoost vs Manual GBM — Side-by-Side Comparison
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost with similar settings to our manual GBM
xgb = XGBRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,  # L2 on leaf weights
    reg_alpha=0.0,  # L1 on leaf weights
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print("=" * 45)
print(f"  XGBoost Test RMSE:     {rmse_xgb:.4f}")
print(f"  Manual GBM Train RMSE: {np.sqrt(mse_manual):.4f}  (train, not test)")
print("=" * 45)
print("\nXGBoost advantages over manual GBM:")
print("  ✓ Second-order gradients (Newton boosting)")
print("  ✓ L1/L2 regularisation on leaf weights (λ, α)")
print("  ✓ Minimum gain threshold for splits (γ)")
print("  ✓ Column subsampling (colsample_bytree)")
print("  ✓ Sparse-aware split finding")
print("  ✓ Built-in cross-validation & early stopping")
# %%
# Learning Rate vs Number of Trees — The Fundamental Trade-off
lrs = [0.3, 0.1, 0.05, 0.01]
n_trees_range = list(range(1, 201))
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr, color in zip(lrs, colors):
    train_rmses, test_rmses = [], []
    for n_est in n_trees_range:
        m = XGBRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=3,
            random_state=42,
            verbosity=0,
        )
        m.fit(X_train, y_train)
        train_rmses.append(np.sqrt(mean_squared_error(y_train, m.predict(X_train))))
        test_rmses.append(np.sqrt(mean_squared_error(y_test, m.predict(X_test))))
    axes[0].plot(n_trees_range, train_rmses, color=color, label=f"η={lr}")
    axes[1].plot(n_trees_range, test_rmses, color=color, label=f"η={lr}")

for ax, title in zip(axes, ["Train RMSE", "Test RMSE (generalisation)"]):
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 3)

plt.suptitle("Learning Rate vs Trees — The Core GBM Trade-off", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("lr_vs_trees.png", dpi=120)
plt.show()
print("Low η = better generalisation but needs many more trees.")
print("High η = learns fast but overfits early.")
# %%
# Regularisation — λ and γ in Action
# λ (reg_lambda) penalises large leaf weights (L2 regularisation),
# while γ (min_split_loss) sets a minimum reduction in loss required to make a split
lambdas = [0, 0.1, 1, 5, 10, 50]
results = []

for lam in lambdas:
    m = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=lam,
        random_state=42,
        verbosity=0,
    )
    m.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, m.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, m.predict(X_test)))
    results.append({"lambda": lam, "train_rmse": train_rmse, "test_rmse": test_rmse})
    print(f"λ={lam:5.1f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

# Plot
r = results
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    lambdas, [r["train_rmse"] for r in results], "o-", label="Train", color="steelblue"
)
ax.plot(lambdas, [r["test_rmse"] for r in results], "s-", label="Test", color="coral")
ax.set_xlabel("λ (reg_lambda)")
ax.set_ylabel("RMSE")
ax.set_title("L2 Regularisation (λ) — Effect on Train vs Test RMSE")
ax.legend()
plt.tight_layout()
plt.savefig("lambda_effect.png", dpi=120)
plt.show()
print("\nSmall λ: low bias, higher variance. Large λ: underfits. Sweet spot is key.")
# %%
# Early Stopping — Let the Data Decide When to Stop
# Instead of hand-tuning n_estimators,
# let XGBoost monitor a held-out validation set and stop when performance stops improving.
# Use XGBoost's native early stopping with eval set
xgb_es = XGBRegressor(
    n_estimators=500,  # Set high — early stopping will cut it short
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=20,  # Stop if no improvement for 20 rounds
)

xgb_es.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

best_n = xgb_es.best_iteration
best_score = xgb_es.best_score
y_pred_es = xgb_es.predict(X_test)

print(f"Best iteration:  {best_n}")
print(f"Best RMSE:       {np.sqrt(best_score):.4f}")
print(f"Saved {500 - best_n} unnecessary trees from being built.")

# Plot the full eval metric history
evals_result = xgb_es.evals_result()
rmse_history = [np.sqrt(v) for v in evals_result["validation_0"]["rmse"]]

plt.figure(figsize=(10, 4))
plt.plot(rmse_history, color="steelblue", linewidth=2)
plt.axvline(best_n, color="red", linestyle="--", label=f"Early stop @ tree {best_n}")
plt.xlabel("Boosting Round")
plt.ylabel("Validation RMSE")
plt.title("Early Stopping — Validation RMSE Over Boosting Rounds")
plt.legend()
plt.tight_layout()
plt.savefig("early_stopping.png", dpi=120)
plt.show()
# %%
# XGBoost's Native DMatrix API — The Fast Path
# XGBoost's DMatrix format is the internal optimised data structure — it pre-computes quantile sketches for split finding
# handles sparsity (missing values in EHR are common), and is 2–5× faster than using the sklearn API on large datasets.
# Always use it in production.

import xgboost as xgb_native

# Convert to DMatrix — XGBoost's native format
dtrain = xgb_native.DMatrix(
    X_train, label=y_train, feature_names=["age", "comorbidities", "severity"]
)
dtest = xgb_native.DMatrix(
    X_test, label=y_test, feature_names=["age", "comorbidities", "severity"]
)

params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 0.05,  # same as learning_rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.0,
    "seed": 42,
    "verbosity": 0,
}

evals_log = {}
bst = xgb_native.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=20,
    evals_result=evals_log,
    verbose_eval=False,
)

print(f"Best round:    {bst.best_iteration}")
print(f"Test RMSE:     {np.sqrt(float(bst.best_score)):.4f}")
print(f"\nFeature importance scores:")
importance = bst.get_score(importance_type="gain")
for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feat:20s}  gain={score:.2f}")

# Save and reload model (production pattern)
bst.save_model("/tmp/clinical_los_xgb.ubj")
bst_reloaded = xgb_native.Booster()
bst_reloaded.load_model("/tmp/clinical_los_xgb.ubj")
print("\nModel saved and reloaded successfully ✓")
# %%
# Putting It Together

import pandas as pd

# Final comparison summary
summary = pd.DataFrame(
    {
        "Model": [
            "Baseline (predict mean)",
            "Manual GBM (50 trees, η=0.1)",
            "XGBoost (50 trees, η=0.1)",
            "XGBoost + Early Stopping (η=0.05)",
            "XGBoost DMatrix (native API)",
        ],
        "Test RMSE": [
            round(np.sqrt(np.mean((y_test - y_train.mean()) ** 2)), 4),
            "N/A (train only)",
            round(rmse_xgb, 4),
            round(np.sqrt(mean_squared_error(y_test, y_pred_es)), 4),
            round(np.sqrt(float(bst.best_score)), 4),
        ],
        "Key Insight": [
            "No learning — just average LOS",
            "Proves residual fitting works from scratch",
            "Newton boosting + regularisation helps",
            "Auto-finds best tree count",
            "Production-ready: sparse + fast + save/load",
        ],
    }
)

print("=" * 80)
print("SESSION SUMMARY — Day 1: Gradient Boosting Internals")
print("=" * 80)
print(summary.to_string(index=False))
print("\nKey clinical insight: gradient boosting dominates tabular healthcare")
print("data because EHR features are structured, mixed-type, and sparse —")
print("exactly the conditions where tree ensembles outperform neural networks.")
