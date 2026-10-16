"""
Home Credit Default Risk - FIRST BASELINE
Metric: ROC-AUC. We predict PROBABILITY of TARGET=1 (default).
Folder layout expected:
    competition/
        baseline.py
        input/train.csv, input/test.csv, input/sample_submission.csv
        output/            (created automatically)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "lightgbm is required for this baseline. Install it with: pip install lightgbm"
    ) from exc

SEED = 42                      # competition rule: set seeds for reproducibility
np.random.seed(SEED)
BASE = Path(__file__).parent if "__file__" in globals() else Path(".")
IN_DIR, OUT_DIR = BASE / "input", BASE / "output"
OUT_DIR.mkdir(exist_ok=True)


def normalize_col_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalize_columns(df):
    return df.rename(columns={col: normalize_col_name(col) for col in df.columns})


# ---------- 1. LOAD ----------
train = normalize_columns(pd.read_csv(IN_DIR / "train.csv", sep="\t"))
test = normalize_columns(pd.read_csv(IN_DIR / "test.csv", sep="\t"))
sample_raw = pd.read_csv(IN_DIR / "sample_submission.csv")
sample = normalize_columns(sample_raw)
print("train", train.shape, "| test", test.shape, "| sample", sample.shape)

# ---------- 2. QUICK EXPLORATION ----------
TARGET = "target" if "target" in train.columns else next(col for col in train.columns if "target" in col.lower())
ID = "skidcurr" if "skidcurr" in train.columns else "id" if "id" in train.columns else train.columns[0]
if ID not in test.columns:
    ID = "id" if "id" in test.columns else next(col for col in test.columns if "id" in col.lower())

sample_id = next((col for col in sample.columns if col in {"skidcurr", "id"}), sample.columns[0])

print("\nTarget rate (share of defaults):", round(train[TARGET].mean(), 4))
print("Columns with missing values (train):")
miss = train.isna().sum()
print(miss[miss > 0].sort_values(ascending=False))

# ---------- 3. SEPARATE FEATURES / TARGET ----------
y = train[TARGET].to_numpy(dtype=np.int8)
X = train.drop(columns=[ID, TARGET])
X_test = test.drop(columns=[ID])
assert list(X.columns) == list(X_test.columns)

# ---------- 4. PREPROCESSING ----------
def preprocess(df):
    df = df.copy()
    for col in list(df.columns):
        if normalize_col_name(col) == "daysemployed":
            df[col] = df[col].replace(365243, np.nan)
    # "XNA" means unknown
    df = df.replace("XNA", np.nan)
    # LightGBM can handle text columns if we mark them as 'category'
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype("category")
    return df


def safe_roc_auc(y_true, y_score):
    if np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


full = preprocess(pd.concat([X, X_test], axis=0, ignore_index=True))   # same categories in train+test
X, X_test = full.iloc[:len(X)].reset_index(drop=True), full.iloc[len(X):].reset_index(drop=True)

# ---------- 5. MODEL + 5-FOLD VALIDATION ----------
params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
              min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=SEED, verbose=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))            # out-of-fold predictions
test_pred = np.zeros(len(X_test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = LGBMClassifier(**params)
    model.fit(X.iloc[tr_idx], y[tr_idx])
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1] # pyright: ignore[reportIndexIssue, reportCallIssue]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    score = safe_roc_auc(y[va_idx], oof[va_idx])
    print(f"Fold {fold} AUC = {score:.5f}" if not np.isnan(score) else f"Fold {fold} AUC = n/a (not enough class diversity)")
print(f"\nOverall CV AUC = {safe_roc_auc(y, oof):.5f}")

# ---------- 6. BUILD SUBMISSION ----------
sample_id_column = next(
    col for col in sample_raw.columns if normalize_col_name(col) == sample_id
)
sample_target_column = next(
    col for col in sample_raw.columns if normalize_col_name(col) == TARGET
)
submission = pd.DataFrame(
    {sample_id_column: test[ID].values, sample_target_column: test_pred}
)[sample_raw.columns]
submission.to_csv(OUT_DIR / "submission.csv", index=False)

# ---------- 7. VERIFY BEFORE UPLOAD ----------
sub = pd.read_csv(OUT_DIR / "submission.csv")
print("\n=== SUBMISSION CHECKS ===")
print("Rows:", len(sub), "(expected", len(test), ")")
print("Columns:", list(sub.columns), "(expected", list(sample_raw.columns), ")")
print(sub.head())
print("Missing values:", int(sub.isna().sum().sum()))
print(
    "TARGET range:",
    sub[sample_target_column].min(),
    "to",
    sub[sample_target_column].max(),
    "(must be within 0..1)",
)

ids_match = len(sub) == len(test) and set(sub[sample_id_column]) == set(test[ID])
order_match = len(sub) == len(test) and (sub[sample_id_column].values == test[ID].values).all()
print("IDs match test:", ids_match, "| duplicates:", sub[sample_id_column].duplicated().any())
print("Same ID order as test:", order_match)
if len(sample_raw) != len(test):
    print(
        "Note: sample_submission has",
        len(sample_raw),
        "example rows; validated against the",
        len(test),
        "test rows instead.",
    )