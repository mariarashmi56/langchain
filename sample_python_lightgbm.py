import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer, StandardScaler, OneHotEncoder
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# 1) Load
df_train = pd.read_csv('path/to/train.csv')
df_test  = pd.read_csv('path/to/test.csv')

# 2) Identify columns
target = 'target'
nums   = df_train.select_dtypes(include='number').columns.drop(target)
cats   = df_train.select_dtypes(include=['object','category']).columns

# 3) Detect skew for log‑transform
skew = df_train[nums].skew().abs()
skewed = skew[skew > 1.0].index.tolist()
others = [c for c in nums if c not in skewed]

# 4) Split cats by cardinality
high_card = [c for c in cats if df_train[c].nunique() > 10]
low_card  = [c for c in cats if df_train[c].nunique() <= 10]

# 5) Build the ColumnTransformer — each branch *must* impute first:
preprocessor = ColumnTransformer([
    # skewed numerics
    ("lognum", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log",     FunctionTransformer(np.log1p, validate=False)),
        ("scale",   StandardScaler())
    ]), skewed),

    # the rest of numerics
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale",   StandardScaler())
    ]), others),

    # high‑cardinality cats → mode‑impute + frequency encode
    ("cat_freq", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq",    FunctionTransformer(
                        lambda df: df.apply(
                            lambda col: col.map(col.value_counts(normalize=True))
                        ), validate=False))
    ]), high_card),

    # low‑cardinality cats → mode‑impute + one‑hot
    ("cat_ohe", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]), low_card),

], remainder="drop")  # anything else (none) gets dropped

# 6) Feature selector
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, random_state=0),
    max_features=30, threshold=-np.inf
)

# 7) Two competing pipelines
pipes = {
    "RF": Pipeline([
        ("pre", preprocessor),
        ("sel", selector),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "HGB": Pipeline([
        ("pre", preprocessor),
        ("sel", selector),
        ("clf", HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.1, random_state=42))
    ]),
}

# 8) Cross‑val
X = df_train.drop(columns=target)
y = df_train[target].astype(int)
cv = StratifiedKFold(3, shuffle=True, random_state=42)
results = {}
for name, pipe in pipes.items():
    roc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pr  = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
    ll  = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")
    print(f"{name:20s} ROC‑AUC: {roc.mean():.3f} ± {roc.std():.3f} | PR‑AUC: {pr.mean():.3f} ± {pr.std():.3f}")
    results[name] = (roc.mean(), pr.mean(), ll.mean())

# 9) Fit best on full train
best = max(results, key=lambda k: results[k][0])
print(f"Best model: {best}")
best_pipe = pipes[best].fit(X, y)

# 10) Predict & eval on test
X_test = df_test.drop(columns=target, errors="ignore")
y_test = df_test.get(target, None)
proba  = best_pipe.predict_proba(X_test)[:,1]

if y_test is not None:
    print("Test ROC‑AUC:", roc_auc_score(y_test, proba))
    print("Test PR‑AUC: ", average_precision_score(y_test, proba))
    print("Test LogLoss:", log_loss(y_test, proba))



    # Get Selected Features
    # 1) Grab the fitted ColumnTransformer
pre = best_pipe.named_steps['pre']

# 2) Get the full list of feature names after preprocessing
#    (sklearn ≥1.0 required for get_feature_names_out on ColumnTransformer)
feature_names = pre.get_feature_names_out()

# 3) Grab the selector and its support mask
sel_mask = best_pipe.named_steps['sel'].get_support()

# 4) Index into the names
selected_features = feature_names[sel_mask]

print("Selected features:")
for f in selected_features:
    print(" •", f)




import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error

# ==== 1) Load Data ====
df_train = pd.read_csv('path/to/train.csv')
df_test  = pd.read_csv('path/to/test.csv')

# ==== 2) Identify columns ====
target_col       = 'target_reg'  # replace with your regression target name
numeric_cols     = df_train.select_dtypes(include=['number']).columns.drop(target_col).tolist()
categorical_cols = df_train.select_dtypes(include=['object','category']).columns.tolist()

# ==== 3) Detect skewed numerics ====
skew = df_train[numeric_cols].skew().abs()
skewed_cols = skew[skew > 1.0].index.tolist()
other_nums  = [c for c in numeric_cols if c not in skewed_cols]

# ==== 4) Split categoricals ====
high_card = [c for c in categorical_cols if df_train[c].nunique() > 10]
low_card  = [c for c in categorical_cols if df_train[c].nunique() <= 10]

# ==== 5) Build ColumnTransformer ====
preprocessor = ColumnTransformer([
    # skewed numerics: median impute → log1p → scale
    ('log_num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log',     FunctionTransformer(np.log1p, validate=False)),
        ('scale',   StandardScaler())
    ]), skewed_cols),

    # other numerics: median impute → scale
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale',   StandardScaler())
    ]), other_nums),

    # high-cardinality cats: mode impute → frequency encode
    ('cat_freq', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode',  FunctionTransformer(
            lambda df: df.apply(lambda col: col.map(col.value_counts(normalize=True))),
            validate=False))
    ]), high_card),

    # low-cardinality cats: mode impute → one-hot
    ('cat_ohe', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]), low_card),
], remainder='drop')

# ==== 6) Feature selector using RandomForestRegressor importances ====
selector = SelectFromModel(
    RandomForestRegressor(n_estimators=50, random_state=0),
    max_features=30,
    threshold=-np.inf
)

# ==== 7) Define regression pipelines (with feature selection) ====
pipelines = {
    'RandomForest': Pipeline([
        ('pre', preprocessor),
        ('sel', selector),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'HistGBM': Pipeline([
        ('pre', preprocessor),
        ('sel', selector),
        ('reg', HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42))
    ]),
}

# ==== 8) Prepare data ====
X_train = df_train.drop(columns=target_col)
y_train = df_train[target_col].astype(float)
X_test  = df_test.drop(columns=target_col, errors='ignore')
y_test  = df_test.get(target_col, None)

# ==== 9) 3‑fold CV and metrics ====
cv = KFold(n_splits=3, shuffle=True, random_state=42)
results = {}
for name, pipe in pipelines.items():
    neg_mse = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    mse     = -neg_mse
    rmse    = np.sqrt(mse)
    results[name] = {
        'MSE':  (mse.mean(), mse.std()),
        'RMSE': (rmse.mean(), rmse.std())
    }

print("=== CV Results (Regression) ===")
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE'][0]:.3f} ± {metrics['MSE'][1]:.3f} | "
          f"RMSE = {metrics['RMSE'][0]:.3f} ± {metrics['RMSE'][1]:.3f}")

# ==== 10) Select and train best model on full data ====
best_name = min(results, key=lambda k: results[k]['MSE'][0])
best_pipe = pipelines[best_name].fit(X_train, y_train)
print(f"\nBest model: {best_name}")

# ==== 11) Predict and evaluate on test ====
y_pred = best_pipe.predict(X_test)
if y_test is not None:
    mse_test  = mean_squared_error(y_test, y_pred)
    rmse_test = np.sqrt(mse_test)
    print("\n=== Test Set Evaluation ===")
    print(f"Test MSE:  {mse_test:.3f}")
    print(f"Test RMSE: {rmse_test:.3f}")
