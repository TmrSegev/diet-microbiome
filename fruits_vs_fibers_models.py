# reduced_feature_models.py
import os
import re
import gc
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

# -----------------------
# Config
# -----------------------
DATA_ROOT = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
RESULT_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/results/fiber_fruits'

TARGETS = 'abundance'     # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
PROBLEM = 'regression'
SPECIES = 'segal_species'

# Exact raw names as they appear before sanitization
FRUIT_FEATURE_NAME_RAW = 'Fruits'
FIBER_FEATURE_NAME_RAW = 'Fiber, total dietary'

# Covariates
ADJUST_FOR_COVARS = True
USE_AGE = True
USE_SEX_OR_GENDER = True   # detects 'sex' or 'gender' and encodes male=1, female=0

# Optional subset of targets
USE_TARGETS_SUBSET = False
TARGETS_SUBSET = []        # e.g. ['Bifidobacterium longum', 'Lawsonibacter asaccharolyticus']

MIN_TRAIN_ROWS = 30
MIN_TEST_ROWS = 30

# -----------------------
# Helpers
# -----------------------
def sanitize_columns(df_or_list):
    if isinstance(df_or_list, pd.DataFrame):
        df_or_list.columns = df_or_list.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        return df_or_list
    return [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in df_or_list]

def sanitize_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def load_lists(species, pathways_suffix):
    with open(f'{DATA_ROOT}data/{species}/my_lists{pathways_suffix}.pkl', 'rb') as fh:
        base_features, features, target_input = pickle.load(fh)
    features = sanitize_columns(features)
    target_input = sanitize_columns(target_input)
    return base_features, features, target_input

def load_data():
    reverse = 'reverse/' if PROBLEM == 'reverse' else ''
    pathways_suffix = '' if TARGETS != 'pathways' else '_pathways'

    train_baseline = pd.read_pickle(f"{DATA_ROOT}data/{SPECIES}/{reverse}diet_mb{pathways_suffix}_baseline_train.pkl")
    test_baseline  = pd.read_pickle(f"{DATA_ROOT}data/{SPECIES}/{reverse}diet_mb{pathways_suffix}_baseline_test.pkl")

    train_baseline = sanitize_columns(train_baseline)
    test_baseline  = sanitize_columns(test_baseline)

    base_features, features, target_input = load_lists(SPECIES, pathways_suffix)

    if PROBLEM == 'reverse':
        diet_features = features
        features = target_input + ['Richness', 'Shannon_diversity'] + base_features
        target_input = [feat for feat in diet_features if feat not in ['age', 'gender']]

    if TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS in ['abundance', 'pathways'] or PROBLEM == 'reverse':
        loop_targets = target_input
    elif TARGETS == 'health':
        loop_targets = ['modified_HACK_top17_score', 'GMWI2_score']
    else:
        loop_targets = target_input

    return train_baseline, test_baseline, features, loop_targets

def find_feature_index(features, raw_name):
    target = sanitize_name(raw_name)
    if target in features:
        return features.index(target), target
    lower = [f.lower() for f in features]
    if target.lower() in lower:
        return lower.index(target.lower()), features[lower.index(target.lower())]
    raise ValueError(f"Feature '{raw_name}' not found after sanitization as '{target}'.")

def pick_covariates(df):
    covars = []
    if ADJUST_FOR_COVARS:
        if USE_AGE and 'age' in df.columns:
            covars.append('age')
        if USE_SEX_OR_GENDER:
            if 'sex' in df.columns:
                covars.append('sex')
            elif 'gender' in df.columns:
                covars.append('gender')
    return covars

def encode_male1_female0(series):
    # If already numeric, try to coerce but only remap if values are {0,1} and you want to flip
    if pd.api.types.is_numeric_dtype(series):
        return series.astype('float64')
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        'male': 1, 'm': 1, '1': 1, 'true': 1,
        'female': 0, 'f': 0, '0': 0, 'false': 0
    }
    return s.map(mapping).astype('float64')

def fit_and_eval_linear(train_df, test_df, target, predictors, standardize=True):
    cols_needed = predictors + [target]
    df_tr = train_df[cols_needed].dropna()
    df_te = test_df[cols_needed].dropna()

    if len(df_tr) < MIN_TRAIN_ROWS or len(df_te) < MIN_TEST_ROWS:
        return None

    X_tr = df_tr[predictors].copy()
    X_te = df_te[predictors].copy()

    # Encode sex or gender to male=1, female=0 if present and not numeric
    for col in X_tr.columns:
        if col.lower() in ['sex', 'gender']:
            X_tr[col] = encode_male1_female0(X_tr[col])
            X_te[col] = encode_male1_female0(X_te[col])

    if standardize:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
    else:
        X_tr_scaled = X_tr.values
        X_te_scaled = X_te.values

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_tr_scaled, df_tr[target].values)

    y_pred = lr.predict(X_te_scaled)
    y_true = df_te[target].values

    r, p = stats.pearsonr(y_pred, y_true) if len(y_true) > 2 else (np.nan, np.nan)
    r2 = r2_score(y_true, y_pred)

    coefs = dict(zip(predictors, lr.coef_.tolist()))
    intercept = float(lr.intercept_)

    return {
        'n_train': len(df_tr),
        'n_test': len(df_te),
        'pearson_r': float(r),
        'pearson_p': float(p),
        'r2': float(r2),
        'coefficients': coefs,
        'intercept': intercept,
    }

# -----------------------
# Core
# -----------------------
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    train_df, test_df, features, loop_targets = load_data()

    if USE_TARGETS_SUBSET and len(TARGETS_SUBSET) > 0:
        target_set = set([sanitize_name(t) for t in TARGETS_SUBSET])
        loop_targets = [t for t in loop_targets if sanitize_name(t) in target_set]

    # Locate fruit and fiber
    fruit_idx, fruit_name = find_feature_index(features, FRUIT_FEATURE_NAME_RAW)
    fiber_idx, fiber_name = find_feature_index(features, FIBER_FEATURE_NAME_RAW)

    results = []
    for i, target in enumerate(loop_targets):
        print(f"[{i+1}/{len(loop_targets)}] target={target}")

        covars = pick_covariates(train_df)

        preds_fiber = [fiber_name] + covars
        preds_fruit = [fruit_name] + covars

        # Build trimmed frames with needed cols
        needed_cols = list(set(preds_fiber + preds_fruit + [target]))
        df_tr = train_df[needed_cols + [fruit_name, fiber_name]].dropna().copy()
        df_te = test_df[needed_cols + [fruit_name, fiber_name]].dropna().copy()

        if len(df_tr) < MIN_TRAIN_ROWS or len(df_te) < MIN_TEST_ROWS:
            print("  Skipped due to insufficient rows")
            continue

        # Standardize fruit and fiber to create interaction term
        zscaler = StandardScaler()
        FF_tr = zscaler.fit_transform(df_tr[[fruit_name, fiber_name]])
        FF_te = zscaler.transform(df_te[[fruit_name, fiber_name]])
        df_tr['__fruit_z__'] = FF_tr[:, 0]
        df_tr['__fiber_z__'] = FF_tr[:, 1]
        df_te['__fruit_z__'] = FF_te[:, 0]
        df_te['__fiber_z__'] = FF_te[:, 1]
        df_tr['__fruit_x_fiber__'] = df_tr['__fruit_z__'] * df_tr['__fiber_z__']
        df_te['__fruit_x_fiber__'] = df_te['__fruit_z__'] * df_te['__fiber_z__']

        covars_present = [c for c in covars if c in df_tr.columns]

        # 1) Fiber only
        res1 = fit_and_eval_linear(
            train_df=df_tr, test_df=df_te, target=target,
            predictors=[fiber_name] + covars_present, standardize=True
        )

        # 2) Fruit only
        res2 = fit_and_eval_linear(
            train_df=df_tr, test_df=df_te, target=target,
            predictors=[fruit_name] + covars_present, standardize=True
        )

        # 3) Fruit + Fiber + Interaction
        res3 = fit_and_eval_linear(
            train_df=df_tr, test_df=df_te, target=target,
            predictors=[fruit_name, fiber_name, '__fruit_x_fiber__'] + covars_present,
            standardize=True
        )

        for name, res in [('fiber_only', res1), ('fruit_only', res2), ('fruit_fiber_inter', res3)]:
            if res is None:
                continue
            results.append({
                'target': target,
                'model': name,
                'n_train': res['n_train'],
                'n_test': res['n_test'],
                'pearson_r': res['pearson_r'],
                'pearson_p': res['pearson_p'],
                'r2': res['r2'],
                'coefficients': res['coefficients'],
                'intercept': res['intercept'],
                'fruit_feature': fruit_name,
                'fiber_feature': fiber_name,
                'adjusted_for_covars': ADJUST_FOR_COVARS,
                'covariates_used': covars_present,
                'sex_encoding_note': 'male=1, female=0'
            })

        gc.collect()

    if len(results) == 0:
        print("No results computed. Check data availability and feature names.")
        return

    res_df = pd.DataFrame(results)

    # Deltas vs fiber_only
    def add_deltas(group):
        base = group[group['model'] == 'fiber_only']
        if base.empty:
            group['delta_r2_vs_fiber'] = np.nan
            group['delta_r_vs_fiber'] = np.nan
            return group
        base_r2 = float(base['r2'].iloc[0])
        base_r  = float(base['pearson_r'].iloc[0])
        group['delta_r2_vs_fiber'] = group['r2'] - base_r2
        group['delta_r_vs_fiber']  = group['pearson_r'] - base_r
        return group

    res_df = res_df.groupby('target', group_keys=False).apply(add_deltas)

    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f"reduced_feature_models_{TARGETS}{'_adjcov' if ADJUST_FOR_COVARS else ''}.csv")
    pkl_path = os.path.join(RESULT_DIR, f"reduced_feature_models_{TARGETS}{'_adjcov' if ADJUST_FOR_COVARS else ''}.pkl")

    res_df.to_csv(csv_path, index=False)
    res_df.to_pickle(pkl_path)

    print("Saved results to:")
    print(" ", csv_path)
    print(" ", pkl_path)

if __name__ == "__main__":
    main()
