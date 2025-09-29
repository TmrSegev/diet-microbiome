# SHAP_interactions.py
import os
import re
import gc
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import shap

# -----------------------
# Config
# -----------------------
outdir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
MODEL = 'LGBM'            # this script expects LGBM for interaction values
TARGETS = 'abundance'     # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
PROBLEM = 'regression'    # this script targets regression
SPECIES = 'segal_species' # 'segal_species' or 'mpa_species'

# The exact feature names as used in your modeling table before sanitization
FRUIT_FEATURE_NAME_RAW = 'Fruits'
FIBER_FEATURE_NAME_RAW = 'Fiber, total dietary'

# Compute interactions on this many rows per target for speed
NSAMPLE = 500
RANDOM_STATE = 1

# Also save how strongly fruit and fiber interact with every other feature
SAVE_PAIRWISE_WITH_ALL = True
TOPK = 25  # top K interactions to save per target per anchor feature

# -----------------------
# Helpers
# -----------------------
def sanitize_columns(df_or_list):
    """Apply the same sanitization your pipeline uses."""
    if isinstance(df_or_list, pd.DataFrame):
        df_or_list.columns = df_or_list.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        return df_or_list
    # list of names
    return [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in df_or_list]

def sanitize_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def load_lists(species, pathways_suffix):
    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{species}/my_lists{pathways_suffix}.pkl', 'rb') as fh:
        base_features, features, target_input = pickle.load(fh)
    features = sanitize_columns(features)
    target_input = sanitize_columns(target_input)
    return base_features, features, target_input

def load_data():
    reverse = 'reverse/' if PROBLEM == 'reverse' else ''
    pathways_suffix = '' if TARGETS != 'pathways' else '_pathways'

    train_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/{reverse}diet_mb{pathways_suffix}_baseline_train.pkl")
    test_baseline  = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/{reverse}diet_mb{pathways_suffix}_baseline_test.pkl")
    test_02_visit  = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways_suffix}_02_visit.pkl")

    train_baseline = sanitize_columns(train_baseline)
    test_baseline  = sanitize_columns(test_baseline)
    test_02_visit  = sanitize_columns(test_02_visit)

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

    return train_baseline, test_baseline, test_02_visit, features, loop_targets

def load_models():
    models_list = pickle.load(open(
        f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/{SPECIES}/models_{MODEL}_{TARGETS}_longitudinal.pkl',
        'rb'
    ))
    return models_list

def safe_subsample_df(df, n, random_state):
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=random_state)

def find_feature_index(features, raw_name):
    """
    Find the index of a feature by its raw name, accounting for your sanitization.
    Falls back to case-insensitive and substring heuristics if exact match not found.
    """
    target = sanitize_name(raw_name)
    # 1) exact sanitized match
    if target in features:
        return features.index(target), target

    # 2) case-insensitive exact
    lower = [f.lower() for f in features]
    if target.lower() in lower:
        return lower.index(target.lower()), features[lower.index(target.lower())]

    # 3) heuristics for fiber in case the exact sanitized changes
    if raw_name.lower().startswith('fiber'):
        # look for something like fiber_total_dietary
        candidates = [i for i, f in enumerate(features)
                      if 'fiber' in f.lower() and 'total' in f.lower() and 'dietary' in f.lower()]
        if candidates:
            return candidates[0], features[candidates[0]]

    # 4) heuristics for fruits
    if raw_name.lower() == 'fruits':
        candidates = [i for i, f in enumerate(features) if f.lower() == 'fruits']
        if candidates:
            return candidates[0], features[candidates[0]]

    raise ValueError(f"Feature '{raw_name}' not found in features. Tried sanitized='{target}'.")

def summarize_pair(inter_vals, i, j):
    """
    inter_vals: [n_samples, n_features, n_features]
    Returns:
      main_i_abs_mean, main_j_abs_mean, inter_ij_abs_mean
    """
    main_i = float(np.mean(np.abs(inter_vals[:, i, i])))
    main_j = float(np.mean(np.abs(inter_vals[:, j, j])))
    inter_ij = float(np.mean(np.abs(inter_vals[:, i, j])))
    return main_i, main_j, inter_ij

def vector_interactions_with_anchor(inter_vals, anchor_idx):
    """
    Mean abs interaction of a single anchor feature with every feature.
    Returns a vector [n_features]
    """
    v = np.mean(np.abs(inter_vals[:, anchor_idx, :]), axis=0)  # [n_features]
    return v

# -----------------------
# Core
# -----------------------
def compute_interactions_for_target(train_df, features, target, model,
                                    fruit_raw='Fruits',
                                    fiber_raw='Fiber, total dietary'):
    # rows with target present
    df = train_df.dropna(subset=[target])

    # subsample for speed
    df_s = safe_subsample_df(df, NSAMPLE, RANDOM_STATE)

    X = df_s[features]

    # identify indices of the two features of interest
    fruit_idx, fruit_name_sanitized = find_feature_index(features, fruit_raw)
    fiber_idx, fiber_name_sanitized = find_feature_index(features, fiber_raw)

    # compute interaction tensor
    explainer = shap.TreeExplainer(model)
    inter_vals = explainer.shap_interaction_values(X)  # [n_samples, n_features, n_features]

    # summarize the main and pair interaction
    fruit_main, fiber_main, fruit_fiber_inter = summarize_pair(inter_vals, fruit_idx, fiber_idx)

    # quick ratios
    interaction_to_fruit_main = np.nan if fruit_main == 0 else fruit_fiber_inter / fruit_main
    denom = fruit_main + fiber_main
    interaction_to_sum_main = np.nan if denom == 0 else fruit_fiber_inter / denom

    # correlations between the two input features in training data (full, not subsample)
    # use sanitized names to grab columns from full training df
    fruit_col = sanitize_name(fruit_raw)
    fiber_col = sanitize_name(fiber_raw)
    r_pearson, p_pearson = (np.nan, np.nan)
    r_spearman, p_spearman = (np.nan, np.nan)
    if fruit_col in train_df.columns and fiber_col in train_df.columns:
        sub = train_df[[fruit_col, fiber_col]].dropna()
        if len(sub) > 3:
            r_pearson, p_pearson = stats.pearsonr(sub[fruit_col], sub[fiber_col])
            r_spearman, p_spearman = stats.spearmanr(sub[fruit_col], sub[fiber_col])

    # optionally, interactions of each anchor with all features
    pairs_with_fruit = None
    pairs_with_fiber = None
    if SAVE_PAIRWISE_WITH_ALL:
        v_fruit = vector_interactions_with_anchor(inter_vals, fruit_idx)
        v_fiber = vector_interactions_with_anchor(inter_vals, fiber_idx)

        pairs_with_fruit = pd.DataFrame({
            'anchor_feature': fruit_name_sanitized,
            'other_feature': features,
            'mean_abs_interaction': v_fruit
        }).sort_values('mean_abs_interaction', ascending=False)

        pairs_with_fiber = pd.DataFrame({
            'anchor_feature': fiber_name_sanitized,
            'other_feature': features,
            'mean_abs_interaction': v_fiber
        }).sort_values('mean_abs_interaction', ascending=False)

    res = {
        'target': target,
        'n_train_used_for_interactions': inter_vals.shape[0],
        'fruit_feature': fruit_name_sanitized,
        'fiber_feature': fiber_name_sanitized,
        'fruit_main_abs_mean': fruit_main,
        'fiber_main_abs_mean': fiber_main,
        'fruit_fiber_interaction_abs_mean': fruit_fiber_inter,
        'interaction_to_fruit_main_ratio': interaction_to_fruit_main,
        'interaction_to_sum_main_ratio': interaction_to_sum_main,
        'pearson_r_fruit_fiber': r_pearson,
        'pearson_p_fruit_fiber': p_pearson,
        'spearman_rho_fruit_fiber': r_spearman,
        'spearman_p_fruit_fiber': p_spearman,
    }

    return res, pairs_with_fruit, pairs_with_fiber

def main():
    if MODEL != 'LGBM':
        raise ValueError("This script expects LightGBM models because shap_interaction_values is used.")

    # load data and models
    train_baseline, test_baseline, test_02_visit, features, loop_targets = load_data()
    models_list = load_models()

    # outputs
    summary_rows = []
    per_target_pairs_fruit = {}
    per_target_pairs_fiber = {}

    # iterate targets (species)
    for i, target in enumerate(loop_targets):
        model = models_list[i]
        if model is None:
            print(f"[skip] No model for {target}")
            continue

        print(f"[{i+1}/{len(loop_targets)}] target={target}")
        try:
            res, pairs_fruit, pairs_fiber = compute_interactions_for_target(
                train_df=train_baseline,
                features=features,
                target=target,
                model=model,
                fruit_raw=FRUIT_FEATURE_NAME_RAW,
                fiber_raw=FIBER_FEATURE_NAME_RAW
            )
            summary_rows.append(res)
            if SAVE_PAIRWISE_WITH_ALL:
                per_target_pairs_fruit[target] = pairs_fruit
                per_target_pairs_fiber[target] = pairs_fiber
        except Exception as e:
            print(f"Error on target {target}: {e}")

        gc.collect()

    # aggregate summary
    summary_df = pd.DataFrame(summary_rows).sort_values('interaction_to_fruit_main_ratio', ascending=True)

    # save
    out_dir = os.path.join(outdir, f"data/{PROBLEM}/{SPECIES}")
    os.makedirs(out_dir, exist_ok=True)

    summary_path_pkl = os.path.join(out_dir, f"shap_interactions_summary_{MODEL}_{TARGETS}.pkl")
    summary_path_csv = os.path.join(out_dir, f"shap_interactions_summary_{MODEL}_{TARGETS}.csv")
    summary_df.to_pickle(summary_path_pkl)
    summary_df.to_csv(summary_path_csv, index=False)

    if SAVE_PAIRWISE_WITH_ALL:
        pair_dir = os.path.join(out_dir, f"shap_interactions_pairwise_{MODEL}_{TARGETS}")
        os.makedirs(pair_dir, exist_ok=True)
        for tgt, df in per_target_pairs_fruit.items():
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', tgt)
            df.head(TOPK).to_csv(os.path.join(pair_dir, f"{safe_name}__top{TOPK}_with_{sanitize_name(FRUIT_FEATURE_NAME_RAW)}.csv"), index=False)
        for tgt, df in per_target_pairs_fiber.items():
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', tgt)
            df.head(TOPK).to_csv(os.path.join(pair_dir, f"{safe_name}__top{TOPK}_with_{sanitize_name(FIBER_FEATURE_NAME_RAW)}.csv"), index=False)

    print("Saved:")
    print("  Summary:", summary_path_csv)
    if SAVE_PAIRWISE_WITH_ALL:
        print("  Pairwise top lists:", os.path.join(out_dir, f"shap_interactions_pairwise_{MODEL}_{TARGETS}"))

if __name__ == "__main__":
    main()
