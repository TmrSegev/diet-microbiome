import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
from functools import reduce
import matplotlib.pyplot as plt
import re

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
data_dir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/'
phenotype_df_path = data_dir + 'phenotypes_diet_microbiome_new_study_ids.pkl'
lists_path = data_dir + 'my_lists_new_study_ids.pkl'

# Outputs
results_path = data_dir + 'phenotypes_ablation_results_df_new_study_ids.pkl'
predictions_path = data_dir + 'phenotypes_ablation_predictions_new_study_ids.pkl'
figure_path = data_dir + 'phenotypes_ablation_barplot_new_study_ids.png'

# --------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------
df = pd.read_pickle(phenotype_df_path)
initial_samples_num = df.shape[0]

with open(lists_path, 'rb') as f:
    base_features, diet_features, target_phenotypes, microbial_features = pickle.load(f)

# --------------------------------------------------------------------
# Sanitize feature names for LightGBM (no special JSON chars)
# --------------------------------------------------------------------
def sanitize_names_in_df(df, feature_lists):
    """
    Replace non-alphanumeric/underscore chars in feature names with '_'
    and update df + feature lists accordingly.
    """
    # Flatten list of lists -> set of all feature names to be sanitized
    all_feats = []
    for lst in feature_lists:
        all_feats.extend(lst)
    all_feats = list(dict.fromkeys(all_feats))  # preserve order, remove dups

    mapping = {}
    used_new_names = set(df.columns)  # to avoid collisions with existing names

    for old in all_feats:
        if old not in df.columns:
            # Might happen if some list elements are outdated; just skip
            continue
        # Replace anything that's not a-zA-Z0-9_ with '_'
        new = re.sub(r'[^0-9a-zA-Z_]', '_', old)

        # Ensure uniqueness
        if new in used_new_names and new != old:
            # Append a short hash to disambiguate
            suffix = abs(hash(old)) % 10000
            new = f"{new}_{suffix}"

        mapping[old] = new
        used_new_names.add(new)

    # Actually rename df
    df = df.rename(columns=mapping)

    # Update feature lists
    def remap_list(lst):
        return [mapping.get(col, col) for col in lst]

    base_new = remap_list(base_features)
    diet_new = remap_list(diet_features)
    micro_new = remap_list(microbial_features)

    return df, base_new, diet_new, micro_new, mapping

# Apply sanitization to base + diet + micro features
df, base_features, diet_features, microbial_features, name_mapping = sanitize_names_in_df(
    df,
    [base_features, diet_features, microbial_features]
)

# (target_phenotypes and RegistrationCode are untouched – they are not model features)

metrics = []
all_predictions = []

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def get_model(n_samples: int) -> LGBMRegressor:
    """Return a LightGBM regressor with different n_estimators by sample size."""
    if n_samples < 3000:
        return LGBMRegressor(
            objective='regression',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_samples=5,
            n_jobs=8,
            verbose=-1
        )
    else:
        return LGBMRegressor(
            objective='regression',
            n_estimators=800,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_samples=5,
            n_jobs=8,
            verbose=-1
        )

def get_oof_preds(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """5-fold OOF predictions for a given feature matrix X and target y."""
    y_pred = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_samples = len(y)

    for train_idx, test_idx in kf.split(X):
        model = get_model(n_samples)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = model.predict(X.iloc[test_idx])

    return y_pred

# --------------------------------------------------------------------
# Main loop: build 4 models per phenotype (ablation)
# --------------------------------------------------------------------
for target in target_phenotypes:
    print(f"\n=== Target: {target} ===")

    if target not in df.columns:
        print(f"  [WARNING] Target {target} not found in df.columns, skipping.")
        continue

    # Keep only non-missing target rows
    df_target = df[df[target].notna()].copy()
    print(f'{df_target.shape[0]} / {initial_samples_num} samples left with non-NA values')

    if df_target.shape[0] < 20:
        print("  [WARNING] Too few samples, skipping.")
        continue

    y_true = df_target[target]

    # Feature sets
    X_base = df_target[base_features]

    X_base_diet = df_target[base_features + diet_features]

    X_base_micro = df_target[base_features + microbial_features]

    full_cols = list(dict.fromkeys(base_features + diet_features + microbial_features))
    X_full = df_target[full_cols]

    # OOF predictions for each model
    y_pred_base = get_oof_preds(X_base, y_true)
    y_pred_diet = get_oof_preds(X_base_diet, y_true)
    y_pred_micro = get_oof_preds(X_base_micro, y_true)
    y_pred_full = get_oof_preds(X_full, y_true)

    # Correlations and p-values
    r_base, p_base = pearsonr(y_true, y_pred_base)
    r_diet, p_diet = pearsonr(y_true, y_pred_diet)
    r_micro, p_micro = pearsonr(y_true, y_pred_micro)
    r_full, p_full = pearsonr(y_true, y_pred_full)

    # Store metrics
    metrics.append({
        'target': target,
        'r_base': r_base,
        'r_diet': r_diet,
        'r_micro': r_micro,
        'r_full': r_full,
        'delta_r_diet': r_diet - r_base,
        'delta_r_micro': r_micro - r_base,
        'delta_r_full': r_full - r_base,
        'p_base': max(p_base, 1e-308),
        'p_diet': max(p_diet, 1e-308),
        'p_micro': max(p_micro, 1e-308),
        'p_full': max(p_full, 1e-308)
    })

    # Store predictions (all 4 models) for potential downstream use
    all_predictions.append(pd.DataFrame({
        'RegistrationCode': df_target['RegistrationCode'],
        f'{target}_pred_base': y_pred_base,
        f'{target}_pred_diet': y_pred_diet,
        f'{target}_pred_micro': y_pred_micro,
        f'{target}_pred_full': y_pred_full,
    }))

# --------------------------------------------------------------------
# Merge predictions across targets
# --------------------------------------------------------------------
if len(all_predictions) > 0:
    final_df = reduce(lambda left, right: pd.merge(left, right, on='RegistrationCode', how='outer'),
                      all_predictions)

    # Add base features (age/sex/BMI etc.) for convenience
    info_cols = ['RegistrationCode'] + base_features
    info_cols = list(dict.fromkeys(info_cols))
    final_df = final_df.merge(df[info_cols], on='RegistrationCode', how='left')

    # Reorder columns: RegistrationCode, base_features, then predictions
    pred_cols = [c for c in final_df.columns if c not in ['RegistrationCode'] + base_features]
    final_df = final_df[['RegistrationCode'] + base_features + pred_cols]

    # # Save predictions
    final_df.to_pickle(path=predictions_path)
else:
    print("No predictions were generated (no valid targets?). Skipping saving predictions.")
    final_df = None

# --------------------------------------------------------------------
# Build metrics DataFrame + FDR + sorting
# --------------------------------------------------------------------
results_df = pd.DataFrame(metrics)

if not results_df.empty:
    # FDR corrections per model
    for col in ['p_base', 'p_diet', 'p_micro', 'p_full']:
        fdr_col = 'fdr_' + col.split('_')[1]
        results_df[fdr_col] = fdrcorrection(results_df[col])[1]

    # Reorder columns nicely
    results_df = results_df[[
        'target',
        'r_base', 'r_diet', 'r_micro', 'r_full',
        'delta_r_diet', 'delta_r_micro', 'delta_r_full',
        'p_base', 'p_diet', 'p_micro', 'p_full',
        'fdr_base', 'fdr_diet', 'fdr_micro', 'fdr_full'
    ]]

    # Sort by overall gain when using everything (full model vs base)
    results_df = results_df.sort_values('delta_r_full', ascending=False).reset_index(drop=True)

    # Save metrics
    results_df.to_pickle(path=results_path)

    # ----------------------------------------------------------------
    # Visualization: grouped barplot for top-N phenotypes
    # ----------------------------------------------------------------
    top_n = 15  # adjust if you want more/less in the figure
    plot_df = results_df.head(top_n).copy()

    x = np.arange(len(plot_df))  # phenotype indices
    width = 0.2

    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5 * width, plot_df['r_base'], width, label='Base (age/sex/BMI)')
    plt.bar(x - 0.5 * width, plot_df['r_diet'], width, label='Base + diet')
    plt.bar(x + 0.5 * width, plot_df['r_micro'], width, label='Base + microbiome')
    plt.bar(x + 1.5 * width, plot_df['r_full'], width, label='Base + diet + microbiome')

    plt.xticks(x, plot_df['target'], rotation=90)
    plt.ylabel('Pearson r (OOF)')
    plt.title('Ablation: predictive performance by feature set (top phenotypes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()
else:
    print("results_df is empty, skipping FDR and plotting.")

