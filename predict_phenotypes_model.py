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
import os

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
home_dir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
phenotype_df_path = home_dir + 'data/phenotypes_diet_microbiome_cmi_new.pkl'
lists_path = home_dir + 'data/my_lists_new.pkl'

# Outputs
results_path = home_dir + 'data/phenotypes_ablation_results_df_cmi_new.pkl'
predictions_path = home_dir + 'data/phenotypes_ablation_predictions_cmi_new.pkl'
figure_path = home_dir + 'figures/diet_intervention/phenotypes_ablation_barplot_cmi_new.png'
models_out_path = '/net/mraid20/export/genie/LabData/Analyses/barakdan/models_mb_phenotypes_cmi_new.pkl'

os.makedirs(os.path.dirname(models_out_path), exist_ok=True)

# --------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------
df = pd.read_pickle(phenotype_df_path)

initial_samples_num = df.shape[0]

with open(lists_path, 'rb') as f:
    base_features, diet_features, target_phenotypes, microbial_features = pickle.load(f)

target_phenotypes.append('CMI_PC1') 

# --------------------------------------------------------------------
# Sanitize feature names (for LightGBM safety)
# --------------------------------------------------------------------
def sanitize_names_in_df(df, feature_lists):
    all_feats = [f for sub in feature_lists for f in sub]
    mapping = {}
    used_new_names = set(df.columns)

    for old in dict.fromkeys(all_feats):
        if old not in df.columns:
            continue
        new = re.sub(r'[^0-9a-zA-Z_]', '_', old)
        if new in used_new_names and new != old:
            suffix = abs(hash(old)) % 10000
            new = f"{new}_{suffix}"
        mapping[old] = new
        used_new_names.add(new)

    df = df.rename(columns=mapping)

    def remap(lst): return [mapping.get(x, x) for x in lst]
    base_new, diet_new, micro_new = map(remap, [base_features, diet_features, microbial_features])
    return df, base_new, diet_new, micro_new, mapping

df, base_features, diet_features, microbial_features, name_mapping = sanitize_names_in_df(
    df, [base_features, diet_features, microbial_features]
)

metrics, all_predictions = [], {}
phenotypes_models_dict = {}

# --------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------
def get_model(n_samples: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective='regression',
        n_estimators=300 if n_samples < 3000 else 800,
        learning_rate=0.05 if n_samples < 3000 else 0.01,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_samples=5,
        n_jobs=8,
        verbose=-1
    )

def get_oof_preds(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    y_pred = np.zeros(len(y))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_samples = len(y)
    for tr, te in kf.split(X):
        model = get_model(n_samples)
        model.fit(X.iloc[tr], y.iloc[tr])
        y_pred[te] = model.predict(X.iloc[te])
    return y_pred

# --------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------
for target in target_phenotypes:
    print(f"\n=== Target: {target} ===")
    if target not in df.columns:
        print(f"  [WARNING] Target {target} not found, skipping.")
        continue

    df_t = df[df[target].notna()].copy()
    if df_t.shape[0] < 20:
        print("  [WARNING] Too few samples, skipping.")
        continue

    y_true = df_t[target]
    X_base = df_t[base_features]
    X_diet = df_t[base_features + diet_features]
    X_micro = df_t[base_features + microbial_features]
    X_full = df_t[list(dict.fromkeys(base_features + diet_features + microbial_features))]

    # Fit final phenotype model (Base + Microbiome) for simulation
    final_model = get_model(len(df_t))
    final_model.fit(X_micro, y_true)
    phenotypes_models_dict[target] = final_model

    # Compute out-of-fold predictions for ablation plot
    preds = {
        'base': get_oof_preds(X_base, y_true),
        'diet': get_oof_preds(X_diet, y_true),
        'micro': get_oof_preds(X_micro, y_true),
        'full': get_oof_preds(X_full, y_true),
    }

    # Correlations
    corr = {m: pearsonr(y_true, preds[m]) for m in preds}
    metrics.append({
        'target': target,
        **{f"r_{m}": corr[m][0] for m in preds},
        **{f"p_{m}": max(corr[m][1], 1e-308) for m in preds},
        'delta_r_diet': corr['diet'][0] - corr['base'][0],
        'delta_r_micro': corr['micro'][0] - corr['base'][0],
        'delta_r_full': corr['full'][0] - corr['base'][0],
    })

    all_predictions[target] = pd.DataFrame({
        'RegistrationCode': df_t['RegistrationCode'],
        **{f"{target}_pred_{m}": preds[m] for m in preds},
    })

# --------------------------------------------------------------------
# Save models dict + metadata
# --------------------------------------------------------------------
# payload = {
#     "models": phenotypes_models_dict,
#     "name_mapping": name_mapping,
#     "base_features": base_features,
#     "microbial_features": microbial_features
# }

payload = phenotypes_models_dict

with open(models_out_path, 'wb') as f:
    pickle.dump(payload, f)
print(f"✅ Saved phenotype models and metadata to: {models_out_path}")

# --------------------------------------------------------------------
# Save predictions + results
# --------------------------------------------------------------------
if all_predictions:
    final_df = reduce(lambda l, r: pd.merge(l, r, on='RegistrationCode', how='outer'),
                      all_predictions.values())
    info_cols = ['RegistrationCode'] + base_features
    final_df = final_df.merge(df[info_cols], on='RegistrationCode', how='left')
    final_df.to_pickle(predictions_path)

results_df = pd.DataFrame(metrics)
if not results_df.empty:
    for col in ['p_base', 'p_diet', 'p_micro', 'p_full']:
        results_df['fdr_' + col.split('_')[1]] = fdrcorrection(results_df[col])[1]
    results_df = results_df.sort_values('delta_r_full', ascending=False)
    results_df.to_pickle(results_path)

    # Plot
    plot_df = results_df.head(15)
    x = np.arange(len(plot_df)); w = 0.2
    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5*w, plot_df['r_base'], w, label='Base')
    plt.bar(x - 0.5*w, plot_df['r_diet'], w, label='Base+Diet')
    plt.bar(x + 0.5*w, plot_df['r_micro'], w, label='Base+Microbiome')
    plt.bar(x + 1.5*w, plot_df['r_full'], w, label='Full')
    plt.xticks(x, plot_df['target'], rotation=90)
    plt.ylabel('Pearson r (OOF)')
    plt.title('Ablation: predictive performance by feature set')
    plt.legend(); plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.show()
