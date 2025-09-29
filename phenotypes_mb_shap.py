import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import re
import os
from lightgbm import LGBMRegressor  # ✅ LightGBM instead of XGBoost

SPECIES = "segal_species"

# === Helper function to sanitize target names for filenames ===
def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '_', name)

# === Load Data ===
df = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_mb.pkl')

with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/target_phenotypes.pkl', 'rb') as f:
    target_phenotypes = pickle.load(f)

# === Setup ===
base_features = ['age', 'sex']
flat_targets = sum(target_phenotypes, [])
predict_subset = ['liver_sound_speed_mps', 'total_scan_vat_mass', 'bt__triglycerides']
exclude_cols = set(flat_targets + base_features + ['RegistrationCode'])
microbial_features = [col for col in df.columns if col not in exclude_cols]

# Output paths
shap_output_path = "/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/SHAP_plots/phenotypes_predictions/new_lgbm_subset"

# Store lists of lists: rows = targets, columns = samples, each cell = list of SHAP values
shap_matrix = []
models_dict = {}

for target in predict_subset:
    print(f"\n[SHAP] Processing target: {target}")
    safe_target = sanitize_filename(target)

    df_target = df[df[target].notna()].copy()
    y = df_target[target]
    X = df_target[base_features + microbial_features]

    # Select model based on sample size
    if len(y) < 3000:
        model = LGBMRegressor(
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
        model = LGBMRegressor(
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

    model.fit(X, y)
    models_dict[target] = model 
    models_dict[safe_target] = model 

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_array = shap_values.values  # shape (n_samples, n_features)

    # Create figure manually to control DPI
    fig = plt.figure(figsize=(10, 9), dpi=300)

    shap.summary_plot(
        shap_array,
        features=X,
        feature_names=X.columns,
        show=False
    )

    # Font adjustments
    plt.title(target, fontsize=18)
    plt.xlabel("SHAP Value", fontsize=16)
    # plt.ylabel("Features", fontsize=16)

    # Tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)

    cbar_ax = plt.gcf().axes[-1]

    # Force update of colorbar title with bigger font
    for ax in plt.gcf().axes:
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Feature value':
            ax.set_ylabel('Feature value', fontsize=16)

    # Set tick font size ("Low", "High")
    cbar_ax.tick_params(labelsize=14)

    plt.title(target, fontsize=16)
    plt.gcf().axes[-1].set_aspect('auto')
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.tight_layout()

    plot_path_base = os.path.join(shap_output_path, safe_target)
    plt.savefig(f"{plot_path_base}.png", dpi=300, facecolor="white", bbox_inches='tight')
    plt.savefig(f"{plot_path_base}.pdf", dpi=300, facecolor="white", bbox_inches='tight')

    plt.close()
    # plt.show()

    # Save this row: a list of SHAP vectors (one per person)
    row = [list(shap_array[i]) for i in range(shap_array.shape[0])]
    shap_matrix.append(row)

# Save models
save_dir = f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{SPECIES}/'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'models_mb_phenotypes_subset.pkl')

print(f"\nSaving models dictionary to: {model_save_path}")
with open(model_save_path, 'wb') as f:
    pickle.dump(models_dict, f)

# Save SHAP values
shap_df = pd.DataFrame(shap_matrix)
shap_df.columns = [f'column{i}' for i in shap_df.columns]
shap_df.index = range(len(shap_df))
shap_df.to_pickle(shap_output_path)

print(f"\n✅ Saved SHAP matrix with shape {shap_df.shape} to:\n{shap_output_path}")
