import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import re

# === Helper function to sanitize target names for filenames ===
def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '_', name)

# === Load Data ===
df = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_diet.pkl')

with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/target_phenotypes.pkl', 'rb') as f:
    target_phenotypes = pickle.load(f)

# === Setup ===
base_features = ['age', 'gender']
flat_targets = sum(target_phenotypes, [])
exclude_cols = set(flat_targets + base_features + ['RegistrationCode'])
microbial_features = [col for col in df.columns if col not in exclude_cols]

# Optional: Create output directory
output_dir = r"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/SHAP_plots/phenotypes_predictions"

# === SHAP Summary Plot for Each Target ===
for target in flat_targets:
    print(f"\n[SHAP] Processing target: {target}")
    safe_target = sanitize_filename(target)

    # Filter to non-NA target
    df_target = df[df[target].notna()].copy()
    y = df_target[target]
    X = df_target[base_features + microbial_features]

    # Use same logic as original model selection
    if len(y) < 3000:
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            n_jobs=8,
            verbosity=0
        )
    else:
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=800,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            n_jobs=8,
            verbosity=0
        )

    # Train on full data
    model.fit(X, y)

    # SHAP explainer and values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # SHAP summary plot
    shap.summary_plot(
        shap_values.values,
        features=X,
        feature_names=X.columns,
        show=False
    )
    plt.title(target)

    plt.gcf().axes[-1].set_aspect('auto')
    plt.tight_layout()
    # As mentioned, smaller "box_aspect" value to make colorbar thicker
    plt.gcf().axes[-1].set_box_aspect(100) 

    plt.tight_layout()

    plt.savefig(f"{output_dir}/diet/{safe_target}.png", dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
    plt.savefig(f"{output_dir}/diet/{safe_target}.pdf", dpi=300, facecolor="white", transparent=False, bbox_inches='tight')

    plt.show()
    plt.close()