# import pandas as pd
# import pickle
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from scipy.stats import pearsonr
# from statsmodels.stats.multitest import fdrcorrection
# from functools import reduce
# import matplotlib.pyplot as plt

# phenotypes_mb = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_mb.pkl')

# with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/target_phenotypes.pkl', 'rb') as f:
#     target_phenotypes = pickle.load(f)

# # === Setup ===
# df = phenotypes_mb.copy()
# initial_samples_num = df.shape[0]
# base_features = ['age', 'sex']

# # Encode sex if needed
# if df['sex'].dtype == 'object':
#     df['sex'] = LabelEncoder().fit_transform(df['sex'])

# flat_targets = sum(target_phenotypes, [])
# microbial_features = [col for col in df.columns if col not in flat_targets + base_features + ['RegistrationCode']]

# metrics = []
# all_predictions = []

# # === Loop over all targets ===
# for target in flat_targets:
#     print(f"\n=== Target: {target} ===")

#     # Filter to rows with non-missing target
#     df_target = df[df[target].notna()].copy()

#     print(f'{df_target.shape[0]} / {initial_samples_num} samples left with non-NA values')

#     y_true = df_target[target]
#     X_base = df_target[base_features]
#     X_combined = df_target[base_features + microbial_features]

#     def get_model(n_samples):
#         # For modalities with less samples like blood tests
#         if n_samples < 3000:
#             return XGBRegressor(
#                 objective='reg:squarederror',
#                 n_estimators=300,           
#                 learning_rate=0.05,         
#                 max_depth=3,                
#                 subsample=0.9,              
#                 colsample_bytree=0.8,       
#                 min_child_weight=5,         
#                 n_jobs=8,                   
#                 verbosity=0
#             )
#         else:
#             return XGBRegressor(
#                 objective='reg:squarederror',
#                 n_estimators=800,           
#                 learning_rate=0.01,         
#                 max_depth=3,                
#                 subsample=0.9,
#                 colsample_bytree=0.8,
#                 min_child_weight=5,
#                 n_jobs=8,
#                 verbosity=0
#             )

#     # OOF prediction
#     def get_oof_preds(X, y):
#         y_pred = np.zeros(len(y))
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         for train_idx, test_idx in kf.split(X):
#             model = get_model(len(y))
#             model.fit(X.iloc[train_idx], y.iloc[train_idx])
#             y_pred[test_idx] = model.predict(X.iloc[test_idx])
#         return y_pred

#     y_pred_base = get_oof_preds(X_base, y_true)
#     y_pred_combined = get_oof_preds(X_combined, y_true)

#     r_base, p_base = pearsonr(y_true, y_pred_base)
#     r_combined, p_combined = pearsonr(y_true, y_pred_combined)

#     metrics.append({
#         'target': target,
#         'r_base': r_base,
#         'r_combined': r_combined,
#         'delta_r': r_combined - r_base,
#         'p_base': max(p_base, 1e-308),
#         'p_combined': max(p_combined, 1e-308)
#     })

#     all_predictions.append(pd.DataFrame({
#         'RegistrationCode': df_target['RegistrationCode'],
#         f'{target}_prediction': y_pred_combined,
#     }))

# # === Merge predictions
# final_df = reduce(lambda left, right: pd.merge(left, right, on='RegistrationCode', how='outer'), all_predictions)

# final_df = final_df.merge(df[['RegistrationCode', 'age', 'sex']], on='RegistrationCode', how='left')
# cols = ['RegistrationCode', 'age', 'sex'] + [col for col in final_df.columns if col not in ['RegistrationCode', 'age', 'sex']]
# final_df = final_df[cols]

# final_df.to_pickle(path='/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_mb_predictions.pkl')

# # === Summary metrics
# results_df = pd.DataFrame(metrics)
# results_df['fdr_base'] = fdrcorrection(results_df['p_base'])[1]
# results_df['fdr_combined'] = fdrcorrection(results_df['p_combined'])[1]
# results_df = results_df[['target', 'r_base', 'r_combined', 'delta_r', 'fdr_base', 'fdr_combined']]
# results_df = results_df.sort_values('delta_r', ascending=False).reset_index(drop=True)

# # Display nicely formatted summary
# results_df.to_pickle(path='/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_results_df.pkl')
# final_df.to_pickle(path='/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_final_df.pkl')

import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
from functools import reduce
import matplotlib.pyplot as plt

# Load data
phenotypes_mb = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_mb.pkl')
with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/target_phenotypes.pkl', 'rb') as f:
    target_phenotypes = pickle.load(f)

df = phenotypes_mb.copy()
initial_samples_num = df.shape[0]
base_features = ['age', 'sex']

# Encode sex if needed
if df['sex'].dtype == 'object':
    df['sex'] = LabelEncoder().fit_transform(df['sex'])

flat_targets = sum(target_phenotypes, [])

microbial_features = [col for col in df.columns if col not in flat_targets + base_features + ['RegistrationCode']]

metrics = []
all_predictions = []

# Loop over targets
for target in flat_targets:
    print(f"\n=== Target: {target} ===")

    df_target = df[df[target].notna()].copy()
    print(f'{df_target.shape[0]} / {initial_samples_num} samples left with non-NA values')

    y_true = df_target[target]
    X_base = df_target[base_features]
    X_combined = df_target[base_features + microbial_features]
    print(X_combined.columns)

    def get_model(n_samples):
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

    def get_oof_preds(X, y):
        y_pred = np.zeros(len(y))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            model = get_model(len(y))
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred[test_idx] = model.predict(X.iloc[test_idx])
        return y_pred

    y_pred_base = get_oof_preds(X_base, y_true)
    y_pred_combined = get_oof_preds(X_combined, y_true)

    r_base, p_base = pearsonr(y_true, y_pred_base)
    r_combined, p_combined = pearsonr(y_true, y_pred_combined)

    metrics.append({
        'target': target,
        'r_base': r_base,
        'r_combined': r_combined,
        'delta_r': r_combined - r_base,
        'p_base': max(p_base, 1e-308),
        'p_combined': max(p_combined, 1e-308)
    })

    all_predictions.append(pd.DataFrame({
        'RegistrationCode': df_target['RegistrationCode'],
        f'{target}_prediction': y_pred_combined,
    }))

# Merge predictions
final_df = reduce(lambda left, right: pd.merge(left, right, on='RegistrationCode', how='outer'), all_predictions)
final_df = final_df.merge(df[['RegistrationCode', 'age', 'sex']], on='RegistrationCode', how='left')
cols = ['RegistrationCode', 'age', 'sex'] + [col for col in final_df.columns if col not in ['RegistrationCode', 'age', 'sex']]
final_df = final_df[cols]

final_df.to_pickle(path='/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_mb_predictions.pkl')

# Summary metrics
results_df = pd.DataFrame(metrics)
results_df['fdr_base'] = fdrcorrection(results_df['p_base'])[1]
results_df['fdr_combined'] = fdrcorrection(results_df['p_combined'])[1]
results_df = results_df[['target', 'r_base', 'r_combined', 'delta_r', 'fdr_base', 'fdr_combined']]
results_df = results_df.sort_values('delta_r', ascending=False).reset_index(drop=True)

results_df.to_pickle(path='/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_results_df.pkl')
final_df.to_pickle(path='/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_final_df.pkl')
