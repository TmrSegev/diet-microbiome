import pickle
import pandas as pd
from sklearn.model_selection import KFold
from scipy import stats
import re
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

outdir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
MODEL = 'LGBM' # 'LGBM' or 'ridge' or 'logistic'
TARGETS = 'abundance' # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
PROBLEM = 'regression' # 'regression' or 'classification' or 'given_presence' or 'reverse'
SPECIES = 'segal_species' # 'segal_species' or 'mpa_species'
aggregate_features = True
aggregattion = '_aggregated_features_rms' if aggregate_features else ''

if aggregate_features:
    with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/food_categories.pkl', 'rb') as file:
        feature_to_category, categories = pickle.load(file)

# def stub_subjob(df, features, target, models_dict, i):
#     print(i)
#     all_scores = []
#     all_p_values=[]
#     all_feat_importances = []
#     all_feat_names = []
#     all_preds = []
#     all_targets = []

#     # # Retrieve models for this target
#     # if target not in models_dict:
#     #     raise ValueError(f"Target {target} not found in loaded models!")

#     df = df.dropna(subset=[target])

#     # target = target_input[i]
#     kf = KFold(n_splits=5, shuffle=False, random_state=1)
#     preds = []
#     targets = []
#     shap_values_list = []

#     for fold_number, (train_index, test_index) in enumerate(kf.split(df)):
#         # print(f"Fold {fold_number}: Train indices {train_index[:5]}, Test indices {test_index[:5]}")
#         train = df.iloc[train_index]
#         test = df.iloc[test_index]
#         model = models_dict[i][fold_number]
#         predictions = model.predict(test[features])
#         preds.extend(predictions)
#         targets.extend(test[target])

#         if MODEL == 'LGBM':
#             explainer = shap.TreeExplainer(model)
#         elif MODEL == 'ridge':
#             # masker = shap.maskers.Independent(train[features])
#             explainer = shap.LinearExplainer(model, train[features])
#         shap_values = explainer.shap_values(train[features])
#         shap_values_list.append(shap_values)

#     score = stats.pearsonr(preds, targets)
#     all_scores.append(score[0])
#     all_p_values.append(score[1])
#     if MODEL == 'ridge':
#         all_feat_importances.append(model.coef_)
#         # all_feat_names.append(features)
#     elif MODEL == 'LGBM':
#         all_feat_importances.append(model.feature_importances_)
#         # all_feat_names.append(model.feature_name_)
#     all_preds.append(preds)
#     all_targets.append(targets)

#     # Ensure all SHAP arrays have the same number of rows
#     min_rows = min(shap.shape[0] for shap in shap_values_list)
#     shap_values_list = [shap[:min_rows] for shap in shap_values_list]
#     shap_values_mean = np.mean(shap_values_list, axis=0)


#     return all_scores, all_p_values, all_feat_importances, all_preds, all_targets, shap_values_mean


# def train_classification(df, features, target, i, models_dict):
#     print(i)
#     shap_values_list = []
#     df = df.dropna(subset=[target])
#     df[target] = df[target].apply(lambda x: 0 if x == -4 else 1)

#     kf = KFold(n_splits=5, shuffle=False, random_state=1)
#     # preds = []
#     # targets = []
#     # pred_probs_list = []

#     for fold_number, (train_index, test_index) in enumerate(kf.split(df)):
#         train = df.iloc[train_index]
#         test = df.iloc[test_index]
        
#         model = models_dict[i][fold_number]
#         # predictions = model.predict(test[features])
#         # pred_probs = model.predict_proba(test[features])[:, 1]  # For AUC calculation
    
#         # preds.extend(predictions)
#         # targets.extend(test[target])
#         # pred_probs_list.extend(pred_probs)

#         if MODEL == 'LGBM':
#             explainer = shap.TreeExplainer(model)
#         elif MODEL == 'logistic':
#             explainer = shap.LinearExplainer(model, train[features])
#         shap_values = explainer.shap_values(train[features])
#         if isinstance(shap_values, list):  # LightGBM binary classification case
#             shap_values = shap_values[1]  # Select SHAP values for the positive class (1)
#         shap_values_list.append(shap_values)
    
#     # Ensure all SHAP arrays have the same number of rows
#     min_rows = min(shap.shape[0] for shap in shap_values_list)
#     shap_values_list = [shap[:min_rows] for shap in shap_values_list]
#     shap_values_mean = np.mean(shap_values_list, axis=0)

#     return [shap_values_mean]

# def aggregate_shap_values(raw_shap_values, features, save_feature_names):

#     unique_categories = set(feature_to_category.values())
#     category_shap = {cat: np.zeros(raw_shap_values.shape[0]) for cat in unique_categories}
#     category_feature_count = {cat: 0 for cat in unique_categories}
#     unmapped_shap = {}

#     # raw_shap_values.shape = (7756, 787)
#     # unique_categories = {'Fruits', 'Poultryanditsproducts', 'Bread', 'Hardcheese'}
#     # category_shap = {'Fruits': array([0., 0., 0., ..., 0., 0., 0.])}
#     # category_feature_count = {'Fruits': 0, 'Poultryanditsproducts': 0}
#     # unmaped_shap = {'Anchovy': array([0., 0., 0., ..., 0., 0., 0.]), 
#     # 'Aperol': array([0., 0., 0., ..., 0., 0., 0.]), 
#     # 'Applesauce': array([0., 0., 0., ..., 0., 0., 0.])}

#     for idx, feature in enumerate(features):

#         if feature in feature_to_category:  # Feature is mapped to a category
#             category = feature_to_category[feature]
        
#         elif feature in unique_categories:  # Feature is itself a category
#             category = feature

#         else:
#             # Keep features that are not mapped to any category
#             unmapped_shap[feature] = raw_shap_values[:, idx]
#             continue  # Continiue to the next feature

#         # Sum SHAP values into category of either mapped or categorical feature
#         category_shap[category] += raw_shap_values[:, idx]
#         category_feature_count[category] += 1  # Track number of features in the category

#         # length of category_shap = 32
#         # length of unmapped_shap = 188
        
#     # Apply averaging 
#     for category in category_shap.keys():  
#             category_shap[category] /= category_feature_count[category] # Normalize by count

#     # Final SHAP values array
#     aggregated_shap_values = []
#     aggregated_feature_names = []

#     # Add category SHAP values 
#     for cat in unique_categories:
#         aggregated_shap_values.append(category_shap[cat])
#         aggregated_feature_names.append(cat)

#     # Add unmapped feature SHAP values
#     for feature, shap_vals in unmapped_shap.items():
#         aggregated_shap_values.append(shap_vals)
#         aggregated_feature_names.append(feature)

#     # aggregated_shap_values is a list of 220 1D np arrays (one per feature), 
#     # each has 7756 shap values (per sample).
#     # Convert this list to a 2D NumPy array
#     aggregated_shap_values = np.column_stack(aggregated_shap_values)

#     # print(f"Final shape of the aggregated SHAP values: {aggregated_shap_values.shape}")

#     if save_feature_names:
#         aggregated_feature_names = pd.Series(aggregated_feature_names)
#         aggregated_feature_names.to_pickle(outdir + f"data/{PROBLEM}/aggregated_feature_names_{MODEL}_{TARGETS}.pkl")

#     return aggregated_shap_values
        

# For median aggregation this is the latest version (below)

import numpy as np
import pandas as pd

def aggregate_shap_values(raw_shap_values, features, save_feature_names):
    """
    Aggregate SHAP values into predefined categories using RMS per feature.

    For each category G with features φ1..φk:
        RMS_G(x) = sqrt(mean(φ_i(x)^2))   # computed per sample

    This gives a size-corrected magnitude for comparing categories across people.
    Note: RMS magnitudes are non-additive (cannot be used in waterfall plots).
    """

    # Unique categories from mapping
    unique_categories = set(feature_to_category.values())
    category_shap = {cat: [] for cat in unique_categories}
    unmapped_shap = {}

    # Assign features to categories
    for idx, feature in enumerate(features):

        if feature in feature_to_category:     # Feature is mapped to a category
            category = feature_to_category[feature]

        elif feature in unique_categories:     # Feature is itself a category
            category = feature

        else:
            # Keep features that are not mapped to any category
            unmapped_shap[feature] = raw_shap_values[:, idx]
            continue

        category_shap[category].append(raw_shap_values[:, idx])

    # Compute RMS per feature for each category
    aggregated_shap_values = []
    aggregated_feature_names = []

    for cat in unique_categories:
        if category_shap[cat]:  # Ensure there are features in the category
            category_shap_array = np.stack(category_shap[cat], axis=1)  # (n_samples, k)
            rms_shap_values = np.sqrt(np.mean(category_shap_array**2, axis=1))
            aggregated_shap_values.append(rms_shap_values)
            aggregated_feature_names.append(cat)

    # Add unmapped features as absolute values (equivalent to RMS of single item)
    for feature, shap_vals in unmapped_shap.items():
        aggregated_shap_values.append(np.abs(shap_vals))
        aggregated_feature_names.append(feature)

    # Convert to 2D NumPy array
    aggregated_shap_values = np.column_stack(aggregated_shap_values)

    # Save feature names if required
    if save_feature_names:
        pd.Series(aggregated_feature_names).to_pickle(
            outdir + f"data/{PROBLEM}/aggregated_feature_names_{MODEL}_{TARGETS}.pkl"
        )

    return aggregated_shap_values


def SHAP_regression(train, test, features, target, models_list, i):
    print(i)
    all_scores = []
    all_p_values=[]
    all_feat_importances = []
    all_feat_names = []
    predicted_abundances = []
    measured_abundances = []
    save_feature_names = False

    model = models_list[i]

    if model is None:
        return all_scores, all_p_values, all_feat_importances, predicted_abundances, measured_abundances
    predictions = model.predict(test[features])
    predicted_abundances.append(list(predictions))
    measured_abundances.append(list(test[target]))

    score = stats.pearsonr(predictions, test[target])
    all_scores.append(score[0])
    all_p_values.append(score[1])

    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
        explainer = shap.TreeExplainer(model)
    elif MODEL == 'ridge':
        all_feat_importances.append(model.coef_)
        # masker = shap.maskers.Independent(train[features])
        explainer = shap.LinearExplainer(model, train[features])
  
    shap_values = explainer.shap_values(train[features])

    if aggregate_features:
        if i==0:
            save_feature_names = True
        shap_values = aggregate_shap_values(
            raw_shap_values=shap_values, features=features, 
            save_feature_names=save_feature_names,
            )
    
    return all_scores, all_p_values, all_feat_importances, predicted_abundances, measured_abundances, shap_values


def SHAP_classification(train, test, features, target, i, models_list):
    print(i)
    save_feature_names = False

    model = models_list[i]
    
    if model is None:
        print("NAN")
        if MODEL == 'LGBM':
            # # Return a properly shaped np.nan array matching SHAP values
            # dummy_shape = (len(test), len(features))  # SHAP normally returns (samples, features)
            # return [[np.full(dummy_shape, np.nan), np.full(dummy_shape, np.nan)]] 
            return np.nan
        elif MODEL == 'logistic':
            return np.nan

    train[target] = train[target].apply(lambda x: 0 if x == -4 else 1)
    test[target] = test[target].apply(lambda x: 0 if x == -4 else 1)
    
    if MODEL == 'LGBM':
        explainer = shap.TreeExplainer(model)
    elif MODEL == 'logistic':
        explainer = shap.LinearExplainer(model, train[features])
    shap_values = explainer.shap_values(train[features])

    # print(f"shap_values[1].shape = {shap_values[1].shape}")
    # print(f"shap_values[1] = {len(shap_values[1])}")

    if MODEL == 'LGBM':
        # print(type([shap_values[1]]))
        # print(len([shap_values[1]]))
        # print([shap_values[1]])

        if aggregate_features:
            if i==0:
                save_feature_names = True
            shap_values[1] = aggregate_shap_values(
                raw_shap_values=shap_values[1], features=features, 
                save_feature_names=save_feature_names,
                aggregation_method='sum', # 'sum' or 'avg'
                )

        return [shap_values[1]]
    elif MODEL == 'logistic': 
        # print(type([shap_values]))
        # print(len([shap_values]))
        # print([shap_values])
        return [shap_values]

    print("Problem in ifs for return")


def stub_job():
    print ('Job started SHAP.py')
    print(MODEL)
    print(TARGETS)
    print(PROBLEM)
    params_results = {}
    print("Loading models")
    # models_dict = pickle.load(open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/models_' + MODEL + '_' + TARGETS + '.pkl', 'rb'))
    models_list = pickle.load(open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/{SPECIES}/models_' + MODEL + '_' + TARGETS + '_longitudinal.pkl', 'rb'))
    
    reverse = 'reverse/' if PROBLEM == 'reverse' else ''
    species = '' if SPECIES == 'segal_species' else '_mpa'
    pathways = '' if TARGETS != 'pathways' else '_pathways'

    # diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb.pkl")
    train_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/{reverse}diet_mb{pathways}_baseline_train.pkl")
    test_02_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_02_visit.pkl")
    test_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/{reverse}diet_mb{pathways}_baseline_test.pkl")
    print(train_baseline.shape)
    print(test_baseline.shape)
    print(test_02_visit.shape)

    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists{pathways}.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists
    # with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/significant_targets.pkl', 'rb') as file:
    #     loaded_lists = pickle.load(file)
    # significant_targets = loaded_lists
    test_baseline.columns = test_baseline.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    test_02_visit.columns = test_02_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    train_baseline.columns = train_baseline.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    # features_to_drop = ['Beans_black_eyed_peas', 'Mille_feuille', 'cereal_type__Cornflakes_with_sugar__cookies__pillows__etc__', 'never_eat__Sugar_or_foods___beverages_that_contain_sugar']
    # features = [feature for feature in features if feature not in features_to_drop]
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]

    if PROBLEM == 'reverse':
        diet_features = features  # Store original features (diet variables)
        features = target_input + ['Richness', 'Shannon_diversity'] + base_features  # Microbial features become predictors
        target_input = [feat for feat in diet_features if feat not in ['age', 'gender']]  # Diet variables become targets

    print(len(features))
    print(len(target_input))
    print('Starting queue')

    if TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'abundance' or TARGETS == 'pathways' or PROBLEM == 'reverse':
        loop_targets = target_input
    elif TARGETS == 'health':
        loop_targets = ['modified_HACK_top17_score', 'GMWI2_score']
    # for i in range(len(significant_targets)):
    for i in range(len(loop_targets)):
        if PROBLEM == 'classification':
            params_results[i] = SHAP_classification(train_baseline, test_baseline, features, loop_targets[i], i, models_list)
        else:
            params_results[i] = SHAP_regression(train_baseline, test_baseline, features, loop_targets[i], models_list, i)

    output = pd.DataFrame(params_results).transpose()
    output = output.apply(lambda x: x.explode(), axis=1)
    output.columns = [f'column{i}' for i in range(len(output.columns))]
    output.to_pickle(outdir + f"data/{PROBLEM}/{SPECIES}/" + "output_SHAP_" + MODEL + '_' + TARGETS + aggregattion + '.pkl')
    print(output)


def main():
    stub_job()

if __name__ == '__main__':
    main()
    output = pd.read_pickle(outdir + f"data/{PROBLEM}/{SPECIES}/" + "output_SHAP_" + MODEL + '_' + TARGETS + aggregattion + '.pkl')
    print(output)
    print("FINISH")
