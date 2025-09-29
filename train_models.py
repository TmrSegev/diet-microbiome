from LabData import config_global as config
from LabUtils import Utils
from LabUtils import addloglevels
import math
import os
import re
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 


### PARAMETERS ###
MODEL = 'LGBM' # 'LGBM' or 'ridge' or 'logistic'
TARGETS = 'abundance' # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
SPLIT = 'longitudinal' # 'kfold' or 'longitudinal'
PROBLEM = 'regression' # 'regression' or 'classification' or 'reverse' or 'given_presence'
SPECIES = 'mpa_species' # 'segal_species' or 'mpa_species'
base_training = False
CLR_flag = False
robust_training = False
ROBUST_REPEATS = 5  # number of subsamples per grid size
no_var_features = False
foods_only = False
##################

suffix = '_longitudinal' if SPLIT == 'longitudinal' else '' 
colsample_bytree = 1 if base_training else 0.3
target_part = 'diet' if PROBLEM == 'reverse' else f'{TARGETS}'
base = 'base_' if base_training else ''
suffix_var = '_no_var_features' if no_var_features else ''
suffix_foods = '_foods_only' if foods_only else ''
pathways = '' if TARGETS != 'pathways' else '_pathways'
CLR_suf = '_CLR' if CLR_flag else ''
# SAVE_MODELS_IN_ROBUSTNESS = False if robust_training else True  # do not pickle models in robustness runs


def stub_subjob(df, features, target, i, models_dict):
    all_scores = []
    all_p_values=[]
    all_feat_importances = []
    # all_feat_names = []
    all_preds = []
    all_targets = []
    models_list = []

    df = df.dropna(subset=[target])

    if PROBLEM == 'given_presence':
        df = df[df[target] > -4].copy()
        prevalence = df[target].shape[0]

    kf = KFold(n_splits=5, shuffle=False) # If shuffle turned on add a random_state=1 kwarg.

    preds = []
    targets = []

    for fold_number, (train_index, test_index) in enumerate(kf.split(df)):
            train = df.iloc[train_index]
            test = df.iloc[test_index]

            if MODEL == 'ridge':
                model = linear_model.RidgeCV(alphas=(0.01, 0.1, 0.5, 1, 10, 100))            
            elif MODEL == 'LGBM':
                if PROBLEM == 'reverse': # Trying hyperparameters modifications, only for reverse for now
                    model = lgb.LGBMRegressor(
                        max_depth=6, n_estimators=2000, subsample=0.5, subsample_freq=1,
                        colsample_bytree=colsample_bytree, learning_rate=0.01, n_jobs=8, random_state=1, 
                        reg_alpha=0, reg_lambda=10, min_data_in_leaf=50, num_leaves=50, verbosity=-1
                    )  
                else:  
                    model = lgb.LGBMRegressor(
                        max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1,
                        colsample_bytree=colsample_bytree, learning_rate=0.001, n_jobs=8, random_state=1, 
                        reg_alpha=1, reg_lambda=1, min_data_in_leaf=20, verbosity=-1
                    )

            model.fit(train[features], train[target])
            predictions = model.predict(test[features])
            preds.extend(predictions)
            targets.extend(test[target])
            models_list.append(model)

    models_dict[i] = models_list
    score = stats.pearsonr(preds, targets)
    all_scores.append(score[0])
    all_p_values.append(score[1])

    if MODEL == 'ridge':
        all_feat_importances.append(model.coef_)
        # all_feat_names.append(features)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
        # all_feat_names.append(model.feature_name_)

    all_preds.append(preds)
    all_targets.append(targets)

    if PROBLEM != 'given_presence':
        return all_scores, all_p_values, all_feat_importances, all_preds, all_targets
    elif PROBLEM == 'given_presence':
        return all_scores, all_p_values, all_feat_importances, all_preds, all_targets, prevalence


def train_baseline(df, features, target, i, models_dict):

    all_scores = []
    all_p_values = []
    all_feat_importances = []
    all_preds = []
    all_targets = []

    # Drop rows where the target column is NaN
    df = df.dropna(subset=[target])

    if PROBLEM == 'given_presence':
        df = df[df[target] > -4].copy()
        prevalence = df[target].shape[0]

    # Train the model on all data in df
    if MODEL == 'ridge':
        model = linear_model.RidgeCV(alphas=(0.01, 0.1, 0.5, 1, 10, 100))
    elif MODEL == 'LGBM':
        if PROBLEM == 'reverse':
            model = lgb.LGBMRegressor(
                max_depth=6, n_estimators=2000, subsample=0.5, subsample_freq=1,
                colsample_bytree=colsample_bytree, learning_rate=0.01, n_jobs=64, random_state=1, 
                reg_alpha=0, reg_lambda=10, min_data_in_leaf=50, num_leaves=50, verbosity=-1
            ) 
        else:
            model = lgb.LGBMRegressor(
                max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1, 
                colsample_bytree=colsample_bytree, learning_rate=0.001, n_jobs=8, random_state=1,
                reg_alpha=1, reg_lambda=1, min_data_in_leaf=20, verbosity=-1
            )

    model.fit(df[features], df[target])
    # Make predictions on the same data (train set predictions)
    preds = model.predict(df[features])
    targets = df[target].values

    # print(pd.Series(preds).value_counts())
    # print(pd.Series(targets).value_counts())

    # Calculate score using Pearson correlation
    score = stats.pearsonr(preds, targets)
    all_scores.append(score[0])
    all_p_values.append(score[1])

    # Store feature importances or coefficients
    if MODEL == 'ridge':
        all_feat_importances.append(model.coef_)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)

    # Store predictions and actual targets
    all_preds.append(preds)
    all_targets.append(targets)

    models_dict[i] = model

    if PROBLEM != 'given_presence':
        return all_scores, all_p_values, all_feat_importances, all_preds, all_targets
    elif PROBLEM == 'given_presence':
        return all_scores, all_p_values, all_feat_importances, all_preds, all_targets, prevalence


def permutation_auc_test(y_true, y_pred_probs, n_permutations=1000):
    """
    Perform a permutation test to compute the p-value for AUC.
    
    Parameters:
    - y_true: array-like, true labels (0/1)
    - y_pred_probs: array-like, predicted probabilities
    - n_permutations: int, number of permutations
    
    Returns:
    - p-value for AUC significance
    """
    observed_auc = roc_auc_score(y_true, y_pred_probs)  # Compute actual AUC
    permuted_aucs = []

    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(y_true)  # Shuffle labels
        perm_auc = roc_auc_score(shuffled_labels, y_pred_probs)  # Compute AUC
        permuted_aucs.append(perm_auc)

    # Compute p-value: fraction of permuted AUCs >= observed AUC
    p_value = np.sum(np.array(permuted_aucs) >= observed_auc) / n_permutations
    return p_value


def train_classification(df, features, target, i, models_dict):

    all_accuracy = []
    all_auc=[]
    all_feat_importances = []
    # all_feat_names = []
    all_preds = []
    all_targets = []
    models_list = []

    df = df.dropna(subset=[target])

    df[target] = df[target].apply(lambda x: 0 if x == -4 else 1)
    if df[target].value_counts()[1] > 0.95 * len(df):
        print("Too many samples of positive class (>95%). Skipping.")
        models_dict[i] = None
        return np.nan, np.nan, [np.nan], [np.nan], [np.nan], [np.nan]

    kf = KFold(n_splits=5, shuffle=False)
    preds = []
    targets = []
    pred_probs_list = []

    for fold_number, (train_index, test_index) in enumerate(kf.split(df)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]        
        
        if MODEL == 'logistic':
            model = linear_model.LogisticRegression(
                max_iter=1000,  # Increase if needed for convergence,
                class_weight='balanced',  # To account for class imbalance
                random_state=1
            )
        elif MODEL == 'LGBM':
            model = lgb.LGBMClassifier(
                max_depth=4, 
                n_estimators=2000, 
                subsample=0.5, 
                subsample_freq=1, 
                colsample_bytree=colsample_bytree, 
                learning_rate=0.001, 
                n_jobs=8, 
                random_state=1,
                objective='binary',
                is_unbalance=True,
                reg_alpha=1, reg_lambda=1, min_data_in_leaf=20, 
                verbosity=-1
            )

        model.fit(train[features], train[target])
        predictions = model.predict(test[features])
        pred_probs = model.predict_proba(test[features])[:, 1]  # For AUC calculation
    
        preds.extend(predictions)
        targets.extend(test[target])
        pred_probs_list.extend(pred_probs)
        models_list.append(model)

    models_dict[i] = models_list
    # Calculate accuracy and AUC
    accuracy = accuracy_score(targets, preds)
    # auc = roc_auc_score(test[target], pred_probs)
    auc = roc_auc_score(targets, pred_probs_list)
    
    # Run permutation test
    auc_p_value = permutation_auc_test(targets, pred_probs_list)

    # all_accuracy.append(accuracy)
    # all_auc.append(auc)

    if MODEL == 'logistic':
        all_feat_importances.append(model.coef_)
        # all_feat_names.append(features)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
        # all_feat_names.append(model.feature_name_)

    all_preds.append(preds)
    all_targets.append(targets)

    return accuracy, auc, all_feat_importances, all_preds, all_targets, [auc_p_value]


def train_baseline_classification(df, features, target, i, models_dict):
    all_accuracy = []
    all_auc=[]
    all_feat_importances = []
    all_preds = []
    all_targets = []

    df = df.dropna(subset=[target])

    df[target] = df[target].apply(lambda x: 0 if x == -4 else 1)
    if df[target].value_counts()[1] > 0.95 * len(df):
        print("Too many samples of positive class (>95%). Skipping.")
        models_dict[i] = None
        return np.nan, np.nan, [np.nan], [np.nan], [np.nan], [np.nan]

    preds = []
    targets = []
    pred_probs_list = []       
    
    if MODEL == 'logistic':
        model = linear_model.LogisticRegression(
            max_iter=1000,  # Increase if needed for convergence,
            class_weight='balanced',  # To account for class imbalance
            random_state=1
        )
    elif MODEL == 'LGBM':
        model = lgb.LGBMClassifier(
            max_depth=4, 
            n_estimators=2000, 
            subsample=0.5, 
            subsample_freq=1, 
            colsample_bytree=colsample_bytree, 
            learning_rate=0.001, 
            n_jobs=8, 
            random_state=1,
            objective='binary',
            is_unbalance=True,
            reg_alpha=1, reg_lambda=1, min_data_in_leaf=20, verbosity=-1
        )

    model.fit(df[features], df[target])
    preds = model.predict(df[features])
    targets = df[target].values
    pred_probs = model.predict_proba(df[features])[:, 1]  # For AUC calculation

    # Calculate accuracy and AUC
    accuracy = accuracy_score(targets, preds)
    # auc = roc_auc_score(test[target], pred_probs)
    auc = roc_auc_score(targets, pred_probs)
    # Run permutation test
    auc_p_value = permutation_auc_test(targets, pred_probs)
    # score = stats.pearsonr(preds, targets)

    # all_accuracy.append(accuracy)
    # all_auc.append(auc)

    if MODEL == 'logistic':
        all_feat_importances.append(model.coef_)
        # all_feat_names.append(features)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
        # all_feat_names.append(model.feature_name_)

    all_preds.append(preds)
    all_targets.append(targets)

    models_dict[i] = model
    return accuracy, auc, all_feat_importances, all_preds, all_targets, [auc_p_value]


def scale_data(train, test, features, target_features, test_02_visit=None, test_04_visit=None):
    # Create copies to ensure the original dataframes passed to the function are not modified.
    train = train.copy()
    test = test.copy()

    # --- 1. Initialize and apply a dedicated scaler for the 'age' feature ---
    age_scaler = StandardScaler()
    age_feature = ['age']
    
    # Fit the scaler on the training data and transform it.
    train.loc[:, age_feature] = age_scaler.fit_transform(train[age_feature])
    # Transform the test data using the same fitted scaler.
    test.loc[:, age_feature] = age_scaler.transform(test[age_feature])

    # --- 2. Initialize and apply the scaler for other diet features ---
    diet_scaler = StandardScaler()
    
    # Identify binary features (columns with only 0 and 1) to exclude them from standardization.
    # Also, exclude 'age' as it has already been scaled separately.
    binary_features = [col for col in features if train[col].nunique() == 2 and sorted(train[col].unique()) == [0, 1]]
    features_to_standardize = [feature for feature in features if feature not in binary_features and feature != 'age']

    # Fit the scaler on the training feature data and transform it.
    # Then, transform the test feature data using the same fitted scaler.
    if features_to_standardize:
        train.loc[:, features_to_standardize] = diet_scaler.fit_transform(train[features_to_standardize])
        test.loc[:, features_to_standardize] = diet_scaler.transform(test[features_to_standardize])

    # --- 3. Initialize and apply the scaler for microbiome (target) features ---
    mb_scaler = StandardScaler()

    # Fit the scaler on the training target data and transform it.
    # Then, transform the test target data using the same fitted scaler.
    if target_features:
        train.loc[:, target_features] = mb_scaler.fit_transform(train[target_features])
        test.loc[:, target_features] = mb_scaler.transform(test[target_features])

    # --- 4. Handle optional dataframes and return values ---
    if (test_02_visit is None) or (test_04_visit is None):
        return train, test, diet_scaler, mb_scaler, age_scaler
    else:
        test_02_visit = test_02_visit.copy()
        test_04_visit = test_04_visit.copy()

        # Apply the fitted age_scaler to the optional dataframes.
        test_02_visit.loc[:, age_feature] = age_scaler.transform(test_02_visit[age_feature])
        test_04_visit.loc[:, age_feature] = age_scaler.transform(test_04_visit[age_feature])

        # Transform the features in the optional dataframes using the fitted diet_scaler.
        if features_to_standardize:
            test_02_visit.loc[:, features_to_standardize] = diet_scaler.transform(test_02_visit[features_to_standardize])
            test_04_visit.loc[:, features_to_standardize] = diet_scaler.transform(test_04_visit[features_to_standardize])
        
        # Transform the targets in the optional dataframes using the fitted mb_scaler.
        if target_features:
            test_02_visit.loc[:, target_features] = mb_scaler.transform(test_02_visit[target_features])
            test_04_visit.loc[:, target_features] = mb_scaler.transform(test_04_visit[target_features])

        return train, test, diet_scaler, mb_scaler, age_scaler, test_02_visit, test_04_visit





def prepare_data():
    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists{pathways}.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists

    if no_var_features:
        features = [x for x in features if x not in [
                    'Foods_per_day_std',
                    'Foods_per_meal_std',
                    'Meals_per_day_std',
                    'plant_foods_per_day_std',
                    'plant_foods_per_week_std',
                    'med_score_per_day_std',
                    'paleo_score_per_day_std',
                    'vegetarian_score_per_day_std',
                    'wfpb_score_per_day_std',
                    'vegan_score_per_day_std',
                    'wfab_score_per_day_std',
                    'fermented_score_per_day_std',
                    'Protein_std',
                    'Total lipid (fat)_std',
                    'Carbohydrate, by difference_std',
                    'Energy_std',
                    'pct_protein_calories_std',
                    'pct_carb_calories_std',
                    'pct_fat_calories_std',
                    'intra_diet_correlation',
                    'pct_protein_calories',
                    'pct_carb_calories',
                    'pct_fat_calories'
                ]]
        
    if foods_only:
        with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/food_shortnames.pkl', 'rb') as file:
            food_shortnames = pickle.load(file)
        features = [x for x in features if x in food_shortnames]
        print(len(features))
        features = features + ['age', 'sex']
        print(len(features))


        
    # Train test split
    diet_mb = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}{CLR_suf}.pkl")
    diet_mb_02_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_02_visit{CLR_suf}.pkl")
    diet_mb_04_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_04_visit{CLR_suf}.pkl")
    print("diet_mb shape:", diet_mb.shape)
    print("diet_mb_02_visit shape:", diet_mb_02_visit.shape)
    print("diet_mb_04_visit shape:", diet_mb_04_visit.shape)

    visit_subjects_02 = list(diet_mb_02_visit.index)
    visit_subjects_04 = list(diet_mb_04_visit.index)
    diet_mb_test = diet_mb.loc[diet_mb.index.isin(visit_subjects_04) | diet_mb.index.isin(visit_subjects_02[:1000])]
    diet_mb_train = diet_mb.loc[~diet_mb.index.isin(diet_mb_test.index)]
    print("Length of all:", len(diet_mb))
    print("Length of train:", len(diet_mb_train))
    print("Length of test:", len(diet_mb_test))

    diet_mb_test_02_visit = diet_mb_02_visit.loc[diet_mb_02_visit.index.isin(diet_mb_test.index)]
    print("Length of test 02 visit:", len(diet_mb_test_02_visit))
    diet_mb_test_04_visit = diet_mb_04_visit.loc[diet_mb_04_visit.index.isin(diet_mb_test.index)]
    print("Length of test 04 visit:", len(diet_mb_test_04_visit))

     # Remove target columns with 0 variance after split
    zero_var_targets = [col for col in target_input if diet_mb_train[col].std() == 0]
    if zero_var_targets:
        print(f"Removing targets with 0 variance: {zero_var_targets}")
    target_input = [col for col in target_input if col not in zero_var_targets]
    diet_mb = diet_mb.drop(columns=zero_var_targets, errors='ignore')
    diet_mb_train = diet_mb_train.drop(columns=zero_var_targets, errors='ignore')
    diet_mb_test = diet_mb_test.drop(columns=zero_var_targets, errors='ignore')
    diet_mb_test_02_visit = diet_mb_test_02_visit.drop(columns=zero_var_targets, errors='ignore')
    diet_mb_test_04_visit = diet_mb_test_04_visit.drop(columns=zero_var_targets, errors='ignore')
    # Save updated target_input to file
    if not (no_var_features or foods_only):
        with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists{pathways}.pkl', 'wb') as file:
            pickle.dump((base_features, features, target_input), file)

    diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_train.columns = diet_mb_train.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_test.columns = diet_mb_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_test_02_visit.columns = diet_mb_test_02_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_test_04_visit.columns = diet_mb_test_04_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    original_features = features[:]
    features = base_features if base_training else features
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]
    
    # Scaling the data
    if PROBLEM != 'reverse':
        diet_mb_train, diet_mb_test, diet_scaler, mb_scaler, age_scaler, diet_mb_test_02_visit_copy, diet_mb_test_04_visit_copy = scale_data(diet_mb_train, diet_mb_test, features, target_input, diet_mb_test_02_visit, diet_mb_test_04_visit)
        # print(diet_mb_train.describe())
        # print(diet_mb_test.describe())
        # print(diet_mb_test_02_visit.describe())
        diet_mb_train.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_baseline{CLR_suf}{suffix_foods}_train.pkl")
        diet_mb_test.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_baseline{CLR_suf}{suffix_foods}_test.pkl")
        diet_mb_test_02_visit_copy.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_02_visit{CLR_suf}{suffix_foods}_test.pkl")
        diet_mb_test_04_visit_copy.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_04_visit{CLR_suf}{suffix_foods}_test.pkl")
        with open(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_scaler{pathways}{CLR_suf}{suffix_foods}.pkl", 'wb') as file:
            pickle.dump(diet_scaler, file)
        with open(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/mb_scaler{pathways}{CLR_suf}{suffix_foods}.pkl", 'wb') as file:
            pickle.dump(mb_scaler, file)
        with open(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/age_scaler{pathways}{CLR_suf}{suffix_foods}.pkl", 'wb') as file:
            pickle.dump(age_scaler, file)
        print("SAVED")
        return diet_mb_train, features, target_input

    if PROBLEM == 'reverse':
        mb_features = (target_input + ['Richness', 'Shannon_diversity'] + base_features) if not base_training else base_features
        diet_targets = [feat for feat in original_features if feat not in ['age', 'gender']]

        # Standardize the data
        diet_mb_train, diet_mb_test, scaler = scale_data(diet_mb_train, diet_mb_test, mb_features)
        # diet_mb_train.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/reverse/diet_mb_baseline_train_reverse.pkl")
        # diet_mb_test.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/reverse/diet_mb_baseline_test_reverse.pkl")
        # with open("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/scaler.pkl", 'wb') as file:
        #     pickle.dump(scaler, file)
        
        return diet_mb_train, mb_features, diet_targets  
        
        
def training_loop(diet_mb, features, target_input):
    params_results = {}
    models_dict = {}
    if TARGETS == 'abundance' or TARGETS == 'pathways' or PROBLEM == 'reverse':
        # lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/regression/output_LGBM_abundance.pkl"))
        # top_microbes = lgbm_diet_scores.sort_values(ascending=False)
        # top_microbes = top_microbes.head(100)
        # target_input = [target_input[i] for i in top_microbes.index if i < len(target_input)]
        loop_targets = target_input
    elif TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'health':
        loop_targets = ['modified_HACK_top17_score', 'GMWI2_score']
        diet_mb = diet_mb.dropna(subset=loop_targets)
    

    with tqdm(total=len(loop_targets), desc="Training Progress", leave=True, dynamic_ncols=True, position=0) as pbar:
        pbar.refresh()
        for i in range(len(loop_targets)):
            if SPLIT == 'kfold':
                if PROBLEM == 'classification':
                    params_results[i] = train_classification(diet_mb, features, loop_targets[i], i, models_dict)
                else:
                    params_results[i] = stub_subjob(diet_mb, features, loop_targets[i], i, models_dict)
            elif SPLIT == 'longitudinal':
                if PROBLEM == 'classification':
                    params_results[i] = train_baseline_classification(diet_mb, features, loop_targets[i], i, models_dict)
                else:
                    params_results[i] = train_baseline(diet_mb, features, loop_targets[i], i, models_dict)

            pbar.update(1)

    pickle.dump(models_dict, open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/{SPECIES}/models_{base}{MODEL}_{target_part}{suffix}{suffix_var}{CLR_suf}{suffix_foods}.pkl', "wb"))
    output = pd.DataFrame(params_results).transpose().apply(lambda x: x.explode(), axis=1)
    output.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/output_{base}{MODEL}_{target_part}{suffix}{suffix_var}{CLR_suf}{suffix_foods}.pkl")


def read_results(df):
    output = []
    for col in df.columns:
        output.append(df[col])
    return tuple(output)


def _select_loop_targets_for_robustness(target_input):
    """
    Mirrors training_loop target selection to keep behavior consistent.
    For abundance/pathways/reverse: use top targets from prior run.
    For div: ['Richness', 'Shannon_diversity'].
    For health: ['modified_HACK_top17_score', 'GMWI2_score'].
    """
    if TARGETS in ('abundance', 'pathways') or PROBLEM == 'reverse':
        # Read prior results with the same naming convention used elsewhere
        prior_output_path = (
            f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/"
            f"data/{PROBLEM}/{SPECIES}/"
            f"output_{base}{MODEL}_{target_part}{suffix}{suffix_var}{CLR_suf}{suffix_foods}.pkl"
        )
        lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(
            pd.read_pickle(prior_output_path)
        )
        top_microbes = lgbm_diet_scores.sort_values(ascending=False).head(100)
        loop_targets = [target_input[i] for i in top_microbes.index if i < len(target_input)]
        return loop_targets
    elif TARGETS == 'div':
        return ['Richness', 'Shannon_diversity']
    elif TARGETS == 'health':
        return ['modified_HACK_top17_score', 'GMWI2_score']
    else:
        # Fallback - just mirror training_loop default
        return target_input


def robustness_training(diet_mb, features, target_input):
    loop_targets = _select_loop_targets_for_robustness(target_input)
    N = len(diet_mb)
    sample_sizes = sorted({min(N, s) for s in [250, 500, 1000, 2000, 4000, 6000, N]})

    for it, n_samples in enumerate(sample_sizes, start=1):
        for rep in range(1, ROBUST_REPEATS + 1):
            print(f"[Robustness] Size {n_samples} - repeat {rep}/{ROBUST_REPEATS}")

            # Different seed per (size, repeat) - deterministic
            seed = it * 10_000 + rep
            current_df = diet_mb.sample(n=n_samples, random_state=seed)

            params_results = {}
            models_dict = {}

            for i, targ in enumerate(loop_targets):
                if PROBLEM == 'classification':
                    params_results[i] = train_classification(current_df, features, targ, i, models_dict)
                else:
                    params_results[i] = stub_subjob(current_df, features, targ, i, models_dict)

            if not robust_training:
                models_out_path = (
                    f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/"
                    f"models/{PROBLEM}/{SPECIES}/robustness/"
                    f"models_{base}{MODEL}_{target_part}_{n_samples}_samples"
                    f"{suffix}{suffix_var}{CLR_suf}{suffix_foods}_rep{rep}.pkl"
                )
                os.makedirs(os.path.dirname(models_out_path), exist_ok=True)
                pickle.dump(models_dict, open(models_out_path, "wb"))

            output = pd.DataFrame(params_results).transpose().apply(lambda x: x.explode(), axis=1)
            data_out_path = (
                f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/"
                f"data/{PROBLEM}/{SPECIES}/"
                f"output_{base}{MODEL}_{target_part}_robust_{n_samples}"
                f"{suffix}{suffix_var}{CLR_suf}{suffix_foods}_rep{rep}.pkl"
            )
            os.makedirs(os.path.dirname(data_out_path), exist_ok=True)
            output.to_pickle(data_out_path)



if __name__ == '__main__':
    print(f"MODEL: {MODEL}")
    print(f"TARGETS: {TARGETS}")
    print(f"PROBLEM: {PROBLEM}")
    print(f"SPLIT: {SPLIT}")
    print(f"base_training: {base_training}")
    print(f"robust_training: {robust_training}")
    print(f"no_var_features: {no_var_features}")
    print(f"food features only: {foods_only}")

    if TARGETS == 'div' and PROBLEM == 'classification':
        raise ValueError("Cannot perform classification on diversity targets. Change TARGETS to 'abundance' or PROBLEM to ''.")

    diet_mb, features, target_input = prepare_data()

    if robust_training:
        robustness_training(diet_mb, features, target_input)
    else:
        training_loop(diet_mb, features, target_input)
        
    output = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/output_{base}{MODEL}_{target_part}{suffix}{suffix_var}{CLR_suf}{suffix_foods}.pkl")
    print(output)