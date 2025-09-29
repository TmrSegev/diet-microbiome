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
TARGETS = 'diet' # 'div' or 'abundance' or 'diet'
SPLIT = 'longitudinal' # 'kfold' or 'longitudinal'
PROBLEM = 'reverse' # 'regression' or 'classification' or 'reverse' or 'given_presence'
base_training = False
robust_training = False
no_var_features = False
##################

suffix = '_longitudinal' if SPLIT == 'longitudinal' else '' 
colsample_bytree = 1 if base_training else 0.3
target_part = 'diet' if PROBLEM == 'reverse' else f'{TARGETS}'
base = 'base_' if base_training else ''
suffix_var = '_no_var_features' if no_var_features else ''


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
                        colsample_bytree=colsample_bytree, learning_rate=0.01, n_jobs=64, random_state=1, 
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
                colsample_bytree=colsample_bytree, learning_rate=0.001, n_jobs=16, random_state=1,
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


def scale_data(train, test, features):
    train = train.copy()
    test = test.copy()

    # Identify binary features (features with only values 0 and 1)
    binary_features = [col for col in train.columns if train[col].nunique() == 2 and sorted(train[col].unique()) == [0, 1]]

    # Exclude binary features from standardization
    features_to_standardize = [feature for feature in features if feature not in binary_features]

    scaler = StandardScaler()
    train.loc[:, features_to_standardize] = pd.DataFrame(
    scaler.fit_transform(train[features_to_standardize]),
    columns=train[features_to_standardize].columns,
    index=train[features_to_standardize].index
    )

    # with open(path + 'scaler_diet_mb.pkl', 'rb') as scaler_file:
        # scaler = pickle.load(scaler_file)
    test.loc[:, features_to_standardize] = pd.DataFrame(
        scaler.transform(test[features_to_standardize]), 
        columns=test[features_to_standardize].columns, 
        index=test[features_to_standardize].index
    )

    return train, test, scaler



def prepare_data():
    diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb.pkl")
    print(f"Shape of the data: {diet_mb.shape}")
    # if SPLIT == 'longitudinal':

    #     diet_mb_02_visit = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_02_visit.pkl")
    #     # print(diet_mb_02_visit.shape)
    #     # visit_subjects = list(diet_mb_02_visit.index)
    #     # diet_mb_test = diet_mb.loc[diet_mb.index.isin(visit_subjects[:2000])]
    #     # diet_mb = diet_mb.loc[~diet_mb.index.isin(visit_subjects[:2000])]
    #     # print("Length of train:", len(diet_mb))
    #     # print("Length of test:", len(diet_mb_test))
        
    #     diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_baseline_train.pkl")
    #     diet_mb_test = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_baseline_test.pkl")

    with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/my_lists.pkl', 'rb') as file:
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
        
    # Train test split
    diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb.pkl")
    diet_mb_02_visit = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_02_visit.pkl")
    print(diet_mb_02_visit.shape)
    visit_subjects = list(diet_mb_02_visit.index)
    diet_mb_test = diet_mb.loc[diet_mb.index.isin(visit_subjects[:2000])]
    diet_mb_train = diet_mb.loc[~diet_mb.index.isin(visit_subjects[:2000])]
    print("Length of all:", len(diet_mb))
    print("Length of train:", len(diet_mb_train))
    print("Length of test:", len(diet_mb_test))

    diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_train.columns = diet_mb_train.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_test.columns = diet_mb_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    original_features = features[:]
    features = base_features if base_training else features
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]
    
    # Scaling the data
    if PROBLEM != 'reverse':
        diet_mb_train, diet_mb_test, scaler = scale_data(diet_mb_train, diet_mb_test, features)
        print(diet_mb_train.describe())
        print(diet_mb_test.describe())
        # diet_mb_train.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_baseline_train.pkl")
        # diet_mb_test.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_baseline_test.pkl")
        # with open("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/scaler.pkl", 'wb') as file:
        #     pickle.dump(scaler, file)
        return diet_mb_train, features, target_input

    if PROBLEM == 'reverse':
        mb_features = (target_input + ['Richness', 'Shannon_diversity'] + base_features) if not base_training else base_features
        diet_targets = [feat for feat in original_features if feat not in ['age', 'gender']]

        # Standardize the data
        diet_mb_train, diet_mb_test, scaler = scale_data(diet_mb_train, diet_mb_test, mb_features)
        diet_mb_train.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/reverse/diet_mb_baseline_train_reverse.pkl")
        diet_mb_test.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/reverse/diet_mb_baseline_test_reverse.pkl")
        # with open("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/scaler.pkl", 'wb') as file:
        #     pickle.dump(scaler, file)
        
        return diet_mb_train, mb_features, diet_targets  
        
        
def training_loop(diet_mb, features, target_input):
    params_results = {}
    models_dict = {}
    if TARGETS == 'abundance' or PROBLEM == 'reverse':
        # lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/regression/output_LGBM_abundance.pkl"))
        # top_microbes = lgbm_diet_scores.sort_values(ascending=False)
        # top_microbes = top_microbes.head(100)
        # target_input = [target_input[i] for i in top_microbes.index if i < len(target_input)]
        loop_targets = target_input
    elif TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']

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

    pickle.dump(models_dict, open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/models_{base}{MODEL}_{target_part}{suffix}{suffix_var}.pkl', "wb"))
    output = pd.DataFrame(params_results).transpose().apply(lambda x: x.explode(), axis=1)
    output.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/output_{base}{MODEL}_{target_part}{suffix}{suffix_var}.pkl")


def read_results(df):
    output = []
    for col in df.columns:
        output.append(df[col])
    return tuple(output)


def robustness_training(diet_mb, features, target_input):
    # Filter for only top 100 microbes
    lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/output_LGBM_abundance.pkl"))
    top_microbes = lgbm_diet_scores.sort_values(ascending=False)
    top_microbes = top_microbes.head(100)
    target_input = [target_input[i] for i in top_microbes.index if i < len(target_input)]

    # Hold out data
    diet_mb_02_visit = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_02_visit.pkl")
    baseline_subjects = list(diet_mb.index)
    visit_subjects = list(diet_mb_02_visit.index)
    print("Length of train data:", len(baseline_subjects))
    print("Length of hold out data:", len(visit_subjects))
    diet_mb = diet_mb.loc[~diet_mb.index.isin(visit_subjects)]
    print(diet_mb.shape)    
    
    n_samples = len(diet_mb)
    current_df = diet_mb.copy()  # Start with the full DataFrame

    iteration = 1
    while n_samples > 1:
        print(f"Iteration {iteration}: {n_samples} samples")

        params_results = {}
        models_list = []
        if TARGETS == 'abundance' or PROBLEM == 'reverse':
            loop_targets = target_input
        elif TARGETS == 'div':
            loop_targets = ['Richness', 'Shannon_diversity']
        for i in range(len(loop_targets)):
            params_results[i] = train_baseline(current_df, features, loop_targets[i], i, models_list)


        pickle.dump(models_list, open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/robustness_models/models_' + MODEL + '_' + TARGETS + '_' + str(n_samples) + '_samples' + PROBLEM + '.pkl', "wb"))
        # output = pd.DataFrame(params_results).transpose()
        # output = output.apply(lambda x : x.explode(), axis=1)
        # output.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/output_"+ MODEL + '_' + TARGETS + suffix + PROBLEM + '.pkl')

        # Reduce samples by half for the next iteration
        n_samples = math.ceil(n_samples / 2)  # Use math.ceil to ensure at least one sample
        current_df = current_df.sample(n=n_samples, random_state=iteration).reset_index(drop=True)
        iteration += 1


if __name__ == '__main__':
    print(f"MODEL: {MODEL}")
    print(f"TARGETS: {TARGETS}")
    print(f"PROBLEM: {PROBLEM}")
    print(f"SPLIT: {SPLIT}")
    print(f"base_training: {base_training}")
    print(f"robust_training: {robust_training}")
    print(f"no_var_features: {no_var_features}")

    if TARGETS == 'div' and PROBLEM == 'classification':
        raise ValueError("Cannot perform classification on diversity targets. Change TARGETS to 'abundance' or PROBLEM to ''.")

    diet_mb, features, target_input = prepare_data()

    if robust_training:
        robustness_training(diet_mb, features, target_input)
    else:
        training_loop(diet_mb, features, target_input)
        
    output = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/output_{base}{MODEL}_{target_part}{suffix}{suffix_var}.pkl")
    print(output)