from LabData import config_global as config
from LabUtils import Utils
from LabUtils import addloglevels
import math
import os
import pandas as pd
import pickle
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
import psutil
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODEL = 'LGBM' # 'LGBM' or 'ridge'
TARGETS = 'pathways' # 'div' or 'abundance' or 'pathways'
SPLIT = 'kfold' # 'kfold' or 'all_baseline'
suffix = '_all_baseline' if SPLIT == 'all_baseline' else ''
PROBLEM = 'regression' # 'regression' or 'classification'
SPECIES = 'segal_species' # 'segal_species' or 'mpa_species'
PERMUTATIONS = 1000 # Number of permutations

pathways = '' if TARGETS != 'pathways' else '_pathways'

import pandas as pd

def choose_target_bins(df_log, targets):
    df_mb = df_log[targets]
    
    if TARGETS == "abundance":
        # Calculate prevalence for each target (number of samples where target is not -4)
        prevalence = df_mb.apply(lambda col: (col > -4).sum(), axis=0)
    elif TARGETS == "pathways":
        prevalence = df_mb.apply(lambda col: (col > 0).sum(), axis=0)
    
    # Define bin edges
    bins = list(range(0, 105, 5))  # 0%-100% in increments of 5%
    
    # Normalize prevalence to percentages
    prevalence_percent = (prevalence / len(df_log)) * 100
    
    # Bin the targets based on prevalence
    df_bins = pd.cut(prevalence_percent, bins=bins, right=False, labels=bins[:-1])
    
    # Dictionary to store representative targets for each bin
    binned_targets = {}
    
    # Select one representative microbe per bin
    for target, bin_label in df_bins.items():
        if pd.notna(bin_label) and bin_label not in binned_targets:
            binned_targets[bin_label] = target
    
    # Convert dictionary values to a list of representative microbes
    selected_targets = list(binned_targets.values())
    
    # Create a mapping of all microbes to their bins
    microbe_bin_mapping = dict(zip(targets, df_bins))
    
    return selected_targets, microbe_bin_mapping


def permute_df(df, random_state, features, targets):
    feat = df[features].copy().reset_index(drop=True)  # Keep the features unchanged
    mb = df[targets].copy()  # Copy the targets
    mb_shuffled = mb.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle the targets
    diet_mb_shuffled = pd.concat([feat, mb_shuffled], axis=1)  # Concatenate the original features with shuffled targets
    return diet_mb_shuffled


def stub_subjob(df, features, target, i):
    # print(i)
    all_scores = []
    df = df.dropna(subset=[target])

    kf = KFold(n_splits=5, shuffle=False, random_state=1)
    preds = []
    targets = []

    for train_index, test_index in kf.split(df):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        if MODEL == 'ridge':
            model = linear_model.RidgeCV(alphas=(0.01, 0.1, 0.5, 1, 10, 100))
        elif MODEL == 'LGBM':
            model = lgb.LGBMRegressor(max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1, colsample_bytree=0.3, learning_rate=0.001, n_jobs=16, random_state=1, verbosity=-1)

        # # Fit the model on the entire df
        # start_time = time.time()
        model.fit(train[features], train[target])
        # end_time = time.time()

        # training_time = end_time - start_time
        # print(f"Time taken to train the model: {training_time:.2f} seconds")

        predictions = model.predict(test[features])
        preds.extend(predictions)
        targets.extend(test[target])

    score = stats.pearsonr(preds, targets)
    all_scores.append(score[0])
    # process = psutil.Process(os.getpid())
    # memory_info = process.memory_info()
    # memory_usage_gb = memory_info.rss / (1024 ** 3)
    # print(f"Memory usage of this job: {memory_usage_gb:.2f} GB")
    return all_scores


def train_baseline(df, features, target, i):
    print(i)
    print(target)
    all_scores = []

    # Drop rows where the target column is NaN
    df = df.dropna(subset=[target])

    # Train the model on all data in df
    if MODEL == 'ridge':
        model = linear_model.RidgeCV(alphas=(0.01, 0.1, 0.5, 1, 10, 100))
    elif MODEL == 'LGBM':
        model = lgb.LGBMRegressor(max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1, colsample_bytree=0.3, learning_rate=0.001, n_jobs=8, random_state=1)

    # Fit the model on the entire df
    model.fit(df[features], df[target])

    # Make predictions on the same data (train set predictions)
    preds = model.predict(df[features])
    targets = df[target].values

    # Calculate score using Pearson correlation
    score = stats.pearsonr(preds, targets)
    all_scores.append(score[0])

    return all_scores


def train_classification(df, features, target, i):
    print(i)
    df = df.dropna(subset=[target])
    df[target] = df[target].apply(lambda x: 0 if x == -4 else 1)

    kf = KFold(n_splits=5, shuffle=False, random_state=1)
    preds = []
    targets = []
    pred_probs_list = []

    for train_index, test_index in kf.split(df):
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
                colsample_bytree=0.3, 
                learning_rate=0.001, 
                n_jobs=8, 
                random_state=1,
                objective='binary',
                is_unbalance=True
            )
        model.fit(train[features], train[target])
        predictions = model.predict(test[features])
        pred_probs = model.predict_proba(test[features])[:, 1]  # For AUC calculation
    
        preds.extend(predictions)
        targets.extend(test[target])
        pred_probs_list.extend(pred_probs)

    # Calculate accuracy and AUC
    accuracy = accuracy_score(targets, preds)
    auc = roc_auc_score(targets, pred_probs_list)

    return accuracy, auc


def reverse_prediction(diet_mb, features, target_input):
    mb_features = target_input + ['Richness', 'Shannon_diversity'] + ['age', 'gender']
    diet_targets = [feat for feat in features if feat not in ['age', 'gender']]
    # Exclude "gender" from standardization
    features_to_standardize = [feature for feature in mb_features if feature != "gender"]

    # Standardize only the selected features
    scaler = StandardScaler()
    diet_mb[features_to_standardize] = pd.DataFrame(
        scaler.fit_transform(diet_mb[features_to_standardize]), 
        columns=diet_mb[features_to_standardize].columns, 
        index=diet_mb[features_to_standardize].index
    )

    # Return updated data
    return diet_mb, mb_features, diet_targets


def stub_job(q):
    print ('job started permutations.py')
    print(MODEL)
    print(TARGETS)
    print(SPLIT)
    print(PROBLEM)
    params_methods = {}
    params_results = {}
    diet_mb = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_baseline_train.pkl")

    with open(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/mb_scaler{pathways}.pkl", 'rb') as f:
        mb_scaler = pickle.load(f)

    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists{pathways}.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists
    diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_')
    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]

    diet_mb_log = mb_scaler.inverse_transform(diet_mb[target_input])
    diet_mb_log = pd.DataFrame(diet_mb_log, columns=target_input, index=diet_mb.index)

    if PROBLEM == 'reverse':
        diet_mb, features, target_input = reverse_prediction(diet_mb, features, target_input)

    if TARGETS == 'abundance' or TARGETS == 'pathways' or PROBLEM == 'reverse':
        binned_targets, microbe_bin_mapping = choose_target_bins(diet_mb_log, target_input)
        # for bin in binned_targets:
        #     print(f"Bin {bin}: {microbe_bin_mapping[bin]}")
        loop_targets = binned_targets
        if not microbe_bin_mapping:
            raise RuntimeError("microbe_bin_mapping is empty; aborting save.")
        with open(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/microbe_bin_mapping_perm{pathways}.pkl", 'wb') as f:
            pickle.dump(microbe_bin_mapping, f)
            print("SAVED MAPPING")
    elif TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    print('preparing queue')

    output = pd.Series()
    all_job_stubs = {}
    print("🚀 Submitting jobs for all permutations...")
    for random_state in range(0, PERMUTATIONS):
        permuted_df = permute_df(diet_mb, random_state, features, loop_targets)
        all_job_stubs[random_state] = {}
        
        # print(f"Submitting jobs for permutation: {random_state}") # Optional: for progress tracking
        for i, target in enumerate(loop_targets):
            # Determine which function to call based on your settings
            if PROBLEM == 'classification':
                job_function = train_classification
            elif SPLIT == 'kfold':
                job_function = stub_subjob
            elif SPLIT == 'all_baseline':
                job_function = train_baseline
            
            # Submit the job and store its stub, keyed by random_state and target name
            stub_method = q.method(job_function, (permuted_df, features, target, i,))
            all_job_stubs[random_state][target] = stub_method

    print("\n✅ All jobs have been submitted to the scheduler.\n")

    # A list to hold the processed DataFrame from each permutation
    all_perm_dataframes = []

    print("⏳ Waiting for results and reconstructing output...")
    for random_state in range(0, PERMUTATIONS):
        # Dictionary to hold the results for the current permutation
        current_perm_results = {}
        
        for target in loop_targets:
            # Retrieve the job stub for the specific permutation and target
            stub_to_wait_for = all_job_stubs[random_state][target]
            
            # Wait for the result of this specific job
            current_perm_results[target] = q.waitforresult(stub_to_wait_for)
        
        # Process the collected results for this permutation, same as your original code
        perm_output = pd.DataFrame(current_perm_results).transpose()
        perm_output = perm_output.apply(lambda x : x.explode(), axis=1)
        
        # Append the processed DataFrame to our list
        all_perm_dataframes.append(perm_output)
        # print(f"Collected and processed results for permutation: {random_state}") # Optional progress tracking

    # --- Final Assembly ---

    # Concatenate all the individual permutation DataFrames column-wise
    output = pd.concat(all_perm_dataframes, axis=1)

    # Final processing, same as your original code
    output.dropna(axis=1, inplace=True)
    if target == "abundance":
        output.index = binned_targets
    output.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/output_" + MODEL + '_' + TARGETS + suffix + '_perm.pkl')


def main():
    os.chdir('/net/mraid20/export/genie/LabData/Analyses/tomerse/tmp/')
    with config.qp(jobname='lgbm', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=8, max_u=200, _mem_def='1G') as q:
        os.chdir('/net/mraid20/export/genie/LabData/Analyses/tomerse/tmp/')
        q.startpermanentrun()
        stub_job(q)


if __name__ == '__main__':
    if TARGETS == 'div' and PROBLEM == 'classification':
        raise ValueError("Cannot perform classification on diversity targets. Change TARGETS to 'abundance' or PROBLEM to ''.")
    addloglevels.sethandlers()
    main()
    output = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/output_" + MODEL + '_' + TARGETS + suffix + '_perm.pkl')
    print(output)