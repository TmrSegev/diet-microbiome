from LabData import config_global as config
from LabUtils import Utils
from LabUtils import addloglevels
import math
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODEL = 'LGBM' # 'LGBM' or 'ridge' or 'logistic'
TARGETS = 'abundance' # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
STAGE = 'baseline' # 'baseline' or 'intervention'
PROBLEM = 'regression' # 'classification' or 'regression' or 'given_presence' or 'reverse'
SPECIES = 'segal_species' # 'mpa_species' or 'segal_species'
DATASET = 'HPP' # 'pnp3' or 'AUS' or 'HPP'
CLR_flag = False
base_training = False
no_var_features = False
nutrients_only = False  # Use nutrients_only models for AUS cohort models, if False will use 10k_pnp3 suffix

home_path = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'

pathways = '' if TARGETS != 'pathways' else '_pathways'
CLR_suf = '_CLR' if CLR_flag else ''
suffix_var = '_no_var_features' if no_var_features else ''
suffix_features = '_nutrients_only' if nutrients_only else '_10k_pnp3'
suffix = '_longitudinal'  # Assuming longitudinal split was used
base = 'base_' if base_training else ''
target_part = 'diet' if PROBLEM == 'reverse' else f'{TARGETS}'


def predict(df, features, target, i, models_list):
    print(f"Predicting target {i}: {target}")
    all_scores = []
    all_p_values = []
    all_feat_importances = []
    predicted_abundances = []
    measured_abundances = []

    model = models_list[i]

    if model is None:
        return all_scores, all_p_values, all_feat_importances, predicted_abundances, measured_abundances
    
    # Check if target exists in dataframe
    if target not in df.columns:
        print(f"Warning: Target {target} not found in dataframe. Skipping.")
        return all_scores, all_p_values, all_feat_importances, predicted_abundances, measured_abundances
    
    predictions = model.predict(df[features])
    predicted_abundances.append(list(predictions))
    measured_abundances.append(list(df[target]))

    score = stats.pearsonr(predictions, df[target])
    all_scores.append(score[0])
    all_p_values.append(score[1])
    if MODEL == 'ridge':
        all_feat_importances.append(model.coef_)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
    
    return all_scores, all_p_values, all_feat_importances, predicted_abundances, measured_abundances


def predict_classification(df, features, target, i, models_list):
    print(f"Predicting target {i}: {target} (classification)")

    all_feat_importances = []
    predicted_abundances = []
    measured_abundances = []

    model = models_list[i]

    if target not in df.columns:
        print(f"Warning: Target {target} not found in dataframe. Skipping.")
        return np.nan, np.nan, [np.nan], [np.nan], [np.nan]

    df[target] = df[target].apply(lambda x: 0 if x == -4 else 1)
    if model is None:
        return np.nan, np.nan, [np.nan], [np.nan], [np.nan]
    
    predictions = model.predict(df[features])
    predicted_abundances.append(list(predictions))
    measured_abundances.append(list(df[target]))
    pred_probs = model.predict_proba(df[features])[:, 1]  # For AUC calculation

    # Calculate accuracy and AUC
    accuracy = accuracy_score(df[target].values, predictions)
    auc = roc_auc_score(df[target].values, pred_probs)

    if MODEL == 'logistic':
        all_feat_importances.append(model.coef_)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
    
    return accuracy, auc, all_feat_importances, predicted_abundances, measured_abundances


def apply_scaling(df, feature_list, target_input):
    """Apply the same scaling that was used during training"""
    df = df.copy()
    
    # Load scalers
    diet_scaler_path = f"{home_path}data/{SPECIES}/diet_scaler{pathways}{CLR_suf}{suffix_features}.pkl"
    mb_scaler_path = f"{home_path}data/{SPECIES}/mb_scaler{pathways}{CLR_suf}{suffix_features}.pkl"
    age_scaler_path = f"{home_path}data/{SPECIES}/age_scaler{pathways}{CLR_suf}{suffix_features}.pkl"
    div_scaler_path = f"{home_path}data/{SPECIES}/div_scaler{pathways}{CLR_suf}{suffix_features}.pkl"
    
    with open(diet_scaler_path, 'rb') as file:
        diet_scaler = pickle.load(file)
    # Only load mb_scaler if not predicting div (div uses div_scaler)
    mb_scaler = None
    if TARGETS != 'div' and os.path.exists(mb_scaler_path):
        with open(mb_scaler_path, 'rb') as file:
            mb_scaler = pickle.load(file)
    with open(age_scaler_path, 'rb') as file:
        age_scaler = pickle.load(file)
    
    # Sanitize feature names to match column names
    features_sanitized = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in feature_list]
    # Filter to only features that exist in dataframe and exclude age and sex
    features_to_scale = [f for f in features_sanitized if f in df.columns and f not in ['age', 'sex']]
    
    # Apply diet scaler to features (excluding age and sex)
    if features_to_scale and hasattr(diet_scaler, 'mean_'):
        df.loc[:, features_to_scale] = diet_scaler.transform(df[features_to_scale])
    elif features_to_scale:
        print(f"Warning: diet_scaler is not fitted. Skipping diet feature scaling for {len(features_to_scale)} features.")
    
    # Apply age scaler
    if 'age' in df.columns:
        df.loc[:, ['age']] = age_scaler.transform(df[['age']])
    
    # Apply mb_scaler to targets (only for abundance/pathways, not div)
    # For div targets, use div_scaler instead
    div_features = ['Richness', 'Shannon_diversity']
    if TARGETS == 'div':
        # Use div_scaler for diversity targets
        div_features_present = [feat for feat in div_features if feat in df.columns]
        if div_features_present and os.path.exists(div_scaler_path):
            with open(div_scaler_path, 'rb') as file:
                div_scaler = pickle.load(file)
            if hasattr(div_scaler, 'mean_'):
                if len(div_features_present) == len(div_features):
                    df.loc[:, div_features_present] = div_scaler.transform(df[div_features_present])
                else:
                    # Manual transform for partial features
                    div_to_idx = {d: i for i, d in enumerate(div_features)}
                    indices = [div_to_idx[d] for d in div_features_present]
                    df.loc[:, div_features_present] = (df[div_features_present].values - div_scaler.mean_[indices]) / div_scaler.scale_[indices]
    else:
        # Use mb_scaler for abundance/pathways targets
        targets_present = [t for t in target_input if t in df.columns]
        if targets_present and mb_scaler is not None and hasattr(mb_scaler, 'mean_'):
            # If all targets are present and in correct order, use transform directly
            if len(targets_present) == len(target_input):
                df.loc[:, targets_present] = mb_scaler.transform(df[targets_present])

        # Also apply div_scaler to div features if they exist (as additional features)
        div_features_present = [feat for feat in div_features if feat in df.columns]
        if div_features_present and os.path.exists(div_scaler_path):
            with open(div_scaler_path, 'rb') as file:
                div_scaler = pickle.load(file)
            if hasattr(div_scaler, 'mean_'):
                if len(div_features_present) == len(div_features):
                    df.loc[:, div_features_present] = div_scaler.transform(df[div_features_present])

    return df


def predict_pnp3():
    print(f'Starting {DATASET} prediction')
    print(f"MODEL: {MODEL}")
    print(f"TARGETS: {TARGETS}")
    print(f"STAGE: {STAGE}")
    print(f"PROBLEM: {PROBLEM}")
    print(f"SPECIES: {SPECIES}")
    print(f"DATASET: {DATASET}")
    print(f"nutrients_only: {nutrients_only}")
    
    # Load models
    models_path = f'{home_path}models/{PROBLEM}/{SPECIES}/models_{base}{MODEL}_{target_part}{suffix}{suffix_var}{CLR_suf}{suffix_features}.pkl'
    print(f"Loading models from: {models_path}")
    models_dict = pickle.load(open(models_path, 'rb'))
    
    # Convert models_dict to list of models
    # For baseline training: models_dict is {i: model}
    # For kfold: models_dict is {i: [model1, model2, ...]} (list of models per fold)
    if isinstance(models_dict, dict):
        # Check if values are lists (kfold) or single models (baseline)
        first_value = models_dict[list(models_dict.keys())[0]]
        if isinstance(first_value, list):
            # Kfold: use the first model from each list (or average predictions)
            # For simplicity, using first model - can be extended to average predictions
            models_list = [models_dict[i][0] if models_dict[i] else None for i in sorted(models_dict.keys())]
        else:
            # Baseline: single model per target
            models_list = [models_dict[i] for i in sorted(models_dict.keys())]
    else:
        # If it's already a list
        models_list = models_dict
    
    # Load dataframes based on DATASET
    df_full = None  # For HPP with multiple runs
    n_aus_subjects = None  # For HPP sampling
    if DATASET == 'pnp3':
        if STAGE == 'baseline':
            df = pd.read_pickle(home_path + 'data/diet_mb_pnp3_baseline.pkl')
        elif STAGE == 'intervention':
            df = pd.read_pickle(home_path + 'data/diet_mb_pnp3_intervention.pkl')
        else:
            raise ValueError(f"STAGE must be 'baseline' or 'intervention', got {STAGE}")
    elif DATASET == 'AUS':
        if STAGE == 'baseline':
            df = pd.read_pickle(home_path + 'data/segal_species/diet_mb_AUS_baseline.pkl')
        else:
            raise ValueError(f"For AUS dataset, only 'baseline' STAGE is supported, got {STAGE}")
    elif DATASET == 'HPP':
        # For HPP dataset, load the test file from train_models.py
        # This matches the file saved in train_models.py line 599
        if nutrients_only:
            # Load pre-scaled test file for nutrients_only
            df_full = pd.read_pickle(f"{home_path}data/{SPECIES}/diet_mb{pathways}_baseline{CLR_suf}{suffix_features}_test.pkl")
            print(f"Loaded pre-scaled HPP dataset (diet_mb{pathways}_baseline{CLR_suf}{suffix_features}_test.pkl)")
            
            # Sample the same number of subjects as AUS dataset
            aus_df = pd.read_pickle(home_path + 'data/segal_species/diet_mb_AUS_baseline.pkl')
            n_aus_subjects = len(aus_df)
            print(f"AUS dataset has {n_aus_subjects} subjects")
            print(f"HPP dataset has {len(df_full)} subjects before sampling")
            
            # Store full dataframe for multiple runs (will sample in prediction loop)
            if len(df_full) < n_aus_subjects:
                print(f"Warning: HPP dataset has fewer subjects ({len(df_full)}) than AUS ({n_aus_subjects}). Using all available subjects.")
                df = df_full  # Use all available for single run
                n_aus_subjects = None  # Don't do multiple runs if not enough subjects
            else:
                # Will sample in the prediction loop with different random seeds
                df = df_full.sample(n=n_aus_subjects, random_state=0)  # Initial sample for feature/target setup
        else:
            # For nutrients_only=False, load the test file with _10k_pnp3 suffix (already pre-scaled)
            df_full = pd.read_pickle(f"{home_path}data/{SPECIES}/diet_mb{pathways}_baseline{CLR_suf}{suffix_features}_test.pkl")
            print(f"Loaded pre-scaled HPP dataset (diet_mb{pathways}_baseline{CLR_suf}{suffix_features}_test.pkl)")
            
            # Sample the same number of subjects as PNP3 dataset (for _10k_pnp3 models)
            pnp3_df = pd.read_pickle(home_path + 'data/diet_mb_pnp3_baseline.pkl')
            n_pnp3_subjects = len(pnp3_df)
            print(f"PNP3 dataset has {n_pnp3_subjects} subjects")
            print(f"HPP dataset has {len(df_full)} subjects before sampling")
            
            # Store full dataframe for multiple runs (will sample in prediction loop)
            if len(df_full) < n_pnp3_subjects:
                print(f"Warning: HPP dataset has fewer subjects ({len(df_full)}) than PNP3 ({n_pnp3_subjects}). Using all available subjects.")
                df = df_full  # Use all available for single run
                n_aus_subjects = None  # Don't do multiple runs if not enough subjects
            else:
                # Will sample in the prediction loop with different random seeds
                df = df_full.sample(n=n_pnp3_subjects, random_state=0)  # Initial sample for feature/target setup
                n_aus_subjects = n_pnp3_subjects  # Use PNP3 size for sampling
        if STAGE != 'baseline':
            raise ValueError(f"For HPP dataset, only 'baseline' STAGE is supported, got {STAGE}")
    else:
        raise ValueError(f"DATASET must be 'pnp3', 'AUS', or 'HPP', got {DATASET}")
    
    print(f"Loaded {DATASET} {STAGE} dataframe with shape: {df.shape}")
    
    # Load features based on nutrients_only flag
    if nutrients_only:
        with open(home_path + 'data/nutr_list_aus.pkl', 'rb') as file:
            feature_list = pickle.load(file)
    else:
        with open(home_path + 'data/my_lists_pnp3.pkl', 'rb') as file:
            feature_list, _ = pickle.load(file)
    
    # Load targets from my_lists_pnp3.pkl (targets are the same)
    with open(home_path + 'data/my_lists_pnp3.pkl', 'rb') as file:
        _, targets_10k = pickle.load(file)
    
    # Sanitize column names in dataframe
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    
    # Sanitize feature and target names
    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in feature_list]
    # Add 'age' and 'sex' if not already present
    for feat in ['age', 'sex']:
        if feat not in features:
            features.append(feat)
    
    # If base_training is True, only use base features (age, sex)
    if base_training:
        base_features = ['age', 'sex']
        features = [f for f in base_features if f in df.columns]
    
    # For TARGETS=abundance, use targets_10k; otherwise use appropriate targets
    if TARGETS == 'abundance':
        target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in targets_10k]
    elif TARGETS == 'div':
        target_input = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'health':
        target_input = ['modified_HACK_top17_score', 'GMWI2_score']
    elif TARGETS == 'pathways':
        # For pathways, we'd need to load from the appropriate file
        # For now, use targets_10k as fallback
        target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in targets_10k]
    else:
        raise ValueError(f"Unsupported TARGETS: {TARGETS}")
    
    # Filter features to only those that exist in dataframe
    features = [f for f in features if f in df.columns]
    target_input = [t for t in target_input if t in df.columns]
    
    print(f"Features count: {len(features)}")
    print(f"Targets count: {len(target_input)}")
    
    # Apply scaling (skip if HPP dataset, as it's already pre-scaled in train_models.py)
    if DATASET == 'HPP':
        print("Skipping scaling - HPP dataset is already pre-scaled")
    else:
        print("Applying scaling...")
        df = apply_scaling(df, feature_list, target_input)
    
    # Determine loop targets
    if TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'abundance' or TARGETS == 'pathways' or PROBLEM == 'reverse':
        loop_targets = target_input
    elif TARGETS == 'health':
        loop_targets = ['modified_HACK_top17_score', 'GMWI2_score']
    
    # Filter loop_targets to only those that exist in dataframe
    loop_targets = [t for t in loop_targets if t in df.columns]
    
    print(f"Loop targets: {len(loop_targets)}")
    
    # For HPP dataset, run 5 times and average results (for both nutrients_only and _10k_pnp3)
    if DATASET == 'HPP' and df_full is not None and n_aus_subjects is not None and len(df_full) >= n_aus_subjects:
        n_runs = 5
        all_run_results = []
        
        print(f"\nRunning {n_runs} prediction runs with different random seeds...")
        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")
            # Sample n_aus_subjects from HPP dataset with different random seed for each run
            df_run = df_full.sample(n=n_aus_subjects, random_state=run)
            print(f"Sampled {len(df_run)} subjects from HPP dataset (random_state={run})")
            
            # Make predictions for this run
            params_results = {}
            for i in range(len(loop_targets)):
                if i >= len(models_list):
                    print(f"Warning: Not enough models for target {i}. Skipping.")
                    continue
                
                if PROBLEM == 'classification':
                    params_results[i] = predict_classification(df_run, features, loop_targets[i], i, models_list)
                else:
                    params_results[i] = predict(df_run, features, loop_targets[i], i, models_list)
            
            # Store results for this run
            output_run = pd.DataFrame(params_results).transpose()
            output_run = output_run.apply(lambda x: x.explode(), axis=1)
            all_run_results.append(output_run)
            
            # Print Pearson correlations for this run
            if PROBLEM != 'classification':
                print(f"Run {run + 1} Pearson correlations:")
                for idx in output_run.index:
                    score = output_run.loc[idx, 0]
                    # Handle case where score might be a list
                    if isinstance(score, list) and len(score) > 0:
                        score_val = score[0]
                    else:
                        score_val = score
                    target_name = loop_targets[idx] if idx < len(loop_targets) else f"target_{idx}"
                    print(f"  Target {idx} ({target_name}): {score_val:.4f}")
        
        # Calculate mean of Pearson correlations across all runs
        print(f"\nCalculating mean Pearson correlations across {n_runs} runs...")
        mean_output = all_run_results[0].copy()
        
        # Extract scores (Pearson correlations) from each run and calculate mean
        for idx in mean_output.index:
            if PROBLEM == 'classification':
                # For classification, we have accuracy and AUC, not Pearson correlations
                # Keep the last run's results for classification
                mean_output.loc[idx] = all_run_results[-1].loc[idx]
            else:
                # For regression, calculate mean of Pearson correlations (first column is scores)
                scores_list = [run_results.loc[idx, 0] for run_results in all_run_results if idx in run_results.index]
                if scores_list:
                    # Handle case where scores might be lists
                    score_values = []
                    for s in scores_list:
                        if isinstance(s, list) and len(s) > 0:
                            score_values.append(s[0])
                        else:
                            score_values.append(s)
                    mean_score = np.mean(score_values)
                    mean_output.loc[idx, 0] = mean_score
                    target_name = loop_targets[idx] if idx < len(loop_targets) else f"target_{idx}"
                    print(f"Target {idx} ({target_name}): Mean Pearson correlation = {mean_score:.4f} (from {n_runs} runs)")
        
        # Save mean results
        output = mean_output
        output_path = f"{home_path}data/{PROBLEM}/{SPECIES}/predictions_{DATASET}_{MODEL}_{TARGETS}_{STAGE}{suffix_features}_mean.pkl"
    else:
        # Single run for non-HPP or non-nutrients_only cases
        # Make predictions
        params_results = {}
        for i in range(len(loop_targets)):
            if i >= len(models_list):
                print(f"Warning: Not enough models for target {i}. Skipping.")
                continue
            
            if PROBLEM == 'classification':
                params_results[i] = predict_classification(df, features, loop_targets[i], i, models_list)
            else:
                params_results[i] = predict(df, features, loop_targets[i], i, models_list)
        
        # Save results
        output = pd.DataFrame(params_results).transpose()
        output = output.apply(lambda x: x.explode(), axis=1)
        output_path = f"{home_path}data/{PROBLEM}/{SPECIES}/predictions_{DATASET}_{MODEL}_{TARGETS}_{STAGE}{suffix_features}.pkl"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_pickle(output_path)
    print(f"\nPredictions saved to: {output_path}")
    
    return output


if __name__ == '__main__':
    output = predict_pnp3()
    print("\nPredictions summary:")
    print(output)

