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
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL = 'LGBM' # 'LGBM' or 'ridge' or 'logistic'
TARGETS = 'abundance' # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
STAGE = 'baseline' # 'baseline' or '02_visit' or '04_visit'
PROBLEM = 'regression' # 'classification' or 'regression' or 'given_presence' or 'reverse'
SPECIES = 'segal_species' # 'mpa_species' or 'segal_species'
robust_flag = False

pathways = '' if TARGETS != 'pathways' else '_pathways'
# stage_suf = '' if STAGE == 'baseline' else '_' + STAGE
stage_suf = '_' + STAGE


def predict(df, features, target, i, models_list):
    print(i)
    all_scores = []
    all_p_values=[]
    all_feat_importances = []
    all_feat_names = []
    predicted_abundances = []
    measured_abundances = []

    model = models_list[i]

    if model is None:
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
    print(i)

    all_feat_importances = []
    predicted_abundances = []
    measured_abundances = []

    model = models_list[i]

    df[target] = df[target].apply(lambda x: 0 if x == -4 else 1)
    if model is None:
        return np.nan, np.nan, [np.nan], [np.nan], [np.nan]
    
    predictions = model.predict(df[features])
    predicted_abundances.append(list(predictions))
    measured_abundances.append(list(df[target]))
    pred_probs = model.predict_proba(df[features])[:, 1]  # For AUC calculation

    # Calculate accuracy and AUC
    accuracy = accuracy_score(df[target].values, predictions)
    # auc = roc_auc_score(test[target], pred_probs)
    auc = roc_auc_score(df[target].values, pred_probs)

    if MODEL == 'logistic':
        all_feat_importances.append(model.coef_)
    if MODEL == 'LGBM':
        all_feat_importances.append(model.feature_importances_)
    
    return accuracy, auc, all_feat_importances, predicted_abundances, measured_abundances


def stub_job():
    print('job started train_models.py')
    print(MODEL)
    print(TARGETS)
    params_results = {}
    print(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/{SPECIES}/models_' + MODEL + '_' + TARGETS + '_longitudinal.pkl')
    models_list = pickle.load(open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/{SPECIES}/models_' + MODEL + '_' + TARGETS + '_longitudinal.pkl', 'rb'))
    train_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_baseline_train.pkl")
    test_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}{stage_suf}_test.pkl")
    test_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_baseline_test.pkl")
    baseline_subjects = list(test_baseline.index)
    visit_subjects = list(test_visit.index)
    print("Length of baseline_subjects:", len(baseline_subjects))
    print("Length of visit_subjects:", len(visit_subjects))
    # test_baseline = diet_mb.loc[diet_mb.index.isin(visit_subjects)]
    # test_visit = test_visit.loc[test_visit.index.isin(test_baseline.index)]
    # print(test_baseline.shape)  
    # print(test_visit.shape)  

    if TARGETS == 'health':
        train_baseline.dropna(inplace=True)
        test_baseline.dropna(inplace=True)
        test_visit.dropna(inplace=True)

    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists{pathways}.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists
    test_baseline.columns = test_baseline.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_')
    test_visit.columns = test_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_')
    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]

    if PROBLEM == 'reverse':
        diet_features = features  # Store original features (diet variables)
        features = target_input + ['Richness', 'Shannon_diversity'] + base_features  # Microbial features become predictors
        target_input = [feat for feat in diet_features if feat not in ['age', 'gender']]  # Diet variables become targets

    if TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'abundance' or TARGETS == 'pathways' or PROBLEM == 'reverse':
        loop_targets = target_input
    elif TARGETS == 'health':
        loop_targets = ['modified_HACK_top17_score', 'GMWI2_score']
    print('preparing queue')
    for i in range(len(loop_targets)):
        if PROBLEM == 'classification':
            if STAGE == 'baseline':
                params_results[i] = predict_classification(test_baseline, features, loop_targets[i], i, models_list)
            elif STAGE != 'baseline':
                params_results[i] = predict_classification(test_visit, features, loop_targets[i], i, models_list)
        else:
            if STAGE == 'baseline':
                params_results[i] = predict(test_baseline, features, loop_targets[i], i, models_list)
            elif STAGE != 'baseline':
                params_results[i] = predict(test_visit, features, loop_targets[i], i, models_list)


    output = pd.DataFrame(params_results).transpose()
    output = output.apply(lambda x : x.explode(), axis=1)
    output.to_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/predictions_"+ MODEL + '_' + TARGETS + '_' + STAGE + '.pkl')


def read_results(df):
    output = []
    for col in df.columns:
        output.append(df[col])
    return tuple(output)


def predict_robustness():
    print ('job started predict.py robustness!')
    print(MODEL)
    print(TARGETS)
    params_results = {}
    diet_mb = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}.pkl")
    test_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb{pathways}_{stage_suf}.pkl")
    
    baseline_subjects = list(diet_mb.index)
    visit_subjects = list(test_visit.index)
    print("Length of baseline_subjects:", len(baseline_subjects))
    print("Length of visit_subjects:", len(visit_subjects))
    test_baseline = diet_mb.loc[diet_mb.index.isin(visit_subjects)]
    test_visit = test_visit.loc[test_visit.index.isin(test_baseline.index)]
    print(test_baseline.shape)  
    print(test_visit.shape) 
    # test_baseline is my holdout set 
    diet_mb = diet_mb.loc[~diet_mb.index.isin(visit_subjects)]

    with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/my_lists.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists

    test_baseline.columns = test_baseline.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_')
    test_visit.columns = test_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_')
    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]

    # Filter for only top 100 microbes
    lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/output_LGBM_abundance.pkl"))
    top_microbes = lgbm_diet_scores.sort_values(ascending=False)
    top_microbes = top_microbes.head(100)
    target_input = [target_input[i] for i in top_microbes.index if i < len(target_input)]
    
    n_samples = len(diet_mb)
    current_df = diet_mb.copy()  # Start with the full DataFrame

    iteration = 1
    while n_samples > 1:
        print(f"Iteration {iteration}: {n_samples} samples")

        params_results = {}
        models_list = pickle.load(open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/robustness_models/models_' + MODEL + '_' + TARGETS + '_' + str(n_samples) + '_samples' + PROBLEM + '.pkl', "rb"))
        if TARGETS == 'div':
            loop_targets = ['Richness', 'Shannon_diversity']
        elif TARGETS == 'abundance':
            loop_targets = target_input
        print('preparing queue')
        for i in range(len(loop_targets)):
            params_results[i] = predict(test_baseline, features, loop_targets[i], i, models_list)

        output = pd.DataFrame(params_results).transpose()
        output = output.apply(lambda x : x.explode(), axis=1)
        output.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/robustness/output_" + MODEL + '_' + TARGETS + '_' + str(n_samples) + '_samples' + PROBLEM + '.pkl')

        # Reduce samples by half for the next iteration
        n_samples = math.ceil(n_samples / 2)  # Use math.ceil to ensure at least one sample
        current_df = current_df.sample(n=n_samples, random_state=iteration).reset_index(drop=True)
        iteration += 1


if __name__ == '__main__':
    print(f"MODEL: {MODEL}")
    print(f"TARGETS: {TARGETS}")
    print(f"STAGE: {STAGE}")
    print(f"PROBLEM: {PROBLEM}")
    print(f"robust_training: {robust_flag}")
    # print(f"no_var_features: {no_var_features}")
    if robust_flag:
        predict_robustness()
    else:
        stub_job()   
    output = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/predictions_"+ MODEL + '_' + TARGETS + '_' + STAGE + '.pkl')
    print(output)