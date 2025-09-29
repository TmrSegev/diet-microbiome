import math
import os
import pickle
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
import shap
from sklearn.model_selection import cross_val_predict


outdir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/test/'
TARGETS = 'div' # 'div' or 'abundance'

def stub_subjob(diet_mb, features, target, models_list):
    all_scores = []
    all_p_values=[]
    all_feat_importances = []
    # all_feat_names = []
    all_preds = []
    all_targets = []
    # target = target_input[i]
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    preds = []
    targets = []

    for train_index, test_index in kf.split(diet_mb):
        train = diet_mb.iloc[train_index]
        test = diet_mb.iloc[test_index]

        model = lgb.LGBMRegressor(max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1, colsample_bytree=0.3, learning_rate=0.001, n_jobs=16, random_state=1)
        model.fit(train[features], train[target])
        predictions = model.predict(test[features])
        preds.extend(predictions)
        targets.extend(test[target])

    score, p_value = stats.pearsonr(preds, targets)
    # all_scores.append(score[0])
    # all_p_values.append(score[1])
    all_feat_importances.append(model.feature_importances_)
    # all_feat_names.append(model.feature_name_)
    all_preds.append(preds)
    all_targets.append(targets)

    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(train[features])

    # shap.summary_plot(shap_values, train[features], show=False)
    # plt.title("LGBM " + target)
    # plt.savefig("/home/tomerse/PycharmProjects/pythonProject/lgbm_div_SHAP_{}.png".format(target), dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
    # plt.close()
    # plt.clf()

    # Train final model on the entire dataset
    model.fit(diet_mb[features], diet_mb[target])
    models_list.append(model)

    return [score], [p_value], all_feat_importances, all_preds, all_targets #, shap_values


def stub_job():
    print ('job started')
    print(TARGETS)
    params_methods = {}
    params_results = {}
    models_list = []
    diet_mb_10k = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/data/diet_mb.pkl")
    # with open('/home/tomerse/PycharmProjects/pythonProject/my_lists.pkl', 'rb') as file:
    #     loaded_lists = pickle.load(file)
    # base_features, features, target_input = loaded_lists
    # with open('/home/tomerse/PycharmProjects/pythonProject/data/overlap_features.pkl', 'rb') as file:
    #     loaded_lists = pickle.load(file)
    # overlap_features = loaded_lists
    # with open('/home/tomerse/PycharmProjects/pythonProject/data/new_targets.pkl', 'rb') as file:
    #     loaded_lists = pickle.load(file)
    # new_targets = loaded_lists
    with open('/home/tomerse/PycharmProjects/pythonProject/data/my_lists_pnp3.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    overlap_features, overlap_targets = loaded_lists
    diet_mb_10k.columns = diet_mb_10k.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_').str.replace('(', '_').str.replace(')', '_').str.replace('.', '_').str.replace(',', '_')
    overlap_features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in overlap_features]
    # features_to_drop = ['Beans_black_eyed_peas', 'Mille_feuille', 'cereal_type__Cornflakes_with_sugar__cookies__pillows__etc__', 'never_eat__Sugar_or_foods___beverages_that_contain_sugar']
    # features = [feature for feature in features if feature not in features_to_drop]
    # target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]
    overlap_targets = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in overlap_targets]

    # Remove targets that are in PNP3 but not in 10K:
    # targets_to_remove = ['fBin__314_gBin__1271_sBin__1892', 'fBin__376_gBin__1543_sBin__2294', 'fBin__100_gBin__473_sBin__693', 'fBin__555_gBin__2161_sBin__3325', 'fBin__73_gBin__358_sBin__515', 'fBin__384_gBin__1593_sBin__2378', 'fBin__100_gBin__473_sBin__694', 'fBin__35_gBin__81_sBin__119', 'fBin__383_gBin__1579_sBin__2360', 'fBin__96_gBin__457_sBin__669', 'fBin__97_gBin__462_sBin__676', 'fBin__376_gBin__1549_sBin__2305', 'fBin__318_gBin__1278_sBin__1907', 'fBin__94_gBin__449_sBin__656']

    # diet_mb.columns = [col for col in diet_mb.columns if col not in targets_to_remove]
    # new_targets = [target for target in new_targets if target not in targets_to_remove]

    # for i in range(0, len(target_input)):
    if TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'abundance':
        loop_targets = overlap_targets
    for i in range(0, len(loop_targets)):
        print(i)
        params_methods[i] = stub_subjob(diet_mb_10k, overlap_features, overlap_targets[i], models_list)

    pickle.dump(models_list, open('/home/tomerse/PycharmProjects/pythonProject/models/models_lgbm_overlap_features_' + TARGETS + '.pkl', "wb"))
    output = pd.DataFrame(params_methods).transpose()
    output = output.apply(lambda x : x.explode(), axis=1)
    output.columns = [f'column{i}' for i in range(len(output.columns))]
    output.to_csv("/net/mraid20/export/genie/LabData/Analyses/tomerse/test/lightGBM_overlap_features_" + TARGETS + ".csv", index=False)
    



def main():
    stub_job()

if __name__ == '__main__':
    main()
    output = pd.read_csv("/net/mraid20/export/genie/LabData/Analyses/tomerse/test/lightGBM_overlap_features_" + TARGETS + ".csv")
    print(output)