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


outdir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/test/'

# def stub_subjob(diet_mb, features, target, models_list, target_input, mb_names):
#     print(target)
#     model = models_list[target]
#
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(diet_mb[features])
#     # print(shap_values.shape)
#     # shap.summary_plot(shap_values, diet_mb[features], show=False)
#     # # shap.summary_plot(shap_values)
#     # microbe_name = mb_names.loc[target_input[int(target)], 'species']
#     # print(microbe_name)
#     # plt.title(str(microbe_name))
#     # plt.savefig("/home/tomerse/PycharmProjects/pythonProject/SHAP_plots/lgbm_mic_SHAP_{}_{}.png".format(target, microbe_name), dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
#     # plt.close()
#     # plt.clf()
#
#     return shap_values

def stub_subjob(diet_mb, features, target, models_list, i):
    print(i)
    all_scores = []
    all_p_values=[]
    all_feat_importances = []
    all_feat_names = []
    all_preds = []
    all_targets = []
    # target = target_input[i]
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    preds = []
    targets = []

    for train_index, test_index in kf.split(diet_mb):
        train = diet_mb.iloc[train_index]
        test = diet_mb.iloc[test_index]

        model = models_list[i]
        predictions = model.predict(test[features])
        preds.extend(predictions)
        targets.extend(test[target])

    score = stats.pearsonr(preds, targets)
    all_scores.append(score[0])
    all_p_values.append(score[1])
    all_feat_importances.append(model.feature_importances_)
    all_feat_names.append(model.feature_name_)
    all_preds.append(preds)
    all_targets.append(targets)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train[features])

    # shap.summary_plot(shap_values, train[features], show=False)
    # plt.title("LGBM " + target)
    # plt.savefig("/home/tomerse/PycharmProjects/pythonProject/lgbm_div_SHAP_{}.png".format(target), dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
    # plt.close()
    # plt.clf()

    return all_scores, all_p_values, all_feat_importances, all_feat_names, all_preds, all_targets, shap_values


def stub_job():
    print ('job started')
    params_methods = {}
    params_results = {}
    print("Loading models")
    models_list = pickle.load(open('/home/tomerse/PycharmProjects/pythonProject/models_lgbm_overlap_features.pkl', 'rb'))
    diet_mb = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/diet_mb.pkl")
    with open('/home/tomerse/PycharmProjects/pythonProject/my_lists.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists
    with open('/home/tomerse/PycharmProjects/pythonProject/new_targets.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    new_targets = loaded_lists
    with open('/home/tomerse/PycharmProjects/pythonProject/overlap_features.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    overlap_features = loaded_lists
    mb_names = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/mb_names.pkl")
    diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_').str.replace('(', '_').str.replace(')', '_').str.replace('.', '_').str.replace(',', '_')
    overlap_features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in overlap_features]
    # features_to_drop = ['Beans_black_eyed_peas', 'Mille_feuille', 'cereal_type__Cornflakes_with_sugar__cookies__pillows__etc__', 'never_eat__Sugar_or_foods___beverages_that_contain_sugar']
    # features = [feature for feature in features if feature not in features_to_drop]
    # target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]
    print('Starting queue')
    # for i in range(0, len(new_targets)):
    #     print(i)
    # max_shap = [417, 24, 32, 22, 7, 150, 141, 371, 253, 352]
    # top_corr_microbes = [32, 417, 141, 24, 22, 140, 343, 410, 403, 128, 150, 212, 429, 202, 190, 130]
    # shared_max_top_corr = [x for x in max_shap if x in top_corr_microbes]
    # extreme_shap = [99]
    # Bifidobacterium = [19, 20, 21, 22, 23, 24]
    # liron = [335]
    for i in range(len(target_input)):
        params_methods[i] = stub_subjob(diet_mb, overlap_features, target_input[i], models_list, i)

    output = pd.DataFrame(params_methods).transpose()
    output = output.apply(lambda x: x.explode(), axis=1)
    output.columns = [f'column{i}' for i in range(len(output.columns))]
    output.to_json(outdir + "overlap_features_SHAP.json")




def main():
    stub_job()

if __name__ == '__main__':
    main()
    output = pd.read_json(outdir + "overlap_features_SHAP.json")
    print(output)
    print("FINISH")
