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
TARGETS = 'div' # 'div' or 'abundance'


def stub_subjob(diet_mb, features, target, models_list, i):
    print(i)
    all_scores = []
    all_p_values=[]
    all_feat_importances = []
    # all_feat_names = []
    predicted_abundances = []
    measured_abundances = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    model = models_list[i]
    predictions = model.predict(diet_mb[features])
    predicted_abundances.append(list(predictions))
    measured_abundances.append(list(diet_mb[target]))

    score = stats.pearsonr(predictions, diet_mb[target])
    all_scores.append(score[0])
    all_p_values.append(score[1])
    all_feat_importances.append(model.feature_importances_)
    # all_feat_names.append(model.feature_name_)
    

    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(train[features])

    # shap.summary_plot(shap_values, train[features], show=False)
    # plt.title("LGBM " + target)
    # plt.savefig("/home/tomerse/PycharmProjects/pythonProject/lgbm_div_SHAP_{}.png".format(target), dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
    # plt.close()
    # plt.clf()
    return all_scores, all_p_values, all_feat_importances, predicted_abundances, measured_abundances #, shap_values


def stub_job():
    print ('job started')
    params_methods = {}
    params_results = {}
    print("Loading models")
    models_list = pickle.load(open('/home/tomerse/PycharmProjects/pythonProject/models/models_lgbm_overlap_features_' + TARGETS +'.pkl', 'rb'))
    diet_mb = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/data/diet_mb_pnp3_intervention.pkl")
    with open('/home/tomerse/PycharmProjects/pythonProject/data/my_lists_pnp3.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    overlap_features, overlap_targets = loaded_lists
    print(len(overlap_features))
    print(len(overlap_targets))
    mb_names = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/data/mb_names_pnp3.pkl")
    diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_').str.replace('(', '_').str.replace(')', '_').str.replace('.', '_').str.replace(',', '_')
    overlap_features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in overlap_features]
    overlap_targets = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in overlap_targets]
    print('Starting queue')
    if TARGETS == 'div':
        loop_targets = ['Richness', 'Shannon_diversity']
    elif TARGETS == 'abundance':
        loop_targets = overlap_targets
    for i in range(len(loop_targets)):
        params_methods[i] = stub_subjob(diet_mb, overlap_features, loop_targets[i], models_list, i)

    output = pd.DataFrame(params_methods).transpose()
    output = output.apply(lambda x: x.explode(), axis=1)
    output.columns = [f'column{i}' for i in range(len(output.columns))]
    output.to_pickle(outdir + "pnp3_intervention_" + TARGETS + ".pkl")
    



def main():
    stub_job()

if __name__ == '__main__':
    main()
    output = pd.read_pickle(outdir + "pnp3_intervention_" + TARGETS + ".pkl")
    print(output.shape)
    print(output)
    print("FINISH")














# import math
# import os
# import pickle
# import pandas as pd
# from sklearn import linear_model
# import numpy as np
# from sklearn.model_selection import KFold
# from scipy import stats
# import matplotlib.pyplot as plt
# import lightgbm as lgb
# import re
# import shap


# outdir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/test/'


# def stub_subjob(diet_mb, features, target, models_list, i):
#     print(i)
#     all_scores = []
#     all_p_values=[]
#     all_feat_importances = []
#     all_feat_names = []
#     all_preds = []
#     all_targets = []
#     kf = KFold(n_splits=5, shuffle=True, random_state=1)
#     preds = []
#     targets = []

#     for train_index, test_index in kf.split(diet_mb):
#         train = diet_mb.iloc[train_index]
#         test = diet_mb.iloc[test_index]

#         model = models_list[i]
#         predictions = model.predict(test[features])
#         preds.extend(predictions)
#         targets.extend(test[target])

#     score = stats.pearsonr(preds, targets)
#     all_scores.append(score[0])
#     all_p_values.append(score[1])
#     all_feat_importances.append(model.feature_importances_)
#     all_feat_names.append(model.feature_name_)
#     all_preds.append(preds)
#     all_targets.append(targets)

#     # explainer = shap.TreeExplainer(model)
#     # shap_values = explainer.shap_values(train[features])

#     # shap.summary_plot(shap_values, train[features], show=False)
#     # plt.title("LGBM " + target)
#     # plt.savefig("/home/tomerse/PycharmProjects/pythonProject/lgbm_div_SHAP_{}.png".format(target), dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
#     # plt.close()
#     # plt.clf()

#     return all_scores, all_p_values, all_feat_importances, all_feat_names, all_preds, all_targets #, shap_values


# def stub_job():
#     print ('job started')
#     params_methods = {}
#     params_results = {}
#     print("Loading models")
#     models_list = pickle.load(open('/home/tomerse/PycharmProjects/pythonProject/models_lgbm_overlap_features.pkl', 'rb'))
#     diet_mb = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/diet_mb_pnp3_intervention.pkl")
#     with open('/home/tomerse/PycharmProjects/pythonProject/my_lists_pnp3.pkl', 'rb') as file:
#         loaded_lists = pickle.load(file)
#     overlap_features, target_input = loaded_lists
#     mb_names = pd.read_pickle("/home/tomerse/PycharmProjects/pythonProject/mb_names_pnp3.pkl")
#     diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_').str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_').str.replace('(', '_').str.replace(')', '_').str.replace('.', '_').str.replace(',', '_')
#     overlap_features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in overlap_features]
#     # features_to_drop = ['Beans_black_eyed_peas', 'Mille_feuille', 'cereal_type__Cornflakes_with_sugar__cookies__pillows__etc__', 'never_eat__Sugar_or_foods___beverages_that_contain_sugar']
#     # features = [feature for feature in features if feature not in features_to_drop]
#     # target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]
#     print('Starting queue')
#     # for i in range(0, len(new_targets)):
#     #     print(i)
#     # max_shap = [417, 24, 32, 22, 7, 150, 141, 371, 253, 352]
#     # top_corr_microbes = [32, 417, 141, 24, 22, 140, 343, 410, 403, 128, 150, 212, 429, 202, 190, 130]
#     # shared_max_top_corr = [x for x in max_shap if x in top_corr_microbes]
#     # extreme_shap = [99]
#     # Bifidobacterium = [19, 20, 21, 22, 23, 24]
#     # liron = [335]
#     for i in range(len(target_input)):
#         params_methods[i] = stub_subjob(diet_mb, overlap_features, target_input[i], models_list, i)

#     output = pd.DataFrame(params_methods).transpose()
#     output = output.apply(lambda x: x.explode(), axis=1)
#     output.columns = [f'column{i}' for i in range(len(output.columns))]
#     output.to_csv(outdir + "pnp3_intervention.csv", index=False)
    



# def main():
#     stub_job()

# if __name__ == '__main__':
#     main()
#     output = pd.read_csv(outdir + "pnp3_intervention.csv")
#     print(output)
#     print("FINISH")
