import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import shap
import os

style_path = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/code/"
single_style = "nature_single.mplstyle"
double_style = "nature_double.mplstyle"
third_style = "nature_third.mplstyle"
plt.style.use(style_path + single_style)


# Increase the font size globally
# plt.rcParams.update({
#     'font.size': 14,  # Change this value to increase or decrease font size
#     'axes.labelsize': 16,
#     'axes.titlesize': 18,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14
# })

outdir = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
MODEL = 'LGBM' # 'LGBM' or 'ridge'
TARGETS = 'div' # 'div' or 'abundance' or 'diet' or 'health' or 'pathways'
GROUP = 'div' # 'Health', 'div', 'Top', 'probiotics', "pathogenic", "TMAO", "SCFA", "butyrate", "acetate", "propionate", 'etc', 'Top_pathways', "Health_shared_species", "Triglycerides"
PROBLEM = 'regression' # 'regression' or 'classification' or 'given_presence' or 'reverse'
SPECIES = "segal_species" # "mpa_species" or "segal_species"
PLOT_TYPE = 'bar' # 'dot' or 'bar'

div_targets = ['Richness', 'Shannon_diversity']
health_targets = ['modified_HACK_top17_score', 'GMWI2_score']

def read_results(df):
    output = []
    for col in df.columns:
        output.append(df[col])
    return tuple(output)


def filter_selected_probiotics(df, pattern, column='Microbe_Name'):
    probiotic_pattern = (r'Bifidobacterium|B\. |'
               r'Streptococcus thermophilus|Streptococcus salivarius|'
               r'Akkermansia muciniphila|'
               r'Faecalibacterium prausnitzii|'
               r'Lactococcus lactis|'
               r'Bacteroides')
    pathogenic_pattern = (
        r'Clostridium difficile|Clostridium perfringens|Clostridium septicum|'
        r'Clostridium sordellii|Clostridium botulinum|Clostridium tetani|'
        r'Escherichia coli|Enterotoxigenic E\. coli|Enteropathogenic E\. coli|'
        r'Enterohemorrhagic E\. coli|Enteroaggregative E\. coli|Uropathogenic E\. coli|'
        r'Enterobacter cloacae|Enterobacter aerogenes|'
        r'Klebsiella pneumoniae|Klebsiella oxytoca|'
        r'Proteus mirabilis|Proteus vulgaris|'
        r'Salmonella|'
        r'Shigella|'
        r'Campylobacter jejuni|Campylobacter coli|'
        r'Helicobacter pylori|Helicobacter hepaticus|'
        r'Fusobacterium nucleatum|Fusobacterium varium|'
        r'Bacteroides fragilis|Bacteroides vulgatus|'
        r'Prevotella copri|'
        r'Ruminococcus gnavus|Ruminococcus torques|'
        r'Parabacteroides distasonis|'
        r'Streptococcus gallolyticus|Streptococcus mutans|Streptococcus pyogenes|'
        r'Staphylococcus aureus|'
        r'Veillonella parvula|'
        r'Morganella morganii|'
        r'Eubacterium rectale|'
        # r'Akkermansia muciniphila|'
        r'Hafnia alvei|'
        r'Pseudomonas aeruginosa|'
        r'Mycobacterium avium paratuberculosis|'
        r'Desulfovibrio' 
    )
    butyrate_pattern = (
    r'Anaerobutyricum hallii|Anaerobutyricum soehngenii|'
    r'Anaerostipes hadrus|Anaerostipes amylophilus|'
    r'Agathobaculum butyriciproducens|Agathobaculum hominis|'
    r'Coprococcus eutactus|Coprococcus sp000154245|Coprococcus sp000433075|Coprococcus sp900548215|'
    r'Eubacterium_G ventriosum|Eubacterium_I ramulus|'
    r'Faecalibacterium prausnitzii|Faecalibacterium duncaniae|'
    r'Roseburia hominis|Roseburia intestinalis|Roseburia inulinivorans|'
    r'Ruminococcus_E bromii_B|Ruminococcus_E intestinalis|'
    r'Butyricicoccus_A intestinisimiae|Lawsonibacter asaccharolyticus'
    )
    acetate_pattern = (
    r'Akkermansia muciniphila|'
    r'Bacteroides thetaiotaomicron|Bacteroides fragilis|Bacteroides ovatus|'
    r'Bifidobacterium adolescentis|Bifidobacterium bifidum|Bifidobacterium longum|'
    r'Blautia_A obeum|Blautia_A wexlerae|'
    r'Collinsella bouchesdurhonensis|'
    r'Coprococcus eutactus|'
    r'Eubacterium_G ventriosum|'
    r'Escherichia coli|'
    r'Faecalibacterium prausnitzii|'
    r'Lactococcus lactis|'
    r'Phascolarctobacterium faecium|'
    r'Roseburia hominis|Roseburia intestinalis|'
    r'Streptococcus thermophilus'
    )
    propionate_pattern = (
    r'Akkermansia muciniphila|'
    r'Bacteroides fragilis|Bacteroides thetaiotaomicron|Bacteroides ovatus|'
    r'Bacteroides vulgatus|Bacteroides uniformis|'
    r'Coprococcus eutactus|'
    r'Dialister hominis|Dialister sp000434475|Dialister sp900543455|'
    r'Phascolarctobacterium faecium|'
    r'Roseburia inulinivorans|'
    r'Prevotella copri|'
    r'Veillonella atypica'
    )
    tmao_pattern = (
    r'Bilophila wadsworthia|'
    r'Desulfovibrio piger|'
    r'Escherichia coli|'
    r'Eggerthella lenta|'
    r'Klebsiella pneumoniae|'
    r'Proteus mirabilis|Proteus vulgaris|'
    r'Ruminococcus gnavus|'
    r'Salmonella'
    )
    health_species_pattern = (
    r'Alistipes shahii|'
    r'Eubacterium eligens|Lachnospira eligens|'
    r'Roseburia intestinalis|'
    r'Odoribacter splanchnicus|'
    r'Butyrivibrio crossotus'
    )
    if pattern == "probiotics":
        pattern = probiotic_pattern
    elif pattern == "pathogenic":
        pattern = pathogenic_pattern
    elif pattern == "butyrate":
        pattern = butyrate_pattern
    elif pattern == "acetate":
        pattern = acetate_pattern
    elif pattern == "propionate":
        pattern = propionate_pattern
    elif pattern == "SCFA":
        pattern = butyrate_pattern + acetate_pattern + propionate_pattern
    elif pattern == "TMAO":
        pattern = tmao_pattern
    elif pattern == "Health_shared_species":
        pattern = health_species_pattern
    return df.loc[df[column].str.contains(pattern, regex=True)]


def find_indices_triglycerides(df, column='Microbe_Name'):
    pattern = (r'Otoolea fessa|'
    r'Lachnosopira pectinoschiza_A|'
    r'Agathobacter rectralis|'
    r'Bacteroides cellulosilyticus|'
    r'UBA11524 sp000437595|'
    r'Roseburia intestinalis|'
    r'Acetatifactor intestinalis_1|'
    r'Bifidobacterium longum|'
    r'Ligilactobacillus ruminis|'
    r'Blautia_A sp900066355'
    )
    return df.loc[df[column].str.contains(pattern, regex=True)].index

def filter_selected_pathways(df, pattern):
    butyrate_pattern = r'CENTFERM-PWY|PWY-5676|PWY-6590'
    propionate_pattern = r'P108-PWY'
    if pattern == "butyrate":
        pattern = butyrate_pattern
    elif pattern == "propionate":
        pattern = propionate_pattern
    mic_indices = df[df["targets"].str.contains(pattern, case=False, regex=True)].index
    return mic_indices


def convert_series_to_species(feature_series, mb_names):
    """Convert feature names to species, fallback to genus, then family, otherwise retain original name."""
    if not isinstance(feature_series, pd.Series):
        feature_series = pd.Series(feature_series)

    mb_names.index = mb_names.index.str.strip().str.lower()
    feature_series = feature_series.str.strip().str.lower()

    # Create mapping dictionaries
    name_to_species = mb_names['species_new'].to_dict()
    name_to_genus = mb_names['genus_new'].to_dict()
    name_to_family = mb_names['family_new'].to_dict()

    # Map species names
    species_series = feature_series.map(name_to_species)

    # Replace "unknown" species with genus name
    species_series = species_series.where(species_series != "unknown", feature_series.map(name_to_genus))

    # Replace "unknown" genus with family name
    species_series = species_series.where(species_series != "unknown", feature_series.map(name_to_family))

    # Ensure NaNs are replaced with the original feature name
    species_series = species_series.fillna(feature_series)

    return species_series


def stub_subjob(train, test, features, target, models_list, targets, mb_names, lgbm_diet_scores=None):
    print(target)
    model = models_list[target]

    if MODEL == 'LGBM':
        explainer = shap.TreeExplainer(model)
    elif MODEL == 'ridge':
        # masker = shap.maskers.Independent(train[features])
        explainer = shap.LinearExplainer(model, train[features])
    shap_values = explainer.shap_values(train[features])
    
    if isinstance(shap_values, list):  # New behavior for binary classifiers
        shap_values = shap_values[1]  # Get SHAP values for the positive class

    if PROBLEM == 'reverse':
        # Convert feature names for a reverse problem
        feature_series = pd.Series(features)
        renamed_features = convert_series_to_species(feature_series, mb_names).tolist()
        shap.summary_plot(shap_values, train[features], feature_names=renamed_features, show=False)
    else:
        shap.summary_plot(shap_values, train[features], show=False, plot_type=PLOT_TYPE, max_display=15)

    score = lgbm_diet_scores[target]
    print(score)
    if TARGETS == 'abundance':
        if SPECIES == "segal_species":
            title = mb_names.loc[targets[int(target)], 'species_new']
        elif SPECIES == "mpa_species":
            title = targets[int(target)]
    elif TARGETS == 'div':
        title = div_targets[target]
    elif TARGETS == 'diet':
        title = targets[int(target)]
    elif TARGETS == 'health':
        title = health_targets[target]
    elif TARGETS == 'pathways':
        title = targets[int(target)]

    print(title)
    plt.title(f"{title} (Corr: {str(round(score, 2))})")
    output_folder = f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/SHAP_plots/{PROBLEM}/{SPECIES}/{GROUP}/"
    os.makedirs(output_folder, exist_ok=True)
    plottype = '' if PLOT_TYPE == 'beeswarm' else "_"+PLOT_TYPE
    plt.savefig(output_folder + MODEL + "_" + TARGETS + '_SHAP_{}_{}{}.png'.format(target, title, plottype), dpi=300, facecolor="white", transparent=False, bbox_inches='tight')
    plt.close()
    plt.clf()

    return shap_values


def stub_job():
    print ('Job started')
    print(MODEL)
    print(TARGETS)
    print(GROUP)
    params_methods = {}
    params_results = {}
    print("Loading models and data")
    models_list = pickle.load(open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{PROBLEM}/{SPECIES}/models_' + MODEL + '_' + TARGETS + '_longitudinal.pkl', 'rb'))

    reverse = 'reverse/' if PROBLEM == 'reverse' else ''

    train_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/{reverse}diet_mb_baseline_train.pkl")
    test_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/{reverse}diet_mb_baseline_test.pkl")

    results = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/output_LGBM_{TARGETS}.pkl")
    if PROBLEM == 'regression':
        lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(results)
    elif PROBLEM == 'given_presence':
        lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets, lgbm_diet_prevalence = read_results(results)
    elif PROBLEM == 'classification':
        lgbm_diet_acc, lgbm_diet_auc, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets, lgbm_diet_pvalues = read_results(results)
        lgbm_diet_scores = lgbm_diet_auc

    pathways = '_pathways' if TARGETS == 'pathways' else ''
    map_df = pd.read_csv(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/map_df{pathways}.csv', index_col=0)

    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists{pathways}.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists
    # with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/significant_targets.pkl', 'rb') as file:
    #     loaded_lists = pickle.load(file)
    # significant_targets = loaded_lists
    mb_names = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/mb_names.pkl")
    mb_names.index = mb_names.index.str.replace(r'[^\w]', '_', regex=True)
    
    train_baseline.columns = train_baseline.columns.str.replace(r'[^\w]', '_', regex=True)
    test_baseline.columns = test_baseline.columns.str.replace(r'[^\w]', '_', regex=True)

    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    # features_to_drop = ['Beans_black_eyed_peas', 'Mille_feuille', 'cereal_type__Cornflakes_with_sugar__cookies__pillows__etc__', 'never_eat__Sugar_or_foods___beverages_that_contain_sugar']
    # features = [feature for feature in features if feature not in features_to_drop]
    original_targets = target_input
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]

    if PROBLEM == 'reverse':
        mb_features = target_input + ['Richness', 'Shannon_diversity'] + base_features
        diet_targets = [feat for feat in features if feat not in ['age', 'gender']]

    print('Starting queue')

    from_my_10k = [24]
    if PROBLEM == 'reverse':
        diet_adherence_targets = [712, 713, 714, 715, 716, 717, 718, 783]
    elif GROUP == 'Top': # Top
        if SPECIES == "segal_species":
            if PROBLEM == 'regression' and TARGETS == 'abundance':
                mic_indices = [594, 46, 223, 34, 221, 507, 32, 586, 309, 237, 579, 201, 76, 191, 190, 573]
            if PROBLEM == 'regression' and TARGETS == 'pathways':
                mic_indices = [594, 46, 223, 34, 221, 507, 32, 309, 586, 237, 579, 201, 76, 610, 191, 573] 
            elif PROBLEM == 'classification':
                mic_indices = [76, 596, 231, 46, 27, 456, 613, 618, 226, 307, 612, 34, 116, 468, 28, 324]
            elif PROBLEM == 'given_presence':
                mic_indices = [223, 221, 595, 453, 380, 718, 472, 57, 280, 627, 452, 46, 508, 34, 201, 518, 580, 284, 319, 282]   
        elif SPECIES == "mpa_species":
            mic_indices = [445, 523, 208, 45, 499, 41, 116, 513, 123, 449, 119, 93, 136, 344, 366, 43]
    elif GROUP in ["probiotics", "pathogenic", "TMAO", "SCFA", "butyrate", "acetate", "propionate", "Health_shared_species"]:
        if TARGETS == 'pathways' and GROUP == 'butyrate':
            mic_indices = filter_selected_pathways(map_df, GROUP)
        elif TARGETS == 'pathways' and GROUP == 'propionate':
            mic_indices = filter_selected_pathways(map_df, GROUP)
        else:
            pattern_df = filter_selected_probiotics(map_df, GROUP)
            mic_indices = pattern_df.index
        
    elif GROUP == "Health":
        mic_indices = [0, 1]
    elif GROUP == "div":
        mic_indices = [0, 1]
    elif GROUP == 'Top_pathways':
        mic_indices = [20, 94, 170, 276, 41, 177, 204, 233, 35, 270, 6, 0, 168, 179, 63]
    elif GROUP == 'Triglycerides':
        mic_indices = find_indices_triglycerides(map_df)
        print("In correct if")
        print(mic_indices)

    if PROBLEM == 'reverse':
        for diet_target_index in diet_adherence_targets:
            params_methods[diet_target_index] = stub_subjob(train_baseline,test_baseline, mb_features, diet_target_index, models_list, diet_targets, mb_names)
    else:
        for mic_index in mic_indices:
            # mic_index = original_targets.index(mic_index) # Used for targets of shape "fBin__541|gBin__2135|sBin__3290" 
            params_methods[mic_index] = stub_subjob(train_baseline, test_baseline, features, mic_index, models_list, target_input, mb_names, lgbm_diet_scores)

    # output = pd.DataFrame(params_methods).transpose()
    # output = output.apply(lambda x: x.explode(), axis=1)
    # output.columns = [f'column{i}' for i in range(len(output.columns))]
    # output.to_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/lightGBM_overlap_features.pkl")
    

def main():
    stub_job()


if __name__ == '__main__':
    main()
    print("FINISH")
