import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import shap
import os

# -------------------------
# STYLE
# -------------------------
style_path = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/code/"
single_style = "nature_single.mplstyle"
plt.style.use(style_path + single_style)

# -------------------------
# THESIS FONT SETTINGS
# -------------------------
TITLE_FONTSIZE = 16
XLABEL_FONTSIZE = 16
YLABEL_FONTSIZE = 16
XTICK_FONTSIZE = 14
YTICK_FONTSIZE = 16
CBAR_LABEL_FONTSIZE = 16
CBAR_TICK_FONTSIZE = 14

ADD_TITLE = False
TITLE_PAD = 8

# ✅ NEW: a bit of padding for the x-axis label
XLABEL_PAD = 10

mpl.rcParams.update({
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": XLABEL_FONTSIZE,
    "xtick.labelsize": XTICK_FONTSIZE,
    "ytick.labelsize": YTICK_FONTSIZE,
})

# -------------------------
# USER SETTINGS
# -------------------------
outdir = "/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/"

MODEL = "LGBM"
TARGETS = "abundance"
GROUP = "bifido"
PROBLEM = "regression"
SPECIES = "segal_species"

PLOT_TYPE = "dot"           # 'dot' (beeswarm) or 'bar'
ONLY_MIC_INDEX = 34         # set None to disable
MAX_DISPLAY = 15            # thesis: keep clean in multi-panel figure

FIGSIZE = (10, 7)
DPI = 300

CBAR_BOX_ASPECT = 100
CBAR_WIDTH_MULT = 1.8
CBAR_RIGHT = 0.86

CUSTOM_XLABEL_SHORT = True  # True -> "B.longum", False -> full species name

div_targets = ["Richness", "Shannon_diversity"]
health_targets = ["modified_HACK_top17_score", "GMWI2_score"]

# -------------------------
# PRETTY FEATURE NAMES (plot-only)
# -------------------------
USE_PRETTY_FEATURE_NAMES = True

FEATURE_NAME_MAP = {
    "sex": "Sex",
    "age": "Age",

    "pct_carb_calories": "% kcal from carbohydrates",
    "pct_fat_calories": "% kcal from fat",

    "NOVA_pct": "Ultra-processed foods (NOVA, % kcal)",
    "wfpb_score_per_day": "WFPB score / day",
    "wfpb_score_per_day_cv": "WFPB score / day (CV)",
    "aHEI_score_per_day": "aHEI score / day",
    "vegetarian_score_per_day": "Vegetarian score / day",

    "Vitamin_B_6": "Vitamin B6",
    "Vitamin_C__total_ascorbic_acid": "Vitamin C (ascorbic acid)",
    "Zinc__Zn": "Zinc (Zn)",
    "Iodine": "Iodine",
    "Choline__total": "Choline (total)",
    "Potassium__K": "Potassium (K)",

    "Fatty_acids__total_monounsaturated": "Total monounsaturated fatty acids",
    "sat_to_total_lipids_ratio": "Saturated / total lipids ratio",

    "Nutseedsandproducts": "Nuts & seeds",
    "Beefveallambandothermeatproducts": "Red meat (beef/veal/lamb)",
}

def nice_feature_name(name: str) -> str:
    s = str(name)

    if s in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[s]

    m = re.match(r"^pct_(.+)_calories$", s)
    if m:
        return f"% kcal from {m.group(1).replace('_', ' ')}"

    m = re.match(r"^Vitamin_([A-Za-z])_(\d+)$", s)
    if m:
        return f"Vitamin {m.group(1)}{m.group(2)}"

    if "__" in s:
        parts = s.split("__")
        if len(parts) == 2 and parts[1]:
            return f"{parts[0].replace('_', ' ')} ({parts[1].replace('_', ' ')})"

    return s.replace("_", " ")

# -------------------------
# HELPERS
# -------------------------
def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-_. ]", "_", str(name))

def read_results(df):
    return tuple(df[col] for col in df.columns)

def short_species_name(species: str, no_space_after_dot: bool = True) -> str:
    parts = str(species).split()
    if len(parts) >= 2:
        return f"{parts[0][0]}.{parts[1]}" if no_space_after_dot else f"{parts[0][0]}. {parts[1]}"
    return str(species)

# ✅ NEW: italicize species in x-axis label (e.g., B. longum)
def italicize_species_label(label: str) -> str:
    """
    Returns a matplotlib-mathtext string with italics for binomial / abbreviated names.
    Examples:
      "B.longum"  -> r"$\it{B.\,longum}$"
      "B. longum" -> r"$\it{B.\,longum}$"
      "Bifidobacterium longum" -> r"$\it{Bifidobacterium\ longum}$"
    """
    s = str(label).strip()

    m = re.match(r"^([A-Za-z])\.\s*([A-Za-z0-9_]+)$", s)  # "B.longum" or "B. longum"
    if m:
        g, sp = m.group(1), m.group(2)
        sp = sp.replace("_", r"\_")
        return rf"$\it{{{g}.\,{sp}}}$"

    parts = s.split()
    if len(parts) >= 2:
        g = parts[0].replace("_", r"\_")
        sp = parts[1].replace("_", r"\_")
        return rf"$\it{{{g}\ {sp}}}$"

    return s

def filter_selected_probiotics(df, pattern, column="Microbe_Name"):
    probiotic_pattern = (r"Bifidobacterium|B\. |"
                         r"Streptococcus thermophilus|Streptococcus salivarius|"
                         r"Akkermansia muciniphila|"
                         r"Faecalibacterium prausnitzii|"
                         r"Lactococcus lactis|"
                         r"Bacteroides")
    pathogenic_pattern = (
        r"Clostridium difficile|Clostridium perfringens|Clostridium septicum|"
        r"Clostridium sordellii|Clostridium botulinum|Clostridium tetani|"
        r"Escherichia coli|Enterotoxigenic E\. coli|Enteropathogenic E\. coli|"
        r"Enterohemorrhagic E\. coli|Enteroaggregative E\. coli|Uropathogenic E\. coli|"
        r"Enterobacter cloacae|Enterobacter aerogenes|"
        r"Klebsiella pneumoniae|Klebsiella oxytoca|"
        r"Proteus mirabilis|Proteus vulgaris|"
        r"Salmonella|"
        r"Shigella|"
        r"Campylobacter jejuni|Campylobacter coli|"
        r"Helicobacter pylori|Helicobacter hepaticus|"
        r"Fusobacterium nucleatum|Fusobacterium varium|"
        r"Bacteroides fragilis|Bacteroides vulgatus|"
        r"Prevotella copri|"
        r"Ruminococcus gnavus|Ruminococcus torques|"
        r"Parabacteroides distasonis|"
        r"Streptococcus gallolyticus|Streptococcus mutans|Streptococcus pyogenes|"
        r"Staphylococcus aureus|"
        r"Veillonella parvula|"
        r"Morganella morganii|"
        r"Eubacterium rectale|"
        r"Hafnia alvei|"
        r"Pseudomonas aeruginosa|"
        r"Mycobacterium avium paratuberculosis|"
        r"Desulfovibrio"
    )
    butyrate_pattern = (
        r"Anaerobutyricum hallii|Anaerobutyricum soehngenii|"
        r"Anaerostipes hadrus|Anaerostipes amylophilus|"
        r"Agathobaculum butyriciproducens|Agathobaculum hominis|"
        r"Coprococcus eutactus|Coprococcus sp000154245|Coprococcus sp000433075|Coprococcus sp900548215|"
        r"Eubacterium_G ventriosum|Eubacterium_I ramulus|"
        r"Faecalibacterium prausnitzii|Faecalibacterium duncaniae|"
        r"Roseburia hominis|Roseburia intestinalis|Roseburia inulinivorans|"
        r"Ruminococcus_E bromii_B|Ruminococcus_E intestinalis|"
        r"Butyricicoccus_A intestinisimiae|Lawsonibacter asaccharolyticus"
    )
    acetate_pattern = (
        r"Akkermansia muciniphila|"
        r"Bacteroides thetaiotaomicron|Bacteroides fragilis|Bacteroides ovatus|"
        r"Bifidobacterium adolescentis|Bifidobacterium bifidum|Bifidobacterium longum|"
        r"Blautia_A obeum|Blautia_A wexlerae|"
        r"Collinsella bouchesdurhonensis|"
        r"Coprococcus eutactus|"
        r"Eubacterium_G ventriosum|"
        r"Escherichia coli|"
        r"Faecalibacterium prausnitzii|"
        r"Lactococcus lactis|"
        r"Phascolarctobacterium faecium|"
        r"Roseburia hominis|Roseburia intestinalis|"
        r"Streptococcus thermophilus"
    )
    propionate_pattern = (
        r"Akkermansia muciniphila|"
        r"Bacteroides fragilis|Bacteroides thetaiotaomicron|Bacteroides ovatus|"
        r"Bacteroides vulgatus|Bacteroides uniformis|"
        r"Coprococcus eutactus|"
        r"Dialister hominis|Dialister sp000434475|Dialister sp900543455|"
        r"Phascolarctobacterium faecium|"
        r"Roseburia inulinivorans|"
        r"Prevotella copri|"
        r"Veillonella atypica"
    )
    tmao_pattern = (
        r"Bilophila wadsworthia|"
        r"Desulfovibrio piger|"
        r"Escherichia coli|"
        r"Eggerthella lenta|"
        r"Klebsiella pneumoniae|"
        r"Proteus mirabilis|Proteus vulgaris|"
        r"Ruminococcus gnavus|"
        r"Salmonella"
    )
    health_species_pattern = (
        r"Alistipes shahii|"
        r"Eubacterium eligens|Lachnospira eligens|"
        r"Roseburia intestinalis|"
        r"Odoribacter splanchnicus|"
        r"Butyrivibrio crossotus"
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

def find_indices_triglycerides(df, column="Microbe_Name"):
    pattern = (r"Otoolea fessa|"
               r"Lachnosopira pectinoschiza_A|"
               r"Agathobacter rectralis|"
               r"Bacteroides cellulosilyticus|"
               r"UBA11524 sp000437595|"
               r"Roseburia intestinalis|"
               r"Acetatifactor intestinalis_1|"
               r"Bifidobacterium longum|"
               r"Ligilactobacillus ruminis|"
               r"Blautia_A sp900066355")
    return df.loc[df[column].str.contains(pattern, regex=True)].index

def filter_selected_pathways(df, pattern):
    butyrate_pattern = r"CENTFERM-PWY|PWY-5676|PWY-6590"
    propionate_pattern = r"P108-PWY"
    if pattern == "butyrate":
        pattern = butyrate_pattern
    elif pattern == "propionate":
        pattern = propionate_pattern
    mic_indices = df[df["targets"].str.contains(pattern, case=False, regex=True)].index
    return mic_indices

def convert_series_to_species(feature_series, mb_names):
    if not isinstance(feature_series, pd.Series):
        feature_series = pd.Series(feature_series)

    mb_names.index = mb_names.index.str.strip().str.lower()
    feature_series = feature_series.str.strip().str.lower()

    name_to_species = mb_names["species_new"].to_dict()
    name_to_genus = mb_names["genus_new"].to_dict()
    name_to_family = mb_names["family_new"].to_dict()

    species_series = feature_series.map(name_to_species)
    species_series = species_series.where(species_series != "unknown", feature_series.map(name_to_genus))
    species_series = species_series.where(species_series != "unknown", feature_series.map(name_to_family))
    species_series = species_series.fillna(feature_series)

    return species_series

def get_model_from_models_list(models_list, target_key):
    if isinstance(models_list, list):
        return models_list[int(target_key)]
    if target_key in models_list:
        return models_list[target_key]
    if str(target_key) in models_list:
        return models_list[str(target_key)]
    raise KeyError(target_key)

def _find_shap_colorbar_ax(fig):
    for ax in fig.axes[::-1]:
        if (ax.get_ylabel() or "").strip() == "Feature value":
            return ax
    if len(fig.axes) > 1:
        return fig.axes[-1]
    return None

def apply_thesis_fonts(fig, title_text=None):
    main_ax = None
    for ax in fig.axes:
        if ax is not None and ax.get_visible():
            if (ax.get_ylabel() or "").strip() != "Feature value":
                main_ax = ax
                break
    if main_ax is None:
        main_ax = plt.gca()

    if title_text is not None and ADD_TITLE:
        main_ax.set_title(str(title_text), fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)

    main_ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)
    main_ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

    plt.xticks(fontsize=XTICK_FONTSIZE)
    plt.yticks(fontsize=YTICK_FONTSIZE)

    cbar_ax = _find_shap_colorbar_ax(fig)
    if cbar_ax is not None:
        cbar_ax.set_ylabel("Feature value", fontsize=CBAR_LABEL_FONTSIZE)
        cbar_ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

        try:
            cbar_ax.set_aspect("auto")
        except Exception:
            pass
        try:
            cbar_ax.set_box_aspect(CBAR_BOX_ASPECT)
        except Exception:
            pass

def enlarge_shap_colorbar(fig):
    cbar_ax = _find_shap_colorbar_ax(fig)
    if cbar_ax is None:
        return

    cbar_ax.set_ylabel("Feature value", fontsize=CBAR_LABEL_FONTSIZE)
    cbar_ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    try:
        cbar_ax.set_aspect("auto")
    except Exception:
        pass

    try:
        cbar_ax.set_box_aspect(CBAR_BOX_ASPECT)
    except Exception:
        pass

    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0, pos.y0, pos.width * CBAR_WIDTH_MULT, pos.height])

    fig.subplots_adjust(right=CBAR_RIGHT)

# -------------------------
# CORE
# -------------------------
def stub_subjob(train, test, features, target, models_list, targets, mb_names, lgbm_diet_scores=None):
    print(target)

    model = get_model_from_models_list(models_list, target)

    if MODEL == "LGBM":
        explainer = shap.TreeExplainer(model)
    elif MODEL == "ridge":
        explainer = shap.LinearExplainer(model, train[features])
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}")





    X_plot = train[features]

    train_cols = explainer.model.original_model.feature_name()  # Booster feature names
    plot_cols = list(X_plot.columns)

    missing = [c for c in train_cols if c not in plot_cols]
    extra   = [c for c in plot_cols if c not in train_cols]

    print("train:", len(train_cols), "plot:", len(plot_cols))
    print("missing:", missing)
    print("extra:", extra)



    shap_values = explainer.shap_values(X_plot)
    if isinstance(shap_values, list):  # binary classifier
        shap_values = shap_values[1]

    # Title (microbe name) for xlabel only
    if TARGETS == "abundance":
        if SPECIES == "segal_species":
            title = mb_names.loc[targets[int(target)], "species_new"]
        else:
            title = targets[int(target)]
    elif TARGETS == "div":
        title = div_targets[int(target)]
    elif TARGETS == "diet":
        title = targets[int(target)]
    elif TARGETS == "health":
        title = health_targets[int(target)]
    elif TARGETS == "pathways":
        title = targets[int(target)]
    else:
        title = str(target)

    title = str(title)
    short_name = short_species_name(title, no_space_after_dot=True) if CUSTOM_XLABEL_SHORT else title

    # Pretty feature labels (plot-only)
    if USE_PRETTY_FEATURE_NAMES:
        pretty_names = [nice_feature_name(c) for c in X_plot.columns]
    else:
        pretty_names = list(X_plot.columns)

    # Plot
    plt.figure(figsize=FIGSIZE, dpi=DPI)

    if PROBLEM == "reverse":
        feature_series = pd.Series(features)
        renamed_features = convert_series_to_species(feature_series, mb_names).tolist()
        if USE_PRETTY_FEATURE_NAMES:
            renamed_features = [nice_feature_name(x) for x in renamed_features]

        shap.summary_plot(
            shap_values,
            X_plot,
            feature_names=renamed_features,
            show=False,
            max_display=MAX_DISPLAY,
            color_bar=True
        )
    else:
        shap.summary_plot(
            shap_values,
            X_plot,
            feature_names=pretty_names,
            show=False,
            plot_type=PLOT_TYPE,
            max_display=MAX_DISPLAY,
            color_bar=True
        )

    fig = plt.gcf()

    # Fonts
    apply_thesis_fonts(fig, title_text=title)

    # Colorbar geometry tweaks for dot
    if PLOT_TYPE == "dot":
        enlarge_shap_colorbar(fig)

    # ✅ X-axis label: italicize species + add label padding
    italic_species = italicize_species_label(short_name)
    if PLOT_TYPE == "bar":
        plt.xlabel(
            f"mean(|SHAP value|) for {italic_species}",
            fontsize=XLABEL_FONTSIZE,
            labelpad=XLABEL_PAD
        )
    else:
        plt.xlabel(
            f"SHAP value for {italic_species} (z-score)",
            fontsize=XLABEL_FONTSIZE,
            labelpad=XLABEL_PAD
        )

    # Save
    output_folder = f"{outdir}/SHAP_plots/{PROBLEM}/{SPECIES}/{GROUP}/"
    os.makedirs(output_folder, exist_ok=True)

    safe_title = sanitize_filename(title)
    suffix = "" if PLOT_TYPE in ["dot", "beeswarm"] else f"_{PLOT_TYPE}"
    outfile = os.path.join(
        output_folder,
        f"{MODEL}_{TARGETS}_SHAP_{target}_{safe_title}_top{MAX_DISPLAY}{suffix}.png"
    )

    print("Saving to:", outfile)
    plt.savefig(outfile, dpi=DPI, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close()
    plt.clf()

    return shap_values

def stub_job():
    print("Job started")
    print(MODEL)
    print(TARGETS)
    print(GROUP)

    print("Loading models and data")
    models_list = pickle.load(open(
        f"{outdir}/models/{PROBLEM}/{SPECIES}/models_{MODEL}_{TARGETS}_longitudinal.pkl", "rb"
    ))

    reverse = "reverse/" if PROBLEM == "reverse" else ""

    train_baseline = pd.read_pickle(f"{outdir}/data/{SPECIES}/{reverse}diet_mb_baseline_train.pkl")
    test_baseline = pd.read_pickle(f"{outdir}/data/{SPECIES}/{reverse}diet_mb_baseline_test.pkl")

    results = pd.read_pickle(f"{outdir}/data/{PROBLEM}/{SPECIES}/output_LGBM_{TARGETS}.pkl")
    if PROBLEM == "regression":
        lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets = read_results(results)
    elif PROBLEM == "given_presence":
        lgbm_diet_scores, lgbm_diet_pvalues, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets, lgbm_diet_prevalence = read_results(results)
    elif PROBLEM == "classification":
        lgbm_diet_acc, lgbm_diet_auc, lgbm_diet_coefs, lgbm_diet_preds, lgbm_diet_targets, lgbm_diet_pvalues = read_results(results)
        lgbm_diet_scores = lgbm_diet_auc
    else:
        lgbm_diet_scores = None

    pathways = "_pathways" if TARGETS == "pathways" else ""
    map_df = pd.read_csv(f"{outdir}/data/{PROBLEM}/{SPECIES}/map_df{pathways}.csv", index_col=0)

    with open(f"{outdir}/data/{SPECIES}/my_lists{pathways}.pkl", "rb") as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists

    mb_names = pd.read_pickle(f"{outdir}/data/mb_names.pkl")
    mb_names.index = mb_names.index.str.replace(r"[^\w]", "_", regex=True)

    train_baseline.columns = train_baseline.columns.str.replace(r"[^\w]", "_", regex=True)
    test_baseline.columns = test_baseline.columns.str.replace(r"[^\w]", "_", regex=True)

    features = [re.sub(r"[^a-zA-Z0-9_]", "_", x) for x in features]
    target_input = [re.sub(r"[^a-zA-Z0-9_]", "_", x) for x in target_input]

    print("Starting queue")

    if PROBLEM == "reverse":
        mic_indices = []
    elif GROUP == "Top":
        mic_indices = [594, 46, 223, 34, 221, 507, 32, 586, 309, 237, 579, 201, 76, 191, 190, 573]
    elif GROUP == "Triglycerides":
        mic_indices = find_indices_triglycerides(map_df)
    elif GROUP in ["probiotics", "pathogenic", "TMAO", "SCFA", "butyrate", "acetate", "propionate", "Health_shared_species"]:
        pattern_df = filter_selected_probiotics(map_df, GROUP)
        mic_indices = pattern_df.index
    else:
        mic_indices = []

    if ONLY_MIC_INDEX is not None and PROBLEM != "reverse":
        mic_indices = [int(ONLY_MIC_INDEX)]
        print(f"Overriding mic_indices → {mic_indices}")

    if PROBLEM == "reverse":
        raise RuntimeError("This script is currently configured for diet->microbe (not reverse).")
    else:
        for mic_index in mic_indices:
            stub_subjob(
                train_baseline,
                test_baseline,
                features,
                mic_index,
                models_list,
                target_input,
                mb_names,
                lgbm_diet_scores,
            )

def main():
    stub_job()

if __name__ == "__main__":
    main()
    print("FINISH")
