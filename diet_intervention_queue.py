print("START")

from LabData import config_global as config
from LabUtils import Utils
from LabUtils import addloglevels
import os
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import linprog
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import re
import matplotlib.patches as mpatches  # Needed for legend
from sklearn.preprocessing import StandardScaler
import warnings

foods_only = True  # Set to True if you want to use only food features, False for all features
suffix_foods = '_foods_only' if foods_only else ''
PHENOTYPE = 'bt__wbc'  # The phenotype to predict, e.g., 'bt__triglycerides'

def main(q):
    home_path = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
    SPECIES = 'segal_species' # 'mpa_species' or 'segal_species'
    PROBLEM = 'regression'

    figures_path = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/figures/diet_intervention'
    phenotypes_mb = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/phenotypes_mb.pkl')

    with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/target_phenotypes.pkl', 'rb') as f:
        target_phenotypes = pickle.load(f)

    base_features = ['age', 'sex']
    flat_targets = sum(target_phenotypes, [])

    microbial_features = [col for col in phenotypes_mb.columns if col not in flat_targets + ['RegistrationCode']]

    microbial_features_df = phenotypes_mb[microbial_features]
    phenotype_df = phenotypes_mb[[PHENOTYPE]]
    species = '' if SPECIES == 'segal_species' else '_mpa'

    diet_mb = pd.read_pickle(home_path + f"data/{SPECIES}/diet_mb.pkl")
    diet_mb_test = pd.read_pickle(home_path + f"data/{SPECIES}/diet_mb_baseline_test.pkl")
    test_subjects = diet_mb_test.index
    print(test_subjects)
    diet_mb = diet_mb.loc[test_subjects, :]
    print(diet_mb.shape)
    with open(home_path + f'data/{SPECIES}/my_lists.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, all_features, targets = loaded_lists

    # all_features_formatted = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in all_features]
    all_features_formatted = all_features


    with open(home_path + f'data/{SPECIES}/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

    # Filter non significant correlations from the permutations
    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/significant_targets.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    significant_targets = loaded_lists
    significant_targets_indices = [targets.index(item) for item in significant_targets]

    with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/food_shortnames.pkl', 'rb') as file:
        food_shortnames = pickle.load(file)
    food_shortnames_formatted = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in food_shortnames]
    significant_targets_df = pd.DataFrame(columns=significant_targets)
    mb_names = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/mb_names.pkl")
    
    # Pre-compute once
    if foods_only:
        all_features = [f for f in all_features if f in ["age", "sex"] + list(food_shortnames)]
        # all_features_formatted = [f for f in all_features_formatted if f in ["age", "sex"] + list(food_shortnames_formatted)]
        all_features_formatted = all_features
        print("Len all features:", len(all_features))
        print("Len all features formatted:", len(all_features_formatted))
    DIET_ONLY_FEATURES = [f for f in all_features_formatted if f not in ("age", "sex")]


    def rename_microbiome_columns(diet_mb_scaled: pd.DataFrame, mb_names: pd.DataFrame, targets: list) -> pd.DataFrame:
        """
        Rename microbiome columns in a full DataFrame (diet + microbiome + metadata)
        using species/genus/family mapping from mb_names.
        Only columns in `targets` will be renamed.
        """
        # --- Normalize ---
        mb_names.index = mb_names.index.str.strip()
        species_map = mb_names['species_new'].str.strip()
        genus_map = mb_names['genus_new'].str.strip()
        family_map = mb_names['family_new'].str.strip()

        # --- Build mapping ---
        final_mapping = {}
        for col in targets:
            name = species_map.get(col, None)
            if name == "unknown" or pd.isna(name):
                name = genus_map.get(col, None)
            if name == "unknown" or pd.isna(name):
                name = family_map.get(col, None)
            if name is None or name == "unknown":
                name = col  # fallback
            final_mapping[col] = name

        # --- Rename only microbiome columns ---
        df = diet_mb_scaled.copy()
        df.rename(columns=final_mapping, inplace=True)

        # --- Deduplicate renamed columns ---
        col_counts = Counter(df.columns)
        name_counter = defaultdict(int)
        new_cols = []
        for col in df.columns:
            if col_counts[col] > 1:
                name_counter[col] += 1
                new_cols.append(f"{col}_{name_counter[col]}")
            else:
                new_cols.append(col)
        df.columns = new_cols

        return df

    microbial_features_df = rename_microbiome_columns(diet_mb[targets], mb_names, targets)
    significant_targets_df = rename_microbiome_columns(significant_targets_df, mb_names, targets)

    def rename_microbiome_series(series_data: pd.Series, mb_names: pd.DataFrame) -> pd.Series:
        """
        Renames the index of a pandas Series based on a provided microbiome mapping.
        """
            # --- Input Validation and Setup ---
        if not isinstance(series_data, pd.Series):
            raise TypeError("Input 'series_data' must be a pandas Series.")
        
        # --- Normalize ---
        mb_names.index = mb_names.index.str.strip()
        data_copy = series_data.copy()
        data_copy.index = data_copy.index.str.strip()

        # --- Extract maps ---
        species_map = mb_names['species_new'].str.strip()
        genus_map = mb_names['genus_new'].str.strip()
        family_map = mb_names['family_new'].str.strip()

        # --- Build name mapping ---
        final_mapping = {}
        for original_name in data_copy.index:
            name = species_map.get(original_name, None)
            if name == "unknown" or pd.isna(name):
                name = genus_map.get(original_name, None)
            if name == "unknown" or pd.isna(name):
                name = family_map.get(original_name, None)
            if name is None or name == "unknown":
                name = original_name  # fallback
            final_mapping[original_name] = name

        # --- Rename index ---
        data_copy.rename(index=final_mapping, inplace=True)

        # --- Deduplicate ---
        idx_counts = Counter(data_copy.index)
        name_counter = defaultdict(int)
        new_idx = []

        for idx_val in data_copy.index:
            if idx_counts[idx_val] > 1:
                name_counter[idx_val] += 1
                new_idx.append(f"{idx_val}_{name_counter[idx_val]}")
            else:
                new_idx.append(idx_val)

        data_copy.index = new_idx

        return data_copy

    # Load SHAP values for phenotypes and microbiome
    directional_phenotypes_shap = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_intervention/directional_phenotypes_shap.pkl')

    # Remove age and gender
    directional_phenotypes_shap = directional_phenotypes_shap[~directional_phenotypes_shap.index.isin(['age', 'sex'])]
    directional_phenotypes_shap = directional_phenotypes_shap.iloc[significant_targets_indices, :]
    directional_microbiome_shap = pd.read_pickle('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_intervention/directional_microbiome_shap.pkl')

    # Remove age and gender
    directional_microbiome_shap = directional_microbiome_shap[~directional_microbiome_shap.index.isin(['age', 'sex'])]


    def get_subject(diet_mb, i):
        subject = diet_mb.iloc[i]

        subject_gender = subject['sex']
        subject_age = subject['age']

        subject_diet = subject[all_features + ['Energy']]
        subject_calories = subject_diet['Energy']
        subject_diet = subject_diet[~subject_diet.index.isin(['age', 'sex'])]


        # subject_diet = subject_diet[food_shortnames]

        subject_mb = subject[targets].copy()
        # subject_mb.index = targets


        return subject_diet, subject_mb, subject_calories, subject_gender, subject_age


    # indices_to_find = [
    #     "Parabacteroides distasonis",
    #     "Fusicatenibacter saccharivorans",
    #     "Faecalibacterium prausnitzii_D",
    #     "Bacteroides uniformis",
    #     "Phocaeicola vulgatus"
    # ]

    # for index_to_find in indices_to_find:
    #     location = directional_triglycerides_shap.index.get_loc("Bifidobacterium longum")
    #     print(f"Index '{index_to_find}' is at location: {location}")
    directional_triglycerides_shap = directional_phenotypes_shap.loc[:, PHENOTYPE]
    # directional_triglycerides_shap[29] = directional_triglycerides_shap[29] * 1000000 # For testing!
    directional_triglycerides_shap.abs().sort_values(ascending=False)

    triglyceride_species = [
        "Otoolea fessa",
        "Lachnospira pectinoschiza_A",
        "Bifidobacterium longum",
        "Alistipescatomonas sp900066785",
        "Faecalibacterium longum_1",
        "Ligilactobacillus ruminis",
        "Acetatifactor intestinalis_1",
        "Choladosuia sp902363665",
        "Agathobacter rectalis",
        "Enterocloster sp000431375",
        "Roseburia intestinalis",
        "Bacteroides cellulosilyticus",
        "UBA11524 sp000437595",
        "Streptococcus parasanguinis",
        "Lachnospira hominis",
        "Klebsiella pneumoniae",
        "Merdivicinus sp934539585",
        "Faecalibacterium sp900539885"
    ]
    directional_triglycerides_shap[[639, 251, 466, 625, 615]]
    directional_microbiome_shap = directional_microbiome_shap.loc[food_shortnames, :]

    prevalence_count = [(diet_mb[target] > -4).sum() for target in significant_targets]
    prevalence_count = pd.Series(prevalence_count, index=directional_phenotypes_shap.index)
    prevalence_count.sort_values()
    diet_foods_df = pd.read_csv('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_adherence_foods.csv', index_col=0)
    diet_foods_df = diet_foods_df.loc[:,diet_foods_df.columns.isin(food_shortnames)]
    nova_foods_df = diet_foods_df.loc['NOVA', :]

    # Food shortnames is the list of foods, you should already have this loaded.
    positive_diets = diet_mb[diet_mb > 0]

    # No worries that there are also non food names here (like bacteria), will be filtered inside the function
    foods_upper_bounds = positive_diets.quantile(0.95).fillna(0.0)
    foods_upper_bounds = foods_upper_bounds[all_features]

    foods_lower_bounds = positive_diets.quantile(0.05).fillna(0.0)
    foods_lower_bounds = foods_lower_bounds[all_features]

    # #TODO Sort bacteria by effect size
    ### Greedy search optimization
    with open(home_path + f'data/{SPECIES}/diet_scaler{suffix_foods}.pkl', 'rb') as f:
        diet_scaler = pickle.load(f)

    with open(home_path + f'data/{SPECIES}/age_scaler{suffix_foods}.pkl', 'rb') as f:
        age_scaler = pickle.load(f)

    with open(home_path + f'data/{SPECIES}/mb_scaler{suffix_foods}.pkl', 'rb') as f:
        mb_scaler = pickle.load(f)

    # diet_species_model = pickle.load(open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/regression/segal_species/models_' + 'LGBM' + '_' + 'abundance_longitudinal.pkl', 'rb'))

    phenotypes_models_dict = pickle.load(open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/{SPECIES}/models_mb_phenotypes.pkl', 'rb'))
    triglycerids_model = phenotypes_models_dict[PHENOTYPE]


    warnings.filterwarnings("ignore", message="Trying to unpickle estimator StandardScaler")

    alcoholic_features = {
        "Beer", "Campari", "Cocktail", "Dessert Wine", "Gin and tonic", "Light Beer",
        "Malt beverage", "Ouzo", "Sweet wine", "Vodka or Arak", "Whiskey", "Wine"
    }


    def normalize_predicted_microbiome_optimized(predicted_mb_log_z: pd.DataFrame,
                                                mb_scaler: StandardScaler,
                                                non_existing_mask: np.ndarray,
                                                targets: list):
        # Split off age/sex
        baseline = predicted_mb_log_z.iloc[:, :2]

        # Work entirely in NumPy
        z_arr = predicted_mb_log_z[targets].values.reshape(-1)  # shape (n_species,)
        # inverse‐scale (z → log10)
        inv_log10 = mb_scaler.inverse_transform(z_arr.reshape(1, -1))[0]
        # exponentiate to linear
        lin = 10 ** inv_log10

        # clamp non‐existing
        lin[non_existing_mask] = 1e-4

        # normalize only existing
        existing = ~non_existing_mask
        lin[existing] /= lin[existing].sum()

        # log10 back
        log10_arr = np.log10(lin)

        # build output DF once
        norm_df = pd.DataFrame([log10_arr], columns=targets, index=predicted_mb_log_z.index)
        return pd.concat([baseline, norm_df], axis=1)


    def check_constraints(candidate: pd.Series,
                        original: pd.Series,
                        kcal: float,
                        gender: str,
                        max_kcal_pct: float,
                        alcohol_idx: np.ndarray) -> bool:
        # total kcal change
        delta_kcal = (candidate - original) * kcal
        if abs(delta_kcal.sum()) > kcal * max_kcal_pct:
            return False

        # alcohol %
        if candidate[alcohol_idx].sum() > (0.05 if gender == "female" else 0.07):
            return False

        return True


    def predict_microbiome_from_diet_optimized(candidate_all: pd.Series,
                                            model_map: dict,
                                            diet_scaler: StandardScaler,
                                            mb_scaler: StandardScaler,
                                            age_scaler: StandardScaler,
                                            subject_age: float,
                                            subject_gender,
                                            all_features: list,
                                            diet_features: list,
                                            targets: list) -> pd.DataFrame:
        # one‐time scaled vector
        vec = pd.Series(index=all_features, dtype=float)

        # diet features
        vec[diet_features] = diet_scaler.transform([candidate_all[diet_features]])[0]
        # age
        vec["age"] = age_scaler.transform([[subject_age]])[0, 0]
        # sex (pass through)
        vec["sex"] = subject_gender

        # build input DF ONCE
        df_in = pd.DataFrame([vec.values], columns=all_features)

        # vectorized predicts
        preds = [model_map[i].predict(df_in)[0] for i in range(len(targets))]
        z_arr = np.array(preds)

        # assemble
        meta = pd.DataFrame([[vec["age"], vec["sex"]]], columns=["age", "sex"])
        z_df = pd.DataFrame([z_arr], columns=targets)
        return pd.concat([meta, z_df], axis=1)


    def predict_triglycerides(subject_mb: pd.DataFrame,
                            targets: list,
                            mb_scaler: StandardScaler,
                            age_scaler: StandardScaler,
                            subject_age: float,
                            subject_gender,
                            tg_model) -> float:
        # scale microbiome
        mb_scaled = mb_scaler.transform(subject_mb[targets])
        df = pd.DataFrame(mb_scaled, columns=targets, index=subject_mb.index)

        # add age, sex
        df["age"] = age_scaler.transform([[subject_age]])[0, 0]
        df["sex"] = subject_gender

        # reorder
        ordered = df[["age", "sex"] + targets]
        return tg_model.predict(ordered)[0]


    def recommend_diet_change_greedy(
        subject_id: int,
        subject_diet: pd.Series,
        subject_mb: pd.Series,
        all_features: list,
        diet_scaler: StandardScaler,
        mb_scaler: StandardScaler,
        age_scaler: StandardScaler,
        model_diet_to_mb_path: str,
        model_tg_path: str,
        nova_df: pd.DataFrame,
        food_upper_bounds: pd.Series,
        food_lower_bounds: pd.Series,
        subject_calories: float,
        subject_gender: str,
        subject_age: float,
        targets: list,
        max_total_kcal_pct_change: float,
        max_iter: int
    ):
        
        # --- LOAD MODELS ON WORKER NODE ---
        with open(model_diet_to_mb_path, 'rb') as f:
            model_diet_to_mb = pickle.load(f)

        with open(model_tg_path, 'rb') as f:
            phenotypes_models_dict = pickle.load(f)
        model_tg = phenotypes_models_dict[PHENOTYPE]
        # -----------------------------------

        # original diet %
        orig_pct = subject_diet[food_shortnames].copy()
        total_steps = pd.Series(0.0, index=orig_pct.index)
        used_steps = set()

        # masks
        non_exist = np.isin(targets,
                            subject_mb[subject_mb == -4.0].index)
        print(non_exist)
        alcohol_idx = np.isin(orig_pct.index, list(alcoholic_features))

        # baseline triglycerides
        baseline_mb = predict_microbiome_from_diet_optimized(
                subject_diet, model_diet_to_mb,
                diet_scaler, mb_scaler,
                age_scaler, subject_age,
                subject_gender,
                all_features, DIET_ONLY_FEATURES, targets
            )
        
        baseline_norm_mb = normalize_predicted_microbiome_optimized(
            baseline_mb, mb_scaler, non_exist, targets
        )

        baseline_tg = predict_triglycerides(
            baseline_norm_mb, targets,
            mb_scaler, age_scaler,
            subject_age, subject_gender,
            model_tg
        )
        print("Baseline triglycerides:", baseline_tg)
        current_tg = baseline_tg

        # greedy loop
        direction_phases = [(1, -1), (0.5, -0.5)]
        for phase, directions in enumerate(direction_phases, start=1):
            print(f"--- Phase {phase}: using directions {directions} ---")
            for it in range(max_iter):
                print(f"-------------- Iteration: {it + 1} --------------")
                best_pred = current_tg
                best_step = None

                for food in orig_pct.index[orig_pct > 0]:

                    if food not in food_upper_bounds.index or food not in food_lower_bounds.index:
                        print(f"Skipping {food}: no bounds defined")
                        continue

                    upper_bound = food_upper_bounds[food]
                    lower_bound = food_lower_bounds[food]

                    print(f"Food: {food}")
                    base = orig_pct[food]

                    # If the food > 95th or < 5th percentiles we skip it! (logging issues)
                    if base > upper_bound or base < lower_bound:
                        print(f"Skipping {food}: baseline {base:.4f} outside [{lower_bound:.4f}, {upper_bound:.4f}]")
                        continue

                    for direction in directions:
                        raw = direction * base 
                        current_val = orig_pct[food] + total_steps[food]

                        # Upper limit check
                        if direction > 0:
                            # --- Increase branch ---
                            remaining_room = upper_bound - current_val
                            step_sz = min(raw, remaining_room)
                            if nova_df[food] == 4 or remaining_room <= 0:
                                continue  # No room to go up

                        elif direction < 0:
                            # --- Decrease branch ---
                            remaining_room = current_val  # distance to 0
                            step_sz = -min(abs(raw), remaining_room)  # ensure negative
                            if remaining_room <= 0:
                                continue  # Already at 0

                        # proposed pct
                        cand_pct = orig_pct + total_steps + step_sz * (orig_pct.index == food)
                        cand_pct = cand_pct / cand_pct.sum()

                        # full‐feature vector
                        cand_all = subject_diet.copy()
                        cand_all.update(cand_pct)

                        # constraint check
                        if not check_constraints(cand_pct,
                                                orig_pct,
                                                subject_calories,
                                                subject_gender,
                                                max_total_kcal_pct_change,
                                                alcohol_idx):
                            # print("Nobody cares about constraints!")
                            continue

                        # predict microbiome
                        pred_mb = predict_microbiome_from_diet_optimized(
                            cand_all, model_diet_to_mb,
                            diet_scaler, mb_scaler,
                            age_scaler, subject_age,
                            subject_gender,
                            all_features, DIET_ONLY_FEATURES, targets
                        )
                        norm_mb = normalize_predicted_microbiome_optimized(
                            pred_mb, mb_scaler, non_exist, targets
                        )

                        # predict TG
                        pred_tg = predict_triglycerides(
                            norm_mb, targets,
                            mb_scaler, age_scaler,
                            subject_age, subject_gender,
                            model_tg
                        )
                        print("Predicted triglycerides for this change: ", pred_tg)

                        if pred_tg < best_pred:
                            print("NEW BEST PRED!")
                            best_pred = pred_tg
                            best_step = (food, direction, step_sz)

                # no improvement?
                if best_step is None:
                    if phase < len(direction_phases):
                        print(f"No improvement with {directions}.  Switching to next phase.")
                    else:
                        print("No improvement. Stopping.")
                    break

                # apply best step
                food, direction, step_sz = best_step
                total_steps[food] += step_sz
                used_steps.add(f"{food}_{direction}")
                current_tg = best_pred

        # how many iterations we actually did
        iterations = it + 1

        # build final diets
        final_pct = orig_pct + total_steps
        final_pct /= final_pct.sum()
        final_diet = subject_diet.copy()
        final_diet.update(final_pct)

        # final microbiome
        final_mb = predict_microbiome_from_diet_optimized(
            final_diet, model_diet_to_mb,
            diet_scaler, mb_scaler,
            age_scaler, subject_age,
            subject_gender,
            all_features, DIET_ONLY_FEATURES, targets
        )
        final_norm_mb = normalize_predicted_microbiome_optimized(
            final_mb, mb_scaler, non_exist, targets
        )

        # final TG
        final_tg = predict_triglycerides(
            final_norm_mb, targets,
            mb_scaler, age_scaler,
            subject_age, subject_gender,
            model_tg
        )

        species_cols = [c for c in baseline_norm_mb.columns if c not in ("age", "sex")]

        baseline_lin_mb = 10 ** baseline_norm_mb[species_cols]
        
        final_lin_mb = 10 ** final_norm_mb[species_cols]

        return {
            "subject_id": subject_id,
            "original_diet": orig_pct,
            "recommended_diet": final_diet,
            "original_tg": baseline_tg,
            "final_tg": final_tg,
            "predicted_baseline_microbiome": baseline_lin_mb,
            "final_microbiome": final_lin_mb,
            "delta_microbiome": final_lin_mb.subtract(baseline_lin_mb, axis=1),
            "total_diet_change": total_steps,
            "steps_taken": len(used_steps),
            "iterations": iterations,
            "Energy": subject_calories
        }


    # CALL QUEUE HERE
    results = []
    param_results = {}
    param_methods = {}

    if foods_only:
        diet_species_model_path = f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/regression/{SPECIES}/models_LGBM_abundance_longitudinal{suffix_foods}.pkl' # FOOD FEATURES ONLY!
    else:
        diet_species_model_path = f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/regression/segal_species/models_LGBM_abundance_longitudinal.pkl'
    phenotypes_models_path = f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/models/segal_species/models_mb_phenotypes.pkl'

    for i in range(len(diet_mb_test)):  # or range(N) for multiple subjects
        print(f"Submitting job for subject {i}...")

        subject_diet, subject_mb, subject_kcal, subject_sex, subject_age = get_subject(diet_mb, i)

        param_methods[i] = q.method(
            recommend_diet_change_greedy, 
                (i,
                subject_diet,
                subject_mb,
                all_features_formatted,
                diet_scaler,
                mb_scaler,
                age_scaler,
                # diet_species_model,
                # triglycerids_model,
                diet_species_model_path,
                phenotypes_models_path,
                nova_foods_df,
                foods_upper_bounds,
                foods_lower_bounds,
                subject_kcal,
                subject_sex,
                subject_age,
                targets,
                0.2,
                10,) # Iterations
            )
    print("All jobs have been submitted to the queue.")

    for (param, stub_method) in param_methods.items():
        print(f"Waiting for result from subject {param}...")
        results.append(q.waitforresult(stub_method))
        print(f"Received result from subject {param}.")
        

    results_df = pd.DataFrame(results)
    results_df.to_pickle(home_path + f'data/diet_intervention_results_queue{suffix_foods}_{PHENOTYPE}.pkl')

    # diet_intervention_results = pd.read_pickle(home_path + 'data/kl')

# Set up queue
addloglevels.sethandlers()
os.chdir('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/code')
with config.qp(jobname='lgbm', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=8, max_u=200, _mem_def='1G') as q:
    # os.chdir('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/code')
    q.startpermanentrun()
    main(q)
    print("FINISHED")