import pandas as pd
import pickle
import re
import os

# Configuration (matching longitudinal_analysis.ipynb)
SPECIES = 'segal_species'  # 'mpa_species' or 'segal_species'
home_path = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
TARGETS = 'abundance'  # 'abundance' or 'div'
PROBLEM = 'regression'

# Helper function to read results (matching notebook)
def read_results(df):
    output = []
    for col in df.columns:
        output.append(df[col])
    return tuple(output)

def map_changes_to_timepoint(master_df, baseline_df, visit02_df, visit04_df, prefix):
    """Helper function to map change dataframes based on Time_Point"""
    changes_list = []
    for _, row in master_df.iterrows():
        regcode = row['RegistrationCode']
        tp = row['Time_Point']
        if tp == 0:
            if regcode in baseline_df.index:
                changes_list.append(baseline_df.loc[regcode].to_dict())
            else:
                changes_list.append({col: 0 for col in baseline_df.columns})
        elif tp == 2:
            if regcode in visit02_df.index:
                changes_list.append(visit02_df.loc[regcode].to_dict())
            else:
                changes_list.append({col: 0 for col in visit02_df.columns})
        elif tp == 4:
            if regcode in visit04_df.index:
                changes_list.append(visit04_df.loc[regcode].to_dict())
            else:
                changes_list.append({col: 0 for col in visit04_df.columns})
        else:
            changes_list.append({col: 0 for col in baseline_df.columns})
    
    changes_df = pd.DataFrame(changes_list, index=master_df.index)
    changes_df.columns = [f'{prefix}_{col}' for col in changes_df.columns]
    return changes_df


def create_master_df():
    """
    Create master_df from measured and predicted dataframes, including covariates.
    Returns the master_df DataFrame.
    """
    # --- Load diet_mb and metadata ---
    diet_mb = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb.pkl")

    # Load feature lists
    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/my_lists.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, all_features, targets = loaded_lists

    # Load significant targets
    with open(f'/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/significant_targets.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    significant_targets = loaded_lists

    # Clean column names (replace special characters with underscores)
    all_features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in all_features]
    targets = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in targets]
    significant_targets = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in significant_targets]

    # --- Load measured dataframes ---
    measured_baseline = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb_baseline_test.pkl")
    measured_02_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb_02_visit_test.pkl")
    measured_04_visit = pd.read_pickle(f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{SPECIES}/diet_mb_04_visit_test.pkl")

    # Clean column names in measured dataframes
    measured_baseline.columns = measured_baseline.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    measured_02_visit.columns = measured_02_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    measured_04_visit.columns = measured_04_visit.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    # Filter measured dataframes to only include significant_targets
    measured_baseline = measured_baseline[significant_targets]
    measured_02_visit = measured_02_visit[significant_targets]
    measured_04_visit = measured_04_visit[significant_targets]

    # --- Load prediction results and create predicted dataframes ---
    pred_baseline_scores, pred_baseline_pvalues, pred_baseline_coefs, pred_baseline_preds, pred_baseline_targets = read_results(
        pd.read_pickle(home_path + f"data/{PROBLEM}/{SPECIES}/predictions_LGBM_" + TARGETS + "_baseline.pkl")
    )
    pred_02_visit_scores, pred_02_visit_pvalues, pred_02_visit_coefs, pred_02_visit_preds, pred_02_visit_targets = read_results(
        pd.read_pickle(home_path + f"data/{PROBLEM}/{SPECIES}/predictions_LGBM_" + TARGETS + "_02_visit.pkl")
    )
    pred_04_visit_scores, pred_04_visit_pvalues, pred_04_visit_coefs, pred_04_visit_preds, pred_04_visit_targets = read_results(
        pd.read_pickle(home_path + f"data/{PROBLEM}/{SPECIES}/predictions_LGBM_" + TARGETS + "_04_visit.pkl")
    )

    # Create predicted dataframes (matching notebook exactly)
    predicted_baseline_df = pd.DataFrame(pred_baseline_preds.to_list()).T
    predicted_baseline_df.columns = targets
    predicted_baseline_df = predicted_baseline_df.loc[:, significant_targets]
    predicted_baseline_df.index = measured_baseline.index

    predicted_02_visit_df = pd.DataFrame(pred_02_visit_preds.to_list()).T
    predicted_02_visit_df.columns = targets
    predicted_02_visit_df = predicted_02_visit_df.loc[:, significant_targets]
    predicted_02_visit_df.index = measured_02_visit.index

    predicted_04_visit_df = pd.DataFrame(pred_04_visit_preds.to_list()).T
    predicted_04_visit_df.columns = targets
    predicted_04_visit_df = predicted_04_visit_df.loc[:, significant_targets]
    predicted_04_visit_df.index = measured_04_visit.index

    # --- Filter to common subjects (matching notebook preprocessing) ---
    # Filter baseline to only include subjects present in 02_visit
    measured_baseline = measured_baseline.loc[measured_baseline.index.isin(measured_02_visit.index)]
    predicted_baseline_df = predicted_baseline_df.loc[predicted_baseline_df.index.isin(measured_02_visit.index)]

    # 1. Define your dictionary of dataframes
    data_map = {
        0: (measured_baseline, predicted_baseline_df),
        2: (measured_02_visit, predicted_02_visit_df),
        4: (measured_04_visit, predicted_04_visit_df)
    }

    long_dfs = []

    # --- Loop through time points ---
    for time_point, (df_measured, df_predicted) in data_map.items():
        # Process Measured Data
        temp_meas = df_measured.reset_index()
        if 'index' in temp_meas.columns:
            temp_meas.rename(columns={'index': 'RegistrationCode'}, inplace=True)
        melted_meas = temp_meas.melt(id_vars='RegistrationCode', var_name='Microbe', value_name='Observed_Abundance')
        
        # Process Predicted Data
        temp_pred = df_predicted.reset_index()
        if 'index' in temp_pred.columns:
            temp_pred.rename(columns={'index': 'RegistrationCode'}, inplace=True)
        melted_pred = temp_pred.melt(id_vars='RegistrationCode', var_name='Microbe', value_name='Predicted_Abundance')

        # Merge Measured and Predicted
        merged_tp = pd.merge(melted_meas, melted_pred, on=['RegistrationCode', 'Microbe'], how='inner')
        merged_tp['Time_Point'] = time_point
        long_dfs.append(merged_tp)

    master_df = pd.concat(long_dfs, ignore_index=True)

    # --- METADATA MERGE & AGE UPDATE ---

    # 2. Prepare metadata (Age and Sex)
    metadata_to_merge = diet_mb[["age", "sex"]].copy()
    metadata_to_merge.index.name = 'RegistrationCode'

    # 3. Merge baseline metadata
    master_df = pd.merge(master_df, metadata_to_merge, on='RegistrationCode', how='left')

    # 4. Update 'age' in place to reflect age at each visit
    master_df['age'] = master_df['age'] + master_df['Time_Point']

    # --- END REVISION ---

    # Sorting
    master_df.sort_values(by=['Microbe', 'RegistrationCode', 'Time_Point'], inplace=True)

    # --- ADD COVARIATES ---
    # Load additional covariates for mixed effect models
    # BMI and comorbidities are already in diet_mb
    # Get covariate column names from covariates.pkl
    with open(home_path + 'data/covariates.pkl', 'rb') as file:
        bmi_col, comorbidity_cols, _ = pickle.load(file)

    comorbidity_cols = list(comorbidity_cols)

    # Extract BMI and comorbidities from diet_mb
    bmi_comorb_df = diet_mb[[bmi_col] + comorbidity_cols].copy()
    bmi_comorb_df.index.name = 'RegistrationCode'
    bmi_comorb_df = bmi_comorb_df.reset_index()

    # Load demographics, lifestyle, and medications change dataframes
    demog_baseline_change = pd.read_pickle(home_path + f"data/demog_baseline_change.pkl")
    demog_02_visit_change = pd.read_pickle(home_path + f"data/demog_02_visit_change.pkl")
    demog_04_visit_change = pd.read_pickle(home_path + f"data/demog_04_visit_change.pkl")

    lifestyle_baseline_change = pd.read_pickle(home_path + f"data/lifestyle_baseline_change.pkl")
    lifestyle_02_visit_change = pd.read_pickle(home_path + f"data/lifestyle_02_visit_change.pkl")
    lifestyle_04_visit_change = pd.read_pickle(home_path + f"data/lifestyle_04_visit_change.pkl")

    medications_baseline_change = pd.read_pickle(home_path + f"data/medications_baseline_change.pkl")
    medications_02_visit_change = pd.read_pickle(home_path + f"data/medications_02_visit_change.pkl")
    medications_04_visit_change = pd.read_pickle(home_path + f"data/medications_04_visit_change.pkl")

    # Create change dataframes aligned to master_df rows
    demog_changes_df = map_changes_to_timepoint(
        master_df, demog_baseline_change, demog_02_visit_change, demog_04_visit_change, 'demog_change'
    )
    lifestyle_changes_df = map_changes_to_timepoint(
        master_df, lifestyle_baseline_change, lifestyle_02_visit_change, lifestyle_04_visit_change, 'lifestyle_change'
    )
    medications_changes_df = map_changes_to_timepoint(
        master_df, medications_baseline_change, medications_02_visit_change, medications_04_visit_change, 'medications_change'
    )

    # Merge all covariates into master_df
    # 1. Merge BMI and comorbidities (same for all time points for a subject)
    master_df = pd.merge(master_df, bmi_comorb_df, on='RegistrationCode', how='left')

    # 2. Merge change dataframes (already row-aligned to master_df)
    master_df = pd.concat(
        [master_df, demog_changes_df, lifestyle_changes_df, medications_changes_df],
        axis=1
    )

    print("Covariates loaded and merged.")
    print(
        "New covariate columns:",
        [col for col in master_df.columns
         if col not in ['RegistrationCode', 'Microbe', 'Observed_Abundance',
                        'Predicted_Abundance', 'Time_Point', 'age', 'sex']]
    )
    print("Master_df shape after adding covariates:", master_df.shape)

    # --- APPLY SCALING ---
    # Load age_scaler and BMI_scaler and apply to master_df
    age_scaler = pd.read_pickle(home_path + f"data/{SPECIES}/age_scaler.pkl")
    bmi_scaler = pd.read_pickle(home_path + f"data/{SPECIES}/bmi_scaler.pkl")

    # Reshape to 2D arrays as required by sklearn transformers
    master_df.loc[:, 'age'] = age_scaler.transform(master_df['age'].values.reshape(-1, 1)).flatten()
    master_df.loc[:, 'bmi'] = bmi_scaler.transform(master_df['bmi'].values.reshape(-1, 1)).flatten()

    print("Applied scaling to age and BMI.")

    return master_df


def main():
    """
    Main function to create and save master_df.
    """
    print("Creating master_df...")
    print(f"SPECIES: {SPECIES}")
    print(f"TARGETS: {TARGETS}")
    print(f"PROBLEM: {PROBLEM}")
    
    master_df = create_master_df()
    
    print("Final Data Shape:", master_df.shape)
    print(master_df[['RegistrationCode', 'Time_Point', 'age', 'sex']].head(10))
    
    # Save master_df (matching notebook save path)
    output_path = f"{home_path}data/{PROBLEM}/{SPECIES}/master_df.pkl"
    master_df.to_pickle(output_path)
    print(f"\n✅ master_df saved to: {output_path}")


if __name__ == '__main__':
    main()

