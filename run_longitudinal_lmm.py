from LabData import config_global as config
from LabUtils import addloglevels
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import re
import os

# Configuration (matching longitudinal_analysis.ipynb)
SPECIES = 'segal_species'  # 'mpa_species' or 'segal_species'
home_path = '/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/'
TARGETS = 'abundance'  # 'abundance' or 'div'
PROBLEM = 'regression'
TEST_MODE = False  # Set to True to test with only 3 species

# Load master_df
master_df_path = f"{home_path}data/{PROBLEM}/{SPECIES}/master_df.pkl"
master_df = pd.read_pickle(master_df_path)
print(f"Loaded master_df from: {master_df_path}")
print(f"master_df shape: {master_df.shape}")

# Load covariate column names from saved lists
# Load BMI and comorbidities from covariates.pkl
with open(home_path + 'data/covariates.pkl', 'rb') as file:
    bmi_col, comorbidity_cols, _ = pickle.load(file)
comorbidity_cols = list(comorbidity_cols)

# Load change dataframe column names from change_covariates.pkl
with open(home_path + 'data/change_covariates.pkl', 'rb') as file:
    demog_change_cols, lifestyle_change_cols, medications_change_cols = pickle.load(file)

# Build the full list of covariate column names as they appear in master_df
# Age and sex are direct columns
covariate_cols = ['age', 'sex']

# BMI and comorbidities are direct columns (need to sanitize to match master_df)
bmi_col_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', bmi_col)
comorbidity_cols_sanitized = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in comorbidity_cols]
covariate_cols.extend([bmi_col_sanitized] + comorbidity_cols_sanitized)

# Change columns have prefixes: demog_change_, lifestyle_change_, medications_change_
covariate_cols.extend([f'demog_change_{col}' for col in demog_change_cols])
covariate_cols.extend([f'lifestyle_change_{col}' for col in lifestyle_change_cols])
covariate_cols.extend([f'medications_change_{col}' for col in medications_change_cols])

# Sanitize all column names to match what's in master_df
covariate_cols_sanitized = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in covariate_cols]
# Remove multiple consecutive underscores
covariate_cols_sanitized = [re.sub(r'_+', '_', col).strip('_') for col in covariate_cols_sanitized]

# Filter to only columns that actually exist in master_df
covariate_columns = [col for col in covariate_cols_sanitized if col in master_df.columns]

print(f"Loaded {len(covariate_columns)} covariate columns from saved lists")
print(f"  Age/sex: 2")
print(f"  BMI/comorbidities: {len([bmi_col_sanitized] + comorbidity_cols_sanitized)}")
print(f"  Demographics changes: {len([col for col in covariate_columns if col.startswith('demog_change_')])}")
print(f"  Lifestyle changes: {len([col for col in covariate_columns if col.startswith('lifestyle_change_')])}")
print(f"  Medications changes: {len([col for col in covariate_columns if col.startswith('medications_change_')])}")

# Rename columns with special characters to valid Python identifiers for formula parsing
# This avoids "error tokenizing input" issues with patsy
column_mapping = {}
for col in master_df.columns:
    # Check if column name needs sanitization (contains special chars that aren't underscores)
    if any(char in col for char in [' ', '(', ')', '-', '/', '.', '[', ']']):
        sanitized = col.replace(' ', '_').replace('(', '_').replace(')', '_').replace('-', '_').replace('/', '_').replace('.', '_').replace('[', '_').replace(']', '_')
        # Remove multiple consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        if sanitized != col:
            column_mapping[col] = sanitized

if column_mapping:
    print(f"Renaming {len(column_mapping)} columns with special characters...")
    master_df = master_df.rename(columns=column_mapping)
    # Update covariate_columns to use sanitized names
    covariate_columns = [column_mapping.get(col, col) for col in covariate_columns if col in column_mapping or col in master_df.columns]
    print(f"Sample renames: {list(column_mapping.items())[:3]}...")

# Check for columns with no variation (all same value) - after renaming
print("\n=== Checking for columns with no variation ===")
core_columns = ['RegistrationCode', 'Microbe', 'Observed_Abundance', 'Predicted_Abundance', 'Time_Point']
columns_to_check = [col for col in master_df.columns if col not in core_columns]

no_variance_cols = []
for col in columns_to_check:
    unique_values = master_df[col].nunique()
    if unique_values <= 1:
        constant_value = master_df[col].iloc[0] if len(master_df) > 0 else None
        no_variance_cols.append((col, constant_value, unique_values))

if no_variance_cols:
    print(f"\nFound {len(no_variance_cols)} columns with no variation:")
    for col, value, n_unique in no_variance_cols:
        print(f"  - {col}: constant value = {value} (n_unique = {n_unique})")
else:
    print("\nNo columns with zero variation found.")
print("=" * 60)

# Filter covariate_columns again to ensure they exist after renaming
covariate_columns = [col for col in covariate_columns if col in master_df.columns]

# Build formula: Observed ~ Predicted + Time + all covariates
# All column names are now valid Python identifiers, so no escaping needed
formula_parts = ['Observed_Abundance ~ Predicted_Abundance + Time_Point']
formula_parts.extend(covariate_columns)
model_formula = ' + '.join(formula_parts)

print(f"\nModel formula includes {len(covariate_columns)} covariates")
print(f"Sample covariates: {covariate_columns[:5]}...")


# Function to fit LMM for a single microbe (runs on queue)
def fit_lmm_model(master_df, microbe, i):
    """
    Fit linear mixed model for a single microbe.
    This function will be executed on the queue.
    """
    sub_df = master_df[master_df['Microbe'] == microbe].copy()
    
    # Skip if data is sparse
    if len(sub_df) < 50:
        return {
            'Microbe': microbe,
            'Coeff_Predicted': np.nan,
            'CI_Lower': np.nan,
            'CI_Upper': np.nan,
            'P_Value_Predicted': np.nan,
            'Coeff_Time': np.nan,
            'P_Value_Time': np.nan,
            'Converged': False,
            'Error': f'Insufficient data (n={len(sub_df)})'
        }
    
    # Build a per-microbe covariate list that exists and has variation
    available_covs = [c for c in covariate_columns if c in sub_df.columns]
    covariates_with_variation = []
    for c in available_covs:
        series = sub_df[c]
        # Require at least two non-NA values and more than one unique value
        if series.notna().sum() < 2:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        covariates_with_variation.append(c)

    # Build formula specific to this microbe
    formula_parts = ['Observed_Abundance ~ Predicted_Abundance + Time_Point']
    if covariates_with_variation:
        formula_parts.extend(covariates_with_variation)
    model_formula_local = ' + '.join(formula_parts)

    try:
        # Model: Observed ~ Predicted + Time + available covariates
        model = smf.mixedlm(
            model_formula_local,
            data=sub_df,
            groups=sub_df["RegistrationCode"],
            missing='drop'
        )
        
        result = model.fit(reml=False)
        
        # Extract Confidence Intervals
        # 0 = Lower, 1 = Upper
        conf_ints = result.conf_int(alpha=0.05)
        ci_lower = conf_ints.loc['Predicted_Abundance', 0] if 'Predicted_Abundance' in conf_ints.index else np.nan
        ci_upper = conf_ints.loc['Predicted_Abundance', 1] if 'Predicted_Abundance' in conf_ints.index else np.nan
        
        return {
            'Microbe': microbe,
            'Coeff_Predicted': result.params.get('Predicted_Abundance', np.nan),
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'P_Value_Predicted': result.pvalues.get('Predicted_Abundance', np.nan),
            'Coeff_Time': result.params.get('Time_Point', np.nan),
            'P_Value_Time': result.pvalues.get('Time_Point', np.nan),
            'Converged': result.converged,
            'Error': None
        }
        
    except Exception as e:
        return {
            'Microbe': microbe,
            'Coeff_Predicted': np.nan,
            'CI_Lower': np.nan,
            'CI_Upper': np.nan,
            'P_Value_Predicted': np.nan,
            'Coeff_Time': np.nan,
            'P_Value_Time': np.nan,
            'Converged': False,
            'Error': str(e)
        }


def stub_job(q):
    """
    Main job function that submits LMM fitting jobs to the queue.
    """
    print('Job started run_longitudinal_lmm.py')
    print(f"SPECIES: {SPECIES}")
    print(f"TARGETS: {TARGETS}")
    print(f"PROBLEM: {PROBLEM}")
    
    microbe_list = master_df['Microbe'].unique()
    
    # Test mode: limit to 3 species
    if TEST_MODE:
        microbe_list = microbe_list[:3]
        print(f"TEST MODE: Limiting to {len(microbe_list)} species for testing")
    
    print(f"Total microbes to fit: {len(microbe_list)}")
    
    # Dictionary to store job stubs
    all_job_stubs = {}
    
    print("🚀 Submitting jobs for all microbes...")
    for i, microbe in enumerate(microbe_list):
        # Submit the job and store its stub
        stub_method = q.method(fit_lmm_model, (master_df, microbe, i,))
        all_job_stubs[microbe] = stub_method
    
    print("\n✅ All jobs have been submitted to the scheduler.\n")
    
    # Collect results
    results_list = []
    print("⏳ Waiting for results...")
    for i, microbe in enumerate(microbe_list):
        # Retrieve the job stub for this microbe
        stub_to_wait_for = all_job_stubs[microbe]
        
        # Wait for the result
        result = q.waitforresult(stub_to_wait_for)
        results_list.append(result)
        
        # Progress feedback
        if (i + 1) % 10 == 0 or (i + 1) == len(microbe_list):
            print(f"Completed {i + 1}/{len(microbe_list)} microbes...")
    
    # Convert results to DataFrame
    lmm_results = pd.DataFrame(results_list)
    
    # Save results
    output_path = f"/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/{PROBLEM}/{SPECIES}/lmm_results_longitudinal.pkl"
    lmm_results.to_pickle(output_path)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Display summary
    print("\n=== LMM Results Summary ===")
    print(f"Total microbes processed: {len(lmm_results)}")
    print(f"Successfully converged: {lmm_results['Converged'].sum()}")
    print(f"Failed or insufficient data: {(~lmm_results['Converged']).sum()}")
    print("\nFirst 10 results:")
    print(lmm_results.head(10))
    
    return lmm_results


def main():
    """
    Main function that sets up the queue and runs the job.
    """
    os.chdir('/net/mraid20/export/genie/LabData/Analyses/tomerse/tmp/')
    with config.qp(jobname='lmm_longitudinal', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=8, max_u=200, _mem_def='1G') as q:
        os.chdir('/net/mraid20/export/genie/LabData/Analyses/tomerse/tmp/')
        q.startpermanentrun()
        stub_job(q)


if __name__ == '__main__':
    addloglevels.sethandlers()
    main()
