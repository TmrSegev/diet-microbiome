from LabData import config_global as config
from LabUtils import Utils
from LabUtils import addloglevels
import math
import os
import re
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import linear_model
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 


os.environ["JOBLIB_START_METHOD"] = "fork"

### PARAMETERS ###
MODEL = 'LGBM' # 'LGBM' or 'ridge' or 'logistic'
TARGETS = 'diet' # 'div' or 'abundance' or 'diet'
SPLIT = 'kfold' # 'kfold' or 'longitudinal'
PROBLEM = 'reverse' # 'regression' or 'classification' or 'reverse' or 'given_presence'
base_training = False
robust_training = False
no_var_features = False
##################

suffix = '_longitudinal' if SPLIT == 'longitudinal' else '' 
colsample_bytree = 1 if base_training else 0.3
target_part = 'diet' if PROBLEM == 'reverse' else f'{TARGETS}'
base = 'base_' if base_training else ''
suffix_var = '_no_var_features' if no_var_features else ''


from sklearn.model_selection import RandomizedSearchCV
import joblib  # For saving the best model efficiently

from itertools import product
from sklearn.model_selection import train_test_split

# def find_best_model(df, features, targets):
#     """
#     Finds the best hyperparameters using 60 random targets,
#     optimizing for mean R² across all targets.
#     """

#     print(f"🔍 Performing hyperparameter search on 60 randomly selected targets as a batch...")

#     # Define the hyperparameter grid
#     # param_grid = {
#     #     'max_depth': [3, 4, 6],  
#     #     'n_estimators': [500, 1000, 2000],  
#     #     'learning_rate': [0.001, 0.01, 0.05],  
#     #     'colsample_bytree': [0.5, 0.7, 0.9],  
#     #     'subsample': [0.5, 0.7, 0.9],  
#     #     'min_data_in_leaf': [20, 50, 100],  
#     #     'reg_alpha': [0.5, 1, 2],  
#     #     'reg_lambda': [0.5, 1, 2],  
#     # }

#     param_grid = {
#         'reg_alpha': [0.5],  
#         'reg_lambda': [0.1, 10],  
#     }

#     # Generate all possible parameter combinations
#     param_combinations = list(product(*param_grid.values()))

#     print(f"param_combinations = {param_combinations}")

#     # Select 60 random targets
#     selected_targets = np.random.choice(targets, 5, replace=False)

#     print(f"selected_targets = {selected_targets}")

#     # Split dataset **once** for all models
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#     # Store results
#     results = []

#     for param_values in param_combinations:
#         params = dict(zip(param_grid.keys(), param_values))
#         print(f"params = {params}")
#         r2_scores = []
        
#         for target in selected_targets:
#             # Drop NaN values for this specific target
#             train_subset = train_df.dropna(subset=[target])

#             # print(train_subset.shape) #7717 instances, 1514 features + trargets

#             test_subset = test_df.dropna(subset=[target])

#             if len(train_subset) < 10 or len(test_subset) < 10:
#                 continue  # Skip if too few samples for this target

#             # Train a model for this target
#             # model = lgb.LGBMRegressor(**params, n_jobs=-1, random_state=1, subsample_freq=1, verbosity=-1)    
#             model = lgb.LGBMRegressor(**params,
#                     max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1,
#                     colsample_bytree=colsample_bytree, learning_rate=0.001, n_jobs=-1, random_state=1, 
#                     min_data_in_leaf=20, verbosity=-1
#                 )       
#             model.fit(train_subset[features], train_subset[target])

#             # Predict on test set
#             predictions = model.predict(test_subset[features])

#             if np.all(predictions == predictions[0]) or np.all(test_subset[target] == test_subset[target].iloc[0]):
#                 print(f"⚠️ Constant predictions or actual values for '{target}'. Assigning R² = 0.")
#                 r2 = 0.0
#             else:
#                 try:
#                     r2 = stats.pearsonr(predictions, test_subset[target])[0] ** 2
#                     print(f"r2 = {r2}")
#                     r2_scores.append(r2)
#                 except Exception:
#                     print("?")
#                     r2 = 0.0  # Fallback to 0 if calculation fails

#         # Compute mean R² across all targets
#         mean_r2 = np.mean(r2_scores) if r2_scores else float('-inf')
#         results.append((params, mean_r2))

#     # Sort results by mean R² score
#     results.sort(key=lambda x: x[1], reverse=True)

#     print("\n🔍 **Hyperparameter Combinations Tried and Their Mean R² Scores:**")
#     for i, (params, mean_r2) in enumerate(results):
#         print(f"Combination {i+1}: {params} -> Mean R²: {mean_r2:.4f}")

#     # Select best parameters based on highest mean R²
#     best_params = results[0][0]
#     print(f"\n✅ Best parameters found: {best_params}")

#     return best_params
        


from itertools import product
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgb
from scipy import stats

# # Define the hyperparameter grid
# param_grid = {
#     'max_depth': [3, 4, 6],  
#     'n_estimators': [500, 1000, 2000],  
#     'learning_rate': [0.001, 0.01, 0.05],  
#     'colsample_bytree': [0.5, 0.7, 0.9],  
#     'subsample': [0.5, 0.7, 0.9],  
#     'min_data_in_leaf': [20, 50, 100],  
#     'reg_alpha': [0.5, 1, 2],  
#     'reg_lambda': [0.5, 1, 2],  
# }

from itertools import product
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from scipy import stats

# def find_best_model(df, features, targets):
#     """
#     Finds the best hyperparameters using 60 random targets,
#     optimizing for mean R² across all targets.
#     """

#     print(f"🔍 Performing hyperparameter search on 60 randomly selected targets as a batch...")

#     # Define the hyperparameter grid
#     param_grid = {
#         'reg_alpha': [0.5],  
#         'reg_lambda': [0.1, 10],  
#     }

#     # Generate all possible parameter combinations
#     param_combinations = list(product(*param_grid.values()))

#     print(f"param_combinations = {param_combinations}")

#     # Select 60 random targets
#     selected_targets = np.random.choice(targets, 5, replace=False)

#     print(f"selected_targets = {selected_targets}")

#     # Split dataset **once** for all models
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#     # Store results
#     results = []

#     for param_values in param_combinations:
#         params = dict(zip(param_grid.keys(), param_values))
#         print(f"params = {params}")

#         # Train a single model for all targets
#         base_model = lgb.LGBMRegressor(**params,
#                                        max_depth=4, n_estimators=2000, subsample=0.5, subsample_freq=1,
#                                        colsample_bytree=0.7, learning_rate=0.001, n_jobs=-1, random_state=1, 
#                                        min_data_in_leaf=20, verbosity=-1)

#         r2_scores = []  # Store individual R² scores for each target

#         for target in selected_targets:
#             # Drop NaN values for this specific target
#             train_subset = train_df.dropna(subset=[target])
#             test_subset = test_df.dropna(subset=[target])

#             if len(train_subset) < 10 or len(test_subset) < 10:
#                 print(f"⚠️ Skipping target '{target}' due to insufficient data.")
#                 continue  # Skip if too few samples for this target

#             # Train and predict
#             model = base_model.fit(train_subset[features], train_subset[target])
#             predictions = model.predict(test_subset[features])

#             if np.all(predictions == predictions[0]) or np.all(test_subset[target] == test_subset[target].iloc[0]):
#                 print(f"⚠️ Constant predictions or actual values for '{target}'. Assigning R² = 0.")
#                 r2 = 0.0
#             # Compute R² for this target
#             else:
#                 try:
#                     r2 = stats.pearsonr(predictions, test_subset[target])[0] ** 2
#                     print(f"r2 = {r2}")
#                     r2_scores.append(r2)
#                 except Exception:
#                     print(f"⚠️ Pearson correlation failed for '{target}'. Assigning R² = 0.")
#                     r2_scores.append(0.0)  

#         # Compute **mean** R² across all targets
#         mean_r2 = np.mean(r2_scores) if r2_scores else float('-inf')
#         results.append((params, mean_r2))

#     # Sort results by mean R² score
#     results.sort(key=lambda x: x[1], reverse=True)

#     print("\n🔍 **Hyperparameter Combinations Tried and Their Mean R² Scores:**")
#     for i, (params, mean_r2) in enumerate(results):
#         print(f"Combination {i+1}: {params} -> Mean R²: {mean_r2:.4f}")

#     # Select best parameters based on highest mean R²
#     best_params = results[0][0]
#     print(f"\n✅ Best parameters found: {best_params}")

#     return best_params




def find_best_model(df, features, targets):
    """
    Finds the best hyperparameters using 60 random targets,
    optimizing for mean R² across all targets.
    """

    # Define the hyperparameter grid
    # param_grid = {
    #     'learning_rate': [0.005, 0.01, 0.05, 0.1],  # Allow better learning
    #     'max_depth': [4, 6],  # Allow deeper splits
    #     'num_leaves': [31, 50, 100],  # Allow more flexible decision boundaries
    #     'subsample': [0.5, 0.8, 1.0],  # Use more data per tree
    #     'colsample_bytree': [0.3, 0.8, 1.0],  # Use more features per tree
    #     'min_data_in_leaf': [5, 10, 20, 50],  # Test smaller AND larger groups
    #     'reg_alpha': [0, 0.1],  # Reduce over-regularization
    #     'reg_lambda': [0, 0.1],  # Reduce over-regularization
    #     'n_estimators': [1000, 2000, 3000],  # Test different numbers of boosting rounds
    # }

    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],  # Allow better learning
        'max_depth': [6],  # Allow deeper splits
        'num_leaves': [50],  # Allow more flexible decision boundaries 
        'subsample': [0.5],  # Use more data per tree 
        'colsample_bytree': [0.3],  # Use more features per tree 
        'min_data_in_leaf': [50],  # Test smaller AND larger groups
        'reg_alpha': [0],  # Reduce over-regularization
        'reg_lambda': [10],  # Reduce over-regularization
        'n_estimators': [2000],  # Test different numbers of boosting rounds
    }

    # Generate all possible parameter combinations
    param_combinations = list(product(*param_grid.values()))

    # Split dataset **once** for all models
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Store results
    results = []

    for param_values in param_combinations:
        params = dict(zip(param_grid.keys(), param_values))
        print(f"params = {params}")

        r_scores = []  # Store individual R² scores for each target

        for i, target in enumerate(targets):
            print(f"target = {target}, {i+1}/{len(targets)}")
            # Drop NaN values for this specific target
            train_subset = train_df.dropna(subset=[target])
            test_subset = test_df.dropna(subset=[target])

            if len(train_subset) < 10 or len(test_subset) < 10:
                print(f"⚠️ Skipping target '{target}' due to insufficient data.")
                continue  # Skip if too few samples for this target

            # Train model for this target **only once**
            model = lgb.LGBMRegressor(**params, subsample_freq=1, n_jobs=1, random_state=1, verbosity=-1)

            # model = linear_model.RidgeCV(alphas=(0.01, 0.1, 0.5, 1, 10, 100))  
            model.fit(train_subset[features], train_subset[target])

            # Predict on test set
            predictions = model.predict(test_subset[features])

            if np.all(predictions == predictions[0]) or np.all(test_subset[target] == test_subset[target].iloc[0]):
                print(f"⚠️ Constant predictions or actual values for '{target}'. Assigning R² = 0.")
                r_scores.append(0.0)
            else:
                try:
                    r = stats.pearsonr(predictions, test_subset[target])[0] 
                    print(f"r = {r} ")
                    r_scores.append(r)
                except Exception:
                    print(f"⚠️ Pearson correlation failed for '{target}'. Assigning R² = 0.")
                    r_scores.append(0.0)  

        # Compute **mean** R² across all targets
        mean_r = np.mean(r_scores) if r_scores else float('-inf')
        print(f"mean_r2 = {mean_r}")
        results.append((params, mean_r))

    # Sort results by mean R² score
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n🔍 **Hyperparameter Combinations Tried and Their Mean R² Scores:**")
    for i, (params, mean_r) in enumerate(results):
        print(f"Combination {i+1}: {params} -> Mean R: {mean_r:.4f}")

    # Select best parameters based on highest mean R²
    best_params = results[0][0]
    print(f"\n✅ Best parameters found: {best_params}")

    return best_params






def scale_data(train, test, features):
    train = train.copy()
    test = test.copy()

    # Identify binary features (features with only values 0 and 1)
    binary_features = [col for col in train.columns if train[col].nunique() == 2 and sorted(train[col].unique()) == [0, 1]]

    # Exclude binary features from standardization
    features_to_standardize = [feature for feature in features if feature not in binary_features]

    scaler = StandardScaler()
    train.loc[:, features_to_standardize] = pd.DataFrame(
    scaler.fit_transform(train[features_to_standardize]),
    columns=train[features_to_standardize].columns,
    index=train[features_to_standardize].index
    )

    # with open(path + 'scaler_diet_mb.pkl', 'rb') as scaler_file:
        # scaler = pickle.load(scaler_file)
    test.loc[:, features_to_standardize] = pd.DataFrame(
        scaler.transform(test[features_to_standardize]), 
        columns=test[features_to_standardize].columns, 
        index=test[features_to_standardize].index
    )

    return train, test, scaler


def prepare_data():
    diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb.pkl")
    print(f"Shape of the data: {diet_mb.shape}")
    # if SPLIT == 'longitudinal':

    #     diet_mb_02_visit = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_02_visit.pkl")
    #     # print(diet_mb_02_visit.shape)
    #     # visit_subjects = list(diet_mb_02_visit.index)
    #     # diet_mb_test = diet_mb.loc[diet_mb.index.isin(visit_subjects[:2000])]
    #     # diet_mb = diet_mb.loc[~diet_mb.index.isin(visit_subjects[:2000])]
    #     # print("Length of train:", len(diet_mb))
    #     # print("Length of test:", len(diet_mb_test))
        
    #     diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_baseline_train.pkl")
    #     diet_mb_test = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_baseline_test.pkl")

    with open('/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/my_lists.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    base_features, features, target_input = loaded_lists

    if no_var_features:
        features = [x for x in features if x not in [
                    'Foods_per_day_std',
                    'Foods_per_meal_std',
                    'Meals_per_day_std',
                    'plant_foods_per_day_std',
                    'plant_foods_per_week_std',
                    'med_score_per_day_std',
                    'paleo_score_per_day_std',
                    'vegetarian_score_per_day_std',
                    'wfpb_score_per_day_std',
                    'vegan_score_per_day_std',
                    'wfab_score_per_day_std',
                    'fermented_score_per_day_std',
                    'Protein_std',
                    'Total lipid (fat)_std',
                    'Carbohydrate, by difference_std',
                    'Energy_std',
                    'pct_protein_calories_std',
                    'pct_carb_calories_std',
                    'pct_fat_calories_std',
                    'intra_diet_correlation',
                    'pct_protein_calories',
                    'pct_carb_calories',
                    'pct_fat_calories'
                ]]
        
    # Train test split
    diet_mb = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb.pkl")
    diet_mb_02_visit = pd.read_pickle("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/diet_mb_02_visit.pkl")
    print(diet_mb_02_visit.shape)
    visit_subjects = list(diet_mb_02_visit.index)
    diet_mb_test = diet_mb.loc[diet_mb.index.isin(visit_subjects[:2000])]
    diet_mb_train = diet_mb.loc[~diet_mb.index.isin(visit_subjects[:2000])]
    print("Length of all:", len(diet_mb))
    print("Length of train:", len(diet_mb_train))
    print("Length of test:", len(diet_mb_test))

    diet_mb.columns = diet_mb.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_train.columns = diet_mb_train.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    diet_mb_test.columns = diet_mb_test.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    features = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in features]
    original_features = features[:]
    features = base_features if base_training else features
    target_input = [re.sub(r'[^a-zA-Z0-9_]', '_', x) for x in target_input]
    
    # Scaling the data

    if PROBLEM == 'reverse':
        mb_features = (target_input + ['Richness', 'Shannon_diversity'] + base_features) if not base_training else base_features
        diet_targets = [feat for feat in original_features if feat not in ['age', 'gender']]

        # Standardize the data
        diet_mb_train, diet_mb_test, scaler = scale_data(diet_mb_train, diet_mb_test, mb_features)
    
        # with open("/net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/data/scaler.pkl", 'wb') as file:
        #     pickle.dump(scaler, file)

    top_lgbm = ['wfpb_score_per_day', 'NOVA_food_score', 'med_score_per_day', 'vegan_score_per_day', 'pct_carb_calories', 'paleo_score_per_day', 'plant_protein_pct', 'plant_energy_pct', 'Nutsseedsandproducts', 'Potassium__K', 'pescatarian_score_per_day', 'carnivore_score_per_day', 'Fruits', 'vegetarian_score_per_day', 'pct_fat_calories', 'plant_fat_pct', 'Fiber__total_dietary', 'Energy', 'sat_to_total_lipids_ratio', 'Magnesium__Mg', 'Cholesterol']
    
    problematic_features = ['Vitamin_B_6', 'Plum', 'Valine', 'Arginine', 'Threonine', 'Tyrosine', 'omega_3', 'Fatty_acids__total_trans', 'Lysine', 'Zinc__Zn', 'Phenylalanine', 'Leucine', 'Methionine', 'Isoleucine', 'Tryptophan', 'Serine', 'Histidine', 'Manganese__Mn', 'Iron__Fe', 'Thiamin', 'Pantothenic_acid', 'Cucumber', 'Cystine', 'Aspartic_acid', 'Niacin', 'Vitamin_B_12', 'Hydroxyproline', 'Fatty_acids__total_polyunsaturated', 'Vitamin_D__D2___D3_', 'Fatty_acids__total_saturated', 'Glycine', 'Proline', 'Lettuce', 'Alanine', 'Riboflavin', 'Glutamic_acid', 'Copper__Cu', 'omega_6', 'Total_lipid__fat__std'] 

    # diet_targets = top_lgbm + problematic_features
    
    diet_targets = top_lgbm + problematic_features

    return diet_mb, mb_features, diet_targets   
        
        
def read_results(df):
    output = []
    for col in df.columns:
        output.append(df[col])
    return tuple(output)


if __name__ == '__main__':

    diet_mb, features, target_input = prepare_data()

    find_best_model(diet_mb, features, target_input)