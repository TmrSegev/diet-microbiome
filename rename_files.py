# import os
# import sys

# # --- Configuration ---
# ROOT_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/'  # Start searching from the current directory
# SUFFIX_TO_FIND = '_with_covariates.pkl'
# SUFFIX_TO_REPLACE = '.pkl'
# NEW_SUFFIX = '_age_sex_only.pkl'
# # ---------------------

# def rename_files(root_dir, dry_run=True):
#     """
#     Recursively searches for files ending in SUFFIX_TO_FIND.
#     For each corresponding file without that suffix, it performs a rename.
#     """
#     if dry_run:
#         print("--- STARTING DRY RUN ---")
#         print("Files will NOT be modified. Showing planned changes only.\n")
#     else:
#         print("--- STARTING LIVE EXECUTION ---")
#         print("Files WILL be modified.\n")

#     # Walk through all directories and files starting from the root_dir
#     for dirpath, dirnames, filenames in os.walk(root_dir):
        
#         # We only care about files that have the SUFFIX_TO_FIND
#         for filename in filenames:
#             if filename.endswith(SUFFIX_TO_FIND):
                
#                 # 1. Construct the path to the file with the suffix
#                 path_with_suffix = os.path.join(dirpath, filename)
                
#                 # 2. Calculate the expected path of the corresponding file WITHOUT the suffix
#                 # (e.g., /path/to/output_base_LGBM_abundance.pkl)
#                 filename_without_suffix = filename.replace(SUFFIX_TO_FIND, SUFFIX_TO_REPLACE)
#                 path_without_suffix = os.path.join(dirpath, filename_without_suffix)
                
#                 # Check if the file to be renamed actually exists
#                 if os.path.exists(path_without_suffix):
                    
#                     # 3. Calculate the new name
#                     # (e.g., /path/to/output_base_LGBM_abundance_age_sex_only.pkl)
#                     new_filename = filename_without_suffix.replace(SUFFIX_TO_REPLACE, NEW_SUFFIX)
#                     new_path = os.path.join(dirpath, new_filename)
                    
#                     print(f"Target found: {path_without_suffix}")
#                     print(f"Planned rename: -> {new_path}\n")
                    
#                     if not dry_run:
#                         try:
#                             os.rename(path_without_suffix, new_path)
#                             print(f"SUCCESS: Renamed {path_without_suffix} to {new_path}")
#                         except OSError as e:
#                             print(f"ERROR: Could not rename {path_without_suffix}. Reason: {e}")
                
#     if dry_run:
#         print("--- DRY RUN COMPLETE ---")
#     else:
#         print("--- LIVE EXECUTION COMPLETE ---")


# if __name__ == "__main__":
#     # By default, run the script in DRY RUN mode
    
#     # Check if the user passed the '--execute' or '-e' argument
#     # If they did, set dry_run to False (i.e., perform the actual renaming)
#     if '--execute' in sys.argv or '-e' in sys.argv:
#         rename_files(ROOT_DIR, dry_run=False)
#     else:
#         # Run the dry run first
#         rename_files(ROOT_DIR, dry_run=True)
#         print("\nTo perform the actual renaming, run the script again with the '--execute' argument:")
#         print("python3 rename_files.py --execute")


import os
import sys

# --- Configuration ---
ROOT_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb'  # Start searching from the current directory
SUFFIX_TO_REMOVE = '_with_covariates.pkl'
NEW_SUFFIX = '.pkl'
# ---------------------

def remove_suffix_from_files(root_dir, dry_run=True):
    """
    Recursively searches for files ending in SUFFIX_TO_REMOVE and renames them
    by replacing the suffix with NEW_SUFFIX.
    """
    if dry_run:
        print("--- STARTING DRY RUN (Removal) ---")
        print("Files will NOT be modified. Showing planned changes only.\n")
    else:
        print("--- STARTING LIVE EXECUTION (Removal) ---")
        print("Files WILL be modified.\n")

    # Walk through all directories and files starting from the root_dir
    for dirpath, dirnames, filenames in os.walk(root_dir):
        
        # We only care about files that have the SUFFIX_TO_REMOVE
        for filename in filenames:
            if filename.endswith(SUFFIX_TO_REMOVE):
                
                # 1. Construct the path to the file with the suffix
                path_with_suffix = os.path.join(dirpath, filename)
                
                # 2. Calculate the new filename by removing the suffix
                # Example: output..._with_covariates.pkl -> output....pkl
                new_filename = filename.replace(SUFFIX_TO_REMOVE, NEW_SUFFIX)
                new_path = os.path.join(dirpath, new_filename)
                
                print(f"Current file: {path_with_suffix}")
                print(f"Planned rename: -> {new_path}\n")
                
                if not dry_run:
                    # Check if the target file already exists to prevent overwriting
                    if os.path.exists(new_path):
                        print(f"SKIPPED: Target file already exists at {new_path}. Rename aborted to prevent overwrite.")
                        continue
                        
                    try:
                        os.rename(path_with_suffix, new_path)
                        print(f"SUCCESS: Renamed {path_with_suffix} to {new_path}")
                    except OSError as e:
                        print(f"ERROR: Could not rename {path_with_suffix}. Reason: {e}")
                
    if dry_run:
        print("--- DRY RUN COMPLETE ---")
    else:
        print("--- LIVE EXECUTION COMPLETE ---")


if __name__ == "__main__":
    # Check if the user passed the '--execute' or '-e' argument
    if '--execute' in sys.argv or '-e' in sys.argv:
        remove_suffix_from_files(ROOT_DIR, dry_run=False)
    else:
        # Run the dry run first
        remove_suffix_from_files(ROOT_DIR, dry_run=True)
        print("\nTo perform the actual renaming, run the script again with the '--execute' argument:")
        print("python3 clean_files.py --execute")