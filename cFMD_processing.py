import pandas as pd
import os
from glob import glob

# ─── CONFIGURE ───
base_path = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/cFMD/cFMD_data/"
# output_file = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/data/merged_cFMD_with_metadata.tsv"
output_path = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/tomerse/diet_mb/data/"

# ─── FIND ALL taxonomic_profiles.tsv FILES ───
all_files = glob(os.path.join(base_path, "*", "*_taxonomic_profiles.tsv"))

metadata_blocks = []
taxa_blocks     = []

for f in all_files:
    # sample prefix (to guarantee unique column names)
    sample = os.path.basename(f).replace("_taxonomic_profiles.tsv", "")
    
    # read entire sheet, using the first column as the row index
    df = pd.read_csv(f, sep="\t", index_col=0)
    
    # boolean mask: “True” for taxa rows (they contain a ‘|’)
    is_taxa = df.index.str.contains(r"\|")
    
    # split
    md = df.loc[~is_taxa].copy()
    tx = df.loc[ is_taxa].copy()
    tx = tx.apply(pd.to_numeric, errors='coerce')  # ensures values are float or NaN
    
    # rename columns to SAMPLE__ORIGINALCOL so they never collide
    md.columns = [f"{sample}__{col}" for col in md.columns]
    tx.columns = [f"{sample}__{col}" for col in tx.columns]
    
    metadata_blocks.append(md)
    taxa_blocks.append(tx)


# ─── MERGE them ───
# side-by-side (outer) on the column axis
all_metadata = pd.concat(metadata_blocks, axis=1)
all_taxa     = pd.concat(taxa_blocks,     axis=1).fillna(0)

# ─── GROUP TAXA BY SPECIES ───
def extract_species(taxon_name):
    parts = taxon_name.split('|')
    species_parts = [p for p in parts if p.startswith("s__")]
    return species_parts[0] if species_parts else taxon_name

species_index = all_taxa.index.to_series().apply(extract_species)
all_taxa_grouped = all_taxa.groupby(species_index).sum(numeric_only=True)

# save
all_taxa_grouped.to_csv(output_path + "cFMD_taxa_only.tsv", sep="\t")
all_metadata.to_csv(output_path + "cFMD_metadata_only.tsv", sep="\t")

print(f"✅ Written merged file to {output_path}")
