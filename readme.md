This repo contains all code used in analysis and figures generation in the manuscript "Diet shapes the gut microbiome: cross-sectional and longitudinal insights from the Human Phenotype Project".
The data used in this paper is part of the Human Phenotype Project (HPP) and is accessible to researchers from universities and other research institutions at: https://humanphenotypeproject.org/data-access.
Interested bona fide researchers should contact info@pheno.ai to obtain instructions for accessing the data

### Instructions of use
The run of each file may take a few minutes up to an hour, depending on the number of features and samples in the dataset. The permutation analysis may take up to a few days.

#### Diet-microbiome predictions pipeline
- `diet_processing.ipynb`
- `create_diet_mb.ipynb`
- `train_models.py`  
  - Parameters:
    - div + ridge
    - div + LGBM
    - abundance + ridge
    - abundance + LGBM
- `permutations_queue.py`  
  - Parameters:
    - div + LGBM
    - abundance + LGBM
- `predict.py`
- `analysis.ipynb`
- `SHAP.py`  
  - Parameters:
    - div + ridge
    - div + LGBM
    - abundance + LGBM
- `shap_analysis.ipynb`
- `SHAP_summary_plot.py`
- `longitudinal_analysis.ipynb`

#### Microbiome-phenotypes prediction and diet intervention simulation pipeline
- `create_phenotypes_mb.ipynb`
- `predict_phenotypes_model.py`
- `predict_phenotypes_analysis.ipynb`
- `phenotypes_mb_shap.py`
- `phenotypes_mb_shap_analysis.ipynb`
- `diet_intervention_queue.py`
- `diet_intervention_analysis.ipynb`


### Software dependencies:
Python 3.7.4, numpy 1.21.0, pandas 1.3.5, scipy 1.7.0, statsmodels 0.13.2, scikit-learn 0.23.2, lightgbm 4.5.0, shap 0.39.0, matplotlib 3.4.3, seaborn 0.11.1, MetaPhlAn 4.0.6, humann 3.6.1

### Demo
Run `demo.ipynb` to train models on a mock data and generate Fig. 3a. Expected time: 30 minutes.


