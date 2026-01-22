# ER-Wikidata-WF
A repo for the paper: A Wikidata-based Workflow for Entity Reconciliation Strategies Evaluation: A Study on Early Modern Polish Personal Names

# Reproducing the paper results

This repo contains three notebooks + a helper module to reproduce the metric evaluation workflow from the paper.

## Files
- `1.-dataset_acquisition.ipynb`: queries Wikidata and builds the validation dataset (labels + aliases + token lists). 
- `2.-distance_calculation.ipynb`: for each configuration (DM/BM/JW/LEV/NLEV), builds token-match sets and computes Dice–Sørensen coverage; writes one parquet per config.
- `3.-metrics_comparison.ipynb`: loads all saved config outputs and selects the best coverage threshold by F2/recall (as in the paper).
- `4. rf-xgb-notebook.ipynb`: trains Random Forest and XGBoost classifiers using 5-fold cross-validation. Generates ROC curves (Fig. 4a/b) and performance tables (Tables 3 & 4).
- `5. labse_cross_validation.ipynb`: fine-tunes LaBSE on the limited balanced subset (direct comparability with Tree-based ensembles) and on the full training dataset (scalability and performance upper bound).
- `functions_for_metrics.py`: shared functions (distance matrices, phonetic matching, Dice–Sørensen coverage, row scoring).

## Run order

### 1) Build the Wikidata dataset
Open and run:
- `1.-dataset_acquisition.ipynb` 

Output expected by later steps:
- `wd_dataset.parquet` 

### 2) Compute per-configuration coverage outputs
Open and run:
- `2.-distance_calculation.ipynb`

What it produces:
- `coverage_outputs/coverage_<CONFIG>.parquet` for each config in `CONFIGS`.

### 3) Evaluate metrics and pick best thresholds
Open and run:
- `3.-metrics_comparison.ipynb`

Inputs:
- `wd_dataset.parquet` (gold pairs) 
- `coverage_outputs/*.parquet` (predicted pairs with `match_count` / coverage) 

Output:
- A summary table (per config) with best F2, recall, and the selected threshold. 

### 4) Train and Evaluate Tree-based Ensembles
Open and run:
- `4. rf-xgb-notebook.ipynb`

Inputs:
- `wd_dataset.parquet`: The main dataset generated in Step 1 (used to create train/validation splits).
- `df_merged.parquet`: The merged dataset that can be divided into train and test

Outputs:
- `figures/roc_rf_similarity.pdf`: ROC curve for Random Forest.
- `figures/roc_xgb_similarity.pdf`: ROC curve for XGBoost.
- Console output: Cross-validated metrics (F2, Recall, Precision) corresponding to Tables 3 and 4 in the paper.

### 5) Neural Baseline evaluation (LaBSE)
Open and run:
- `5. labse_cross_validation.ipynb`: reproduces the ROC curve and metrics for the comparison on the limited subset (Figure 4c, Table 6). Via the FULL_TEST (setting it to True) variable can reproduce the results for the full-dataset experiment (Table 7).

Inputs:
- `wd_dataset.parquet`: The main dataset generated in Step 1 (used to create train/validation splits).
- `df_merged.parquet`: The merged dataset that can be divided into train and test

Outputs:
- `figures/roc_labse_similarity_smaller_tuning.pdf`: ROC curve for the small subset experiment.
- `figures/roc_labse_similarity_big_tuning.pdf`: ROC curve for the full dataset experiment.
- `output/labse_big_finetune_results.csv`: Summary table for the big finetune experiment.
- Printed metrics logs (Precision/Recall/F2 per threshold).

## Notes on reproducibility
Runtime and memory: word-vocabulary distance matrices can be expensive; consider running on a machine with enough RAM/CPU.
Neural models (LaBSE): Training requires a GPU environment. The experiments reported in the paper were conducted on dual NVIDIA RTX A5500 GPUs (approx. 45 min for the full dataset). 
