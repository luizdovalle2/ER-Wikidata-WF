# ER-Wikidata-WF
A repo for the paper: A Wikidata-based Workflow for Entity Reconciliation Strategies Evaluation: A Study on Early Modern Polish Personal Names

# Reproducing the paper results

This repo contains three notebooks + a helper module to reproduce the metric evaluation workflow from the paper.

## Files
- `1.-dataset_acquisition.ipynb`: queries Wikidata and builds the validation dataset (labels + aliases + token lists). 
- `2.-distance_calculation.ipynb`: for each configuration (DM/BM/JW/LEV/NLEV), builds token-match sets and computes Dice–Sørensen coverage; writes one parquet per config.
- `3.-metrics_comparison.ipynb`: loads all saved config outputs and selects the best coverage threshold by F2/recall (as in the paper).
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
- 
### 3) Evaluate metrics and pick best thresholds
Open and run:
- `3.-metrics_comparison.ipynb`

Inputs:
- `wd_dataset.parquet` (gold pairs) 
- `coverage_outputs/*.parquet` (predicted pairs with `match_count` / coverage) 

Output:
- A summary table (per config) with best F2, recall, and the selected threshold. 


## Notes on reproducibility
Runtime and memory: word-vocabulary distance matrices can be expensive; consider running on a machine with enough RAM/CPU.

