# Melanoma Risk Analysis

IQR-based risk scoring pipeline for skin lesion records from the ISIC 2024 tabular dataset. Scores 400k+ benign lesions by how closely their features match the core profile of 393 confirmed malignant melanoma cases.

## Project structure

```
170melanoma/
├── data/
│   ├── raw/              # Source CSVs — place train-metadata.csv and train-labels.csv here
│   └── processed/        # Generated: datasetv1.csv, datasetv2.csv, datasetv3.csv
├── notebooks/
│   └── melanoma_analysis.ipynb
├── outputs/
│   ├── graphs/           # All generated plots (histograms, box plots, heatmap, etc.)
│   └── results/          # feature_ranges.csv, mann_whitney_results.csv, model_metrics.csv
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the raw data

Download the ISIC 2024 tabular dataset and place the two CSV files in `data/raw/`:

```
data/raw/train-metadata.csv
data/raw/train-labels.csv
```

The dataset is available on Kaggle: `isic-2024-challenge` (tabular metadata only, no images needed).

### 4. Launch Jupyter and run the notebook

```bash
jupyter notebook notebooks/melanoma_analysis.ipynb
```

Run all cells top-to-bottom. Generated files are written to:
- `data/processed/` — processed datasets
- `outputs/graphs/` — all plots
- `outputs/results/` — CSV result tables

## Pipeline overview

| Step | What it does |
|------|-------------|
| 1–2  | Load and merge feature + label CSVs on `isic_id` |
| 3–6  | Drop ID columns, impute missing values, clip out-of-range values, engineer 6 new features |
| 7    | Split into malignant (393 records) and benign (400k records) subsets |
| 8–11 | Compute malignant IQR zones per feature; score every benign record 0–16 |
| 12–13 | Rank benign records by risk score; bin features into Low/Medium/High |
| 14   | Generate histograms, box plots, bar charts, pie charts, correlation heatmap |
| 15   | Mann-Whitney U test — identifies which features differ significantly between groups |
| 16   | Gaussian Naive Bayes malignancy probability as a second risk signal |
| 17   | K-Means clustering + PCA visualization |
| 18–20 | LightGBM with StratifiedGroupKFold (patient-level splits) + PR-AUC evaluation |
| 21   | Save final scored dataset (`datasetv3.csv`) |

## Outputs

- **`data/processed/datasetv1.csv`** — 393 malignant records (reference profile)
- **`data/processed/datasetv2.csv`** — 400k benign records with IQR risk scores
- **`data/processed/datasetv3.csv`** — full dataset with IQR score + Naive Bayes probability
- **`outputs/results/feature_ranges.csv`** — Low/Medium/High bin breakdown per feature
- **`outputs/results/mann_whitney_results.csv`** — statistical significance per feature
- **`outputs/results/model_metrics.csv`** — LightGBM PR-AUC, F1, precision, recall
