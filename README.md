# Housing Price Analysis & Prediction

This project aims to enrich housing listing data with external API information (Census, Transit, Police, Google Places/Routes) and build various machine learning models to predict housing prices.

## Project Structure

```
Housing/
├── code/
│   ├── enrich_listings.py       # Data enrichment script
│   ├── services/                # API service modules
│   ├── utils/                   # Utility functions
│   └── models/                  # Machine learning models
│       ├── compare_all_models.py
│       ├── decision_tree_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       └── linear_models.py
├── data/                        # Data directory (input/output CSVs)
├── env.example                  # Environment variables template
└── requirements.txt             # Python dependencies
```

## Prerequisites

- Python 3.8+
- [Google Cloud Platform](https://console.cloud.google.com/) account (for Places & Routes APIs)
- [CUMTD Developer](https://developer.mtd.org/) account (for Bus data)
- Census API Key (optional but recommended)

## Setup

1.  **Clone the repository** (if applicable) and navigate to the project root:
    ```bash
    cd Housing
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables**:
    Copy `env.example` to `.env` and fill in your API keys.
    ```bash
    cp env.example .env
    ```
    *Edit `.env` with your actual API keys.*

## 1. Data Enrichment

The `enrich_listings.py` script adds external data to your housing listings CSV.

**Run from the project root:**

```bash
python code/enrich_listings.py
```

**Options:**
- `--input`: Path to input CSV (default: `data/listings_details_allcities.csv`)
- `--output`: Path to output CSV (default: `data/listings_enriched.csv`)
- `--skip-places`, `--skip-routes`, etc.: Skip specific API calls to save costs/time.
- `--batch-size`: Number of parallel requests (default: 5).

Example:
```bash
python code/enrich_listings.py --skip-places --skip-routes
```

## 2. Running Models

The models are located in `code/models/`. They use the enriched data (`data/listings_enriched.csv`).

**Important:** Run model scripts from the `code/models/` directory so they can correctly locate the data file relative to the script.

1.  **Navigate to the models directory**:
    ```bash
    cd code/models
    ```

2.  **Run Comprehensive Comparison**:
    To train and compare all models (Decision Tree, Random Forest, XGBoost, OLS, Lasso, Ridge):
    ```bash
    python compare_all_models.py
    ```
    This will:
    - Train all models.
    - Generate performance metrics and plots in `figs/`.
    - Save a summary CSV: `comprehensive_model_comparison.csv`.

3.  **Run Individual Models**:
    You can also run specific models independently to tune them or view detailed analysis:

    ```bash
    python decision_tree_model.py
    python random_forest_model.py
    python xgboost_model.py
    python linear_models.py
    ```

    Each script typically performs:
    - Data loading & preprocessing.
    - Model training.
    - Evaluation (RMSE, MAE, R²).
    - Optional hyperparameter tuning.
    - Feature importance analysis.
    - Plot generation (saved to `figs/`).

## Output

- **Enriched Data**: `data/listings_enriched.csv`
- **Model Figures**: `code/models/figs/` (Performance plots, Feature importance, etc.)
- **Model Summaries**: `code/models/*.csv`
- **Detailed Report**: `code/models/MODELS_SUMMARY.md` (Comprehensive analysis of model performance)

## Troubleshooting

### CSV Formatting Issues
If you encounter errors loading the input CSV (e.g., `ParserError` or bad lines), you can try running the fix script:
```bash
python fix_csv.py
```
This script attempts to repair common CSV formatting issues in `data/listings_details_allcities.csv`.

### Import Errors
If running models results in `ModuleNotFoundError`, ensure you are running the scripts from the `code/models/` directory as specified, or add the project root to your `PYTHONPATH`.

