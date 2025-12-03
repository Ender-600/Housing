# Figures Directory

This directory contains all visualization outputs from the house price prediction models.

## 📊 Contents

### Model Comparison Charts

- `comprehensive_models_comparison.png` - Comprehensive comparison of all 6 models
- `models_performance_comparison.png` - Tree-based models (Decision Tree, Random Forest, XGBoost) comparison
- `feature_importance_comparison.png` - Feature importance comparison across tree-based models

### Individual Model Visualizations

#### Decision Tree

- `decision_tree_feature_importance.png`
- `decision_tree_predictions.png`

#### Random Forest

- `random_forest_feature_importance.png`
- `random_forest_predictions.png`
- `random_forest_residuals.png`

#### XGBoost

- `xgboost_feature_importance.png`
- `xgboost_predictions.png`
- `xgboost_residuals.png`

#### Linear Models (OLS, Lasso, Ridge)

- `linear_models_coefficients.png` - Coefficient comparison
- `ols_predictions.png`
- `ols_residuals.png`
- `lasso_predictions.png`
- `lasso_residuals.png`
- `ridge_predictions.png`
- `ridge_residuals.png`

## 🔄 Auto-Generation

These figures are automatically generated when you run:

- Individual model scripts (`*_model.py`)
- Comparison scripts (`compare_models.py`, `compare_all_models.py`)

## 📐 Figure Specifications

- **Format**: PNG
- **DPI**: 300 (high resolution)
- **Size**: Varies by visualization type
- **Total**: 18 figures

## 📝 Notes

All figures use consistent color schemes and styling for easy comparison across models.
