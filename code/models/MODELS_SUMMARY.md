# House Price Prediction Models - Comprehensive Summary

## 📊 All Implemented Models

This project implements **6 machine learning models** for house price prediction:

### 🌳 Tree-Based Models

1. **Decision Tree** - Single decision tree regressor
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting model

### 📈 Linear Models

4. **OLS (Ordinary Least Squares)** - Standard linear regression
5. **Lasso Regression** - L1 regularized linear regression
6. **Ridge Regression** - L2 regularized linear regression

---

## 🏆 Performance Comparison

### Overall Best Model: **XGBoost** 🥇

| Metric            | Decision Tree | Random Forest | **XGBoost**    | OLS      | Lasso    | Ridge    |
| ----------------- | ------------- | ------------- | -------------- | -------- | -------- | -------- |
| **Test R²**       | 0.7916        | 0.8622        | **0.8647** ⭐  | 0.8111   | 0.8112   | 0.8096   |
| **Test RMSE**     | $67,877       | $55,194       | **$54,692** ⭐ | $64,629  | $64,609  | $64,880  |
| **Test MAE**      | $49,516       | $38,273       | **$37,597** ⭐ | $47,467  | $47,400  | $47,621  |
| **CV RMSE**       | $70,837       | $54,283       | **$52,796** ⭐ | $63,430  | $63,401  | $63,627  |
| **Training Time** | 0.02s         | 0.12s         | 0.43s          | 0.01s ⚡ | 0.01s ⚡ | 0.01s ⚡ |

### Key Findings:

✅ **XGBoost** achieves the highest accuracy with:

- Test R² of **0.8647** (explains 86.47% of variance)
- Lowest RMSE of **$54,692**
- Best cross-validation performance with lowest standard deviation

✅ **Random Forest** is second-best with:

- Test R² of **0.8622**
- Fast training time (0.12s)
- Good balance between accuracy and speed

✅ **Linear Models** (OLS, Lasso, Ridge) show:

- Similar performance (~0.81 R²)
- Extremely fast training (<0.02s)
- Good interpretability through coefficients

✅ **Decision Tree** is the fastest but:

- Lowest accuracy (R² = 0.7916)
- Highest overfitting (train R² = 0.9253 vs test R² = 0.7916)

---

## 📁 Generated Files

### Model Files

- `decision_tree_model.py` - Decision Tree implementation
- `random_forest_model.py` - Random Forest implementation
- `xgboost_model.py` - XGBoost implementation
- `linear_models.py` - OLS, Lasso, and Ridge implementations

### Comparison Scripts

- `compare_models.py` - Compare tree-based models (DT, RF, XGB)
- `compare_all_models.py` - **Comprehensive comparison of all 6 models**

### Saved Models

- `decision_tree_model.pkl` - Trained Decision Tree
- `random_forest_model.pkl` - Trained Random Forest
- `xgboost_model.pkl` - Trained XGBoost
- `linear_models.pkl` - All three linear models

### Visualizations (in `figs/` folder)

All visualization files are organized in the `figs/` directory:

#### Comparison Charts

- `figs/comprehensive_models_comparison.png` - All 6 models comparison
- `figs/models_performance_comparison.png` - Tree-based models comparison

#### Feature Importance/Coefficients

- `figs/decision_tree_feature_importance.png`
- `figs/random_forest_feature_importance.png`
- `figs/xgboost_feature_importance.png`
- `figs/linear_models_coefficients.png`
- `figs/feature_importance_comparison.png`

#### Predictions

- `figs/*_predictions.png` - Predicted vs Actual plots for each model

#### Residuals

- `figs/*_residuals.png` - Residual analysis for each model

### Data Files

- `model_comparison.csv` - Tree-based models comparison
- `comprehensive_model_comparison.csv` - **All 6 models comparison**

---

## 🔬 Model Type Analysis

### Tree-Based Models Performance

- **Best Test R²**: 0.8647 (XGBoost)
- **Average Test R²**: 0.8395
- **Strengths**:
  - Capture non-linear relationships
  - Handle feature interactions automatically
  - No need for feature scaling
  - High predictive accuracy

### Linear Models Performance

- **Best Test R²**: 0.8112 (Lasso)
- **Average Test R²**: 0.8106
- **Strengths**:
  - Extremely fast training (<0.02s)
  - Highly interpretable coefficients
  - Good baseline performance
  - Lasso provides feature selection

---

## 🎯 Feature Importance Insights

### XGBoost Top Features:

1. **baths** (44.95%) - Number of bathrooms
2. **area** (9.82%) - Property area
3. **livingArea** (9.05%) - Living area
4. **median_income** (6.20%) - Neighborhood income
5. **beds** (3.60%) - Number of bedrooms

### Linear Models (Lasso Coefficients):

Top positive contributors:

- **area**: +$47,454 per unit
- **baths**: +$44,365 per unit
- **drive_to_carle_hospital_min**: +$35,217 per minute (closer = higher price)

Top negative contributors:

- **latitude**: -$16,690 (geographic factor)
- **daysOnZillow**: -$12,499 per day (longer listings = lower price)

---

## 🚀 How to Use

### Train Individual Models

```bash
# Train Decision Tree
python decision_tree_model.py

# Train Random Forest
python random_forest_model.py

# Train XGBoost
python xgboost_model.py

# Train all linear models (OLS, Lasso, Ridge)
python linear_models.py
```

### Run Comparisons

```bash
# Compare tree-based models only
python compare_models.py

# Compare ALL 6 models (RECOMMENDED)
python compare_all_models.py
```

### Load Saved Models

```python
import joblib

# Load XGBoost (best model)
model_data = joblib.load('xgboost_model.pkl')
model = model_data['model']

# Load linear models
models_data = joblib.load('linear_models.pkl')
ols_model = models_data['ols_model']
lasso_model = models_data['lasso_model']
ridge_model = models_data['ridge_model']
```

---

## 💡 Recommendations

### For Production Use:

**Use XGBoost** - Best overall performance with:

- Highest accuracy (R² = 0.8647)
- Lowest prediction error (RMSE = $54,692)
- Most stable cross-validation results

### For Quick Prototyping:

**Use Random Forest** - Great balance of:

- High accuracy (R² = 0.8622)
- Fast training (0.12s)
- Easy to understand

### For Interpretability:

**Use Lasso** - When you need:

- Clear feature relationships (coefficients)
- Feature selection capability
- Instant training time
- Good baseline performance (R² = 0.8112)

### For Learning/Experimentation:

**Use Decision Tree** - Best for:

- Understanding basic concepts
- Quick iterations
- Visualizing decision paths

---

## 📊 Model Complexity vs Performance

```
Performance (Test R²)
    │
0.87│            ● XGBoost
    │
0.86│         ● Random Forest
    │
0.82│                      ● Lasso
    │                      ● OLS
    │                      ● Ridge
0.79│  ● Decision Tree
    │
    └──────────────────────────────── Complexity
       Low          Medium        High
```

---

## 🔧 Hyperparameter Tuning

All models support hyperparameter tuning. Run with the tuning option:

```python
# Each model's main() function will prompt:
# "Perform Hyperparameter Tuning? Enter 'y' to tune..."
```

- **Decision Tree**: GridSearchCV on max_depth, min_samples_split, min_samples_leaf
- **Random Forest**: RandomizedSearchCV on n_estimators, max_depth, etc.
- **XGBoost**: RandomizedSearchCV on learning_rate, max_depth, subsample, etc.
- **Lasso/Ridge**: GridSearchCV on alpha (regularization strength)

---

## 📈 Cross-Validation Results

All models use **5-fold cross-validation** for robust performance estimation:

| Model         | CV RMSE (Mean) | CV RMSE (Std) | Stability  |
| ------------- | -------------- | ------------- | ---------- |
| XGBoost       | $52,796        | $2,538        | ⭐⭐⭐⭐⭐ |
| Random Forest | $54,283        | $3,380        | ⭐⭐⭐⭐   |
| Lasso         | $63,401        | $4,535        | ⭐⭐⭐     |
| OLS           | $63,430        | $4,445        | ⭐⭐⭐     |
| Ridge         | $63,627        | $4,929        | ⭐⭐⭐     |
| Decision Tree | $70,837        | $4,450        | ⭐⭐       |

**XGBoost** has the lowest standard deviation, indicating the most stable predictions across different data splits.

---

## 📝 Dataset Information

- **Total samples**: 1,879 (after outlier removal)
- **Training set**: 1,503 samples (80%)
- **Test set**: 376 samples (20%)
- **Features**: 22 features including:
  - Property features (area, beds, baths, livingArea)
  - Location features (latitude, longitude, addressCity)
  - Neighborhood features (median_income, amenities nearby)
  - Distance features (to UIUC, hospitals, downtown, etc.)

---

## 🎓 Technical Details

### Data Preprocessing

- Price outlier removal using IQR method
- Missing value imputation with median for numeric features
- Label encoding for categorical features (city, homeType)
- Standard scaling for linear models only

### Evaluation Metrics

- **RMSE** (Root Mean Square Error) - Penalizes large errors
- **MAE** (Mean Absolute Error) - Average prediction error
- **R²** (Coefficient of Determination) - Variance explained
- **MAPE** (Mean Absolute Percentage Error) - Relative error
- **Cross-Validation** - 5-fold CV for generalization

---

## 🏗️ Future Improvements

1. **Ensemble Methods**: Stack XGBoost + Random Forest
2. **Deep Learning**: Neural network models
3. **Feature Engineering**:
   - Polynomial features
   - Interaction terms
   - Time-based features (seasonality)
4. **Advanced Tuning**: Bayesian optimization
5. **Explainability**: SHAP values for feature importance

---

## 📞 Model Selection Guide

Choose your model based on your priorities:

| Priority             | Recommended Model | Why?                                   |
| -------------------- | ----------------- | -------------------------------------- |
| **Accuracy**         | XGBoost           | Highest R², lowest errors              |
| **Speed**            | Lasso/Ridge       | <0.02s training time                   |
| **Balance**          | Random Forest     | Good accuracy + reasonable speed       |
| **Interpretability** | Lasso             | Clear coefficients + feature selection |
| **Simplicity**       | OLS               | Standard linear regression             |
| **Learning**         | Decision Tree     | Easy to visualize and understand       |

---

## 📄 Citation

If you use these models in your research or projects, please reference this repository.

**Dataset**: Champaign-Urbana area housing data with enriched features
**Models**: Decision Tree, Random Forest, XGBoost, OLS, Lasso, Ridge
**Best Performance**: XGBoost with R² = 0.8647, RMSE = $54,692

---

_Last Updated: December 2024_
_All models trained and evaluated on the same train/test split for fair comparison_
