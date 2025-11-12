# House Price Prediction Models

This directory contains two machine learning models for house price prediction:

1. **Decision Tree** - `decision_tree_model.py`
2. **Random Forest** - `random_forest_model.py`

## 📋 Directory Structure

```
models/
├── decision_tree_model.py          # Decision Tree model
├── random_forest_model.py          # Random Forest model
├── compare_models.py               # Model comparison script
├── README.md                       # This document
├── Rmodel for large dataset.R      # R model
└── test.py                         # Test script
```

## 🚀 Quick Start

### 1. Train Decision Tree Model

```bash
cd code/models
python decision_tree_model.py
```

**Output files:**

- `decision_tree_model.pkl` - Trained model
- `decision_tree_feature_importance.png` - Feature importance plot
- `decision_tree_predictions.png` - Prediction comparison plot

### 2. Train Random Forest Model

```bash
cd code/models
python random_forest_model.py
```

**Output files:**

- `random_forest_model.pkl` - Trained model
- `random_forest_feature_importance.png` - Feature importance plot
- `random_forest_predictions.png` - Prediction comparison plot
- `random_forest_residuals.png` - Residual analysis plot

### 3. Compare Both Models

```bash
cd code/models
python compare_models.py
```

**Output files:**

- `model_comparison.csv` - Performance comparison table
- `models_performance_comparison.png` - Performance comparison plot
- `feature_importance_comparison.png` - Feature importance comparison plot

## 📊 Dataset

The models use the `data/listings_enriched.csv` dataset with the following features:

### Numeric Features

- **House basic info**: `area`, `baths`, `beds`, `livingArea`
- **Price related**: `zestimate`, `rentZestimate`, `soldPrice`
- **Location**: `latitude`, `longitude`
- **Market info**: `daysOnZillow`
- **Demographics**: `median_income`
- **Nearby amenities**:
  - `bus_stops_1km` - Number of bus stops within 1km
  - `restaurants_nearby` - Number of nearby restaurants
  - `cafes_nearby` - Number of nearby cafes
  - `schools_nearby` - Number of nearby schools
  - `parks_nearby` - Number of nearby parks
  - `gyms_nearby` - Number of nearby gyms
  - `supermarkets_nearby` - Number of nearby supermarkets
- **Transportation convenience**:
  - `drive_to_uiuc_main_quad_min` - Drive time to UIUC main quad
  - `drive_to_downtown_champaign_min` - Drive time to downtown Champaign
  - `drive_to_carle_hospital_min` - Drive time to Carle Hospital
  - `drive_to_memorial_stadium_min` - Drive time to Memorial Stadium
  - `drive_to_willard_airport_min` - Drive time to Willard Airport

### Categorical Features

- `addressCity` - City (Champaign, Urbana, Savoy, etc.)
- `homeType` - House type (SINGLE_FAMILY, TOWNHOUSE, CONDO, MULTI_FAMILY)

## 🔧 Model Features

### Decision Tree Model Class (`HousePriceDecisionTree`)

```python
from decision_tree_model import HousePriceDecisionTree

# Create model instance
model = HousePriceDecisionTree()

# Load and preprocess data
X, y = model.load_and_preprocess_data()

# Split dataset
model.split_data(X, y, test_size=0.2)

# Train model
model.train_model(max_depth=15, min_samples_split=10, min_samples_leaf=5)

# Evaluate model
metrics = model.evaluate_model()

# Hyperparameter tuning (optional)
best_params = model.hyperparameter_tuning(cv=5)

# Feature importance analysis
feature_importance = model.plot_feature_importance(top_n=20)

# Save model
model.save_model('decision_tree_model.pkl')

# Load model
model.load_model('decision_tree_model.pkl')

# Predict new data
predictions = model.predict(X_new)
```

### Random Forest Model Class (`HousePriceRandomForest`)

```python
from random_forest_model import HousePriceRandomForest

# Create model instance
model = HousePriceRandomForest()

# Load and preprocess data
X, y = model.load_and_preprocess_data()

# Split dataset
model.split_data(X, y, test_size=0.2)

# Train model
model.train_model(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples required to split
    min_samples_leaf=2,    # Minimum samples at leaf node
    max_features='sqrt'    # Number of features to consider per split
)

# Evaluate model
metrics = model.evaluate_model()

# Hyperparameter tuning (optional, uses random search)
best_params = model.hyperparameter_tuning(cv=5, n_iter=20)

# Feature importance analysis
feature_importance = model.plot_feature_importance(top_n=20)

# Residual analysis
model.plot_residuals()

# Save model
model.save_model('random_forest_model.pkl')

# Load model
model.load_model('random_forest_model.pkl')

# Predict new data
predictions = model.predict(X_new)
```

## 📈 Model Evaluation Metrics

The models use the following metrics for evaluation:

1. **RMSE (Root Mean Squared Error)**

   - Measures deviation between predicted and actual values
   - Same unit as target variable (dollars)

2. **MAE (Mean Absolute Error)**

   - Average of prediction errors
   - Easier to interpret

3. **R² Score (Coefficient of Determination)**

   - Model goodness of fit
   - Range: 0-1, closer to 1 is better
   - Represents how much variance the model explains

4. **MAPE (Mean Absolute Percentage Error)**

   - Only provided for Random Forest model
   - Represents prediction error as a percentage

5. **Cross-Validation RMSE** - 5-fold cross-validation
   - More reliable performance estimate
   - Reduces overfitting risk

## 🎯 Model Comparison

| Feature                 | Decision Tree | Random Forest |
| ----------------------- | ------------- | ------------- |
| **Training Speed**      | Fast ⚡       | Slower 🐢     |
| **Prediction Accuracy** | Moderate 📊   | High 🎯       |
| **Overfitting Risk**    | High ⚠️       | Low ✅        |
| **Interpretability**    | High 📖       | Moderate 📚   |
| **Stability**           | Low           | High          |
| **Parameter Tuning**    | Simple        | Complex       |
| **Memory Usage**        | Small 💾      | Large 💾💾💾  |

### When to Use Decision Tree?

- ✅ Need fast training and prediction
- ✅ Small dataset
- ✅ Need highly interpretable model
- ✅ Rapid prototyping

### When to Use Random Forest?

- ✅ Seeking highest prediction accuracy
- ✅ Large dataset
- ✅ Can accept longer training time
- ✅ Production environment deployment
- ✅ Need stable and reliable predictions

## 🔬 Data Preprocessing

Both models include the following preprocessing steps:

1. **Price cleaning**: Remove `$` symbol and commas
2. **Outlier handling**: Remove price outliers using IQR method
3. **Missing value imputation**:
   - Numeric features: Fill with median
   - Categorical features: Fill with "Unknown"
4. **Invalid value handling**: Replace invalid values (e.g., -666666666) with median
5. **Label encoding**: Convert categorical features to numeric

## 📝 Hyperparameter Descriptions

### Decision Tree Key Parameters

- `max_depth`: Maximum tree depth

  - Default: 15
  - Smaller values prevent overfitting, larger values increase model complexity

- `min_samples_split`: Minimum samples required to split a node

  - Default: 10
  - Larger values prevent overfitting

- `min_samples_leaf`: Minimum samples required at leaf node
  - Default: 5
  - Larger values produce smoother models

### Random Forest Key Parameters

- `n_estimators`: Number of trees

  - Default: 100
  - More trees usually perform better but train slower

- `max_depth`: Maximum depth of each tree

  - Default: 20
  - Controls complexity of individual trees

- `min_samples_split`: Minimum samples required to split a node

  - Default: 5

- `min_samples_leaf`: Minimum samples required at leaf node

  - Default: 2

- `max_features`: Number of features to consider per split
  - Default: 'sqrt'
  - Options: 'sqrt', 'log2', or float

## 💡 Usage Recommendations

1. **First use**: Run `compare_models.py` first to understand performance differences

2. **Production environment**: Recommend Random Forest for higher accuracy and stability

3. **Quick experiments**: Use Decision Tree for fast testing and feature exploration

4. **Hyperparameter tuning**:

   - Decision Tree: GridSearchCV (full grid search)
   - Random Forest: RandomizedSearchCV (random search, faster)

5. **Feature engineering**: Both models output feature importance for feature selection

## 🐛 Common Issues

### Q: Training takes too long?

A:

- Reduce Random Forest `n_estimators` parameter
- Use smaller data subset for initial testing
- Skip hyperparameter tuning step

### Q: Model is overfitting?

A:

- Increase `min_samples_split` and `min_samples_leaf`
- Decrease `max_depth`
- Random Forest: Reduce `n_estimators` or increase `max_features`

### Q: How to improve model accuracy?

A:

- Perform hyperparameter tuning
- Add more features
- Improve data quality
- Use larger dataset
- Try ensemble methods

### Q: How to use model on new data?

A:

```python
# Load saved model
model = HousePriceRandomForest()
model.load_model('random_forest_model.pkl')

# Prepare new data (ensure feature order and types match training)
import pandas as pd
new_data = pd.DataFrame({...})  # Include all required features

# Predict
predictions = model.predict(new_data)
```

## 📚 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

## 📧 Contact

For questions or suggestions, please contact the project maintainer.

## 📄 License

This project follows the MIT License.
