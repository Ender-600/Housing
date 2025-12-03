"""
Linear Regression Models for Price Prediction
Includes: OLS, Lasso, and Ridge Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path


class HousePriceLinearModels:
    """Linear regression models (OLS, Lasso, Ridge) for house price prediction"""
    
    def __init__(self, data_path='../../data/listings_enriched.csv'):
        """
        Initialize the models
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.ols_model = None
        self.lasso_model = None
        self.ridge_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        df = pd.read_csv(self.data_path, skipinitialspace=True)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        print(f"Original data shape: {df.shape}")
        
        # Clean soldPrice column - handle formats like $272,000 and $1.08M
        def parse_price(price_str):
            """Convert price string to float, handling M/K suffixes"""
            if pd.isna(price_str):
                return np.nan
            # Strip whitespace and remove $
            price_str = str(price_str).strip().replace('$', '').replace(',', '')
            # Handle M (millions) and K (thousands)
            if price_str.endswith('M'):
                return float(price_str[:-1]) * 1_000_000
            elif price_str.endswith('K'):
                return float(price_str[:-1]) * 1_000
            else:
                return float(price_str)
        
        df['soldPrice'] = df['soldPrice'].apply(parse_price)
        
        # Remove price outliers (using IQR method)
        Q1 = df['soldPrice'].quantile(0.25)
        Q3 = df['soldPrice'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['soldPrice'] >= Q1 - 1.5 * IQR) & (df['soldPrice'] <= Q3 + 1.5 * IQR)]
        
        print(f"Data shape after removing outliers: {df.shape}")
        print(f"Price range: ${df['soldPrice'].min():,.2f} - ${df['soldPrice'].max():,.2f}")
        
        # Select features (excluding zestimate and rentZestimate to avoid data leakage)
        numeric_features = [
            'area', 'baths', 'beds', 'latitude', 'longitude',
            'daysOnZillow', 'median_income',
            'bus_stops_1km', 'restaurants_nearby', 'cafes_nearby', 
            'schools_nearby', 'parks_nearby', 'gyms_nearby', 'supermarkets_nearby',
            'drive_to_uiuc_main_quad_min', 'drive_to_downtown_champaign_min',
            'drive_to_carle_hospital_min', 'drive_to_memorial_stadium_min',
            'drive_to_willard_airport_min'
        ]
        
        categorical_features = ['addressCity', 'homeType']
        
        # Create feature dataframe
        X = df[numeric_features + categorical_features].copy()
        y = df['soldPrice'].copy()
        
        # Handle missing values for numeric features (fill with median)
        for col in numeric_features:
            if col in X.columns:
                median_val = X[col].median()
                X.loc[:, col] = X[col].fillna(median_val)
                # Handle invalid values (e.g., -666666666)
                X.loc[X[col] < -1000000, col] = median_val
        
        # Handle missing values and encode categorical features
        for col in categorical_features:
            if col in X.columns:
                X.loc[:, col] = X[col].fillna('Unknown')
                # Label encoding
                le = LabelEncoder()
                X.loc[:, col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features for Lasso and Ridge (OLS doesn't require scaling but we'll use it for consistency)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
    def train_ols(self):
        """Train OLS (Ordinary Least Squares) model"""
        print("\nTraining OLS (Linear Regression) model...")
        self.ols_model = LinearRegression()
        self.ols_model.fit(self.X_train_scaled, self.y_train)
        print("OLS model training completed!")
        
    def train_lasso(self, alpha=1.0):
        """
        Train Lasso regression model
        
        Args:
            alpha: Regularization strength
        """
        print(f"\nTraining Lasso model (alpha={alpha})...")
        self.lasso_model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        self.lasso_model.fit(self.X_train_scaled, self.y_train)
        print("Lasso model training completed!")
        
    def train_ridge(self, alpha=1.0):
        """
        Train Ridge regression model
        
        Args:
            alpha: Regularization strength
        """
        print(f"\nTraining Ridge model (alpha={alpha})...")
        self.ridge_model = Ridge(alpha=alpha, random_state=42)
        self.ridge_model.fit(self.X_train_scaled, self.y_train)
        print("Ridge model training completed!")
        
    def tune_lasso(self, cv=5):
        """Hyperparameter tuning for Lasso"""
        print("\nTuning Lasso hyperparameters...")
        
        alphas = np.logspace(-2, 4, 50)
        param_grid = {'alpha': alphas}
        
        grid_search = GridSearchCV(
            Lasso(max_iter=10000, random_state=42),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best alpha for Lasso: {grid_search.best_params_['alpha']:.4f}")
        print(f"Best score (neg MSE): {grid_search.best_score_:.2f}")
        
        self.lasso_model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def tune_ridge(self, cv=5):
        """Hyperparameter tuning for Ridge"""
        print("\nTuning Ridge hyperparameters...")
        
        alphas = np.logspace(-2, 4, 50)
        param_grid = {'alpha': alphas}
        
        grid_search = GridSearchCV(
            Ridge(random_state=42),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best alpha for Ridge: {grid_search.best_params_['alpha']:.4f}")
        print(f"Best score (neg MSE): {grid_search.best_score_:.2f}")
        
        self.ridge_model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def evaluate_model(self, model, model_name):
        """Evaluate a specific model's performance"""
        print(f"\n=== {model_name} Model Evaluation ===")
        
        # Training set predictions
        y_train_pred = model.predict(self.X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        print(f"\nTraining Set Performance:")
        print(f"  RMSE: ${train_rmse:,.2f}")
        print(f"  MAE:  ${train_mae:,.2f}")
        print(f"  R²:   {train_r2:.4f}")
        
        # Test set predictions
        y_test_pred = model.predict(self.X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  MAE:  ${test_mae:,.2f}")
        print(f"  R²:   {test_r2:.4f}")
        
        # Calculate MAPE
        mape = np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
        print(f"  MAPE: {mape:.2f}%")
        
        # Cross-validation
        # Create a copy of the model for CV to avoid issues
        from sklearn.base import clone
        model_cv = clone(model)
        
        cv_scores = cross_val_score(
            model_cv, 
            self.X_train_scaled, 
            self.y_train,
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(f"\n5-Fold Cross-Validation RMSE:")
        print(f"  Mean: ${cv_rmse.mean():,.2f}")
        print(f"  Std:  ${cv_rmse.std():,.2f}")
        
        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_mape': mape,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
    
    def evaluate_all_models(self):
        """Evaluate all three models"""
        ols_metrics = self.evaluate_model(self.ols_model, "OLS")
        lasso_metrics = self.evaluate_model(self.lasso_model, "Lasso")
        ridge_metrics = self.evaluate_model(self.ridge_model, "Ridge")
        
        return ols_metrics, lasso_metrics, ridge_metrics
    
    def plot_coefficients_comparison(self, top_n=20):
        """Compare coefficients across the three models"""
        print("\nAnalyzing model coefficients...")
        
        # Get coefficients
        ols_coef = self.ols_model.coef_
        lasso_coef = self.lasso_model.coef_
        ridge_coef = self.ridge_model.coef_
        
        # Create dataframe
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'OLS': ols_coef,
            'Lasso': lasso_coef,
            'Ridge': ridge_coef
        })
        
        # Sort by absolute OLS coefficient
        coef_df['abs_ols'] = np.abs(coef_df['OLS'])
        coef_df = coef_df.sort_values('abs_ols', ascending=False)
        
        print(f"\nTop {top_n} Features by Coefficient Magnitude:")
        print(coef_df.drop('abs_ols', axis=1).head(top_n).to_string(index=False))
        
        # Count non-zero coefficients in Lasso
        non_zero_lasso = np.sum(np.abs(lasso_coef) > 1e-5)
        print(f"\nLasso selected {non_zero_lasso} out of {len(self.feature_names)} features")
        
        # Plot
        top_features = coef_df.head(top_n)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # OLS coefficients
        axes[0].barh(top_features['feature'], top_features['OLS'], alpha=0.8, color='skyblue')
        axes[0].set_xlabel('Coefficient Value')
        axes[0].set_title(f'OLS - Top {top_n} Features')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        # Lasso coefficients
        axes[1].barh(top_features['feature'], top_features['Lasso'], alpha=0.8, color='lightcoral')
        axes[1].set_xlabel('Coefficient Value')
        axes[1].set_title(f'Lasso - Top {top_n} Features')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        # Ridge coefficients
        axes[2].barh(top_features['feature'], top_features['Ridge'], alpha=0.8, color='lightgreen')
        axes[2].set_xlabel('Coefficient Value')
        axes[2].set_title(f'Ridge - Top {top_n} Features')
        axes[2].grid(axis='x', alpha=0.3)
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent / 'figs'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'linear_models_coefficients.png', dpi=300, bbox_inches='tight')
        print(f"\nCoefficients plot saved to: {output_dir / 'linear_models_coefficients.png'}")
        plt.close()
        
        return coef_df
    
    def plot_predictions(self, model, model_name):
        """Plot predicted vs actual prices for a specific model"""
        y_pred = model.predict(self.X_test_scaled)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{model_name}: Predicted vs Actual Prices')
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent / 'figs'
        output_dir.mkdir(exist_ok=True)
        filename = f'{model_name.lower().replace(" ", "_")}_predictions.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to: {output_dir / filename}")
        plt.close()
    
    def plot_all_predictions(self):
        """Plot predictions for all three models"""
        self.plot_predictions(self.ols_model, "OLS")
        self.plot_predictions(self.lasso_model, "Lasso")
        self.plot_predictions(self.ridge_model, "Ridge")
    
    def plot_residuals(self, model, model_name):
        """Plot residuals analysis for a specific model"""
        y_pred = model.predict(self.X_test_scaled)
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residual Plot')
        
        # Residuals histogram
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Residual Distribution')
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent / 'figs'
        output_dir.mkdir(exist_ok=True)
        filename = f'{model_name.lower().replace(" ", "_")}_residuals.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to: {output_dir / filename}")
        plt.close()
    
    def plot_all_residuals(self):
        """Plot residuals for all three models"""
        self.plot_residuals(self.ols_model, "OLS")
        self.plot_residuals(self.lasso_model, "Lasso")
        self.plot_residuals(self.ridge_model, "Ridge")
    
    def save_models(self):
        """Save all models"""
        output_dir = Path(__file__).parent
        
        models_data = {
            'ols_model': self.ols_model,
            'lasso_model': self.lasso_model,
            'ridge_model': self.ridge_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        model_path = output_dir / 'linear_models.pkl'
        joblib.dump(models_data, model_path)
        print(f"\nAll models saved to: {model_path}")
    
    def load_models(self, filename='linear_models.pkl'):
        """Load saved models"""
        model_path = Path(__file__).parent / filename
        models_data = joblib.load(model_path)
        
        self.ols_model = models_data['ols_model']
        self.lasso_model = models_data['lasso_model']
        self.ridge_model = models_data['ridge_model']
        self.scaler = models_data['scaler']
        self.label_encoders = models_data['label_encoders']
        self.feature_names = models_data['feature_names']
        
        print(f"Models loaded from {model_path}")


def main():
    """Main function - Run complete training pipeline"""
    
    # Create model instance
    linear_models = HousePriceLinearModels()
    
    # Load and preprocess data
    X, y = linear_models.load_and_preprocess_data()
    
    # Split data
    linear_models.split_data(X, y)
    
    # Train all models
    print("\n" + "=" * 80)
    print("Training Linear Regression Models")
    print("=" * 80)
    
    linear_models.train_ols()
    linear_models.train_lasso(alpha=100.0)
    linear_models.train_ridge(alpha=100.0)
    
    # Ask for hyperparameter tuning
    print("\n=== Perform Hyperparameter Tuning for Lasso and Ridge? ===")
    tune = input("Enter 'y' to tune, any other key to skip: ").lower()
    if tune == 'y':
        linear_models.tune_lasso(cv=5)
        linear_models.tune_ridge(cv=5)
    
    # Evaluate all models
    ols_metrics, lasso_metrics, ridge_metrics = linear_models.evaluate_all_models()
    
    # Compare models
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Test RMSE', 'Test MAE', 'Test R²', 'Test MAPE', 'CV RMSE (mean)'],
        'OLS': [
            f"${ols_metrics['test_rmse']:,.2f}",
            f"${ols_metrics['test_mae']:,.2f}",
            f"{ols_metrics['test_r2']:.4f}",
            f"{ols_metrics['test_mape']:.2f}%",
            f"${ols_metrics['cv_rmse_mean']:,.2f}"
        ],
        'Lasso': [
            f"${lasso_metrics['test_rmse']:,.2f}",
            f"${lasso_metrics['test_mae']:,.2f}",
            f"{lasso_metrics['test_r2']:.4f}",
            f"{lasso_metrics['test_mape']:.2f}%",
            f"${lasso_metrics['cv_rmse_mean']:,.2f}"
        ],
        'Ridge': [
            f"${ridge_metrics['test_rmse']:,.2f}",
            f"${ridge_metrics['test_mae']:,.2f}",
            f"{ridge_metrics['test_r2']:.4f}",
            f"{ridge_metrics['test_mape']:.2f}%",
            f"${ridge_metrics['cv_rmse_mean']:,.2f}"
        ]
    })
    
    print("\n")
    print(comparison_df.to_string(index=False))
    
    # Coefficient analysis
    coef_df = linear_models.plot_coefficients_comparison(top_n=20)
    
    # Plot predictions
    linear_models.plot_all_predictions()
    
    # Plot residuals
    linear_models.plot_all_residuals()
    
    # Save models
    linear_models.save_models()
    
    print("\n=== Training Complete! ===")
    print(f"OLS Test R²: {ols_metrics['test_r2']:.4f}, RMSE: ${ols_metrics['test_rmse']:,.2f}")
    print(f"Lasso Test R²: {lasso_metrics['test_r2']:.4f}, RMSE: ${lasso_metrics['test_rmse']:,.2f}")
    print(f"Ridge Test R²: {ridge_metrics['test_r2']:.4f}, RMSE: ${ridge_metrics['test_rmse']:,.2f}")


if __name__ == '__main__':
    main()

