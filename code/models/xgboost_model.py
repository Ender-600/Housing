"""
XGBoost Model for Price Prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path


class HousePriceXGBoost:
    """XGBoost model for house price prediction"""
    
    def __init__(self, data_path='../../data/listings_enriched.csv'):
        """
        Initialize the model
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.X_train = None
        self.X_test = None
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
                X[col].fillna(median_val, inplace=True)
                # Handle invalid values (e.g., -666666666)
                X.loc[X[col] < -1000000, col] = median_val
        
        # Handle missing values and encode categorical features
        for col in categorical_features:
            if col in X.columns:
                X[col].fillna('Unknown', inplace=True)
                # Label encoding
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"\nTraining set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
    def train_model(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=1):
        """
        Train the XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of each tree
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            min_child_weight: Minimum sum of instance weight needed in a child
        """
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        print("Model training completed!")
        
    def hyperparameter_tuning(self, cv=5, n_iter=20):
        """
        Hyperparameter tuning using RandomizedSearchCV
        
        Args:
            cv: Number of cross-validation folds
            n_iter: Number of random search iterations
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        print("\nStarting hyperparameter tuning (Randomized Search)...")
        
        param_distributions = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7]
        }
        
        random_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best score (neg MSE): {random_search.best_score_:.2f}")
        
        self.model = random_search.best_estimator_
        
        return random_search.best_params_
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")
        
        # Training set predictions
        y_train_pred = self.model.predict(self.X_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        print(f"\nTraining Set Performance:")
        print(f"  RMSE: ${train_rmse:,.2f}")
        print(f"  MAE:  ${train_mae:,.2f}")
        print(f"  R²:   {train_r2:.4f}")
        
        # Test set predictions
        y_test_pred = self.model.predict(self.X_test)
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
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
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
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to display
        """
        print("\nAnalyzing feature importance...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - XGBoost')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent / 'figs'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {output_dir / 'xgboost_feature_importance.png'}")
        plt.close()
        
        return feature_importance_df
    
    def plot_predictions(self):
        """Plot predicted vs actual prices"""
        y_pred = self.model.predict(self.X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('XGBoost: Predicted vs Actual Prices')
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent / 'figs'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'xgboost_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to: {output_dir / 'xgboost_predictions.png'}")
        plt.close()
    
    def plot_residuals(self):
        """Plot residuals analysis"""
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        
        # Residuals histogram
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent / 'figs'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'xgboost_residuals.png', dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to: {output_dir / 'xgboost_residuals.png'}")
        plt.close()
    
    def plot_learning_curve(self):
        """Plot learning curve from training history"""
        try:
            results = self.model.evals_result()
            
            if results and 'validation_0' in results:
                plt.figure(figsize=(10, 6))
                epochs = len(results['validation_0']['rmse'])
                x_axis = range(0, epochs)
                
                plt.plot(x_axis, results['validation_0']['rmse'], label='Test')
                plt.xlabel('Number of Trees')
                plt.ylabel('RMSE')
                plt.title('XGBoost Learning Curve')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                output_dir = Path(__file__).parent / 'figs'
                output_dir.mkdir(exist_ok=True)
                plt.savefig(output_dir / 'xgboost_learning_curve.png', dpi=300, bbox_inches='tight')
                print(f"Learning curve plot saved to: {output_dir / 'xgboost_learning_curve.png'}")
                plt.close()
            else:
                print("Note: Learning curve not available (model was not trained with eval_set)")
        except Exception as e:
            print(f"Note: Could not plot learning curve - {str(e)}")
    
    def save_model(self, filename='xgboost_model.pkl'):
        """
        Save the model
        
        Args:
            filename: Model filename
        """
        output_dir = Path(__file__).parent
        model_path = output_dir / filename
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to: {model_path}")
    
    def load_model(self, filename='xgboost_model.pkl'):
        """
        Load a saved model
        
        Args:
            filename: Model filename
        """
        model_path = Path(__file__).parent / filename
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, X):
        """
        Predict prices for new data
        
        Args:
            X: Feature data
            
        Returns:
            Predicted prices
        """
        return self.model.predict(X)


def main():
    """Main function - Run complete training pipeline"""
    
    # Create model instance
    xgb_model = HousePriceXGBoost()
    
    # Load and preprocess data
    X, y = xgb_model.load_and_preprocess_data()
    
    # Split data
    xgb_model.split_data(X, y)
    
    # Train base model
    print("\n=== Training Base XGBoost Model ===")
    xgb_model.train_model(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
    
    # Evaluate base model
    metrics = xgb_model.evaluate_model()
    
    # Hyperparameter tuning (optional, time-consuming)
    print("\n=== Perform Hyperparameter Tuning? ===")
    tune = input("Enter 'y' to tune, any other key to skip: ").lower()
    if tune == 'y':
        best_params = xgb_model.hyperparameter_tuning(cv=5, n_iter=20)
        print("\nRe-evaluating model with best parameters:")
        metrics = xgb_model.evaluate_model()
    
    # Feature importance analysis
    feature_importance = xgb_model.plot_feature_importance(top_n=20)
    
    # Plot predictions
    xgb_model.plot_predictions()
    
    # Plot residuals
    xgb_model.plot_residuals()
    
    # Plot learning curve
    xgb_model.plot_learning_curve()
    
    # Save model
    xgb_model.save_model()
    
    print("\n=== Training Complete! ===")
    print(f"Final Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Final Test RMSE: ${metrics['test_rmse']:,.2f}")
    print(f"Final Test MAPE: {metrics['test_mape']:.2f}%")


if __name__ == '__main__':
    main()

