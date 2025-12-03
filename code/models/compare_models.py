"""
Model Comparison Script
Compare performance of Decision Tree, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

from decision_tree_model import HousePriceDecisionTree
from random_forest_model import HousePriceRandomForest
from xgboost_model import HousePriceXGBoost


def compare_models():
    """Compare Decision Tree, Random Forest, and XGBoost models"""
    
    print("=" * 80)
    print("House Price Prediction Model Comparison: Decision Tree vs Random Forest vs XGBoost")
    print("=" * 80)
    
    # Initialize all models
    dt_model = HousePriceDecisionTree()
    rf_model = HousePriceRandomForest()
    xgb_model = HousePriceXGBoost()
    
    # Load data (load once using decision tree model)
    print("\nLoading data...")
    X, y = dt_model.load_and_preprocess_data()
    
    # Use the same data split for all models
    dt_model.split_data(X, y, test_size=0.2, random_state=42)
    
    # Share data with Random Forest
    rf_model.X_train = dt_model.X_train
    rf_model.X_test = dt_model.X_test
    rf_model.y_train = dt_model.y_train
    rf_model.y_test = dt_model.y_test
    rf_model.feature_names = dt_model.feature_names
    rf_model.label_encoders = dt_model.label_encoders
    
    # Share data with XGBoost
    xgb_model.X_train = dt_model.X_train
    xgb_model.X_test = dt_model.X_test
    xgb_model.y_train = dt_model.y_train
    xgb_model.y_test = dt_model.y_test
    xgb_model.feature_names = dt_model.feature_names
    xgb_model.label_encoders = dt_model.label_encoders
    
    # Train Decision Tree model
    print("\n" + "=" * 80)
    print("Training Decision Tree Model...")
    print("=" * 80)
    dt_start_time = time.time()
    dt_model.train_model(max_depth=15, min_samples_split=10, min_samples_leaf=5)
    dt_train_time = time.time() - dt_start_time
    dt_metrics = dt_model.evaluate_model()
    
    # Train Random Forest model
    print("\n" + "=" * 80)
    print("Training Random Forest Model...")
    print("=" * 80)
    rf_start_time = time.time()
    rf_model.train_model(n_estimators=100, max_depth=20, min_samples_split=5, 
                        min_samples_leaf=2, max_features='sqrt')
    rf_train_time = time.time() - rf_start_time
    rf_metrics = rf_model.evaluate_model()
    
    # Train XGBoost model
    print("\n" + "=" * 80)
    print("Training XGBoost Model...")
    print("=" * 80)
    xgb_start_time = time.time()
    xgb_model.train_model(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
    xgb_train_time = time.time() - xgb_start_time
    xgb_metrics = xgb_model.evaluate_model()
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("Model Performance Comparison")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        'Metric': [
            'Training Time (s)',
            'Train RMSE',
            'Train MAE',
            'Train R²',
            'Test RMSE',
            'Test MAE',
            'Test R²',
            'CV RMSE (mean)',
            'CV RMSE (std)'
        ],
        'Decision Tree': [
            f"{dt_train_time:.2f}",
            f"${dt_metrics['train_rmse']:,.2f}",
            f"${dt_metrics['train_mae']:,.2f}",
            f"{dt_metrics['train_r2']:.4f}",
            f"${dt_metrics['test_rmse']:,.2f}",
            f"${dt_metrics['test_mae']:,.2f}",
            f"{dt_metrics['test_r2']:.4f}",
            f"${dt_metrics['cv_rmse_mean']:,.2f}",
            f"${dt_metrics['cv_rmse_std']:,.2f}"
        ],
        'Random Forest': [
            f"{rf_train_time:.2f}",
            f"${rf_metrics['train_rmse']:,.2f}",
            f"${rf_metrics['train_mae']:,.2f}",
            f"{rf_metrics['train_r2']:.4f}",
            f"${rf_metrics['test_rmse']:,.2f}",
            f"${rf_metrics['test_mae']:,.2f}",
            f"{rf_metrics['test_r2']:.4f}",
            f"${rf_metrics['cv_rmse_mean']:,.2f}",
            f"${rf_metrics['cv_rmse_std']:,.2f}"
        ],
        'XGBoost': [
            f"{xgb_train_time:.2f}",
            f"${xgb_metrics['train_rmse']:,.2f}",
            f"${xgb_metrics['train_mae']:,.2f}",
            f"{xgb_metrics['train_r2']:.4f}",
            f"${xgb_metrics['test_rmse']:,.2f}",
            f"${xgb_metrics['test_mae']:,.2f}",
            f"{xgb_metrics['test_r2']:.4f}",
            f"${xgb_metrics['cv_rmse_mean']:,.2f}",
            f"${xgb_metrics['cv_rmse_std']:,.2f}"
        ]
    })
    
    print("\n")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    output_dir = Path(__file__).parent
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\nComparison results saved to: {output_dir / 'model_comparison.csv'}")
    
    # Plot performance comparison
    plot_comparison(dt_metrics, rf_metrics, xgb_metrics, output_dir)
    
    # Compare feature importance
    print("\n" + "=" * 80)
    print("Feature Importance Comparison")
    print("=" * 80)
    
    dt_importance = dt_model.plot_feature_importance(top_n=15)
    rf_importance = rf_model.plot_feature_importance(top_n=15)
    xgb_importance = xgb_model.plot_feature_importance(top_n=15)
    
    # Merge and compare feature importance
    compare_feature_importance(dt_importance, rf_importance, xgb_importance, output_dir)
    
    # Prediction comparison plots
    dt_model.plot_predictions()
    rf_model.plot_predictions()
    xgb_model.plot_predictions()
    
    # Additional plots for XGBoost
    xgb_model.plot_residuals()
    
    # Conclusions
    print("\n" + "=" * 80)
    print("Conclusions")
    print("=" * 80)
    
    # Find the best performing model
    best_r2 = max(dt_metrics['test_r2'], rf_metrics['test_r2'], xgb_metrics['test_r2'])
    if xgb_metrics['test_r2'] == best_r2:
        print(f"✅ XGBoost performs best!")
        print(f"   - Test R²: {xgb_metrics['test_r2']:.4f}")
        print(f"   - Test RMSE: ${xgb_metrics['test_rmse']:,.2f}")
    elif rf_metrics['test_r2'] == best_r2:
        print(f"✅ Random Forest performs best!")
        print(f"   - Test R²: {rf_metrics['test_r2']:.4f}")
        print(f"   - Test RMSE: ${rf_metrics['test_rmse']:,.2f}")
    else:
        print(f"✅ Decision Tree performs best!")
        print(f"   - Test R²: {dt_metrics['test_r2']:.4f}")
        print(f"   - Test RMSE: ${dt_metrics['test_rmse']:,.2f}")
    
    print(f"\nTraining time comparison:")
    print(f"   - Decision Tree: {dt_train_time:.2f}s")
    print(f"   - Random Forest: {rf_train_time:.2f}s")
    print(f"   - XGBoost: {xgb_train_time:.2f}s")
    
    return dt_model, rf_model, xgb_model, dt_metrics, rf_metrics, xgb_metrics


def plot_comparison(dt_metrics, rf_metrics, xgb_metrics, output_dir):
    """Plot model performance comparison charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. R² comparison
    models = ['Decision Tree', 'Random Forest', 'XGBoost']
    train_r2 = [dt_metrics['train_r2'], rf_metrics['train_r2'], xgb_metrics['train_r2']]
    test_r2 = [dt_metrics['test_r2'], rf_metrics['test_r2'], xgb_metrics['test_r2']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=15)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. RMSE comparison
    train_rmse = [dt_metrics['train_rmse'], rf_metrics['train_rmse'], xgb_metrics['train_rmse']]
    test_rmse = [dt_metrics['test_rmse'], rf_metrics['test_rmse'], xgb_metrics['test_rmse']]
    
    axes[0, 1].bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
    axes[0, 1].bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
    axes[0, 1].set_ylabel('RMSE ($)')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=15)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. MAE comparison
    train_mae = [dt_metrics['train_mae'], rf_metrics['train_mae'], xgb_metrics['train_mae']]
    test_mae = [dt_metrics['test_mae'], rf_metrics['test_mae'], xgb_metrics['test_mae']]
    
    axes[1, 0].bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.8)
    axes[1, 0].bar(x + width/2, test_mae, width, label='Test MAE', alpha=0.8)
    axes[1, 0].set_ylabel('MAE ($)')
    axes[1, 0].set_title('MAE Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=15)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Cross-validation RMSE comparison
    cv_rmse_mean = [dt_metrics['cv_rmse_mean'], rf_metrics['cv_rmse_mean'], xgb_metrics['cv_rmse_mean']]
    cv_rmse_std = [dt_metrics['cv_rmse_std'], rf_metrics['cv_rmse_std'], xgb_metrics['cv_rmse_std']]
    
    axes[1, 1].bar(x, cv_rmse_mean, yerr=cv_rmse_std, capsize=10, alpha=0.8)
    axes[1, 1].set_ylabel('CV RMSE ($)')
    axes[1, 1].set_title('Cross-Validation RMSE')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=15)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    figs_dir = output_dir / 'figs'
    figs_dir.mkdir(exist_ok=True)
    plt.savefig(figs_dir / 'models_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPerformance comparison plot saved to: {figs_dir / 'models_performance_comparison.png'}")
    plt.close()


def compare_feature_importance(dt_importance, rf_importance, xgb_importance, output_dir):
    """Compare feature importance between the three models"""
    
    # Merge top features
    top_n = 15
    dt_top = dt_importance.head(top_n).copy()
    rf_top = rf_importance.head(top_n).copy()
    xgb_top = xgb_importance.head(top_n).copy()
    
    dt_top.columns = ['feature', 'dt_importance']
    rf_top.columns = ['feature', 'rf_importance']
    xgb_top.columns = ['feature', 'xgb_importance']
    
    # Merge
    merged = pd.merge(dt_top, rf_top, on='feature', how='outer')
    merged = pd.merge(merged, xgb_top, on='feature', how='outer').fillna(0)
    merged = merged.sort_values('xgb_importance', ascending=False)
    
    print("\nFeature Importance Comparison (Top 15):")
    print(merged.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Decision Tree
    dt_data = merged.sort_values('dt_importance', ascending=True).tail(15)
    axes[0].barh(dt_data['feature'], dt_data['dt_importance'], alpha=0.8, color='skyblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Decision Tree - Top 15 Features')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Random Forest
    rf_data = merged.sort_values('rf_importance', ascending=True).tail(15)
    axes[1].barh(rf_data['feature'], rf_data['rf_importance'], alpha=0.8, color='lightcoral')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Random Forest - Top 15 Features')
    axes[1].grid(axis='x', alpha=0.3)
    
    # XGBoost
    xgb_data = merged.sort_values('xgb_importance', ascending=True).tail(15)
    axes[2].barh(xgb_data['feature'], xgb_data['xgb_importance'], alpha=0.8, color='lightgreen')
    axes[2].set_xlabel('Importance')
    axes[2].set_title('XGBoost - Top 15 Features')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    figs_dir = output_dir / 'figs'
    figs_dir.mkdir(exist_ok=True)
    plt.savefig(figs_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature importance comparison plot saved to: {figs_dir / 'feature_importance_comparison.png'}")
    plt.close()


if __name__ == '__main__':
    dt_model, rf_model, xgb_model, dt_metrics, rf_metrics, xgb_metrics = compare_models()
