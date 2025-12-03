"""
Comprehensive Model Comparison Script
Compare all models: Decision Tree, Random Forest, XGBoost, OLS, Lasso, Ridge
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
from linear_models import HousePriceLinearModels


def compare_all_models():
    """Compare all six models"""
    
    print("=" * 100)
    print("Comprehensive House Price Prediction Model Comparison")
    print("Decision Tree | Random Forest | XGBoost | OLS | Lasso | Ridge")
    print("=" * 100)
    
    # Initialize all models
    dt_model = HousePriceDecisionTree()
    rf_model = HousePriceRandomForest()
    xgb_model = HousePriceXGBoost()
    linear_models = HousePriceLinearModels()
    
    # Load data (load once)
    print("\nLoading data...")
    X, y = dt_model.load_and_preprocess_data()
    
    # Split data
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
    
    # Share data with Linear Models (they handle scaling internally)
    linear_models.X_train = dt_model.X_train
    linear_models.X_test = dt_model.X_test
    linear_models.y_train = dt_model.y_train
    linear_models.y_test = dt_model.y_test
    linear_models.feature_names = dt_model.feature_names
    linear_models.label_encoders = dt_model.label_encoders
    
    # Scale the data for linear models
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    linear_models.X_train_scaled = scaler.fit_transform(linear_models.X_train)
    linear_models.X_test_scaled = scaler.transform(linear_models.X_test)
    linear_models.scaler = scaler
    
    # Train Decision Tree model
    print("\n" + "=" * 100)
    print("1/6: Training Decision Tree Model...")
    print("=" * 100)
    dt_start_time = time.time()
    dt_model.train_model(max_depth=15, min_samples_split=10, min_samples_leaf=5)
    dt_train_time = time.time() - dt_start_time
    dt_metrics = dt_model.evaluate_model()
    
    # Train Random Forest model
    print("\n" + "=" * 100)
    print("2/6: Training Random Forest Model...")
    print("=" * 100)
    rf_start_time = time.time()
    rf_model.train_model(n_estimators=100, max_depth=20, min_samples_split=5, 
                        min_samples_leaf=2, max_features='sqrt')
    rf_train_time = time.time() - rf_start_time
    rf_metrics = rf_model.evaluate_model()
    
    # Train XGBoost model
    print("\n" + "=" * 100)
    print("3/6: Training XGBoost Model...")
    print("=" * 100)
    xgb_start_time = time.time()
    xgb_model.train_model(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
    xgb_train_time = time.time() - xgb_start_time
    xgb_metrics = xgb_model.evaluate_model()
    
    # Train OLS model
    print("\n" + "=" * 100)
    print("4/6: Training OLS Model...")
    print("=" * 100)
    ols_start_time = time.time()
    linear_models.train_ols()
    ols_train_time = time.time() - ols_start_time
    ols_metrics = linear_models.evaluate_model(linear_models.ols_model, "OLS")
    
    # Train Lasso model
    print("\n" + "=" * 100)
    print("5/6: Training Lasso Model...")
    print("=" * 100)
    lasso_start_time = time.time()
    linear_models.train_lasso(alpha=100.0)
    lasso_train_time = time.time() - lasso_start_time
    lasso_metrics = linear_models.evaluate_model(linear_models.lasso_model, "Lasso")
    
    # Train Ridge model
    print("\n" + "=" * 100)
    print("6/6: Training Ridge Model...")
    print("=" * 100)
    ridge_start_time = time.time()
    linear_models.train_ridge(alpha=100.0)
    ridge_train_time = time.time() - ridge_start_time
    ridge_metrics = linear_models.evaluate_model(linear_models.ridge_model, "Ridge")
    
    # Create comprehensive comparison table
    print("\n" + "=" * 100)
    print("Comprehensive Model Performance Comparison")
    print("=" * 100)
    
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
        ],
        'OLS': [
            f"{ols_train_time:.2f}",
            f"${ols_metrics['train_rmse']:,.2f}",
            f"${ols_metrics['train_mae']:,.2f}",
            f"{ols_metrics['train_r2']:.4f}",
            f"${ols_metrics['test_rmse']:,.2f}",
            f"${ols_metrics['test_mae']:,.2f}",
            f"{ols_metrics['test_r2']:.4f}",
            f"${ols_metrics['cv_rmse_mean']:,.2f}",
            f"${ols_metrics['cv_rmse_std']:,.2f}"
        ],
        'Lasso': [
            f"{lasso_train_time:.2f}",
            f"${lasso_metrics['train_rmse']:,.2f}",
            f"${lasso_metrics['train_mae']:,.2f}",
            f"{lasso_metrics['train_r2']:.4f}",
            f"${lasso_metrics['test_rmse']:,.2f}",
            f"${lasso_metrics['test_mae']:,.2f}",
            f"{lasso_metrics['test_r2']:.4f}",
            f"${lasso_metrics['cv_rmse_mean']:,.2f}",
            f"${lasso_metrics['cv_rmse_std']:,.2f}"
        ],
        'Ridge': [
            f"{ridge_train_time:.2f}",
            f"${ridge_metrics['train_rmse']:,.2f}",
            f"${ridge_metrics['train_mae']:,.2f}",
            f"{ridge_metrics['train_r2']:.4f}",
            f"${ridge_metrics['test_rmse']:,.2f}",
            f"${ridge_metrics['test_mae']:,.2f}",
            f"{ridge_metrics['test_r2']:.4f}",
            f"${ridge_metrics['cv_rmse_mean']:,.2f}",
            f"${ridge_metrics['cv_rmse_std']:,.2f}"
        ]
    })
    
    print("\n")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    output_dir = Path(__file__).parent
    comparison_df.to_csv(output_dir / 'comprehensive_model_comparison.csv', index=False)
    print(f"\nComparison results saved to: {output_dir / 'comprehensive_model_comparison.csv'}")
    
    # Create comprehensive performance visualization
    plot_comprehensive_comparison(
        dt_metrics, rf_metrics, xgb_metrics, 
        ols_metrics, lasso_metrics, ridge_metrics, 
        output_dir
    )
    
    # Conclusions
    print("\n" + "=" * 100)
    print("Conclusions")
    print("=" * 100)
    
    # Find the best performing model
    all_metrics = {
        'Decision Tree': dt_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics,
        'OLS': ols_metrics,
        'Lasso': lasso_metrics,
        'Ridge': ridge_metrics
    }
    
    best_r2_name = max(all_metrics.items(), key=lambda x: x[1]['test_r2'])[0]
    best_rmse_name = min(all_metrics.items(), key=lambda x: x[1]['test_rmse'])[0]
    
    print(f"\n🏆 Best Model by Test R²: {best_r2_name}")
    print(f"   - Test R²: {all_metrics[best_r2_name]['test_r2']:.4f}")
    print(f"   - Test RMSE: ${all_metrics[best_r2_name]['test_rmse']:,.2f}")
    
    print(f"\n🏆 Best Model by Test RMSE: {best_rmse_name}")
    print(f"   - Test RMSE: ${all_metrics[best_rmse_name]['test_rmse']:,.2f}")
    print(f"   - Test R²: {all_metrics[best_rmse_name]['test_r2']:.4f}")
    
    print(f"\n⏱️  Training time comparison:")
    print(f"   - Decision Tree: {dt_train_time:.3f}s")
    print(f"   - Random Forest: {rf_train_time:.3f}s")
    print(f"   - XGBoost: {xgb_train_time:.3f}s")
    print(f"   - OLS: {ols_train_time:.3f}s")
    print(f"   - Lasso: {lasso_train_time:.3f}s")
    print(f"   - Ridge: {ridge_train_time:.3f}s")
    
    # Model categories
    print(f"\n📊 Model Type Performance:")
    print(f"   Tree-based models (Decision Tree, Random Forest, XGBoost):")
    print(f"     - Best Test R²: {max(dt_metrics['test_r2'], rf_metrics['test_r2'], xgb_metrics['test_r2']):.4f}")
    print(f"   Linear models (OLS, Lasso, Ridge):")
    print(f"     - Best Test R²: {max(ols_metrics['test_r2'], lasso_metrics['test_r2'], ridge_metrics['test_r2']):.4f}")
    
    return all_metrics, comparison_df


def plot_comprehensive_comparison(dt_metrics, rf_metrics, xgb_metrics, 
                                   ols_metrics, lasso_metrics, ridge_metrics, 
                                   output_dir):
    """Plot comprehensive model performance comparison charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Model names
    models = ['Decision\nTree', 'Random\nForest', 'XGBoost', 'OLS', 'Lasso', 'Ridge']
    
    # 1. R² comparison
    train_r2 = [
        dt_metrics['train_r2'], rf_metrics['train_r2'], xgb_metrics['train_r2'],
        ols_metrics['train_r2'], lasso_metrics['train_r2'], ridge_metrics['train_r2']
    ]
    test_r2 = [
        dt_metrics['test_r2'], rf_metrics['test_r2'], xgb_metrics['test_r2'],
        ols_metrics['test_r2'], lasso_metrics['test_r2'], ridge_metrics['test_r2']
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='skyblue')
    bars2 = axes[0, 0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='coral')
    axes[0, 0].set_ylabel('R² Score', fontsize=12)
    axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0.75, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. RMSE comparison
    test_rmse = [
        dt_metrics['test_rmse'], rf_metrics['test_rmse'], xgb_metrics['test_rmse'],
        ols_metrics['test_rmse'], lasso_metrics['test_rmse'], ridge_metrics['test_rmse']
    ]
    
    bars = axes[0, 1].bar(x, test_rmse, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[0, 1].set_ylabel('RMSE ($)', fontsize=12)
    axes[0, 1].set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                      f'${height/1000:.1f}K', ha='center', va='bottom', fontsize=8)
    
    # 3. MAE comparison
    test_mae = [
        dt_metrics['test_mae'], rf_metrics['test_mae'], xgb_metrics['test_mae'],
        ols_metrics['test_mae'], lasso_metrics['test_mae'], ridge_metrics['test_mae']
    ]
    
    bars = axes[1, 0].bar(x, test_mae, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[1, 0].set_ylabel('MAE ($)', fontsize=12)
    axes[1, 0].set_title('Test MAE Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                      f'${height/1000:.1f}K', ha='center', va='bottom', fontsize=8)
    
    # 4. Cross-validation RMSE comparison
    cv_rmse_mean = [
        dt_metrics['cv_rmse_mean'], rf_metrics['cv_rmse_mean'], xgb_metrics['cv_rmse_mean'],
        ols_metrics['cv_rmse_mean'], lasso_metrics['cv_rmse_mean'], ridge_metrics['cv_rmse_mean']
    ]
    cv_rmse_std = [
        dt_metrics['cv_rmse_std'], rf_metrics['cv_rmse_std'], xgb_metrics['cv_rmse_std'],
        ols_metrics['cv_rmse_std'], lasso_metrics['cv_rmse_std'], ridge_metrics['cv_rmse_std']
    ]
    
    bars = axes[1, 1].bar(x, cv_rmse_mean, yerr=cv_rmse_std, capsize=5, alpha=0.8,
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[1, 1].set_ylabel('CV RMSE ($)', fontsize=12)
    axes[1, 1].set_title('Cross-Validation RMSE (5-Fold)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    figs_dir = output_dir / 'figs'
    figs_dir.mkdir(exist_ok=True)
    plt.savefig(figs_dir / 'comprehensive_models_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComprehensive performance comparison plot saved to: {figs_dir / 'comprehensive_models_comparison.png'}")
    plt.close()


if __name__ == '__main__':
    all_metrics, comparison_df = compare_all_models()

