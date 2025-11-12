"""
Quick Test Script - Verify models work correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from decision_tree_model import HousePriceDecisionTree
from random_forest_model import HousePriceRandomForest


def test_decision_tree():
    """Test Decision Tree model"""
    print("\n" + "=" * 60)
    print("Testing Decision Tree Model")
    print("=" * 60)
    
    try:
        # Create model
        model = HousePriceDecisionTree()
        print("✅ Model created successfully")
        
        # Load data
        X, y = model.load_and_preprocess_data()
        print(f"✅ Data loaded successfully: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Split data
        model.split_data(X, y, test_size=0.2)
        print(f"✅ Data split successfully: train {len(model.X_train)}, test {len(model.X_test)}")
        
        # Train model (using smaller parameters for speed)
        model.train_model(max_depth=10, min_samples_split=10, min_samples_leaf=5)
        print("✅ Model trained successfully")
        
        # Evaluate model
        metrics = model.evaluate_model()
        print(f"✅ Model evaluated successfully:")
        print(f"   - Test R²: {metrics['test_r2']:.4f}")
        print(f"   - Test RMSE: ${metrics['test_rmse']:,.2f}")
        
        # Test prediction
        predictions = model.predict(model.X_test.iloc[:5])
        print(f"✅ Prediction works, first 5 predictions:")
        for i, pred in enumerate(predictions[:5]):
            actual = model.y_test.iloc[i]
            print(f"   {i+1}. Predicted: ${pred:,.2f}, Actual: ${actual:,.2f}")
        
        print("\n✅ Decision Tree model test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Decision Tree model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_random_forest():
    """Test Random Forest model"""
    print("\n" + "=" * 60)
    print("Testing Random Forest Model")
    print("=" * 60)
    
    try:
        # Create model
        model = HousePriceRandomForest()
        print("✅ Model created successfully")
        
        # Load data
        X, y = model.load_and_preprocess_data()
        print(f"✅ Data loaded successfully: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Split data
        model.split_data(X, y, test_size=0.2)
        print(f"✅ Data split successfully: train {len(model.X_train)}, test {len(model.X_test)}")
        
        # Train model (using smaller parameters for speed)
        model.train_model(n_estimators=10, max_depth=10, min_samples_split=10)
        print("✅ Model trained successfully")
        
        # Evaluate model
        metrics = model.evaluate_model()
        print(f"✅ Model evaluated successfully:")
        print(f"   - Test R²: {metrics['test_r2']:.4f}")
        print(f"   - Test RMSE: ${metrics['test_rmse']:,.2f}")
        
        # Test prediction
        predictions = model.predict(model.X_test.iloc[:5])
        print(f"✅ Prediction works, first 5 predictions:")
        for i, pred in enumerate(predictions[:5]):
            actual = model.y_test.iloc[i]
            print(f"   {i+1}. Predicted: ${pred:,.2f}, Actual: ${actual:,.2f}")
        
        print("\n✅ Random Forest model test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Random Forest model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Starting House Price Prediction Model Tests")
    print("=" * 60)
    
    dt_success = test_decision_tree()
    rf_success = test_random_forest()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Decision Tree Model: {'✅ PASSED' if dt_success else '❌ FAILED'}")
    print(f"Random Forest Model: {'✅ PASSED' if rf_success else '❌ FAILED'}")
    
    if dt_success and rf_success:
        print("\n🎉 All tests passed! Models are ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python decision_tree_model.py' to train full Decision Tree model")
        print("  2. Run 'python random_forest_model.py' to train full Random Forest model")
        print("  3. Run 'python compare_models.py' to compare both models")
    else:
        print("\n⚠️ Some tests failed, please check the error messages above.")


if __name__ == '__main__':
    main()
