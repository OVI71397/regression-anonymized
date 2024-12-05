import pandas as pd
import joblib
import os

def predict():
    
    # Load the test data
    test_set = pd.read_csv('hidden_test.csv')
    X_test = test_set[['6', '7']]
    
    # Load the trained model
    model_path = os.path.join('model', 'random_forest_model.joblib')
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Save predictions to file
    pd.DataFrame({
        'prediction': predictions
    }).to_csv('predictions.csv', index=False)
    
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    predict()