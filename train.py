import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model():

    # Read the training data
    train_set = pd.read_csv('train.csv')
    
    # Prepare features and target
    X = train_set[['6', '7']]
    Y = train_set['target']
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.1, 
        random_state=4545
    )
    
    # Initialize and train the model
    model = RandomForestRegressor(random_state=1515)
    model.fit(X_train, Y_train)
    
    # Make predictions and calculate RMSE
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    # Create model directory if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Save the model
    model_path = os.path.join('model', 'random_forest_model.joblib')
    joblib.dump(model, model_path)
    
    return rmse

if __name__ == "__main__":
    rmse = train_model()
    print(f"Model training completed. RMSE: {rmse:.4f}")
