# Regression for Anonymized Dataset
This repository provides a solution for predicting the target variable in a dataset using 53 anonymized features. The project includes exploratory data analysis, model training, and predictions for unseen test data. The target metric is Root Mean Squared Error (RMSE).
## Repository Contents
- **EDA_on_anonymized_data.ipynb**: Jupyter notebook for exploratory data analysis and feature choice.
- **train.py**: Python script to fit the predictive model.
- **predict.py**: Python script to make predictions on test data.
- **predictions.csv**: File containing the prediction results for hidden_test.csv.
- **requirements.txt**: List of required Python libraries.
- **README.md**: Instructions for project setup and usage.
- **train.csv**: File with train dataset.
- **hidden_test.csv** File with unseen data to make predictions on.
## Setup Instructions
1. Clone the Repository
   ```Shell
   git clone https://github.com/OVI71397/regression-anonymized.git
   cd regression-anonymized
   ```
2. Create and Activate a Virtual Environment
   ```Shell
   python3.11 -m venv my_env
   source my_env/bin/activate # for Windows use my_env\Scripts\activate instead
   ```
3. Install Dependencies
   ```Shell
   pip install -r requirements.txt
   ```
## Usage Instructions
1. Run the train.py script to train the model and save it to disk
   ```Shell
   python train.py
   ```
   You should see the following message: * * Model training completed. RMSE: 0.0014 * *
2. Use the predict.py script to generate predictions for the hidden test dataset.
   ```Shell
   python predict.py
   ```
   You should see the following message: * * Predictions saved to predictions.csv * *
