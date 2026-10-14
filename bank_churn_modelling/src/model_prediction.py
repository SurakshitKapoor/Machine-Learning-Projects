

from pathlib import Path

import os
import joblib
from sklearn.pipeline import make_pipeline

def make_predictions(df):
    # Get paths to saved artifacts
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_path = os.path.join(project_root, "models", "best_model.pkl")
    preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")

    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess input data
    X = preprocessor.transform(df)

    # Generate predictions
    y_pred = model.predict(X)

    # Return predictions
    return y_pred







# testing part !
if __name__ == '__main__':
    print("WE are in prediction.py!!!")
    # Using os
    import os
    #gets the abs path from root directory to current file
    print("file path: ", os.path.abspath(__file__) )   
    # gets the till directoy path of current file!      
    print("1st parent: ", os.path.dirname(os.path.abspath(__file__) ) )
    # gets the path till parent direcoty of current file's parent directory!
    print("2nd parent: ", os.path.dirname( os.path.dirname(os.path.abspath(__file__) )))
    print()
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "best_model.pkl"
    )
    # print("Final path: ", path)

    import pandas as pd

    sample_df = pd.DataFrame({
        'CreditScore': [650, 720, 580],
        'Geography': ['France', 'Spain', 'Germany'],
        'Gender': ['Female', 'Male', 'Female'],
        'Age': [35, 45, 29],
        'Tenure': [5, 8, 3],
        'Balance': [75000.0, 120000.0, 0.0],
        'NumOfProducts': [2, 1, 1],
        'HasCrCard': [1, 1, 0],
        'IsActiveMember': [1, 0, 1],
        'EstimatedSalary': [80000.0, 110000.0, 65000.0],
        'zero_balance': [0, 0, 1],
        'balance_per_product': [37500.0, 120000.0, 0.0]
    })

    # print(sample_df)

    # make the predictions!
    y_pred = make_predictions(sample_df)
    print("Predictions are: ", y_pred)

