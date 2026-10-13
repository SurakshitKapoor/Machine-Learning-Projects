

import joblib

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from model_training import train_models


def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate the trained models on the test set and return their performance metrics.

    Parameters:
    - trained_models: A dictionary of trained models.
    - X_test: The input features for the test set.
    - y_test: The target variable for the test set.

    Returns:
    - A dictionary containing the evaluation metrics for each model.
    """
    evaluation_results = {}

    # trained_models contains a model_name, and a instance of trained model, for prediction!
    for model_name, model in trained_models.items():
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store the results
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    
    return evaluation_results



# for our business case, we are focusing on Recall based model evaluations!
def select_best_model(evaluation_results):
    """
    Select the best model based on the highest recall score.

    Parameters:
    - evaluation_results: A dictionary containing the evaluation metrics for each model.

    Returns:
    - The name of the best model and its corresponding metrics.
    """
    best_model_name = None
    best_recall = 0.0

    # finding the best model based on the highest recall score!
    for model_name, metrics in evaluation_results.items():
        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            best_model_name = model_name

    # return the best model and its corresponding recall metrics score!
    return best_model_name, evaluation_results[best_model_name]



def save_best_model(best_model, model_path):
    """
    Save the best model to a file.

    Parameters:
    - best_model: The best trained model to be saved.
    - model_path: The path where the model should be saved.
    """

    model_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    joblib.dump(best_model, model_path)







if __name__ == '__main__':
    from data_preprocessing import get_inputs_and_outputs, split_data, preprocess_data
    from feature_engineering import create_new_features
    from data_cleaning import load_data, clean_data

    # Load and clean the data
    df = load_data("data/Churn_Modelling.csv")
    cleaned_df = clean_data(df)
    print("Data Cleaning Completed!")
    print("Data Shape after Cleaning:", cleaned_df.shape)
    print()

    # Create new features
    feature_df = create_new_features(cleaned_df)
    print("Feature Engineering Completed!")
    print("Data Shape after Feature Engineering:", feature_df.shape)
    print()

    # Split the data into input features and target variable
    X, y = get_inputs_and_outputs(feature_df, target_column='Exited')
    print("Data Split Completed!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data Split Completed!")
    print("X_train shape:", X_train.shape)
    print() 

    # making the data ready for model training by preprocessing the input features
    X_train, X_test = preprocess_data(X_train, X_test)
    print("Data Preprocessing Completed!")
    print("X_train shape:", X_train.shape)


    # Train the model on the raw feature matrix so the pipeline's
    # built-in preprocessing is applied correctly.

    print("⏳ Training model...", flush=True)
    trained_models = train_models(X_train, y_train)
    print("Model Training Completed!")
    print("⏳ Training model...", flush=True)




    # Evaluate the trained models on the test set
    evaluation_results = evaluate_models(trained_models, X_test, y_test)
    print("Model Evaluation Completed!")
    print()

    # find best model name, and metrics!
    find_best_model_name, best_model_metrics = select_best_model(evaluation_results)
    print(f"Best Model: {find_best_model_name}")
    print(f"Best Model Metrics: {best_model_metrics}")
    print()

