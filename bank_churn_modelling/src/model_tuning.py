

from sklearn.model_selection import GridSearchCV
from pathlib import Path
from utils.model_saving import save_object 


# hyperparameter tuning to improve the model!
def hypertune_best_model( best_model_name, best_model, X_train, y_train ):

    param_grids = {
        "Decision Tree": {
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 5, 10],
            "class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "class_weight": [None, "balanced"]
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "class_weight": [None, "balanced"]
        }
    }

    search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grids[best_model_name],
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )



# retraining the best model with tuned best parameters!
def best_model_with_params(best_model, best_params, X_train, y_train):
    """
    Refit a fresh copy of the selected model using the tuned hyperparameters.
    """
    from sklearn.base import clone

    final_model = clone(best_model)
    final_model.set_params(**best_params)
    final_model.fit(X_train, y_train)

    return final_model







if __name__ == '__main__':
    from data_preprocessing import get_inputs_and_outputs, split_data, preprocess_data
    from feature_engineering import create_new_features
    from data_cleaning import load_data, clean_data
    from model_evaluation import evaluate_models, select_best_model
    from model_training import train_models
    

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
    X_train, X_test, preprocessor = preprocess_data(X_train, X_test)
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


    best_model_name, best_model_metrics = select_best_model(evaluation_results)
    # make the best model as a part of pipeline!
    best_model = trained_models[best_model_name]
    print("Selecting best Model after basic training!")
    print("Best Model name: ", best_model_name)
    print("Best Model Metrics: ", best_model_metrics)
    print()

    
    best_estimator, best_params, best_cv_recall = hypertune_best_model(
        best_model_name, best_model, X_train, y_train
    )
    print("Hyperparameter Tuning is done!")
    print("Best Model:", best_model_name)
    print("Best Parameters:", best_params)
    print("Best CV Recall:", best_cv_recall)
    print()

    finalised_model = best_model_with_params(best_model, best_params, X_train, y_train)
    print("Final Model is completed!")
    print("Final Model: ", finalised_model)
    print()


    model_path = save_object(finalised_model, Path("models") / "best_model.pkl")
    print("Saving the Model")
    print("Path is: ", model_path)
    print()

    preprocessor_path = save_object(preprocessor, Path("models") / "preprocessor.pkl")
    print("Preprocessor is saved!")
    print("Path is: ", preprocessor_path)
    print()