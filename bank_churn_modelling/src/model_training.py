
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# classification problem! -> labels -> 0 (genuine), 1 (churned)!
def train_models(X_train, y_train):

    """
    #     Trains the given model using the provided training data.

    #     Parameters:
    #     X_train (pd.DataFrame): The training input features.
    #     y_train (pd.Series): The training target variable.
    #     model: The machine learning model to be trained.

    #     Returns:
    #     The trained models.
    #     """

    #     starting the model training process for multiple classifiers

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    trained_models = {}

    for model_name, model in models.items():
        # model training initiated!
        model.fit(X_train, y_train)

        trained_models[model_name] = model

    return trained_models




if __name__ == '__main__':
    from data_preprocessing import get_inputs_and_outputs, split_data, preprocess_data
    from feature_engineering import create_new_features
    from data_cleaning import load_data, clean_data
    from sklearn.ensemble import RandomForestClassifier

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

    print()








    # The saved model is already a complete sklearn Pipeline that includes
    # its own preprocessing steps. Fitting it on already-transformed data
    # would pass the wrong shape into the pipeline and break training.
    # PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # model_path = PROJECT_ROOT / "models" / "best_decision_tree_model.pkl"
    # model_pipeline = joblib.load(model_path)

