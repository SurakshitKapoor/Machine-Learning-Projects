
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import column_or_1d

# split data into X, and y
def get_inputs_and_outputs(df, target_column):
    """
    Splits the DataFrame into input features and output target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.

    Returns:
    tuple: A tuple containing the input features (X) and output target variable (y).
    """
    X = df.drop(columns=[target_column])
    y = df[[target_column]]
    y = column_or_1d(y, warn=True)

    return X, y



# split data into training and testing parts
def split_data(X, y):

    # splits data into training and testing parts!
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)

    return X_train, X_test, y_train, y_test



# function to apply preprocessing steps on the data, and return the processed data
# in form of a pipeline!

def preprocess_data(X_train, X_test):
    """
    Preprocesses the input data by applying necessary transformations.

    Parameters:
    X_train (pd.DataFrame): The training input features.
    X_test (pd.DataFrame): The testing input features.

    Returns:
    tuple: A tuple containing the preprocessed training and testing input features.
    """
    num_cols = X_train.select_dtypes(include=['number']).columns.to_list()
    cat_cols = X_train.select_dtypes(exclude=['number']).columns.to_list()
    print("Numerical Columns:", num_cols)
    print("Categorical Columns:", cat_cols)

    transformers = []

    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols))

    if not transformers:
        return X_train.copy(), X_test.copy()

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, preprocessor





# to test the working of current file, or to execute something when server initiatedf rom current file!
if __name__ == "__main__":
    print("In Data_Preprocessing.py")

    from data_cleaning import load_data, clean_data
    from feature_engineering import create_new_features

    df = load_data("data/Churn_Modelling.csv")
    cleaned_df = clean_data(df)
    feature_df = create_new_features(cleaned_df)

    # data seperation!
    X, y = get_inputs_and_outputs(feature_df, "Exited")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data Split Completed!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # data preprocessing steps
    X_train, X_test = preprocess_data(X_train, X_test)
    print("Data Preprocessing Completed!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)




# <!-- 
# - Reassignment: X_train = ... → local variable points to a new object. Return it if the caller needs the updated result.
# - In-place modification: X_train.drop(..., inplace=True) → modifies the original DataFrame. No return is needed to preserve that change.
# Real-world tip: Prefer returning the transformed DataFrame rather than modifying the caller's original data unexpectedly. -->

