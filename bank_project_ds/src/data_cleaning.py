import pandas as pd

def load_data(path):
    """
    Load the dataset from a CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    \

    # Load the dataset
    data = pd.read_csv(path)

    return data


def clean_data(df):
    """
    Clean the dataset by removing duplicates and handling missing values.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """

    # Remove duplicates
    data = df.drop_duplicates()

    # remove unnecessary columns
    data.drop(columns=['RowNumber', 'Surname', 'CustomerId'], inplace=True)


    return data



if __name__ == "__main__":
    df = load_data("data/Churn_Modelling.csv")
    print("Data is loaded!")
    print("Original Data Shape:", df.shape)
    # print(df.head())

    cleaned_df = clean_data(df)
    # print(cleaned_df.head(2))
    print("Cleaned Data Shape:", cleaned_df.shape)
    print("Data is cleaned!")
    