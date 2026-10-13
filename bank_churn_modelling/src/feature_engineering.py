
import numpy as np

def create_new_features(df):
    # 1. Whether customer has zero account balance
    df['zero_balance'] = (df['Balance'] == 0).astype(int)

    # 2. Average balance per product held
    df['balance_per_product'] = (
        df['Balance'] / df['NumOfProducts'].replace(0, np.nan) )

    return df



if __name__ == "__main__":
    from data_cleaning import load_data, clean_data

    df = load_data("data/Churn_Modelling.csv")
    cleaned_df = clean_data(df)
    feature_df = create_new_features(cleaned_df)

    print("Feature Engineering Completed!")
    print("Data Shape after Feature Engineering:", feature_df.shape)