import pandas as pd
import numpy as np


def calculate_fraud_index(df: pd.DataFrame, corr_fraud: pd.Series):
    """
    Calculate a fraud index based on correlation values.

    Steps:
    1. Normalize numerical features to be between -1 and 1.
    2. Clamp boolean features to -1 (False) and 1 (True).
    3. Multiply each column by its correlation with fraud_bool.
    4. Sum across columns to get a fraud index for each row.

    Parameters:
        df (pd.DataFrame): The original dataset.
        fraud_corr (pd.Series): Correlation values of features with fraud_bool.

    Returns:
        pd.Series: Fraud index values for each row.
    """

    # Step 0: Select only the columns that have a correlation with fraud_bool
    selected_features = corr_fraud.index.tolist()
    df_selected = df[selected_features].copy()

    # Step 1: Normalize numerical features
    for col in df_selected.columns:
        if df_selected[col].dtype == "bool":
            # Convert boolean columns: False -> -1, True -> 1
            df_selected[col] = df_selected[col].astype(int) * 2 - 1
        elif np.issubdtype(df_selected[col].dtype, np.number):
            # Scale numerical columns based on mean and standard deviation
            mean = df_selected[col].mean()
            std = df_selected[col].std()
            df_selected[col] = (df_selected[col] - mean) / std
            df_selected[col] = df_selected[col].clip(-1, 1)  # Clamp to [-1, 1]

    # Step 2: Multiply each column by its correlation weight
    fraud_weights = corr_fraud.values
    fraud_index = df_selected.mul(fraud_weights, axis=1).sum(axis=1)

    return fraud_index
