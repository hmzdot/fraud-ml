import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os
from fraud import calculate_fraud_index


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate fraud probability")
    parser.add_argument("model_snapshot", help="Path to the trained model snapshot")
    parser.add_argument("data_path", help="Path to transactions csv")
    return parser.parse_args()


def prepare_data(df: pd.DataFrame):
    if "fraud_bool" in df.columns:
        df = df.drop(columns=["fraud_bool"])

    # Convert categorical variables into numeric
    df = pd.get_dummies(
        df,
        columns=[
            "payment_type",
            "employment_status",
            "source",
            "device_os",
            "housing_status",
        ],
    )

    try:
        # Load the previously saved correlation weights
        with open("data/corr_fraud.pkl", "rb") as f:
            saved_corr_fraud = pickle.load(f)

        fraud_index = calculate_fraud_index(df, saved_corr_fraud)
        df["fraud_index"] = fraud_index
    except FileNotFoundError:
        df["fraud_index"] = 0

    # List the columns where -1 indicates missing data
    missing_cols = [
        "current_address_months_count",
        "prev_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
    ]

    # Replace -1 with np.nan in these columns
    df[missing_cols] = df[missing_cols].replace(-1, np.nan)

    # Identify boolean columns
    bool_cols = df.select_dtypes(include=["bool"]).columns

    # Convert boolean columns to integers (0 or 1)
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def eval(df: pd.DataFrame, model_snapshot: str) -> pd.Series:
    dtest = xgb.DMatrix(df)

    bst = xgb.Booster()
    bst.load_model(model_snapshot)

    y_pred_proba = bst.predict(dtest)
    return y_pred_proba


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.data_path)
    df = prepare_data(df)

    fraud_proba = eval(df, args.model_snapshot)
    df["fraud_probability"] = fraud_proba

    file_name = os.path.basename(args.data_path)
    eval_file_name = f"data/eval_{file_name}"
    fraud_count_low = len(df[df["fraud_probability"] > 0.1])
    fraud_count = len(df[df["fraud_probability"] > 0.5])
    df.to_csv(eval_file_name, index=False)

    print(f"Evaluated {len(df)} transactions")
    print(f"Transactions with fraud probability > 0.1: {fraud_count_low}")
    print(f"Transactions with fraud probability > 0.5: {fraud_count}")
    print(f"Saved to {eval_file_name}")
