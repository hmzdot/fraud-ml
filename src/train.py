import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import random
from typing import Union
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from fraud import calculate_fraud_index


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
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

    # Get correlation values sorted by importance
    corr_fraud = df.corr()["fraud_bool"].sort_values(ascending=False)
    corr_fraud = corr_fraud.drop("fraud_bool")

    # Save the correlation series to disk for future use
    with open("data/corr_fraud.pkl", "wb") as f:
        pickle.dump(corr_fraud, f)

    df["fraud_index"] = calculate_fraud_index(df, corr_fraud)

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


def train(df: pd.DataFrame, seed: Union[int, None] = None):
    X = df.drop(columns=["fraud_bool"])
    y = df["fraud_bool"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    if seed is None:
        seed = random.randint(0, 1000000)
    print(f"Using seed: {seed}")

    # Define XGBoost parameters
    params = {
        "objective": "binary:logistic",  # binary classification
        "eval_metric": "auc",  # AUC is a good metric for fraud detection
        "max_depth": 5,
        "eta": 0.1,
        "seed": seed,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,
    }

    # Specify number of boosting rounds
    num_rounds = 100

    # Set up a watchlist to monitor performance on training and test sets
    watchlist = [(dtrain, "train"), (dtest, "eval")]

    # Train the model with early stopping
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        early_stopping_rounds=10,
    )

    # Use the trained model to predict probabilities on the test set
    y_pred_proba = bst.predict(dtest)

    # Evaluate the model using ROC AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print("Test ROC AUC Score:", auc_score)

    # Optionally, if you want to convert probabilities to binary predictions:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("True Negatives:", cm[0, 0])
    print("False Positives:", cm[0, 1])
    print("False Negatives:", cm[1, 0])
    print("True Positives:", cm[1, 1])

    model_name = "./snapshots/model_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".ubj"
    bst.save_model(model_name)


if __name__ == "__main__":
    df = pd.read_csv("data/fraud.csv")

    df = clean_data(df)
    train(df)
