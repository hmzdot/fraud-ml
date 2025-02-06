import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def calculate_fraud_index(df, corr_fraud):
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
    corr_matrix = df.corr()

    # Get correlation values sorted by importance
    corr_fraud = corr_matrix["fraud_bool"].sort_values(ascending=False)
    corr_fraud.drop("fraud_bool")

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


def train(df: pd.DataFrame):
    X = df.drop(columns=["fraud_bool"])
    y = df["fraud_bool"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create XGBoost DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define XGBoost parameters
    params = {
        "objective": "binary:logistic",  # binary classification
        "eval_metric": "auc",  # AUC is a good metric for fraud detection
        "max_depth": 5,  # maximum depth of trees (tune as necessary)
        "eta": 0.1,  # learning rate (tune as necessary)
        "seed": 42,
        "subsample": 0.8,  # sample ratio of training instances
        "colsample_bytree": 0.8,  # subsample ratio of columns when constructing each tree
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
        verbose_eval=True,
    )

    # Use the trained model to predict probabilities on the test set
    y_pred_proba = bst.predict(dtest)

    # Evaluate the model using ROC AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print("Test ROC AUC Score:", auc_score)

    # Optionally, if you want to convert probabilities to binary predictions:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    model_name = "./snapshots/model_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".ubj"
    bst.save_model(model_name)


if __name__ == "__main__":
    df = pd.read_csv("data/fraud.csv")

    df = clean_data(df)
    train(df)
