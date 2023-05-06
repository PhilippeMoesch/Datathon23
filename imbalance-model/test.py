import pickle

import numpy as np
from lightgbm import LGBMRegressor
from feature_engineering import feature_pipeline
from train import split_features_imbalance, evaluate_model


def evaluate_test_data() -> None:
    df = feature_pipeline("test")
    features, imbalance, _ = split_features_imbalance(df)
    lgbm_model = pickle.load(open("model.pickle", "rb"))
    y_pred = lgbm_model.predict(features)
    evaluate_model(y_pred, imbalance)
    # Using just 0
    print("Using just 0")
    y_pred = np.repeat(0, len(y_pred))
    evaluate_model(y_pred, imbalance)


if __name__ == "__main__":
    evaluate_test_data()
