import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import plotly.express as px
from feature_engineering import feature_pipeline


def report_best_scores(results, n_top: int = 3) -> Dict:
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
    return results


def split_features_imbalance(df: pd.DataFrame) -> List:
    feature_names = df.drop(columns=["imbalance", "ts"]).columns.to_list()
    features = df[feature_names].to_numpy()
    imbalance = df["imbalance"].to_numpy()
    return features, imbalance, feature_names


def evaluate_model(y_pred, imbalance) -> None:
    mse = mean_squared_error(imbalance, y_pred)
    mae = mean_absolute_error(imbalance, y_pred)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")


def model_train(df: pd.DataFrame) -> None:
    features, imbalance, feature_names = split_features_imbalance(df)
    feature_names = df.drop(columns=["imbalance", "ts"]).columns.to_list()
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4),
    }
    xgb_model = xgb.XGBRegressor()
    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=params,
        random_state=42,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        return_train_score=True,
        verbose=10,
    )
    search.fit(features, imbalance)
    best_parameters = report_best_scores(search.cv_results_, 1)

    # Train on full data
    print("Training on full model")
    xgb_model = xgb.XGBRegressor(**best_parameters)
    xgb_model.fit(features, imbalance)
    xgb_model.get_booster().feature_names = feature_names
    pickle.dump(xgb_model, open("model.pickle", "wb"))

    y_pred = xgb_model.predict(features)
    evaluate_model(y_pred, imbalance)

    # Plot feature importance
    feature_importance = xgb_model.get_booster().get_score(importance_type="total_gain")
    feature_importance = pd.json_normalize(feature_importance).T.reset_index()
    feature_importance.columns = ["feature", "importance"]
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    fig = px.bar(feature_importance, x="feature", y="importance")
    fig.write_html("feature_importance.html")


if __name__ == "__main__":
    data = feature_pipeline("train")
    model_train(data)
