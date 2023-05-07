import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from scipy.stats import uniform, randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from lightgbm import LGBMRegressor
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
            'num_leaves': [10, 20, 30,50], #,100], #,1300,1800] #,2000,2500,2800,3000],
            'max_depth': [10, 15, 20], #,40,45,50,55],
            'n_estimators': [5000], #,6000,8000,9000,10000],
            'min_child_weight': [10, 13], #,10,50,100], #,200,500,700,800,100],
            'subsample': [0.3,0.4], #, 0.6, 0.7, 0.8, 0.9, 1.0],
            # 'reg_alpha': [0, 0.5],
            'reg_lambda': [5, 10, 15],
            'learning_rate': [0.05],
	        'metric': ['rmse'],
            'random_state': [42],
            #'early_stopping_round': [130],
	        'verbosity': [-1],
	        'min_child_samples': [20, 50],
    }
    model = LGBMRegressor(n_jobs=-1)
    gbm = GridSearchCV(model, params, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gbm.fit(features, imbalance) 
    best_parameters = report_best_scores(gbm.cv_results_, 1)

    # Train on full data
    print("Training on full model")
    lgbm_model = LGBMRegressor(**best_parameters)
    lgbm_model.fit(features, imbalance)
    pickle.dump(lgbm_model, open("model.pickle", "wb"))

    y_pred = lgbm_model.predict(features)
    evaluate_model(y_pred, imbalance)

    #feature_imp = pd.DataFrame(sorted(zip(lgbm_model.feature_importances_,features)), columns=['Value','Feature'])

    #Plot feature importance
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(30, 50))
    # sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    # plt.title('LightGBM Features')
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    data = feature_pipeline("train")
    model_train(data)
