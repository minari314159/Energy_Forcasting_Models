import itertools
import random
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from .preprocess import create_timeseries_features, add_lags


def cross_validation(df: pd.DataFrame) -> None:
    tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
    df = df.sort_index()

    fold = 0
    preds = []
    scores = []
    # Loop through the TimeSeriesSplit
    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        train = create_timeseries_features(train)
        test = create_timeseries_features(test)
        target_map = df['PJME_MW'].to_dict()
        train = add_lags(train, target_map)
        test = add_lags(test, target_map)

        FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                    'lag1', 'lag2', 'lag3']
        TARGET = 'PJME_MW'

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        model_xgb = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000, early_stopping_rounds=50, objective="reg:linear", max_depth=6,
                                     learning_rate=0.01, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1, random_state=42)
        model_xgb.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=100)

        y_pred = model_xgb.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

        return preds, scores, model_xgb


def tune_hyperparameters(df: pd.DataFrame, params: dict, split_date: str, n_iter: int) -> pd.DataFrame:
    """
    Tune the hyperparameters of the XGBoost model using a basic grid search

    Args:
        df (pd.DataFrame): The dataframe to use
        params (dict): The hyperparameters to tune
        split_date (str): The date to split the dataset on

    Returns:
        pd.DataFrame: The results of the hyperparameter tuning
    """
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]

    train = create_timeseries_features(train)
    test = create_timeseries_features(test)
    target_map = df['PJME_MW'].to_dict()
    train = add_lags(train, target_map)
    test = add_lags(test, target_map)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                'lag1', 'lag2', 'lag3']
    TARGET = 'PJME_MW'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    keys, values = zip(*params.items())
    all_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)]
    random_combinations = random.sample(all_combinations, n_iter)
    results = []

    for hyperparameters in random_combinations:
        model_xgb = xgb.XGBRegressor(base_score=0.5, booster='gbtree', early_stopping_rounds=50, objective="reg:linear",
                                     min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0, random_state=42, **hyperparameters)
        model_xgb.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=100)

        y_pred = model_xgb.predict(X_test)
        score = np.sqrt(root_mean_squared_error(y_test, y_pred))

        results.append({
            **hyperparameters,
            'score': score
        })

    return pd.DataFrame(results)
