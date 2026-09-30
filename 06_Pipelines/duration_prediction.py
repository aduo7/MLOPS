#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error
import pickle

import xgboost as xgb

from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
import os

db_path = os.path.abspath("/workspaces/MLOPS/00-mlflow/mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("nyc-taxi-exp-v3")

#mlflow ui --backend-store-uri sqlite:///mlflow.db

def read_dataframe(year, month):
    url=f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    print(f"URL is: {url}")
    df=pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime-df.lpep_pickup_datetime

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df=df[((df.duration>=1) & (df.duration<=60))]

    categorical = ['PULocationID', 'DOLocationID']

    df[categorical]=df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        dv=DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X=dv.transform(dicts)
        
    return X, dv


def find_best_params(X_train, y_train, X_val, y_val, max_evals=50):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, "validation")],
                early_stopping_rounds=50,
                verbose_eval=100
            )
            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials(),
        verbose=False
    )

    # space_eval resolves the raw fmin output back into the actual
    # hyperparameter values (e.g. int max_depth) and re-adds the
    # fixed (non-tuned) entries such as 'objective' and 'seed'.
    best_params = space_eval(search_space, best_result)

    return best_params


def train_model(X_train, y_train, X_val, y_val, dv, best_params):
  
    with mlflow.start_run() as mlflow_run:
        train=xgb.DMatrix(X_train, label=y_train)
        valid=xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(best_params)

        booster = xgb.train(
                    params=best_params,
                    dtrain=train,
                    num_boost_round=30,
                    evals=[(valid, "validation")],
                    early_stopping_rounds=50
                )

        y_pred = booster.predict(valid)
        rmse=root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse",rmse)


        with open("models/preprocessor.b","wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, name="models_mlflow")

        return mlflow_run.info.run_id


def run(year, month):
    
    print(f"Year for training is : {year}")
    print(f"Month for training is : {month}")
    df_train = read_dataframe(year=year, month=month)
    
    next_year = year if month < 12 else year + 1    
    next_month = month+ 1 if month < 12 else 1
    
    print(f"Year for eval is : {next_year}")
    print(f"Month for eval is : {next_month}")
    
    df_val = read_dataframe(year=next_year, month=next_month)
    
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target='duration'
    y_train = df_train[target].values
    y_val= df_val[target].values

    print("Searching for best hyperparameters with hyperopt...")
    best_params = find_best_params(X_train, y_train, X_val, y_val)

    run_id = train_model(X_train, y_train, X_val, y_val, dv, best_params)
    
    return run_id
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on.')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on.')
    args= parser.parse_args()
    run_id=run(year=args.year, month=args.month)
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)




