import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm

from ExtremeNN import ENeuralN
from Utils import MSE


def prepare_dataset(train_path: str, test_path: str = None):
    train_set = pd.read_csv(train_path, header=None, index_col=0)
    # We have training set tr_x 1182 columns and 10 rows
    tr_x, tr_y = train_set.iloc[:, :10].to_numpy(), train_set.iloc[:, 10:].to_numpy()

    ts_x, ts_y = None, None
    if test_path is not None:
        test_set = pd.read_csv(test_path, header=None, index_col=0)
        ts_x, ts_y = test_set.iloc[:, :10].to_numpy().T, test_set.iloc[:, 10:].to_numpy().T

    x_train, x_val, y_train, y_val = train_test_split(tr_x, tr_y, test_size=0.33)
    x_train, x_val = x_train.T, x_val.T
    y_train, y_val = y_train.T, y_val.T

    return (x_train, y_train), (x_val, y_val), (ts_x, ts_y)


def fit_cholesky(x_train: np.ndarray, y_train: np.ndarray, hidden: int,
                 lambda_: float, activation=None, features_x: int = 10) -> ENeuralN:
    model = ENeuralN(features_x, hidden, lambda_, activation)
    model.fit_cholesky(x_train, y_train)
    return model


def fit_fista(x_train: np.ndarray, y_train: np.ndarray, hidden: int, lambda_: float,
              max_inters: int = None, activation=None, features_x: int = 10) -> Tuple[ENeuralN, list[float]]:
    model = ENeuralN(features_x, hidden, lambda_, activation)
    mse_errors = model.fit_fista(x_train, y_train, max_inters)
    return model, mse_errors


def get_results(model: ENeuralN, x: np.ndarray, y: np.ndarray):
    y_pred = model(x)
    return MSE(y, y_pred)


def GridSearch_cholesky(train_path: str, test_path: str = None, configs: dict = None):
    (x_train, y_train), (x_val, y_val), (ts_x, ts_y) = prepare_dataset(train_path, test_path)
    min_error, best_model, best_conf = sys.maxsize, None, None

    progress = tqdm(ParameterGrid(configs))
    for single_conf in progress:

        hidden = single_conf["hidden"]
        lambda_ = single_conf["regularization"]
        activation = single_conf["activation_fun"]

        model = fit_cholesky(x_train, y_train, hidden, lambda_, activation)
        mse_train = get_results(model, x_train, y_train)
        mse_val = get_results(model, x_val, y_val)

        if mse_val < min_error:
            best_model = model
            best_conf = (single_conf, mse_train, mse_val)
            min_error = mse_val

        progress.set_postfix(minMSE=min_error)

    print("\nThe best configuration is ", best_conf[0])
    print("Train error ", best_conf[1], " Validation error", best_conf[2])

    if test_path is not None:
        y_test_pred = best_model(ts_x)
        test_error = MSE(ts_y, y_test_pred)
        print("Test error ", test_error)


def GridSearch_fista(train_path: str, test_path: str = None, configs: dict = None):

    (x_train, y_train), (x_val, y_val), (ts_x, ts_y) = prepare_dataset(train_path, test_path)
    min_error, best_model, best_conf = sys.maxsize, None, None

    progress = tqdm(ParameterGrid(configs))
    for single_conf in progress:

        hidden = single_conf["hidden"]
        lambda_ = single_conf["regularization"]
        activation = single_conf["activation_fun"]
        max_iter = single_conf["max_iter"]

        model = fit_fista(x_train, y_train, hidden, lambda_,max_iter, activation)
        mse_train = get_results(model[0], x_train, y_train)
        mse_val = get_results(model[0], x_val, y_val)

        if mse_val < min_error:
            best_model = model[0]
            best_conf = (single_conf, mse_train, mse_val)
            min_error = mse_val

        progress.set_postfix(minMSE=min_error)

    print("\nThe best configuration is ", best_conf[0])
    print("Train error ", best_conf[1], " Validation error", best_conf[2])

    if test_path is not None:
        y_test_pred = best_model(ts_x)
        test_error = MSE(ts_y, y_test_pred)
        print("Test error ", test_error)
