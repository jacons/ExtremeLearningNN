import sys
from typing import Tuple, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm

from ExtremeNN import ENeuralN
from Utils import MSE, sigmoid


def prepare_dataset(train_path: str, test_path: str = None, unique: bool = False):
    train_set = pd.read_csv(train_path, header=None, index_col=0)
    # We have training set tr_x 1182 columns and 10 rows
    tr_x, tr_y = train_set.iloc[:, :10].to_numpy(), train_set.iloc[:, 10:].to_numpy()

    if not unique:
        ts_x, ts_y = None, None
        if test_path is not None:
            test_set = pd.read_csv(test_path, header=None, index_col=0)
            ts_x, ts_y = test_set.iloc[:, :10].to_numpy().T, test_set.iloc[:, 10:].to_numpy().T

        x_train, x_val, y_train, y_val = train_test_split(tr_x, tr_y, test_size=0.33)
        x_train, x_val = x_train.T, x_val.T
        y_train, y_val = y_train.T, y_val.T

        return (x_train, y_train), (x_val, y_val), (ts_x, ts_y)
    else:
        return tr_x.T, tr_y.T


def fit_cholesky(x_train: np.ndarray, y_train: np.ndarray, hidden: int, lambda_: float,
                 resevoir: np.ndarray = None, activation=sigmoid, features_x: int = 10) -> ENeuralN:
    """
    :param x_train: array X [ feature, examples ]
    :param y_train: array target [ 2, examples ]
    :param hidden: Hidden size
    :param lambda_: L2-Regularization term
    :param activation: activation function
    :param features_x: Number of input features
    :param resevoir:
    """
    model = ENeuralN(features_x, hidden, lambda_, activation, resevoir)
    model.fit_cholesky(x_train, y_train)
    return model


def fit_fista(x_train: np.ndarray, y_train: np.ndarray, w_star: np.ndarray, hidden: int, lambda_: float,
              activation=None, max_inters: int = None, eps: float = 0,
              resevoir: np.ndarray = None, features_x: int = 10) -> Tuple[ENeuralN, DataFrame, float]:
    """
    :param resevoir:
    :param x_train: array X [ feature, examples ]
    :param y_train: array target [ 2, examples ]
    :param hidden: Hidden size
    :param lambda_: L2-Regularization term
    :param activation: activation function
    :param features_x: Number of input features
    :param max_inters: Number of max iteration
    :param eps: gradient thresholds
    :return:
    """
    model = ENeuralN(features_x, hidden, lambda_, activation, resevoir)
    mse_errors, gap, output_gap = model.fit_fista(x_train, y_train, max_inters, eps, w_star)

    dt = pd.DataFrame({"MSE_error": mse_errors, "Gap": gap, "Output Gap": output_gap})
    dt["iters"] = dt.index
    dt = dt[["iters", "MSE_error","Gap", "Output Gap"]].set_index("iters")

    return model, dt, round(mse_errors[-1], 4)


def fit_sgd(x_train: np.ndarray, y_train: np.ndarray, hidden: int, w_star: np.ndarray, lambda_: float = 0,
            activation=sigmoid, max_inters: int = None, eps: float = 0, resevoir: np.ndarray = None,
            lr: float = 0, beta: float = 0, features_x: int = 10) -> Tuple[ENeuralN, DataFrame, float]:
    """
    :param resevoir:
    :param x_train: array X [ feature, examples ]
    :param y_train: array target [ 2, examples ]
    :param hidden: Hidden size
    :param lambda_: L2-Regularization term
    :param activation: activation function
    :param features_x: Number of input features
    :param max_inters: Number of max iteration
    :param eps: gradient thresholds
    :param lr: learning rate if 0 then will be used 1/L
    :param beta: momentum term

    """
    model = ENeuralN(features_x, hidden, lambda_, activation, resevoir)
    mse_errors, gap, output_gap = model.fit_SDG(x_train, y_train, max_inters, lr, beta, eps, w_star)

    dt = pd.DataFrame({"MSE_error": mse_errors, "Gap": gap, "Output Gap": output_gap})
    dt["iters"] = dt.index
    dt = dt[["iters", "MSE_error", "Gap", "Output Gap"]].set_index("iters")

    return model, dt, round(mse_errors[-1], 4)


def get_results(model: ENeuralN, x: np.ndarray, y: np.ndarray):
    y_pred = model(x)
    return MSE(y, y_pred)


def grid_search_cholesky(configs: dict, train: tuple[np.ndarray, np.ndarray], valid: tuple[np.ndarray, np.ndarray],
                         test: tuple[np.ndarray, np.ndarray] = None):
    min_error, best_model, best_conf = sys.maxsize, None, None

    progress = tqdm(ParameterGrid(configs))
    for single_conf in progress:

        model = fit_cholesky(x_train=train[0], y_train=train[1], **single_conf)
        mse_train = get_results(model, train[0], train[1])
        mse_val = get_results(model, valid[0], valid[1])

        if mse_val < min_error:
            best_model = model
            best_conf = (single_conf, mse_train, mse_val)
            min_error = mse_val

        progress.set_postfix(minMSE=min_error)

    print("\nThe best configuration is ", best_conf[0])
    print("Train error ", best_conf[1], " Validation error", best_conf[2])

    if test is not None:
        y_test_pred = best_model(test[0])
        test_error = MSE(test[1], y_test_pred)
        print("Test error ", test_error)


def grid_search_iterative(configs: dict, train: tuple[np.ndarray, np.ndarray], valid: tuple[np.ndarray, np.ndarray],
                          optimizer: Literal["FISTA", "SGD"] = "SGD",
                          test: tuple[np.ndarray, np.ndarray] = None):
    min_error, best_model, best_conf = sys.maxsize, None, None

    progress = tqdm(ParameterGrid(configs))
    for single_conf in progress:

        model = fit_fista(x_train=train[0], y_train=train[1], **single_conf) if optimizer == "FISTA" \
            else fit_sgd(x_train=train[0], y_train=train[1], **single_conf)

        mse_train = get_results(model[0], train[0], train[1])
        mse_val = get_results(model[0], valid[0], valid[1])

        if mse_val < min_error:
            best_model = model[0]
            best_conf = (single_conf, mse_train, mse_val)
            min_error = mse_val

        progress.set_postfix(minMSE=min_error)

    print("\nThe best configuration is ", best_conf[0])
    print("Train error ", best_conf[1], " Validation error", best_conf[2])

    if test is not None:
        y_test_pred = best_model(test[0])
        test_error = MSE(test[1], y_test_pred)
        print("Test error ", test_error)
