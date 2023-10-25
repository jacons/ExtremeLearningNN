import sys
import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm

from ExtremeNN import ENeuralN
from Utils import MSE


def GridSearch(train_path: str, test_path: str, configs: dict):
    train_set = pd.read_csv(train_path, header=None, index_col=0)
    test_set = pd.read_csv(test_path, header=None, index_col=0)

    # We have training set tr_x 1182 columns and 10 rows
    tr_x, tr_y = train_set.iloc[:, :10].to_numpy(), train_set.iloc[:, 10:].to_numpy()
    ts_x, ts_y = test_set.iloc[:, :10].to_numpy().T, test_set.iloc[:, 10:].to_numpy().T

    x_train, x_val, y_train, y_val = train_test_split(tr_x, tr_y, test_size=0.33)
    x_train, x_val = x_train.T, x_val.T
    y_train, y_val = y_train.T, y_val.T

    min_error, best_model, best_conf = sys.maxsize, None, None

    progress = tqdm(ParameterGrid(configs))
    for single_conf in progress:
        hidden = single_conf["hidden"]
        lambda_ = single_conf["regularization"]
        activation = single_conf["activation_fun"]

        model = ENeuralN(10, hidden, lambda_, activation)
        model.fit_cholesky(x_train, y_train)
        y_train_pred, y_val_pred = model(x_train), model(x_val)

        mse_train, mse_val = MSE(y_train, y_train_pred), MSE(y_val, y_val_pred)

        if mse_val < min_error:
            best_model = model
            best_conf = (single_conf, mse_train, mse_val)
            min_error = mse_val

        progress.set_postfix(minMSE=min_error)

    print("\nThe best configuration is ", best_conf[0])
    print("Train error ", best_conf[1], " Validation error", best_conf[2])

    y_test_pred = best_model(ts_x)
    test_error = MSE(ts_y, y_test_pred)
    print("Test error ", test_error)


def GridSearch_Time(train_path: str, configs: dict) -> DataFrame:
    train_set = pd.read_csv(train_path, header=None, index_col=0)

    # We have training set tr_x 1182 columns and 10 rows
    tr_x, tr_y = train_set.iloc[:, :10].to_numpy(), train_set.iloc[:, 10:].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(tr_x, tr_y, test_size=0.33)
    x_train, x_val = x_train.T, x_val.T

    all_configuration = []

    for single_conf in tqdm(ParameterGrid(configs)):
        hidden = single_conf["hidden"]
        lambda_ = single_conf["regularization"]
        activation = single_conf["activation_fun"]

        model = ENeuralN(10, hidden, lambda_, activation)

        start_train = time.time()
        model.fit_cholesky(x_train, y_train)
        end_train = time.time()

        start_inference = time.time()
        y_train_pred, y_val_pred = model(x_train), model(x_val)
        end_inference = time.time()

        mse_train, mse_val = MSE(y_train, y_train_pred), MSE(y_val, y_val_pred)

        current_conf = (mse_train, mse_val, single_conf,
                        {"train_time": end_train - start_train,
                         "inference_time": end_inference - start_inference})
        all_configuration.append(current_conf)

    rank = sorted(all_configuration, key=lambda conf: conf[1])
    best_conf = rank[0]

    print("\nThe best configuration is ", best_conf[0])
    print("Train error ", best_conf[1], " Validation error", best_conf[2])

    time_matrix = np.zeros((len(all_configuration), 3))

    for idx, config in enumerate(all_configuration):
        time_matrix[idx, 0] = config[2]["hidden"]
        time_matrix[idx, 1] = config[0]
        time_matrix[idx, 2] = config[1]

    return DataFrame(time_matrix, columns=["Hidden", "Train time", "Inference time"])
