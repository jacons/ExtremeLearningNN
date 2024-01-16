import datetime
from typing import Literal, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

from ExtremeNN import ENeuralN
from Utils import mse, sigmoid, get_gap_sol


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


def fit_cholesky(x_train: np.ndarray, y_train: np.ndarray,
                 hidden: int = 0,
                 lambda_: float = 0,
                 resevoir: np.ndarray = None,
                 w_star: np.ndarray = None,
                 mode: Literal["verbose", "testing"] = "verbose") -> Union[ENeuralN, dict]:
    """
    :param x_train: array X [ feature, examples ]
    :param y_train: array target [ 2, examples ]
    :param hidden: Hidden size
    :param lambda_: L2-Regularization term
    :param resevoir: Resevoir layer (random if not provided)
    :param w_star: 
    :param mode: if "testing", get_mse_residuals will not be computed
    """
    start = datetime.datetime.now()

    model = ENeuralN(hidden, lambda_, resevoir)
    model.fit_cholesky(x_train, y_train)
    end = datetime.datetime.now()

    if mode == "verbose":
        output = {
            "model": model,
            "elapsed_time": (end - start).microseconds,
            "gap_sol": get_gap_sol(model.w2, w_star)
        }
    else:
        output = model
    return output


def fit_sgd(x_train: np.ndarray, y_train: np.ndarray,
            hidden: int = 100,
            lambda_: float = 0,
            max_inters: int = None,
            eps: float = 0,
            resevoir: np.ndarray = None,
            w_star: np.ndarray = None,
            f_star: np.ndarray = None,
            mode: Literal["verbose", "testing"] = "verbose",
            ) -> Union[ENeuralN, dict]:
    """

    :param x_train: array X [ feature, examples ]
    :param y_train: array target [ 2, examples ]
    :param hidden: Hidden size
    :param lambda_: L2-Regularization term
    :param max_inters: Number of max iteration
    :param eps: gradient thresholds
    :param resevoir: Resevoir layer (random if not provided)
    :param w_star:
    :param f_star: 
    :param mode: if "testing", get_mse_residuals will not be computed
    """
    start = datetime.datetime.now()

    model = ENeuralN(hidden, lambda_, resevoir)
    weights, norm_grad = model.fit_SDG(x=x_train, y=y_train, max_iter=max_inters, eps=eps, testing=(mode == "testing"))

    end = datetime.datetime.now()

    if mode == "verbose":
        H = model.resevoir(x_train)
        dt = get_metrics(weights, y_train, H, w_star, f_star)

        output = {
            "model": model,
            "elapsed_time": (end - start).microseconds,
            "iterations": len(weights),
            "norm_grad": norm_grad,
            "metrics": dt,
        }
    else:
        output = model

    return output


def fit_fista(x_train: np.ndarray, y_train: np.ndarray,
              hidden: int = 100,
              lambda_: float = 0,
              max_inters: int = None,
              eps: float = 0,
              resevoir: np.ndarray = None,
              w_star: np.ndarray = None,
              f_star: np.ndarray = None,
              mode: Literal["verbose", "testing"] = "verbose",
              ) -> Union[ENeuralN, dict]:
    """
    :param x_train: array X [ feature, examples ]
    :param y_train: array target [ 2, examples ]
    :param hidden: Hidden size
    :param lambda_: L2-Regularization term
    :param max_inters: Number of max iteration
    :param eps: gradient thresholds
    :param resevoir: Resevoir layer (random if not provided)
    :param w_star:
    :param f_star: 
    :param mode: if "testing", get_mse_residuals will not be computed
    :return:
    """
    start = datetime.datetime.now()

    model = ENeuralN(hidden, lambda_, resevoir)
    weights, norm_grad = model.fit_fista2(x_train, y_train, max_inters, eps)

    end = datetime.datetime.now()

    if mode == "verbose":
        H = model.resevoir(x_train)
        dt = get_metrics(weights, y_train, H, w_star, f_star)

        output = {
            "model": model,
            "elapsed_time": (end - start).microseconds,
            "iterations": len(weights),
            "norm_grad": norm_grad,
            "metrics": dt,
        }
    else:
        output = model

    return output


def get_metrics(weights: list[np.ndarray],
                y_train: np.ndarray,
                H: np.ndarray,
                w_star: np.ndarray,
                f_star: np.ndarray):
    # tensor (num iteration) x (2) x (hidden)
    weights = np.asarray(weights)

    # tensor (num iteration) x (2) x (num examples)
    y_pred = np.matmul(weights, H)

    # tensor (num iteration) x (2) x (num examples)
    y_diff = y_pred - y_train[np.newaxis, :, :]

    # tensor (num iteration) x (2) x (h)
    w_diff = weights - w_star[np.newaxis, :, :]

    # tensor (num iteration) x (2) x (num examples)
    gap_pred = y_pred - f_star[np.newaxis, :, :]

    # tensor (num iteration) x (1)
    mse_errors = np.power(y_diff, 2).mean(axis=(1, 2))

    relative_gap_sol = norm(w_diff, axis=(1, 2), ord="fro") / norm(w_star, ord="fro")
    relative_gap_pred = norm(gap_pred, axis=(1, 2), ord="fro") / norm(f_star, ord="fro")

    results = pd.DataFrame({"MSE": mse_errors, "Gap_Sol": relative_gap_sol, "Gap_Pred": relative_gap_pred})
    results["iters"] = results.index
    results = results[["iters", "MSE", "Gap_Sol", "Gap_Pred"]].set_index("iters")

    return results


def get_results(model: ENeuralN, x: np.ndarray, y: np.ndarray):
    y_pred = model(x)
    return mse(y, y_pred)


def test_over_regularization(tr_x: np.ndarray, tr_y: np.ndarray, parameters: dict,
                             regs: list[float], resevoir_: list[float]):
    MAX_ITER = parameters["MAX_ITER"]
    PRECISION = parameters["PRECISION"]

    results = []

    for size_resevoir in resevoir_:
        resevoir = np.random.uniform(-1, 1, (size_resevoir, 10))
        H = sigmoid(resevoir @ tr_x)

        for LAMBDA_REG in regs:
            E = H @ H.T + np.power(LAMBDA_REG, 2) * np.eye(H.shape[0])

            # Perform the conditional number
            condition_number = np.linalg.cond(E)

            # Calculate the optimal solution
            start_optimal = datetime.datetime.now()
            w_star, _, _, _ = np.linalg.lstsq(E, H @ tr_y.T, rcond=-1)
            end_optimal = datetime.datetime.now()

            # -----  CHOLESKY -----
            cholesky = fit_cholesky(tr_x, tr_y, lambda_=LAMBDA_REG, resevoir=resevoir, w_star=w_star.T, mode="verbose")
            f_star = cholesky["model"](tr_x)
            # -----  CHOLESKY -----

            # -----  CLASSICAL SGD -----
            classical_sgd = fit_sgd(x_train=tr_x, y_train=tr_y, lambda_=LAMBDA_REG,
                                    max_inters=MAX_ITER, eps=PRECISION,
                                    resevoir=resevoir, w_star=w_star.T, f_star=f_star, mode="verbose")
            # -----  CLASSICAL SGD -----

            # ----- FISTA -----
            fista = fit_fista(x_train=tr_x, y_train=tr_y, lambda_=LAMBDA_REG,
                              max_inters=MAX_ITER, eps=PRECISION,
                              resevoir=resevoir, w_star=w_star.T, f_star=f_star, mode="verbose")

            # ----- FISTA -----

            result = {
                "Size": size_resevoir,
                "Lambda": LAMBDA_REG,
                "Conditional number": condition_number,
                "Optimal MSE": mse(w_star.T @ H, tr_y),
                "Optimal Time": (end_optimal - start_optimal).microseconds,

                "Cholesky MSE": mse(cholesky["model"](tr_x), tr_y),
                "Cholesky Time": cholesky["elapsed_time"],
                "Cholesky Gap_sol": cholesky["gap_sol"],

                "SGD MSE": mse(classical_sgd["model"](tr_x), tr_y),
                "SGD Time": classical_sgd["elapsed_time"],
                "SGD Iterations": classical_sgd["iterations"],
                "SGD Gap_sol": classical_sgd["metrics"]["Gap_Sol"].tail(1).values[0],
                "SGD Gap_pred": classical_sgd["metrics"]["Gap_Pred"].tail(1).values[0],


                "Fista MSE": mse(fista["model"](tr_x), tr_y),
                "Fista Time": fista["elapsed_time"],
                "Fista Iterations": fista["iterations"],
                "Fista Gap_sol": fista["metrics"]["Gap_Sol"].tail(1).values[0],
                "Fista Gap_pred": fista["metrics"]["Gap_Pred"].tail(1).values[0],

            }
            results.append(result)

    return pd.DataFrame(results).T
