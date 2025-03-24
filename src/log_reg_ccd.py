import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.model_selection import KFold
from .measures import ProbMeasure, ClassMeasure

class LogRegCCD:
    """
    Implementation of regularized logistic regression for binary classification problem using
    cyclic coordinate descent (CCD) method, based solely on https://www.jstatsoft.org/article/view/v033i01
    """

    def __init__(self, verbose: bool = False) -> None:
        self.is_fitted: bool = False
        self.lambdas: NDArray[np.float64] = np.empty(0)
        self.betas: NDArray[np.float64] = np.empty(0)
        self.best_beta: NDArray[np.float64] = np.empty(0)
        self.verbose: bool = verbose

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int_],
        eps=1e-3,
        lam_max=10,
        lam_count=100,
        k_fold=10,
    ) -> None:
        """
        Calculates beta (1 x p+1) coefficients vector for X_train (n x p) matrix of standardized features
        and y_train (n x 1) vector of binary class labels (int 0 or 1). Input space R^p. After fitting
        predict method can be used to predict class of new observation.

        Parameters:
            X_train (NDArray[np.float64]): (n x p) matrix of n observations, each one have p features.
            y_train (NDArray[np.int]): (n x 1) vector of n binary class labels (0 or 1).
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise RuntimeError(
                f"LogRegCCD fit: X_train ({X_train.shape[0]}) and y_train({y_train.shape[0]}) shapes does not match."
            )

        in_features = X_train.shape[1]
        self._log(f"Number of features: {in_features}")

        if lam_count == 1:
            self.lambdas = np.array([lam_max])
            self.betas = np.empty((1, in_features + 1))
            self.betas[0] = self._coordinate_descent(
                X_train, y_train, lam_max, np.zeros(in_features + 1)
            )
            self.best_beta = self.betas[0]
            self._log(f"Lambda {lam_max}, Beta: {self.best_beta}")
        else:
            self.lambdas = np.logspace(
                np.log10(eps * lam_max), np.log10(lam_max), lam_count, dtype=np.float64
            )[::-1]
            self._log(f"Lambdas space: {self.lambdas}")

            lambda_mse = {lam: [] for lam in self.lambdas}
            lambda_betas = { lam: [] for lam in self.lambdas }

            # k-fold cross-validation
            kf = KFold(n_splits=k_fold, shuffle=True)
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
                self._log(f"Fold {fold + 1}")

                cw_X_train, cw_y_train = X_train[train_idx], y_train[train_idx]
                cw_X_valid, cw_y_valid = X_train[valid_idx], y_train[valid_idx]

                self._log(f"Train shape: {cw_X_train.shape}")
                self._log(f"Validation shape: {cw_X_valid.shape}")

                # Initialize with zeros
                current_beta = np.zeros(in_features + 1)


                for lam in self.lambdas:
                    new_beta = self._coordinate_descent(
                        cw_X_train, cw_y_train, lam, current_beta
                    )

                    # Validation step
                    val_predictions = np.clip(self._sigmoid(-(cw_X_valid @ new_beta[1:]) + new_beta[0]), 1e-10, 1 - 1e-10)
                    # Mean deviance
                    val_loss = -2 * np.mean(cw_y_valid * np.log(val_predictions) + (1 - cw_y_valid) * np.log(1 - val_predictions))

                    self._log(
                        f"Lambda {lam}: Fold {fold+1} Validation Loss = {val_loss:.4f} Beta: {new_beta}"
                    )

                    # Store results
                    lambda_mse[lam].append(val_loss)
                    lambda_betas[lam].append(new_beta)

                    # Warm start for next lambda
                    current_beta = new_beta

            # Compute average MSE for each lambda
            avg_mse = {lam: np.mean(mse_list) for lam, mse_list in lambda_mse.items()}

            # Select lambda with lowest avg MSE
            best_lambda = min(avg_mse, key=avg_mse.get)
            self._log(f"Avg MSE of best lambda ({best_lambda}): {avg_mse[best_lambda]}")
            self.betas = np.empty((lam_count, in_features + 1))
            current_beta = np.zeros(in_features + 1)
            for idx, lam in enumerate(self.lambdas):
                self.betas[idx] = self._coordinate_descent(
                    X_train, y_train, lam, current_beta
                )
                if lam == best_lambda:
                    self.best_beta = self.betas[idx]

    def validate(self, X_valid, y_valid, measure: ProbMeasure | ClassMeasure):
        method = self.predict if isinstance(measure, ClassMeasure) else self.predict_proba
        max_measure_value = float('-inf')
        max_measure_idx = -1
        max_measure_lam = -1
        for idx, lam in enumerate(self.lambdas):
            value = measure(y_valid, method(X_valid, beta_idx=idx))
            if value > max_measure_value:
                max_measure_value = value
                max_measure_idx = idx
                max_measure_lam = lam
        self.best_beta = self.betas[max_measure_idx]
        self._log(f"Max measure value: {max_measure_value:.4f} for lambda: {max_measure_lam}, beta values: {self.best_beta}")

    def predict_proba(self, X, beta_idx=None):
        beta = self.best_beta if not beta_idx else self.betas[beta_idx]
        return self._sigmoid((-X @ beta[1:]) + beta[0])

    def predict(self, X, beta_idx=None):
        beta = self.best_beta if not beta_idx else self.betas[beta_idx]
        return ((beta[0] + (X @ beta[1:])) > 0.0).astype("int")

    def _coordinate_descent(
        self, X, y, lam: float, beta: NDArray[np.float64], max_iter=1000, eps=1e-10
    ) -> NDArray[np.float64]:
        n = X.shape[0]
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        l = float("inf")

        for i in range(0, max_iter):
            atol = 1e-3
            posteriors = np.clip(self._sigmoid(-(X @ beta)), atol, 1.-atol)
            weights = posteriors * (1 - posteriors)

            # Calculate common part of partial derivatives over beta_j
            y_minus_posteriors = y - posteriors
            new_beta = beta.copy()

            # Calculate partial derivatives of beta_j for every j
            for j in range(0, beta.shape[0]):
                denom = (np.square(X[:, j]).T @ weights)
                result = ((X[:, j].T @ y_minus_posteriors) + (beta[j] * denom))
                if j == 0:
                    # Intercept shouldn't be regularized
                    new_beta[j] = result / denom
                else:
                    # Penalty included using soft threshold function
                    new_beta[j] = self._soft_thresh(result, n * np.float64(lam)) / denom
            if not np.isclose(np.linalg.norm(new_beta), 0.):
                new_beta = new_beta / np.linalg.norm(new_beta)

            l_old = l
            # Exact loss
            # l = (-np.sum((y @ (X @ new_beta)) - np.log(1 + np.exp(np.clip(X @ new_beta, -500, 500)))) / n) + lam * np.linalg.norm(new_beta, ord=1)
            # Quadratic approx
            l = -1/(2*n) * weights.T @ np.square(X @ beta + (y_minus_posteriors / weights) - X @ new_beta)        

            # Stop condition
            if abs(l - l_old) < eps:
                self._log(f"Early stopping Coordinate Descent in iteration: {i}")
                break

            beta = new_beta

        return beta

    # def _sigmoid(self, s, clip=300):
    #     return 1.0 / (1.0 + np.exp(np.clip(s, -clip, clip)))

    # https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _negative_sigmoid(self, x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)


    def _sigmoid(self, x):
        positive = x >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains junk hence will be faster to allocate
        # Zeros has to zero-out the array after allocation, no need for that
        # See comment to the answer when it comes to dtype
        result = np.empty_like(x, dtype=np.float64)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])

        return result
  
    def _soft_thresh(self, z, gamma):
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

    def plot_lasso_path(self):
        """
        Plots the Lasso path with -log(lambda) on the x-axis and beta values on the y-axis.
        """
        plt.figure(figsize=(8, 6))
        for i in range(1, self.betas.shape[1]):
            plt.plot(self.lambdas, self.betas[:, i], label=f'{i+1}')
        
        plt.xscale("log")
        plt.xlabel('lambda')
        plt.ylabel('Beta coefficients')
        plt.title('Lasso Regularization Path')
        plt.grid(True)
        plt.show()

    def _log(self, message) -> None:
        """
        Logs message to stdout using print method only if verbose flag of the class is set to true.
        """
        if self.verbose:
            print("LogRegCCD: ", message)
