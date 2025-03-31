"""This module contains implementation of regularized logistic regression using CCD."""

# pylint: disable=too-many-arguments, too-many-positional-arguments, invalid-name, too-many-locals

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.model_selection import KFold
from .measures import ProbMeasure, ClassMeasure

EPS = 1e-8


class LogRegCCD:
    """
    Implementation of regularized logistic regression for binary classification problem using cyclic
    coordinate descent (CCD) method, based solely on https://www.jstatsoft.org/article/view/v033i01
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initializes the logistic regression model.

        Parameters:
            verbose (bool): Whether to enable logging.
        """
        self.is_fitted: bool = False
        self.lambdas: NDArray[np.float64] = np.empty(0)
        self.betas: NDArray[np.float64] = np.empty(0)
        self.best_beta: NDArray[np.float64] = np.empty(0)
        self.verbose: bool = verbose

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int_],
        eps: float = 1e-3,
        lam_max: float = 10.0,
        lam_count: int = 100,
        k_fold: int = 10,
    ) -> None:
        """
        Fits the logistic regression model using CCD. Log-scale space from eps * lam_max to lam_max.

        Parameters:
            X_train (NDArray[np.float64]): Feature matrix (n x p) where n is the number of samples
                                           and p is the number of features.
            y_train (NDArray[np.int_]): Binary class labels (0 or 1), shape (n,).
            eps (float): Smallest lambda value as a fraction of lam_max.
            lam_max (float): Maximum lambda value for regularization.
            lam_count (int): Number of lambda values to evaluate.
            k_fold (int): Number of folds for cross-validation.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise RuntimeError(
                f"LogRegCCD fit: X_train ({X_train.shape[0]}) "
                + f"and y_train({y_train.shape[0]}) shapes do not match."
            )

        in_features = X_train.shape[1]
        self._log(f"Number of features: {in_features}")

        # If only one lambda value calculate coordinate descent for it
        if lam_count == 1:
            self.lambdas = np.array([lam_max])
            self.betas = np.empty((1, in_features + 1))
            self.betas[0] = self._coordinate_descent(
                X_train, y_train, lam_max, np.zeros(in_features + 1)
            )
            self.best_beta = self.betas[0]
            self._log(f"Lambda {lam_max}, Beta: {self.best_beta}")
        else:
            # Lambda values in log-scale
            self.lambdas = np.logspace(
                np.log10(eps * lam_max), np.log10(lam_max), lam_count, dtype=np.float64
            )[::-1]
            self._log(f"Lambdas space: {self.lambdas}")

            # Perform k-fold cross-validation
            avg_loss, _ = self._k_fold_cross_validation(
                X_train, y_train, k_fold, self.lambdas, in_features
            )

            # Find the best lambda
            best_lambda = min(avg_loss, key=avg_loss.get)
            self._log(
                f"Avg Mean Deviance of best lambda ({best_lambda}): {avg_loss[best_lambda]}"
            )

            # Perform CCD on whole training dataset and save betas
            self.betas = np.empty((lam_count, in_features + 1))
            current_beta = np.zeros(in_features + 1)
            for idx, lam in enumerate(self.lambdas):
                self.betas[idx] = self._coordinate_descent(
                    X_train, y_train, lam, current_beta
                )
                if lam == best_lambda:
                    self.best_beta = self.betas[idx]

    def validate(
        self,
        X_valid: NDArray[np.float64],
        y_valid: NDArray[np.int_],
        measure: ProbMeasure | ClassMeasure,
    ) -> tuple:
        """
        Validates all betas calculated using cyclical coordinate descent for logistic regression
        for the specified measure class. It chooses and returns the best beta and the corresponding
        lambda based on the maximum validation measure value.

        Parameters:
            X_valid (NDArray[np.float64]): Validation feature matrix (n x p).
            y_valid (NDArray[np.int_]): Validation target vector (n,).
            measure (ProbMeasure | ClassMeasure): The measure function to evaluate predictions.

        Returns:
            tuple: (best_lambda (float), best_beta (NDArray[np.float64]))
                best_lambda: The lambda value that gave the highest measure.
                best_beta: The beta values corresponding to the best lambda.
        """
        method = (
            self.predict if isinstance(measure, ClassMeasure) else self.predict_proba
        )

        max_measure_value = float("-inf")
        max_measure_idx = -1
        max_measure_lam = -1

        for idx, lam in enumerate(self.lambdas):
            value = measure(y_valid, method(X_valid, beta_idx=idx))

            if value > max_measure_value:
                max_measure_value = value
                max_measure_idx = idx
                max_measure_lam = lam

        self.best_beta = self.betas[max_measure_idx]

        self._log(
            f"Max measure value: {max_measure_value:.4f} for lambda: {max_measure_lam}, "
            f"beta values: {self.best_beta}"
        )

        return max_measure_lam, self.best_beta

    def predict_proba(
        self, X: NDArray[np.float64], beta_idx: int | None = None
    ) -> NDArray[np.float64]:
        """
        Predicts probabilities using the logistic model.

        Parameters:
            X (NDArray[np.float64]): Feature matrix (n x p).
            beta_idx (int, optional): Index of beta coefficients to use.

        Returns:
            NDArray[np.float64]: Probability predictions.
        """
        beta = self.best_beta if beta_idx is None else self.betas[beta_idx]
        return self._sigmoid((X @ beta[1:]) + beta[0])

    def predict(
        self, X: NDArray[np.float64], beta_idx: int | None = None
    ) -> NDArray[np.int_]:
        """
        Predicts binary class labels.

        Parameters:
            X (NDArray[np.float64]): Feature matrix (n x p).
            beta_idx (int, optional): Index of beta coefficients to use.

        Returns:
            NDArray[np.int_]: Binary class predictions (0 or 1).
        """
        beta = self.best_beta if beta_idx is None else self.betas[beta_idx]
        return ((beta[0] + (X @ beta[1:])) > 0.0).astype(np.int_)

    def _k_fold_cross_validation(
        self, X_train, y_train, k_fold, lambdas, in_features
    ) -> tuple:
        """
        Performs k-fold cross-validation to evaluate loss for different lambda values.

        Args:
            X_train (NDArray[np.float64]): Training feature matrix.
            y_train (NDArray[np.int_]): Training target vector.
            k_fold (int): Number of cross-validation folds.
            lambdas (NDArray[np.float64]): Lambda values to evaluate.
            in_features (int): Number of features in the model.

        Returns:
            tuple of dicts: avg_loss and lambda_betas containing losses and betas for each lambda.
        """
        lambda_losses = {lam: [] for lam in lambdas}
        lambda_betas = {lam: [] for lam in lambdas}

        kf = KFold(n_splits=k_fold, shuffle=True)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            self._log(f"Fold {fold + 1}")
            cw_X_train, cw_y_train = X_train[train_idx], y_train[train_idx]
            cw_X_valid, cw_y_valid = X_train[valid_idx], y_train[valid_idx]

            # Initialize with zeros
            current_beta = np.zeros(in_features + 1)

            for lam in lambdas:
                new_beta = self._coordinate_descent(
                    cw_X_train, cw_y_train, lam, current_beta
                )
                val_loss = self._evaluate_validation_loss(
                    cw_X_valid, cw_y_valid, new_beta
                )

                self._log(
                    f"Lambda {lam}: Fold {fold+1} Validation Loss = {val_loss:.4f} Beta: {new_beta}"
                )

                lambda_losses[lam].append(val_loss)
                lambda_betas[lam].append(new_beta)

                # Warm start for next lambda
                current_beta = new_beta

        # Calculate average losses
        avg_loss = {lam: np.mean(loss_list) for lam, loss_list in lambda_losses.items()}
        return avg_loss, lambda_betas

    def _evaluate_validation_loss(self, X_valid, y_valid, beta) -> float:
        """
        Evaluates the mean deviance loss on the validation set.

        Args:
            X_valid (NDArray[np.float64]): Validation feature matrix.
            y_valid (NDArray[np.int_]): Validation target vector.
            beta (NDArray[np.float64]): Model parameters.

        Returns:
            float: Validation loss.
        """
        val_predictions = np.clip(
            self._sigmoid((X_valid @ beta[1:] + beta[0])), 1e-5, 1 - 1e-5
        )
        return np.mean(
            -2 * y_valid * np.log(val_predictions)
            + (1 - y_valid) * np.log(1 - val_predictions)
        )

    def _coordinate_descent(
        self, X, y, lam: float, beta: NDArray[np.float64], max_iter=1000, eps=1e-10
    ) -> NDArray[np.float64]:
        """
        Performs cyclical coordinate descent to optimize beta coefficients for logistic regression.

        The method iteratively updates each beta coefficient using a quadratic approximation
        and applies L1 regularization (lasso penalty) to all but the intercept term.

        Args:
            X: Feature matrix.
            y: Target vector.
            lam: Regularization parameter (lambda).
            beta: Initial beta coefficients.
            max_iter: Maximum number of iterations.
            eps: Convergence threshold for early stopping.

        Returns:
            NDArray[np.float64]: Optimized beta coefficients.
        """
        n, _ = X.shape
        X_ext = np.hstack((np.ones((n, 1)), X))  # Add intercept term
        atol = 1e-5
        l_old = float("inf")
        weights = np.repeat(0.25, n)

        for i in range(max_iter):
            posteriors = np.clip(self._sigmoid(X_ext @ beta), atol, 1.0 - atol)

            # alternatively instead of 0.25 for every weight
            # weights = posteriors * (1 - posteriors)

            y_minus_posteriors = y - posteriors
            new_beta = beta.copy()

            # Compute denominators for all beta coefficients
            denom = (
                np.einsum("ij,i,ij->j", X_ext, weights, X_ext) + 1e-12
            )  # Avoid division by zero

            # Compute soft-thresholding step efficiently
            num = X_ext.T @ y_minus_posteriors
            num += beta * denom
            num[1:] = self._soft_thresh(num[1:], n * lam)  # Exclude intercept

            new_beta = num / denom

            # Compute loss using the quadratic approximation
            l = self._quadratic_approx_loss(
                X_ext, beta, new_beta, weights, y_minus_posteriors, lam
            )

            # Check for convergence
            if abs(l - l_old) < eps:
                self._log(f"Early stopping Coordinate Descent in iteration: {i}")
                break

            beta, l_old = new_beta, l  # Update beta and loss

        return beta

    def _exact_loss(self, X, y, beta, lam) -> float:
        """Computes the exact logistic regression loss with L1 regularization."""
        n = X.shape[0]
        logits = X @ beta
        log_likelihood = np.sum(
            y * logits - np.log(1 + np.exp(np.clip(logits, -500, 500)))
        )
        return (-log_likelihood / n) + lam * np.linalg.norm(beta[1:], ord=1)

    def _quadratic_approx_loss(
        self, X, beta, new_beta, weights, y_minus_posteriors, lam
    ) -> float:
        """Computes the quadratic approximation of the logistic regression loss."""
        n = X.shape[0]
        return (-1 / (2 * n)) * weights.T @ np.square(
            X @ beta + (y_minus_posteriors / weights) - X @ new_beta
        ) + lam * np.linalg.norm(new_beta[1:], ord=1)

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
            plt.plot(self.lambdas, self.betas[:, i], label=f"{i+1}")

        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("Beta coefficients")
        plt.title("Lasso Regularization Path")
        plt.grid(True)
        plt.show()

    def plot(
        self,
        X_valid: NDArray[np.float64],
        y_valid: NDArray[np.int_],
        measure: ProbMeasure | ClassMeasure,
        file_path: str | None = None,
    ):
        """
        Plots how the given evaluation measure changes with lambda.
        The maximum measure value and corresponding lambda are highlighted.

        Parameters:
            X_valid (NDArray[np.float64]): Validation feature matrix.
            y_valid (NDArray[np.int_]): Validation target vector.
            measure (ProbMeasure | ClassMeasure): Evaluation measure.
            file_path (str | None): Path to save the plot.
        """
        method = (
            self.predict if isinstance(measure, ClassMeasure) else self.predict_proba
        )

        values = []
        for idx, _ in enumerate(self.lambdas):
            values.append(measure(y_valid, method(X_valid, beta_idx=idx)))

        max_idx = np.argmax(values)
        best_lambda = self.lambdas[max_idx]
        best_value = values[max_idx]

        plt.figure(figsize=(8, 6))
        plt.plot(
            self.lambdas, values, marker="o", linestyle="-", color="b", label="Measure"
        )

        plt.scatter(best_lambda, best_value, color="red", zorder=3, label="Max Measure")
        plt.annotate(
            f"λ={best_lambda:.2e}\nValue={best_value:.4f}",
            xy=(best_lambda, best_value),
            xytext=(1.2 * best_lambda, best_value),
            arrowprops={"arrowstyle": "->", "color": "red"},
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "edgecolor": "red",
                "facecolor": "white",
            },
        )

        plt.xscale("log")
        plt.xlabel("Lambda")
        plt.ylabel(measure)
        plt.title(f"{measure} vs. Lambda")
        plt.legend()
        plt.grid(True)

        if file_path:
            plt.savefig(file_path)

        plt.show()

    def _log(self, message) -> None:
        """
        Logs message to stdout using print method only if verbose flag of the class is set to true.
        """
        if self.verbose:
            print("LogRegCCD: ", message)
