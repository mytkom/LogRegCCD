"""Utility functions"""
from matplotlib import pyplot as plt

from src.measures import AUCROC, AUCPR, FMeasure, BalancedAccuracy

# pylint: disable=invalid-name
def evaluate_model(model, X, y, model_name):
    """Evaluate models using measure classes"""
    roc_auc = AUCROC()
    pr_auc = AUCPR()
    f1 = FMeasure()
    bal_acc = BalancedAccuracy()

    if model_name == "LogRegCCD":
        y_prob = model.predict_proba(X)
        y_pred = model.predict(X)
    else:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

    return {
        'ROC AUC': roc_auc(y, y_prob),
        'PR AUC': pr_auc(y, y_prob),
        'F1 Score': f1(y, y_pred),
        'Balanced Accuracy': bal_acc(y, y_pred)
    }


def plot_lasso_path(lambdas, betas):
    """Plot the Lasso regularization"""
    plt.figure(figsize=(8, 6))
    for i in range(0, betas.shape[0]):
        plt.plot(lambdas, betas[i, :], label=f'{i + 1}')

    plt.xscale("log")
    plt.xlabel('lambda')
    plt.ylabel('Beta coefficients')
    plt.title('Lasso Regularization Path')
    plt.grid(True)
    plt.show()
