"""Module providing supported measures."""

from abc import ABC, abstractmethod
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

class ClassMeasure(ABC):
    """Abstract base class for classification measures."""

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass

    @abstractmethod
    def __str__(self):
        pass

class ProbMeasure(ABC):
    """Abstract base class for probability-based measures."""

    @abstractmethod
    def __call__(self, y_true, y_prob):
        pass

    @abstractmethod
    def __str__(self):
        pass

class Precision(ClassMeasure):
    """Precision score metric."""

    def __call__(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def __str__(self):
        return 'Precision'

class Recall(ClassMeasure):
    """Recall score metric."""

    def __call__(self, y_true, y_pred):# -> Float | ndarray[Any, Any]:
        return recall_score(y_true, y_pred)

    def __str__(self):
        return 'Recall'

class FMeasure(ClassMeasure):
    """F1-score metric."""

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

    def __str__(self):
        return 'F-measure'

class BalancedAccuracy(ClassMeasure):
    """Balanced accuracy score metric."""

    def __call__(self, y_true, y_pred):
        return balanced_accuracy_score(y_true, y_pred)

    def __str__(self):
        return 'Balanced accuracy'

class AUCROC(ProbMeasure):
    """Area Under the Receiver Operating Characteristic Curve metric."""

    def __call__(self, y_true, y_prob):
        return roc_auc_score(y_true, y_prob)

    def __str__(self):
        return 'AUC ROC'

class AUCPR(ProbMeasure):
    """Area Under the Precision-Recall Curve metric."""

    def __call__(self, y_true, y_prob):
        return average_precision_score(y_true, y_prob)

    def __str__(self):
        return 'AUC PR'
