from abc import ABC, abstractmethod
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

class ClassMeasure(ABC):
    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass

class ProbMeasure(ABC):
    @abstractmethod
    def __call__(self, y_true, y_prob):
        pass

class Precision(ClassMeasure):
    def __call__(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

class Recall(ClassMeasure):
    def __call__(self, y_true, y_pred):
        return recall_score(y_true, y_pred)

class FMeasure(ClassMeasure):
    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

class BalancedAccuracy(ClassMeasure):
    def __call__(self, y_true, y_pred):
        return balanced_accuracy_score(y_true, y_pred)

class AUCROC(ProbMeasure):
    def __call__(self, y_true, y_prob):
        return roc_auc_score(y_true, y_prob)

class AUCPR(ProbMeasure):
    def __call__(self, y_true, y_prob):
        return average_precision_score(y_true, y_prob)