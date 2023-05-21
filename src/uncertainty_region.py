import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


class UncertaintyRegionExperiment:
    def __init__(self, data, target, n_splits=5, coverage_quantiles=[0.25, 0.5, 0.75], random_state=42):
        self.data = data
        self.target = target
        self.n_splits = n_splits
        self.coverage_quantiles = coverage_quantiles
        self.random_state = random_state

    def get_confidence(self, pred, y_true, k):
        ap = y_true.mean()

        lb = pred[:, 1] - k * np.sqrt(pred[:, 1] * pred[:, 0])
        ub = pred[:, 1] + k * np.sqrt(pred[:, 1] * pred[:, 0])

        confidence = np.logical_or(
            lb > ap, ap > ub
        )

        return np.array([
            "reliable" if c else "unreliable"
            for c in confidence
        ])

    def get_closest_k(self, y_pred, y_test, target_size):
        param_interval = np.arange(0, 3, .01)
        sizes = [
            (self.get_confidence(y_pred, y_test, k) == "reliable").mean()
            for k in param_interval
        ]

        index = np.argmin(np.abs(np.array(sizes) - target_size))
        return param_interval[index]

    def get_top_n_accuracy(self, test, pred_proba, y_test, n):
        k = self.get_closest_k(pred_proba, y_test, n)
        test = test.assign(
            y_true=y_test,
            pred=pred_proba[:, 1],
            pred_class=(pred_proba[:, 1] >= 0.5).astype("int"),
            confidence=self.get_confidence(pred_proba, y_test, k)
        )

        reliable = test.loc[test.confidence == "reliable"]

        accuracy = accuracy_score(
            y_true=reliable.y_true,
            y_pred=reliable.pred_class
        )
        f1 = f1_score(
            y_true=reliable.y_true,
            y_pred=reliable.pred_class
        )
        return accuracy, f1

    def run(self):
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        model = LogisticRegression(max_iter=1000, random_state=self.random_state)

        accuracy_dict = {
            'overall_accuracies': [],
            'overall_f1_scores': []
        }
        for q in self.coverage_quantiles:
            accuracy_dict[f"top_{int(q * 100)}_accuracies"] = []
            accuracy_dict[f"top_{int(q * 100)}_f1_scores"] = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred_class = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred_class)
            accuracy_dict["overall_accuracies"].append(accuracy)

            f1 = f1_score(y_test, pred_class)
            accuracy_dict["overall_f1_scores"].append(f1)

            pred_proba = model.predict_proba(X_test)

            for q in self.coverage_quantiles:
                accuracy, f1 = self.get_top_n_accuracy(
                    X_test, pred_proba, y_test, q
                )
                accuracy_dict[f"top_{int(q * 100)}_accuracies"].append(accuracy)
                accuracy_dict[f"top_{int(q * 100)}_f1_scores"].append(f1)

        return pd.DataFrame(accuracy_dict).describe().T[["mean", "std"]]
