import bisect

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score


class InductiveConformalPredictor():
    def __init__(self, predictor):
        self.predictor = predictor
        
        self._le = LabelEncoder()
        self.classes = self._le.fit_transform(predictor.classes_)

    def fit(self, X, y):
        self.calibration_score = self._uncertainty_conformity_score(X)
        self.calibration_class = self._le.transform(y)
        return self

    def _uncertainty_conformity_score(self, data):
        uncertainty_score = 1 - self.predictor.predict_proba(data)
        return uncertainty_score

    def predict_proba(self, X, mondrian=True):
        conformity_score = self._uncertainty_conformity_score(X)
        conformal_pred = np.zeros(conformity_score.shape)

        for c in self.classes:
            if mondrian:
                calibration_filt = self.calibration_score[
                    self.calibration_class == c
                ]
                calib = calibration_filt[:, c]
            else:
                calib = self.calibration_score[
                    range(len(self.calibration_class)), 
                    self.calibration_class
                ]

            sorted_calib = np.sort(calib)
            conformal_pred[:, c] = [
                float(bisect.bisect(sorted_calib, x))/len(calib)
                for x in conformity_score[:, c]
            ]

        return conformal_pred

    def predict(self, X, mondrian=True, alpha=0.05):
        _conformal_proba = self.predict_proba(X=X, mondrian=mondrian)
        conformal_pred = (_conformal_proba > alpha).astype(int)

        mlb = MultiLabelBinarizer()
        mlb.fit([self._le.classes_])
        pred = mlb.inverse_transform(conformal_pred)

        return pred
    
    
class ConformalInferenceExperiment:
    def __init__(self, data, target, n_splits=5, coverage_quantiles=[0.25, 0.5, 0.75], random_state=42):
        self.data = data
        self.target = target
        self.n_splits = n_splits
        self.coverage_quantiles = coverage_quantiles
        self.random_state = random_state

    def get_coverage(self, cfp, X_test, alpha_interval):
        coverage = np.array([
            np.array([len(y_set) == 1
                      for y_set in cfp.predict(X_test, alpha=alpha)]).mean()
            for alpha in alpha_interval
        ])
        return coverage

    def run(self):
        y = self.data[self.target]
        X = self.data.drop(self.target, axis=1)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)

        metrics = {
            'overall_accuracy': [],
            'overall_f1_score': []
        }
        for q in self.coverage_quantiles:
            metrics[f"top_{int(q * 100)}_accuracy"] = []
            metrics[f"top_{int(q * 100)}_f1_score"] = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, test_size=0.3,
                stratify=y_train, random_state=self.random_state
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cfp = InductiveConformalPredictor(predictor=model)
            cfp.fit(X_calib, y_calib)

            accuracy = accuracy_score(y_test, y_pred)
            metrics["overall_accuracy"].append(accuracy)

            f1 = f1_score(y_test, y_pred)
            metrics["overall_f1_score"].append(f1)

            alpha_interval = np.arange(0, .9, .01)
            coverage = self.get_coverage(cfp, X_test, alpha_interval)

            for q in self.coverage_quantiles:
                index = np.argmin(np.abs(coverage - q))

                y_test_sets = cfp.predict(X_test, alpha=alpha_interval[index])
                indexes = np.where(np.vectorize(len)(y_test_sets) == 1)

                accuracy = accuracy_score(y_test.values[indexes], y_pred[indexes])
                metrics[f"top_{int(q * 100)}_accuracy"].append(accuracy)

                f1 = f1_score(y_test.values[indexes], y_pred[indexes])
                metrics[f"top_{int(q * 100)}_f1_score"].append(f1)

        return pd.DataFrame(metrics).describe().T[["mean", "std"]]
