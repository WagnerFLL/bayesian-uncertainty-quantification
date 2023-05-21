import pymc3 as pm
import numpy as np
import arviz as az
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score


class BayesianExperiment:
    def __init__(
        self, 
        data, 
        target, 
        n_splits=5, 
        hdi_prob = 0.95,
        random_state=42, 
        n_iterations=20000, 
        tuning_iterations=12000, 
        probability_threshold=0.5,
        coverage_quantiles=[0.25, 0.5, 0.75]
    ):
        self.data = data
        self.target = target
        self.n_splits = n_splits
        self.hdi_prob = hdi_prob
        self.random_state = random_state
        self.n_iterations = n_iterations
        self.tuning_iterations = tuning_iterations
        self.probability_threshold = probability_threshold
        self.coverage_quantiles = coverage_quantiles
        
        self.features = data.drop(target, axis=1).columns.tolist()

    def predict_posterior(self, trace, X):
        X = X.copy()

        data = dict(intercept=trace['beta_i'])
        for f in self.features:
            data[f] = trace[f"beta_{f}"]

        coef_samples = pd.DataFrame(data)

        X.insert(0, "intercept", 1)
        linear_combinations = np.matmul(coef_samples, X.T)
        probabilities = 1 / (1 + np.exp(-linear_combinations))

        return probabilities

    def fit_model(self, X_train, y_train):
        lower = -10**6
        higher = 10**6

        with pm.Model() as model:
            beta_i = pm.Uniform('beta_i', lower=lower, upper=higher)

            betas = dict()
            for f in self.features:
                betas[f"beta_{f}"] = pm.Uniform(f"beta_{f}", lower=lower, upper=higher)

            linear_combination = beta_i
            for f in self.features:
                linear_combination += betas[f"beta_{f}"] * X_train[f]

            p = pm.Deterministic('p', pm.math.sigmoid(linear_combination))

            observed = pm.Bernoulli("target", p, observed=y_train)
            start = pm.find_MAP()
            step = pm.Metropolis()

            trace = pm.sample(self.n_iterations, step=step, start=start)
            burned_trace = trace[self.tuning_iterations:]

        return burned_trace

    def run(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]

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

            trace = self.fit_model(X_train, y_train)
            posterior = self.predict_posterior(trace, X_test)

            bound = az.hdi(posterior.values, hdi_prob=self.hdi_prob)
            result = pd.DataFrame(dict(
                lb=bound[:, 0],
                ub=bound[:, 1],
                pred_proba=posterior.mean(),
                pred_class=(posterior.mean() > self.probability_threshold).astype(int),
                interval_width=bound[:, 1] - bound[:, 0],
                y_true=y_test.values
            ))

            accuracy = accuracy_score(result.y_true, result.pred_class)
            metrics["overall_accuracy"].append(accuracy)

            f1 = f1_score(result.y_true, result.pred_class)
            metrics["overall_f1_score"].append(f1)

            for q in self.coverage_quantiles:
                width_top_q = result.interval_width.quantile(q)
                reliable = result.loc[result.interval_width < width_top_q]

                accuracy = accuracy_score(reliable.y_true, reliable.pred_class)
                metrics[f"top_{int(q * 100)}_accuracy"].append(accuracy)

                f1 = f1_score(reliable.y_true, reliable.pred_class)
                metrics[f"top_{int(q * 100)}_f1_score"].append(f1)

        self.metrics_ = pd.DataFrame(metrics).describe().T[["mean", "std"]]
        self.trace_ = trace
        
        return self.metrics_
