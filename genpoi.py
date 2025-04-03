import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

from pymc3.distributions.dist_math import bound, factln, logpow
from pymc3.distributions.distribution import draw_values, generate_samples
from pymc3.theanof import intX

EPSILON=0.01

def MSE(y_true, y_pred):
    return np.mean((np.square(y_true - y_pred)))

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def MASE(y:np.array, ypred:np.array, season:int=1):
    return MAE(y, ypred) / MAE(np.array(y[season:]), np.array(y[:-season]))

def sMAPE(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))

def main():
    df = pd.read_csv("data/simulation/"+data+".csv")
    y =  df['click']

    # # Bayesian Optimization for cross-validation
    # # use this to select hyperparameters
    # y_train = y[:int(0.55 * len(y))]
    # y_val = y[int(0.55 * len(y)):int(0.7 * len(y))]
    # y_test = y[int(0.7 * len(y)):]

    split = int(0.7 * len(y))
    y_train = y[:split]
    y_test = y[split:]
    timeIdx = np.arange(len(y_train) + len(y_test))[:, None]
    t = np.arange(len(y_train) + len(y_test))
    t_train = t[:split]
    t_test = t[split:]

    start = time.time()
    def my_callback(trace, draw):
        if len(trace) % 10 == 0:
            end = time.time()
            print("Sample trace: ", len(trace), "Accumulate Running time:", end - start)

    def genpoisson_logp(theta, lam, value):
        theta_lam_value = theta + lam * value
        log_prob = np.log(theta) + logpow(theta_lam_value, value - 1) - theta_lam_value - factln(value)

        # Probability is 0 when value > m, where m is the largest positive integer for which
        # theta + m * lam > 0 (when lam < 0).
        log_prob = tt.switch(theta_lam_value <= 0, -np.inf, log_prob)

        return bound(log_prob, value >= 0, theta > 0, abs(lam) <= 1, -theta / 4 <= lam)

    def genpoisson_rvs(theta, lam, size=None):
        if size is not None:
            assert size == theta.shape
        else:
            size = theta.shape
        lam = lam[0]
        omega = np.exp(-lam)
        X = np.full(size, 0)
        S = np.exp(-theta)
        P = np.copy(S)
        for i in range(size[0]):
            U = np.random.uniform()
            while U > S[i]:
                X[i] += 1
                C = theta[i] - lam + lam * X[i]
                P[i] = omega * C * (1 + lam / C) ** (X[i] - 1) * P[i] / X[i]
                S[i] += P[i]
        return X

    class GenPoisson(pm.Discrete):
        def __init__(self, theta, lam, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.theta = theta
            self.lam = lam
            self.mode = intX(tt.floor(theta / (1 - lam)))

        def logp(self, value):
            theta = self.theta
            lam = self.lam
            return genpoisson_logp(theta, lam, value)

        def random(self, point=None, size=None):
            theta, lam = draw_values([self.theta, self.lam], point=point, size=size)
            return generate_samples(genpoisson_rvs, theta=theta, lam=lam, size=size)

    with pm.Model() as model:
        bias = pm.Normal("beta[0]", mu=0, sigma=0.1)
        beta_recent = pm.Normal("beta[1]", mu=1, sigma=0.1)
        rho = [bias, beta_recent]
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        f = pm.AR("f", rho, sigma=sigma, constant=True, shape=len(y))

        lam = pm.TruncatedNormal("lam", mu=0, sigma=10, lower=-20, upper=20)

        y_past = GenPoisson("y_past", theta=tt.exp(f[:len(y_train)]), lam=lam, observed=y_train) # 'y_past': simple demo; 'y_val': rigor bayesian parameter selection and validation

        trace = pm.sample(
            400,
            tune=400,
            target_accept=0.99,
            max_treedepth=15,
            chains=1,
            cores=1,
            init="adapt_diag",
            random_seed=42,
            callback=my_callback,
        )
    print("predicting for train sample...")
    with model:
        val_samples = pm.sample_posterior_predictive(trace, random_seed=42)

    forecasts_for_train = val_samples['y_past']
    print("predicting for test sample...")

    with model:
        #y_future = GenPoisson("y_future", theta=tt.exp(f[-len(y_test):]), lam=lam, shape=len(y_test))
        y_future = GenPoisson("y_future", theta=tt.exp(f[-len(y_test):]), lam=lam, observed=y_test)
        forecasts = pm.sample_posterior_predictive(trace, var_names=['y_future'], random_seed=42, progressbar=True)

    forecasts_for_test = forecasts["y_future"]

    y_train_pred = np.percentile(forecasts_for_train, 50)
    y_test_pred = np.percentile(forecasts_for_test, 50)

    print("training MSE:", MSE(y_train, y_train_pred), "testing MSE:", MSE(y_test, y_test_pred))
    print("training MAE:", MAE(y_train, y_train_pred), "testing MAE:", MAE(y_test, y_test_pred))
    print("training MASE:", MASE(y_train, y_train_pred), "testing MASE:", MASE(y_test, y_test_pred))
    print("training sMAPE:", sMAPE(y_train, y_train_pred), "testing sMAPE:", sMAPE(y_test, y_test_pred))


#data = 'data145'

data = 'double1'

if __name__ == '__main__':
    main()