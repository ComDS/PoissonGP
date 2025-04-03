'''
Gaussian-Poisson Process with double periodic kernel version
'''

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

def main():
    df = pd.read_csv("data/empirical_2305/"+param['data']+".csv")
    y = df['click'] # column: time t (after anonymization), click (aggregated sales volume, total customer clicking the 'purchase' bottom on ComZ)

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
    # t_train = t[:split]
    # t_test = t[split:]

    start = time.time()
    def my_callback(trace, draw):
        if len(trace) % 10 == 0:
            end = time.time()
            print("Sample trace: ", len(trace), "Accumulate Running time:", end - start)

    with pm.Model() as model:
        # Gaussian Process Prior
        #m = np.mean(y_train)
        Lambda0 = pm.Gamma('Lambda0', alpha=param['Lambda0_gamma(alpha)'], beta=param['Lambda0_gamma(beta)'])

        # se kernel
        #mu2 = pm.Normal('mu2', sigma=10)
        meanG = pm.gp.mean.Constant(c=param['muG'])
        aG = pm.HalfNormal('amplitudeG', sigma=param['aG_HN(sigma)'])
        gammaG = pm.TruncatedNormal('time-scaleG', mu=param['gammaG_HN(mu)'], sigma=param['gammaG_HN(sigma)'], lower=0)  # new hp
        covG = aG ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=gammaG)
        GP_G = pm.gp.Latent(mean_func=meanG, cov_func=covG)

        # periodic kernel
        meanS = pm.gp.mean.Constant(c=param['muS'])
        aS = pm.HalfNormal('amplitudeS', sigma=param['aS_HN(sigma)'])
        gammaS = pm.TruncatedNormal('time-scaleS', mu=param['gammaS_HN(mu)'], sigma=param['gammaS_HN(sigma)'], lower=0)
        covS = aS ** 2 * pm.gp.cov.Periodic(input_dim=1, period=param['w'], ls=gammaS)
        GP_S = pm.gp.Latent(mean_func=meanS, cov_func=covS)

        # periodic kernel1
        meanS1 = pm.gp.mean.Constant(c=param['muS1'])
        aS1 = pm.HalfNormal('amplitudeS1', sigma=param['aS1_HN(sigma)'])
        gammaS1 = pm.TruncatedNormal('time-scaleS1', mu=param['gammaS1_HN(mu)'], sigma=param['gammaS1_HN(sigma)'], lower=0)
        covS1 = aS1 ** 2 * pm.gp.cov.Periodic(input_dim=1, period=param['w1'], ls=gammaS1)
        GP_S1 = pm.gp.Latent(mean_func=meanS1, cov_func=covS1)

        # white noise
        meanW = pm.gp.mean.Constant(c=param['muW'])
        covW = pm.gp.cov.WhiteNoise(sigma=param['sigmaW'])
        GP_W = pm.gp.Latent(mean_func=meanW, cov_func=covW)

        GP = GP_G + GP_S + GP_S1 + GP_W

        f = GP.prior('f', X=timeIdx)

        mu_tr = pm.Deterministic('mu_tr',Lambda0 * tt.exp(f[:split]))
        pm.Poisson('y_val', mu=mu_tr, observed=y_train)
        trace = pm.sample(draws=param['draws'], tune=param['tune'], chains=1, target_accept=.9, random_seed=param['seed'], callback=my_callback)

    par_dt = pd.DataFrame({
        'Lambda0': trace['Lambda0'],
        'amplitudeG': trace['amplitudeG'],
        'time-scaleG': trace['time-scaleG'],
        'amplitudeS': trace['amplitudeS'],
        'time-scaleS': trace['time-scaleS'],
        'amplitudeS1': trace['amplitudeS1'],
        'time-scaleS1': trace['time-scaleS1'],
    })
    par_dt.to_csv("result/empirical/gp_parameter/par_"+param['data']+".csv",index=False)

    with model:
        val_samples = pm.sample_posterior_predictive(trace, random_seed=param['seed'])
    forecasts_for_train = val_samples['y_val']

    with model:
        mu_ts = pm.Deterministic('mu_ts', Lambda0 * tt.exp(f[split:]))
        y_pred = pm.Poisson('y_pred', mu=mu_ts, observed=y_test)
        test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=param['seed'])

    forecasts_for_test = test_samples['y_pred']
    train_result_dt = pd.DataFrame(forecasts_for_train).T
    train_result_dt.to_csv("result/empirical/gp/result_"+param['data']+"_train.csv",index=False)
    test_result_dt = pd.DataFrame(forecasts_for_test).T
    test_result_dt.to_csv("result/empirical/gp/result_"+param['data']+"_test.csv",index=False)

# change here to implement different model versions
param = {
    # data
    'data': 'product2',

    # training
    'draws': 500, # 500
    'tune': 500, # 500
    'seed': 42,

    # model hyperparameter (selected through cross-validation)
    # base rate
    'Lambda0_gamma(alpha)': 2,
    'Lambda0_gamma(beta)': 2,
    # SE kernel
    'muG': 0,
    'aG_HN(sigma)': 2,
    'gammaG_HN(mu)': 85,
    'gammaG_HN(sigma)': 170,
    # periodic kernel
    'muS': 0,
    'aS_HN(sigma)': 2,
    'gammaS_HN(mu)': 20,
    'gammaS_HN(sigma)': 5,
    'w': 24,    # periodic length
    # periodic kernel 1
    'muS1': 0,
    'aS1_HN(sigma)': 2,
    'gammaS1_HN(mu)': 20,
    'gammaS1_HN(sigma)': 5,
    'w1': 168,    # periodic length
    # white noise kernel
    'muW': 0,
    'sigmaW': 1,
}

if __name__ == '__main__':
    main()

