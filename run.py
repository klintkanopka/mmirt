import numpy as np
import pandas as pd
from scipy.special import expit

import os
from util import read_data, init_params

def pi_1(X, theta, beta):
    scale = 10
    return expit((theta[np.intp(X[:,1]),1] - X[:,3]) / scale)

def pi_2(X, theta, beta):
    return X[:,5]

def item_response_funcs(X, theta, beta):
    thetas = theta[np.intp(X[:,1]), 0]
    thetas = np.broadcast_to(thetas, (4, thetas.shape[0])).T
    betas = beta[np.intp(X[:,2]),:]
    return expit(thetas - betas)

def response_prob(X, theta, beta):
    pi1 = pi_1(X, theta, beta)
    pi2 = pi_2(X, theta, beta)

    pi = np.array([pi1*pi2, pi1*(1-pi2), (1-pi1)*pi2, (1-pi1)*(1-pi2)]).T
    p = item_response_funcs(X, theta, beta)

    P = np.sum(pi*p, axis=1)
    return P, p, pi

def grad_base(X, theta, beta):
    P, p, pi = response_prob(X, theta, beta)
    base = X[:,0]/P - (1 - X[:,0])/(1-P)
    return base, p, pi

def grad_theta(X, theta, beta):
    base, p, pi = grad_base(X, theta, beta)
    chain = np.sum(pi*p*(1-p), axis=1)
    dthetas = base * chain
    dt = np.bincount(np.intp(X[:,1]), weights = dthetas)
    pi1 = pi_1(X, theta, beta)
    pi2 = pi_2(X, theta, beta)
    pi = np.array([pi2, (1-pi2), -pi2, -(1-pi2)]).T
    dpi = base * np.sum(pi * p, axis=1) * pi1 * (1-pi1)
    dpi1 = np.bincount(np.intp(X[:,1]), weights=dpi)
    dpi2 = np.zeros(theta.shape[0])
    dtheta = np.concatenate((dt[:,None], dpi1[:,None], dpi2[:,None]), axis=1)
    return dtheta

def grad_beta(X, theta, beta):
    base, p, pi = grad_base(X, theta, beta)
    chain = -pi*p*(1-p)
    db = base[:, None] * chain
    dbeta = np.array([np.bincount(np.intp(X[:,2]), weights=db[:,0]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,1]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,2]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,3])]).T
    return dbeta

def loglik(X, theta, beta):
    P, _, _ = response_prob(X, theta, beta)
    return np.sum( X[:,0] * np.log(P) + (1 - X[:,0]) * np.log(1 - P) )

def main(file_path):
    print('loading data...')
    d = read_data(file_path)
    X, theta, beta = init_params(d)

    eta = 0.01
    eps = 1e0

    ll = prev_ll =  None

    i = 0
    max_iter = 1e3

    ll_data = []

    for _ in range(100):

        it = 0
        prev_ll = None
    
        print('E step...')

        while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # E step:
            dtheta = grad_theta(X, theta, beta)
            theta = theta + eta * dtheta
            prev_ll = ll
            ll = loglik(X, theta, beta)
            ll_data.append([i,ll])
            it += 1
            i += 1

            if it % 10 == 0:
                print('E:', it, 'll:', ll)

        prev_ll = None
        it = 0

        print('M step...')

        while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # M step:
            dbeta = grad_beta(X, theta, beta)
            beta = beta + eta * dbeta
            beta[:,0] = beta[:,0] - np.mean(beta[:,0])
            ll = loglik(X, theta, beta)
            ll_data.append([i,ll])
            it += 1 
            i += 1

            if it % 10 == 0:
                print('M:', it, 'll:', ll)

        prev_ll = None


    ll_dat = pd.DataFrame(np.array(ll_data))
    ll_dat.to_csv('output/loglik.csv', index=False,
            header = ['iteration', 'loglikelihood'])

    final_theta = pd.DataFrame(theta)
    final_theta.to_csv('output/theta.csv', index=False,
            header = ['theta', 'k', 'unused'])

    final_beta = pd.DataFrame(beta)
    final_beta.to_csv('output/beta.csv', index=False,
            header = ['b00', 'b01', 'b10', 'b11'])


if __name__ == '__main__':
    # test data
    #main('data/test.csv')
    # full data
    main('data/allgrade_Spring_6.csv')
