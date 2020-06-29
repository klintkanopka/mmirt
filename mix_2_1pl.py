import numpy as np
import pandas as pd
from scipy.special import expit

import os
from util import read_data, init_params

def pi_1(X, theta, beta):
    s = 10
    return expit((theta[np.intp(X[:,1]),1] - X[:,3]) / s), s

def pi_2(X, theta, beta):
    return X[:,5]

def item_response_funcs(X, theta, beta):
    thetas = theta[np.intp(X[:,1]), 0]
    thetas = np.broadcast_to(thetas, (4, thetas.shape[0])).T
    betas = beta[np.intp(X[:,2]),:]
    return expit(thetas - betas)

def response_prob(X, theta, beta):
    pi1, s = pi_1(X, theta, beta)
    pi2 = pi_2(X, theta, beta)

    pi = np.array([pi1*pi2, pi1*(1-pi2), (1-pi1)*pi2, (1-pi1)*(1-pi2)]).T
    p = item_response_funcs(X, theta, beta)

    P = np.sum(pi*p, axis=1)
    return P, p, pi, s

def grad_base(X, theta, beta):
    P, p, pi, s = response_prob(X, theta, beta)
    base = X[:,0]/P - (1 - X[:,0])/(1-P)
    return base, p, pi, s

def grad_theta(X, theta, beta):
    base, p, pi, s = grad_base(X, theta, beta)
    chain = np.sum(pi*p*(1-p), axis=1)
    dthetas = base * chain
    dt = np.bincount(np.intp(X[:,1]), weights = dthetas)
    pi1, s = pi_1(X, theta, beta)
    pi2 = pi_2(X, theta, beta)
    pi = np.array([pi2, (1-pi2), -pi2, -(1-pi2)]).T
    dpi = base * np.sum(pi * p, axis=1) * pi1 * (1-pi1) / s
    dpi1 = np.bincount(np.intp(X[:,1]), weights=dpi)
    dpi2 = np.zeros(theta.shape[0])
    dtheta = np.concatenate((dt[:,None], dpi1[:,None], dpi2[:,None]), axis=1)
    return dtheta

def grad_beta(X, theta, beta):
    base, p, pi, s = grad_base(X, theta, beta)
    chain = -pi*p*(1-p)
    db = base[:, None] * chain
    dbeta = np.array([np.bincount(np.intp(X[:,2]), weights=db[:,0]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,1]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,2]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,3])]).T
    return dbeta

def loglik(X, theta, beta):
    P, _, _, _ = response_prob(X, theta, beta)
    return np.mean( X[:,0] * np.log(P) + (1 - X[:,0]) * np.log(1 - P) )

def main(file_path):
    print('loading data...')
    d = read_data(file_path)
    X, theta, beta = init_params(d)

    eta = 0.0005
    eps = 1e-6
    max_iter = 1e3
    max_step = 1e2

    ll = E_ll = M_ll =  None

    steps = 0

    ll_data = [loglik(X, theta, beta)]

    while steps < max_step and (E_ll is None or np.abs(M_ll - E_ll) >= eps):

        print('E step', steps, '...')
        prev_ll = None
        it = 0

        while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # E step:
            dtheta = grad_theta(X, theta, beta)
            theta = theta + eta * dtheta
            
            it += 1
            
            prev_ll = ll
            ll = loglik(X, theta, beta)
            ll_data.append(ll)

            if it % 100 == 0:
                print('E:', it, 'll:', ll)

        print('M step', steps, '...')
        prev_ll = None
        it = 0

        while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # M step:
            dbeta = grad_beta(X, theta, beta)
            beta = beta + eta * dbeta
            beta[:,0] = beta[:,0] - np.mean(beta[:,0])

            it += 1 

            prev_ll = ll
            ll = loglik(X, theta, beta)
            ll_data.append(ll)

            if it % 100 == 0:
                print('M:', it, 'll:', ll)

        steps += 1


    ll_dat = pd.DataFrame(np.array(ll_data))
    ll_dat.to_csv('output/mix2_1pl_loglik.csv',
            header = ['loglikelihood'])

    final_theta = pd.DataFrame(theta)
    final_theta.to_csv('output/mix2_1pl_theta.csv', index=False,
            header = ['theta', 'k', 'unused'])

    final_beta = pd.DataFrame(beta)
    final_beta.to_csv('output/mix2_1pl_beta.csv', index=False,
            header = ['b00', 'b01', 'b10', 'b11'])


if __name__ == '__main__':
    # test data
    #main('data/test.csv')
    # full data
    main('data/allgrade_Spring_6.csv')
