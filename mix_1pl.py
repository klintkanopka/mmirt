import numpy as np
import pandas as pd
from scipy.special import expit

import os
from util import read_data, init_1m_1pl_params

def get_pi(X, theta, beta):
    """
    Returns a vector of mixing parameters, pi
    pi is computed by logit( (k-seq)/scale )
    where k is a person-level parameter, seq is the sequence number of the 
    item, and scale is a (tunable) hyperparameter
    """
    s = 10
    return expit((theta[np.intp(X[:,1]),1] - X[:,3]) / s), s

def item_response_funcs(X, theta, beta):
    """
    Returns a vector of predicted response probabilities for each of the 
    mixed IRFs *separately* based upon current theta and beta estimates.
    Output shape is one row per observed item response 
    and one column per model in the mixture
    """
    thetas = theta[np.intp(X[:,1]), 0]
    thetas = np.broadcast_to(thetas, (2, thetas.shape[0])).T
    betas = beta[np.intp(X[:,2]),:]
    return expit(thetas - betas)

def response_prob(X, theta, beta):
    """
    Returns a vector of predicted response probabilities for each person-item
    interaction by mixing the IRFs according to parameter pi.
    Output shape of P is one row per person-item interaction and one column.
    Also returns unmixed probabilities and mixing parameters for use in other
    computations.
    """
    pi1, s = get_pi(X, theta, beta)

    pi = np.array([pi1, 1-pi1]).T
    p = item_response_funcs(X, theta, beta)

    P = np.sum(pi*p, axis=1)
    return P, p, pi, s

def grad_base(X, theta, beta):
    """
    Computes the "base" of the gradient of the log likelihood, 
    as it is the same for all parameters. 
    Application of the chain rule is left to grad_theta and grad_beta.
    """
    P, p, pi, s = response_prob(X, theta, beta)
    base = X[:,0]/P - (1 - X[:,0])/(1-P)
    return base, p, pi, s

def grad_theta(X, theta, beta):
    """
    Computes the gradient of the log likelihood wrt all person parameters, 
    which are the standard rasch theta and the k parameter that governs
    the value of the mixing parameter, pi. Returns a gradient in the same 
    shape as the theta matrix.
    """
    base, p, pi, s = grad_base(X, theta, beta)
    chain = np.sum(pi*p*(1-p), axis=1)
    dthetas = base * chain
    dt = np.bincount(np.intp(X[:,1]), weights = dthetas)
    w = np.array([1.0, -1.0]).T
    dpi_w = base * np.sum(w * p, axis=1) * pi[:,0] * pi[:,1] / s
    dpi = np.bincount(np.intp(X[:,1]), weights=dpi_w)
    dtheta = np.concatenate((dt[:,None], dpi[:,None]), axis=1)
    return dtheta

def grad_beta(X, theta, beta):
    """
    Computes the gradient of the log likelihood wrt all item parameters, 
    which are one standard rasch difficulty per IRF in the mixture.
    Returns a gradient in the same shape as the beta matrix.
    """
    base, p, pi, s = grad_base(X, theta, beta)
    chain = -pi*p*(1-p)
    db = base[:, None] * chain
    dbeta = np.array([np.bincount(np.intp(X[:,2]), weights=db[:,0]),
                      np.bincount(np.intp(X[:,2]), weights=db[:,1])]).T
    return dbeta

def loglik(X, theta, beta):
    """
    Computes the average log likelihood over the entire dataset.
    """
    P, _, _, _ = response_prob(X, theta, beta)
    return np.mean( X[:,0] * np.log(P) + (1 - X[:,0]) * np.log(1 - P) )

def main(file_path):
    print('loading data...')
    d = read_data(file_path)
    X, theta, beta = init_1m_1pl_params(d)

    # hyperparameters - don't forget scale in get_pi()
    # in future, move scale outside of get_pi() and allow these to be
    # set from the command line
    eta = 0.0005
    eps = 1e-7
    max_iter = 1e3
    max_step = 1e2

    ll = E_ll = M_ll =  None

    steps = 0

    ll_data = [loglik(X,theta, beta)]

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

            E_ll = ll

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
            
            M_ll = ll

            if it % 100 == 0:
                print('M:', it, 'll:', ll)

        steps += 1


    ll_dat = pd.DataFrame(np.array(ll_data))
    ll_dat.to_csv('output/1m_1pl_loglik.csv', index=False,
            header = ['loglikelihood'])

    final_theta = pd.DataFrame(theta)
    final_theta.to_csv('output/1m_1pl_theta.csv', index=False,
            header = ['theta', 'k'])

    final_beta = pd.DataFrame(beta)
    final_beta.to_csv('output/1m_1pl_beta.csv', index=False,
            header = ['b0', 'b1'])


if __name__ == '__main__':
    # test data
    #main('data/test.csv')
    # full data
    main('data/allgrade_Spring_6.csv')
