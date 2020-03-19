import numpy as np
import pandas as pd
from scipy.special import expit

def read_data(file_path):
    df = pd.read_csv(file_path)
    
    # get starting theta estimates
    df_0 = df[['id', 'sequence_number', 'diff']]
    df_0 = df_0[df_0.sequence_number == 1]
    df_0 = (df_0 .assign(th_0 = df_0['diff'] + 5)
                .assign(sid = list(range(len(df_0)))))
    df_0 = df_0.drop(columns = ['diff', 'sequence_number'])

    # zero index sequence number
    df = df.assign(seq = df['sequence_number'] - 1)

    # join everything into big df

    d = pd.merge(df, df_0, how='left', on='id')
    d = d.dropna()
    d = d.assign(sid = d['sid'].astype(int))
    d = d.assign(p = expit((d['th_0'].to_numpy()-200)/10 - (d['diff'].to_numpy()-200)/10))

    # zero index items
    items = d['itemkey'].unique()
    item_df = pd.DataFrame(
            {'itemkey' : items,
                'ik' : list(range(len(items)))})
    d = pd.merge(d, item_df, how='left', on='itemkey')
    
    return d

def init_params(d):
    pass

    n_items = d['ik'].nunique()
    n_persons = d['sid'].nunique()

    theta = np.broadcast_to([0, 20, 0], [n_persons, 3])
    beta = np.zeros([n_items, 4])

    X = d[['resp', 'sid', 'ik', 'seq', 'lrt', 'p']].to_numpy(copy=True)

    return X, theta, beta
