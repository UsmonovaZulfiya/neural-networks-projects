#!/usr/bin/python3
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2025

###### 0. Imports and some helper functions ######

import numpy as np
import random
import pandas as pd
from som import *
from util import *
import time, csv, itertools
import matplotlib.pyplot as plt
random.seed(42)

###### 1. Load Data ######

df = pd.read_csv('seeds.txt', delim_whitespace=True, header=None)

print(df.shape)
print(df.head())

X = df.iloc[:, :7].values
y = df.iloc[:, 7].values.astype(int)

X = X.T

###### 2. Scaling ######

mean = X.mean(axis=1, keepdims=True)
std = X.std(axis=1, keepdims=True)
X_scaled = (X - mean) / std


###### 3. Grid-search parameter space ######

GRID_SIZES   = [(18, 12), (21, 9), (15, 15)]

METRICS      = {'L1': lambda a, b: np.sum(np.abs(a-b), axis=-1),
                'L2': lambda a, b: np.linalg.norm(a-b, axis=-1),
                'Linf': lambda a, b: np.max(np.abs(a-b), axis=-1)}

NEIGHBOURS   = {'gauss': False,   # discrete_neighborhood flag
                'disc' : True}

ALPHAS       = [(0.5, 0.02), (0.5, 0.01)]

LAMBDA_F     = {'gauss': [0.5, 0.3, 0.000001],
                'disc' : [1.0, 2.0]}

EPOCHS       = 150

# uncomment for testing
# GRID_SIZES   = [(18, 12)]
#
# METRICS      = {'L1': lambda a, b: np.sum(np.abs(a-b), axis=-1),
#                 'L2': lambda a, b: np.linalg.norm(a-b, axis=-1),
#                 }
# NEIGHBOURS   = {'gauss': False,   # discrete_neighborhood flag
#                 'disc' : True}
#
# ALPHAS       = [(0.5, 0.02)]
#
# LAMBDA_F     = {'gauss': [0.5, 0.3, 0.000001],
#                 'disc' : [1.0]}
#
# EPOCHS       = 50
# RANDOM_SEED  = 42

###### 4.  Produce all combinations ######

def param_product():
    for (rows, cols) in GRID_SIZES:
        for metric_name, metric_fn in METRICS.items():
            for neigh_name, discrete in NEIGHBOURS.items():
                for alpha_s, alpha_f in ALPHAS:
                    for lambda_f in LAMBDA_F[neigh_name]:
                        yield (rows, cols, metric_name, metric_fn,
                               neigh_name, discrete,
                               alpha_s, alpha_f, lambda_f)

###### 5. Running grid ######

results = []
t0 = time.time()

for i, params in enumerate(param_product(), 1):
    (rows, cols, m_name, m_fn,
     n_name, disc, a_s, a_f, l_f) = params

    # initial λ_0 = 0.5 × map diagonal
    diag   = m_fn(np.array((0, 0)), np.array((rows-1, cols-1)))
    lambda_s = 0.5 * diag

    som = SOM(dim_in=7, n_rows=rows, n_cols=cols,
              inputs=X_scaled)

    som.train(X_scaled,
              eps=EPOCHS,
              alpha_s=a_s, alpha_f=a_f,
              lambda_s=lambda_s, lambda_f=l_f,
              discrete_neighborhood=disc,
              grid_metric=m_fn,
              live_plot=False)

    final_qe = som.quant_err[-1]
    results.append([rows, cols, m_name, n_name, a_s, a_f, l_f, final_qe])

    print(f"[{i:02d}] {rows}x{cols}  {m_name:<4}  {n_name:<5}  "
          f"α={a_s}->{a_f:<4}  λ_f={l_f:<3}   QE={final_qe:.4f}")

###### 6.  Saving the results ######

results_df = pd.DataFrame(results,
    columns=['rows','cols','metric','neighb','alpha0','alphaF','lambdaF','final_QE'])

results_df.to_csv('grid_results_1.csv', index=False)
best = results_df.sort_values('final_QE').iloc[0]

print("\nBest configuration")
print(best)
print(f"Total runtime: {time.time()-t0:.1f}s")

