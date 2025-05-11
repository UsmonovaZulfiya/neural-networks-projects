import numpy as np
import random
import pandas as pd
from som import *
from util import *
import time, csv, itertools
import matplotlib.pyplot as plt
from plots import make_all_plots, plot_training_curves

random.seed(42)

df = pd.read_csv('seeds.txt', delim_whitespace=True, header=None)

print(df.shape)
print(df.head())

X = df.iloc[:, :7].values
y = df.iloc[:, 7].values.astype(int)

X = X.T

mean = X.mean(axis=1, keepdims=True)
std = X.std(axis=1, keepdims=True)
X_scaled = (X - mean) / std

t0 = time.time()

rows = 15
cols = 15
m_fn = lambda a, b: np.sum(np.abs(a-b), axis=-1) # L1
disc = False
a_s = 0.5
a_f = 0.02
l_f = 0.000001
EPOCHS = 100

# initial λ₀ = 0.5 × map diagonal
diag = m_fn(np.array((0, 0)), np.array((rows-1, cols-1)))
lambda_s = 0.5 * diag

# build + train
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

print(f"{rows}x{cols}  "
      f"α={a_s}->{a_f:<4}  λ_f={l_f:<3}   QE={final_qe:.4f}")

print(min(som.quant_err), max(som.quant_err))
plt.figure(); plt.plot(som.quant_err); plt.title('Quantisation error'); plt.xlabel('epoch'); plt.show()
plt.figure(); plt.plot(som.avg_adj);   plt.title('Average adjustment'); plt.xlabel('epoch'); plt.show()

make_all_plots(som, X_scaled, y,
               save_dir='figures',
               title_prefix='15x15_L1_best')

plot_training_curves(som,
                     alpha_s=a_s, alpha_f=a_f,
                     lambda_s=lambda_s,
                     lambda_f=l_f,
                     epochs=EPOCHS)
