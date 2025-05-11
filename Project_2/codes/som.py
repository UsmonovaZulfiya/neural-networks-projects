# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2025
import random

from pandas.core.methods.selectn import SelectNSeries

from util import *


class SOM:
    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.avg_adj = None
        self.quant_err = None
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.weights = np.random.rand(n_rows, n_cols, dim_in)

        random.seed(42) # for reproducibility

        if inputs is not None:
            data_min = inputs.min(axis=1)  # (dim,)
            data_max = inputs.max(axis=1)
            span = data_max - data_min
            self.weights = (data_min.reshape(1, 1, -1) +
                            self.weights * span.reshape(1, 1, -1))

    def winner(self, x):
        """
        Find winner neuron and return its coordinates in grid (i.e. its "index").
        Iterate over all neurons and find the neuron with the lowest distance to input x (np.linalg.norm).
        """

        dists = np.linalg.norm(self.weights - x, axis=2)  # ✔ fixed
        return np.unravel_index(np.argmin(dists), dists.shape)



    def train(self,
              inputs,   # Matrix of inputs - each column is one input vector
              eps=100,  # Number of epochs
              alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1,  # Start & end values for alpha & lambda
              discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
              grid_metric=(lambda u, v: 0),  # Grid distance metric
              live_plot=False, live_plot_interval=10  # Draw plots during training process
              ):

        (dim, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        if lambda_s is None:
            diag = grid_metric(np.array((0, 0)),
                               np.array((self.n_rows - 1, self.n_cols - 1)))
            lambda_s = diag / 2

        if live_plot:
            interactive_on()
            (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
            redraw()


        self.quant_err = []    # quantisation error per epoch
        self.avg_adj   = []    # average |Δw| per neuron per epoch
        prev_weights   = self.weights.copy()


        for ep in range(eps):
            alpha_t = alpha_s * (alpha_f / alpha_s) ** (ep/(eps-1))
            lambda_t  = lambda_s * (lambda_f / lambda_s) ** (ep/(eps-1))
            if lambda_t < 1e-6:
                lambda_t = 1e-6

            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        d = grid_metric(np.array([win_r, win_c]), np.array((r, c)))
                        if discrete_neighborhood:
                            h = 1.0 if d <= lambda_t else 0.0
                        else:  # Gaussian neighbourhood                    ✔ added
                            h = np.exp(-(d ** 2) / (2 * (lambda_t ** 2)))

                        self.weights[r, c] += alpha_t * h * (x - self.weights[r, c])


            # quantisation error
            qe = 0.0
            for i in range(count):
                r, c = self.winner(inputs[:, i])
                qe += np.linalg.norm(self.weights[r, c] - inputs[:, i])
            qe /= count
            self.quant_err.append(qe)

            # average adjustment
            delta = np.linalg.norm(self.weights - prev_weights, axis=2).mean()
            self.avg_adj.append(delta)
            prev_weights = self.weights.copy()


            print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}, q_error = {:.3f}'
                  .format(ep+1, eps, alpha_t, lambda_t, qe))

            if live_plot and ((ep+1) % live_plot_interval == 0):
                (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=False)
                redraw()

        if live_plot:
            interactive_off()
        # else:
        #     (plot_grid_3d if plot_in3d else plot_grid_2d)(inputs, self.weights, block=True)
