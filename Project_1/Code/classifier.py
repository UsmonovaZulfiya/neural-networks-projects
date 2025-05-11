# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2025

from mlp import *
from util import *

def logsig(x):
    fx = 1 / (1 + np.exp(-x))
    return fx, fx * (1 - fx)

def tanh(x):
    fx = np.tanh(x)
    return fx, 1 - fx**2

def relu(x):
    fx = np.maximum(0, x)
    d  = (x > 0).astype(x.dtype)
    return fx, d

ACTIVATIONS = {
    "logsig": logsig,
    "tanh"  : tanh,
    "relu"  : relu,
}

class MLPClassifier(MLP):
    def __init__(self, dim_in, dim_hid, n_classes, act_name="logsig"):
        self.n_classes = n_classes
        self.act_name  = act_name
        if act_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{act_name}'")
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    def error(self, targets, outputs):
        """
        Cost / loss / error function
        """
        eps = 1e-10
        return -np.sum(targets * np.log(outputs + eps), axis=0)

    # @override
    def f_hid(self, x):
        fx, _ = ACTIVATIONS[self.act_name](x)
        return fx

    def df_hid(self, x):
        _, dfx = ACTIVATIONS[self.act_name](x)
        return dfx

    # @override
    def f_out(self, x):
        c = np.max(x, axis=0, keepdims=True)  # for numerical stability
        exp_x = np.exp(x - c)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    # @override
    def df_out(self, x):
        return np.ones_like(x)

    def predict(self, inputs):
        """
        Prediction = forward pass
        """
        outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        return outputs, onehot_decode(outputs)

    def test(self, inputs, labels):
        """
        Test model: forward pass on given inputs, and compute errors
        """
        targets = onehot_encode(labels, self.n_classes)
        outputs, predicted = self.predict(inputs)
        CE = np.mean(predicted != labels)
        RE = np.mean(self.error(targets, outputs))
        return CE, RE

    def train(self, inputs, labels, alpha=0.1, eps=100, lr_decay=0.0, live_plot=False, live_plot_interval=10,
              val_inputs=None, val_labels=None, patience=10, min_delta=0.0,):
        """
        Training of the classifier
        inputs: matrix of input vectors (each column is one input vector)
        labels: vector of labels (each item is one class label)
        alpha: learning rate
        eps: number of episodes
        live_plot: plot errors and data during training
        live_plot_interval: refresh live plot every N episodes
        """
        (_, count) = inputs.shape
        targets  = onehot_encode(labels, self.n_classes)

        if live_plot:
            interactive_on()

        CEs = []
        REs = []

        best_val_CE = np.inf
        best_W_hid, best_W_out = self.W_hid.copy(), self.W_out.copy()
        wait = 0

        for ep in range(eps):
            CE = 0.0
            RE = 0.0
            alpha_t = alpha / (1.0 + lr_decay * ep)

            for idx in np.random.permutation(count):
                x = inputs[:, idx:idx + 1]
                d = targets[:, idx:idx + 1]

                a, h, b, y = self.forward(x)
                dW_hid, dW_out = self.backward(x, a, h, b, y, d)

                self.W_hid += alpha_t * dW_hid
                self.W_out += alpha_t * dW_out

                CE += labels[idx] != onehot_decode(y)
                RE += self.error(d, y)

            CE /= count
            RE /= count
            CEs.append(CE)
            REs.append(RE)
            if (ep + 1) % 5 == 0: print('Epoch {:3d}/{}, CE = {:6.2%}, RE = {:.5f}'.format(ep + 1, eps, float(CE), float(RE)))

            if live_plot and ((ep + 1) % live_plot_interval == 0):
                _, predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                plot_areas(self, inputs, block=False)
                redraw()

            if val_inputs is not None:
                val_CE, _ = self.test(val_inputs, val_labels)
                if val_CE + min_delta < best_val_CE:
                    best_val_CE = val_CE
                    best_W_hid, best_W_out = self.W_hid.copy(), self.W_out.copy()
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    print(f"Early‑stopping at epoch {ep + 1} (no improv. in {patience})")
                    break

        if live_plot:
            interactive_off()

        print()

        if val_inputs is not None:
            self.W_hid, self.W_out = best_W_hid, best_W_out

        return CEs, REs
