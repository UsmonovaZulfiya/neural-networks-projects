# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Stefan Pócoš, Iveta Bečková 2017-2025

import csv
import matplotlib
matplotlib.use('TkAgg')  # fixme if plotting doesn't work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
# for 3D visualization
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import atexit
import os
import time
import functools


# # Utilities

def init_weights(model, w_type="gaussian", w_scale=0.01, density=0.1):
    """
    Initialise model.W_hid / model.W_out in‑place.

    w_type  : "gaussian" | "uniform" | "sparse"
    w_scale : std‑dev (gauss) or half‑range / sqrt(3) (uniform)
    density : fraction of non‑zero weights for sparse mode
    """
    if w_type == "gaussian":
        model.W_hid = w_scale * np.random.randn(model.dim_hid,
                                                model.dim_in + 1)
        model.W_out = w_scale * np.random.randn(model.dim_out,
                                                model.dim_hid + 1)

    elif w_type == "uniform":
        limit = np.sqrt(3) * w_scale           # same variance as gaussian
        model.W_hid = np.random.uniform(-limit, limit,
                                        size=(model.dim_hid,
                                              model.dim_in + 1))
        model.W_out = np.random.uniform(-limit, limit,
                                        size=(model.dim_out,
                                              model.dim_hid + 1))

    elif w_type == "sparse":
        model.W_hid = np.zeros((model.dim_hid, model.dim_in + 1))
        model.W_out = np.zeros((model.dim_out, model.dim_hid + 1))

        mask_h = np.random.rand(*model.W_hid.shape) < density
        mask_o = np.random.rand(*model.W_out.shape) < density

        model.W_hid[mask_h] = w_scale * np.random.randn(mask_h.sum())
        model.W_out[mask_o] = w_scale * np.random.randn(mask_o.sum())
    else:
        raise ValueError(f"Unknown w_type '{w_type}'")


def load_dataset(path, label_map={'A': 0, 'B': 1, 'C': 2}):
    """
    Reads the 2‑D classification dataset.
      header line: "2 3 8000"
      data lines :  <x> <y> <A|B|C>
    Returns
      inputs : (2 , N) float32
      labels : (N,)   int
    """
    # structured dtype: two floats + one single‑char string
    dt = [('x', 'f4'), ('y', 'f4'), ('lbl', 'U1')]
    raw = np.genfromtxt(path,
                        skip_header=1,
                        dtype=dt,
                        comments=None,
                        autostrip=True)

    # structured array → separate arrays
    inputs = np.vstack((raw['x'], raw['y']))          # shape (2, N)
    labels = np.vectorize(label_map.get)(raw['lbl'])  # shape (N,)

    return inputs.astype(np.float32), labels.astype(int)


def save_data(results):
    with open("hyper-param-errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["hidden", "lr", "lr_decay", "activation",
                        "init_type", "w_scale", "est_CE", "val_CE"])
        writer.writeheader()
        writer.writerows(results)
    print("Saved grid results → hyper-param-errors.csv")


def onehot_decode(inp):
    return np.argmax(inp, axis=0)


def onehot_encode(idx, num_c):
    if isinstance(idx, int):
        idx = [idx]
    n = len(idx)
    out = np.zeros((num_c, n))
    out[idx, range(n)] = 1
    return np.squeeze(out)


def vector(array, row_vector=False):
    """
    Constructs a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    """
    v = np.array(array)
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    """
    Add bias term to vector, or to every (column) vector in a matrix.
    """
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def timeit(func):
    """
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to measurement
    Returns:
        (*function) New wrapped function with measurement
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('Function [{}] finished in {:.3f} s'.format(func.__name__, elapsed_time))
        return out
    return newfunc


# # Interactive drawing
def clear():
    plt.clf()


def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)


def interactive_off():
    plt.ioff()
    plt.close()


def redraw():
    # plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(0.001)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close()  # skip blocking figures


def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


# # Non-blocking figures still block at end
def finish():
    plt.show(block=True)  # block until all figures are closed


atexit.register(finish)


# # Plotting
palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']


def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))


def plot_errors(title, errors, test_error=None, block=True):
    plt.figure(1)
    use_keypress()
    plt.clf()
    plt.ylim(bottom=0)

    plt.plot(errors)

    if test_error:
        plt.plot([test_error]*len(errors))

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(title)
    plt.show(block=block)


def plot_both_errors(trainCEs, trainREs, testCE=None, testRE=None, pad=None, block=True):
    plt.figure(2)
    use_keypress()
    plt.clf()

    if pad is None:
        pad = max(len(trainCEs), len(trainREs))
    else:
        trainCEs = np.concatentate((trainCEs, [None]*(pad-len(trainCEs))))
        trainREs = np.concatentate((trainREs, [None]*(pad-len(trainREs))))

    ax = plt.subplot(2, 1, 1)
    plt.ylim(bottom=0, top=100)
    plt.title('Classification error [%]')
    plt.plot(100*np.array(trainCEs), label='train set')

    if testCE is not None:
        plt.plot([100*testCE]*pad, label='test set')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylim(bottom=0, top=1)
    plt.title('Model loss [MSE/sample]')
    plt.plot(trainREs, label='train set')

    if testRE is not None:
        plt.plot([testRE]*pad, label='test set')

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title('Error metrics')
    plt.legend()

    plt.show(block=block)


def plot_dots(inputs, labels=None, predicted=None, test_inputs=None, test_labels=None, test_predicted=None, s=60, i_x=0,
              i_y=1, title=None, block=True):
    plt.figure(title or 3)
    use_keypress()
    plt.clf()

    if inputs is not None:
        if labels is None:
            plt.gcf().canvas.manager.set_window_title('Data distribution')
            plt.scatter(inputs[i_x, :], inputs[i_y, :], s=s, c=palette[-1], edgecolors=[0.4]*3, alpha=0.5,
                        label='train data')

        elif predicted is None:
            plt.gcf().canvas.manager.set_window_title('Class distribution')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x, labels == c], inputs[i_y, labels == c], s=s, c=palette[i], edgecolors=[0.4]*3,
                            label='train cls {}'.format(c))

        else:
            plt.gcf().canvas.manager.set_window_title('Predicted vs. actual')
            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x, labels == c], inputs[i_y, labels == c], s=2.0*s, c=palette[i], edgecolors=None,
                            alpha=0.333, label='train cls {}'.format(c))

            for i, c in enumerate(set(labels)):
                plt.scatter(inputs[i_x, predicted == c], inputs[i_y, predicted == c], s=0.5*s, c=palette[i],
                            edgecolors=None, label='predicted {}'.format(c))

        plt.xlim(limits(inputs[i_x, :]))
        plt.ylim(limits(inputs[i_y, :]))

    if test_inputs is not None:
        if test_labels is None:
            plt.scatter(test_inputs[i_x, :], test_inputs[i_y, :], marker='s', s=s, c=palette[-1], edgecolors=[0.4]*3,
                        alpha=0.5, label='test data')

        elif test_predicted is None:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x, test_labels == c], test_inputs[i_y, test_labels == c], marker='s', s=s,
                            c=palette[i], edgecolors=[0.4]*3, label='test cls {}'.format(c))

        else:
            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x, test_labels == c], test_inputs[i_y, test_labels == c], marker='s', s=2.0*s,
                            c=palette[i], edgecolors=None, alpha=0.333, label='test cls {}'.format(c))

            for i, c in enumerate(set(test_labels)):
                plt.scatter(test_inputs[i_x, test_predicted == c], test_inputs[i_y, test_predicted == c], marker='s',
                            s=0.5*s, c=palette[i], edgecolors=None, label='predicted {}'.format(c))

        if inputs is None:
            plt.xlim(limits(test_inputs[i_x, :]))
            plt.ylim(limits(test_inputs[i_y, :]))

    plt.legend()
    if title is not None:
        plt.gcf().canvas.manager.set_window_title(title)
    plt.tight_layout()
    plt.show(block=block)


def plot_areas(model, inputs, labels=None, w=30, h=20, i_x=0, i_y=1, block=True):
    plt.figure(4)
    use_keypress()
    plt.clf()
    plt.gcf().canvas.manager.set_window_title('Decision areas')

    dim = inputs.shape[0]
    data = np.zeros((dim, w*h))

    # # "proper":
    # X = np.linspace(*limits(inputs[i_x,:]), w)
    # Y = np.linspace(*limits(inputs[i_y,:]), h)
    # YY, XX = np.meshgrid(Y, X)
    #
    # for i in range(dim):
    #     data[i,:] = np.mean(inputs[i,:])
    # data[i_x,:] = XX.flat
    # data[i_y,:] = YY.flat

    X1 = np.linspace(*limits(inputs[0, :]), w)
    Y1 = np.linspace(*limits(inputs[1, :]), h)
    X2 = np.linspace(*limits(inputs[2, :]), w)
    Y2 = np.linspace(*limits(inputs[3, :]), h)
    YY1, XX1 = np.meshgrid(Y1, X1)
    YY2, XX2 = np.meshgrid(Y2, X2)
    data[0, :] = XX1.flat
    data[1, :] = YY1.flat
    data[2, :] = XX2.flat
    data[3, :] = YY2.flat

    outputs, *_ = model.predict(data)
    assert outputs.shape[0] == model.dim_out,\
           f'Outputs do not have correct shape, expected ({model.dim_out}, ?), got {outputs.shape}'
    outputs = outputs.reshape((-1, w, h))

    outputs -= np.min(outputs, axis=0, keepdims=True)
    outputs = np.exp(1*outputs)
    outputs /= np.sum(outputs, axis=0, keepdims=True)

    plt.imshow(outputs.T)

    plt.tight_layout()
    plt.show(block=block)


def init_weights(model, w_type="gaussian", w_scale=0.01, density=0.1):
    """
    Initialise model.W_hid / model.W_out in‑place.

    w_type  : "gaussian" | "uniform" | "sparse"
    w_scale : std‑dev (gauss) or half‑range / sqrt(3) (uniform)
    density : fraction of non‑zero weights for sparse mode
    """
    if w_type == "gaussian":
        model.W_hid = w_scale * np.random.randn(model.dim_hid,
                                                model.dim_in + 1)
        model.W_out = w_scale * np.random.randn(model.dim_out,
                                                model.dim_hid + 1)

    elif w_type == "uniform":
        limit = np.sqrt(3) * w_scale           # same variance as gaussian
        model.W_hid = np.random.uniform(-limit, limit,
                                        size=(model.dim_hid,
                                              model.dim_in + 1))
        model.W_out = np.random.uniform(-limit, limit,
                                        size=(model.dim_out,
                                              model.dim_hid + 1))

    elif w_type == "sparse":
        model.W_hid = np.zeros((model.dim_hid, model.dim_in + 1))
        model.W_out = np.zeros((model.dim_out, model.dim_hid + 1))

        mask_h = np.random.rand(*model.W_hid.shape) < density
        mask_o = np.random.rand(*model.W_out.shape) < density

        model.W_hid[mask_h] = w_scale * np.random.randn(mask_h.sum())
        model.W_out[mask_o] = w_scale * np.random.randn(mask_o.sum())
    else:
        raise ValueError(f"Unknown w_type '{w_type}'")


def load_dataset(path, label_map={'A': 0, 'B': 1, 'C': 2}):
    """
    Reads the toy 2‑D classification dataset.
      header line: "2 3 8000"
      data lines :  <x> <y> <A|B|C>
    Returns
      inputs : (2 , N) float32
      labels : (N,)   int
    """
    # structured dtype: two floats + one single‑char string
    dt = [('x', 'f4'), ('y', 'f4'), ('lbl', 'U1')]
    raw = np.genfromtxt(path,
                        skip_header=1,
                        dtype=dt,
                        comments=None,   # header already skipped
                        autostrip=True)  # trim leading '+'

    # structured array → separate arrays
    inputs = np.vstack((raw['x'], raw['y']))          # shape (2, N)
    labels = np.vectorize(label_map.get)(raw['lbl'])  # shape (N,)

    return inputs.astype(np.float32), labels.astype(int)


def save_data(results):
    with open("hyper-param-errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["hidden", "lr", "lr_decay", "activation",
                        "init_type", "w_scale", "est_CE", "val_CE"])
        writer.writeheader()
        writer.writerows(results)
    print("Saved grid results → hyper-param-errors.csv")