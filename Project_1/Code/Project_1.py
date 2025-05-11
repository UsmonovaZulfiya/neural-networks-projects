# Neural Networks (2-AIN-132/15), FMFI UK BA
# Zulfiya Usmonova, Project 1, April 2025


###### 0. Imports and seed setting for reproducibility ######

import random
import numpy as np
from classifier import MLPClassifier
from util import plot_dots, plot_both_errors, init_weights, load_dataset, save_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

RND_SEED = 42
np.random.seed(RND_SEED)
random.seed(RND_SEED)

def fmt(t):
    return time.strftime("%H:%M:%S", time.gmtime(t)) + f".{int((t%1)*1000):03d}"


###### 1. LOAD DATA ######

train_inputs, train_labels = load_dataset("2d.trn.dat")
test_inputs,  test_labels  = load_dataset("2d.tst.dat")

print(train_inputs.shape)  # ➜ (2, 8000)
print(np.bincount(train_labels))


###### 2. NORMALIZATION ######

mean = np.mean(train_inputs, axis=1, keepdims=True)
std = np.std(train_inputs, axis=1, keepdims=True)

train_inputs = (train_inputs - mean) / std
test_inputs = (test_inputs - mean) / std


###### 3. SPLIT: ESTIMATION & VALIDATION ######

N = train_inputs.shape[1]
indices = np.random.permutation(N)
split = int(0.8 * N)

est_idx = indices[:split]
val_idx = indices[split:]

est_inputs = train_inputs[:, est_idx]
est_labels = train_labels[est_idx]

val_inputs = train_inputs[:, val_idx]
val_labels = train_labels[val_idx]


###### 4. HYPERPARAMETER GRID ######

hidden_sizes = [10, 20, 30]
learning_rates = [0.01, 0.05]
lr_decays = [0.0, 0.01, 0.05]   # 0 -> constant LR
activations = ["logsig", "tanh", "relu"]
init_types = ["gaussian", "uniform", "sparse"]
weight_scales = [0.01, 0.1]

# uncomment for error/table/plot generation testing purposes
# hidden_sizes = [10]
# learning_rates = [0.01]
# lr_decays = [0.0]   # 0 -> constant LR
# activations = ["tanh"]
# init_types = ["gaussian"]
# weight_scales = [0.01, 0.1]

grid_start = time.time() # time-tracking for grid-search

best_model = None
best_val_CE = float('inf')
best_params = {}

results = []

for h in hidden_sizes:
    for lr in learning_rates:
        for act in activations:
            for w_type in init_types:
                for w_scale in weight_scales:
                    for lr_decay in lr_decays:
                        model = MLPClassifier(dim_in=2, dim_hid=h, n_classes=3, act_name=act)

                        init_weights(model, w_type, w_scale)

                        trainCEs, trainREs = model.train(
                            est_inputs, est_labels,
                            alpha=lr, eps=100, # epochs are set from here
                            lr_decay=lr_decay,
                            val_inputs=val_inputs, val_labels=val_labels,  # ← NEW
                            patience=12, min_delta=0.0001
                        )

                        val_CE, val_RE = model.test(val_inputs, val_labels)

                        val_row = {
                            "hidden": h,
                            "lr": lr,
                            "lr_decay": lr_decay,
                            "activation": act,
                            "init_type": w_type,
                            "w_scale": w_scale,
                            "est_CE": trainCEs[-1],  # CE on estimation subset (last epoch)
                            "val_CE": val_CE
                        }
                        results.append(val_row)

                        print(f"H={h:2d}  LR={lr:.3f}  DECAY={lr_decay:.3f}  "
                              f"ACT={act:6s}  INIT={w_type:7s}  Wscale={w_scale:.3f}  "
                              f"Val CE={val_CE:.3f}  Val RE={val_RE:.3f}")

                        if val_CE < best_val_CE:
                            best_val_CE = val_CE
                            best_model = model
                            best_params = {"hidden": h,
                                           "lr": lr,
                                           "act": act,
                                           "init": w_type,
                                           "w_scale": w_scale,
                                           "lr_decay": lr_decay}

save_data(results)
grid_time = time.time() - grid_start
print(f"\nGrid‑search finished in {fmt(grid_time)}")

###### 5. TRAIN BEST MODEL ON FULL TRAIN SET ######

train_start = time.time() # time tracking for final training with best parameters

print("\nBest hyperparameters:", best_params)
final_model = MLPClassifier(dim_in=2,
                            dim_hid=best_params["hidden"],
                            n_classes=3,
                            act_name=best_params["act"])

init_weights(final_model,
             best_params["init"],
             best_params["w_scale"])

final_train_inputs = train_inputs
final_train_labels = train_labels

trainCEs, trainREs = final_model.train(final_train_inputs, final_train_labels,
                                       alpha=best_params['lr'], eps=300, lr_decay=best_params['lr_decay'], )
                                       # FIXME eps=300 for final model, not 100
train_time = time.time() - train_start
print(f"Final model training time: {fmt(train_time)}")

###### 6. TESTING ######

test_start = time.time() # time tracking for final model testing
test_CE, test_RE = final_model.test(test_inputs, test_labels)

test_time = time.time() - test_start
print(f"Final test pass time: {fmt(test_time)}")
print(f"\nFinal Test Results: CE={test_CE:.2%}, RE={test_RE:.5f}")


###### 7. CONFUSION MATRIX ######

_, test_pred = final_model.predict(test_inputs)
conf = confusion_matrix(test_labels, test_pred)
print("\nConfusion Matrix:\n", conf)
conf_pct = 100 * conf / conf.sum(axis=1, keepdims=True)
print(conf_pct)
np.savetxt("confusion_matrix.csv", conf_pct, fmt="%.2f", delimiter=",")


###### 8. PLOTTING ######

plot_both_errors(trainCEs, trainREs, testCE=test_CE, testRE=test_RE, block=False)
plt.savefig("error_curves.png", dpi=300)
plot_dots(final_train_inputs, final_train_labels, block=False, title="Decision Regions for Train Data")
plt.savefig("decision_train.png", dpi=300)
plot_dots(test_inputs, test_labels, test_pred, title="Decision Regions for Test Data", block=False)
plt.savefig("decision_test.png", dpi=300)
