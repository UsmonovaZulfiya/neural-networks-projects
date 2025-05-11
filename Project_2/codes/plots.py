import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors

# helper : U-matrix
def compute_u_matrix(weights):
    rows, cols, dim = weights.shape
    u = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            neigh_dists = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if 0 <= r+dr < rows and 0 <= c+dc < cols:
                    dist = np.linalg.norm(weights[r,c] - weights[r+dr,c+dc])
                    neigh_dists.append(dist)
            u[r, c] = np.mean(neigh_dists)
    return u

# helper : class activation map
def class_activation(bmu_coords, y, rows, cols):
    grid_cls = -np.ones((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            idx = np.where((bmu_coords == [r,c]).all(axis=1))[0]
            if len(idx):
                grid_cls[r,c] = np.bincount(y[idx]).argmax() + 1   # classes 1..3
    return grid_cls


def make_all_plots(som, X, y, save_dir='figures', title_prefix='best'):
    os.makedirs(save_dir, exist_ok=True)
    rows, cols, _ = som.weights.shape

    # quantisation error & avg adjustment
    plt.figure(); plt.plot(som.quant_err, lw=2)
    plt.xlabel('Epoch'); plt.ylabel('Quantisation Error')
    plt.title('QE vs Epoch')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{title_prefix}_qe_curve.png', dpi=300)
    plt.close()

    plt.figure(); plt.plot(som.avg_adj, lw=2)
    plt.xlabel('Epoch'); plt.ylabel('Average adjustment')
    plt.title('Average adjustment vs Epoch')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{title_prefix}_adj_curve.png', dpi=300)
    plt.close()

    # BMU coords for every sample
    bmu = np.array([som.winner(X[:, i]) for i in range(X.shape[1])])

    # neuron-class activation map
    grid_cls = class_activation(bmu, y-1, rows, cols)  # y-1 ==> 0,1,2 for cmap
    cmap_classes = colors.ListedColormap(['#d73027', '#1a9850', '#4575b4'])
    plt.figure(figsize=(6,5))
    im = plt.imshow(grid_cls, cmap=cmap_classes, interpolation='none')
    plt.title('Neuron-class activation map')
    cbar = plt.colorbar(im, ticks=[0,1,2])
    cbar.ax.set_yticklabels(['Class 1','Class 2','Class 3'])
    plt.savefig(f'{save_dir}/{title_prefix}_class_map.png', dpi=300)
    plt.close()

    # attribute heatmaps (7 features)
    feat_names = ['Area A','Perimeter P','Compactness C',
                  'Kernel length','Kernel width','Asymmetry','Groove length']
    for k, name in enumerate(feat_names):
        heat = som.weights[:,:,k]
        plt.figure(figsize=(6,5))
        plt.imshow(heat, cmap='viridis')
        plt.title(f'Heatmap: {name}')
        plt.colorbar()
        plt.savefig(f'{save_dir}/{title_prefix}_feat{k+1}.png', dpi=300)
        plt.close()

    # U-matrix
    u_mat = compute_u_matrix(som.weights)
    plt.figure(figsize=(6,5))
    plt.imshow(u_mat, cmap='inferno')
    plt.title('U-matrix')
    plt.colorbar()
    plt.savefig(f'{save_dir}/{title_prefix}_umatrix.png', dpi=300)
    plt.close()

    # optional lattice overlay in PC space
    try:
        from sklearn.decomposition import PCA
        pc = PCA(2).fit_transform(X.T)
        pc_min, pc_max = pc.min(axis=0), pc.max(axis=0)
        # neuron lattice in PC space
        coords = np.stack(np.meshgrid(np.arange(rows),
                                      np.arange(cols), indexing='ij'), -1).reshape(-1,2)
        coords = coords.astype(float)
        coords[:,0] = pc_min[0] + coords[:,0]/(rows-1)*(pc_max[0]-pc_min[0])
        coords[:,1] = pc_min[1] + coords[:,1]/(cols-1)*(pc_max[1]-pc_min[1])

        plt.figure(figsize=(6,6))
        plt.scatter(pc[:,0], pc[:,1], s=20, alpha=0.5, color='gray', label='Samples')
        # draw lattice
        for r in range(rows):
            idx = (coords[:,0].reshape(rows,cols)[r], coords[:,1].reshape(rows,cols)[r])
            plt.plot(idx[0], idx[1], lw=0.4, color='red')
        for c in range(cols):
            idx = (coords[:,0].reshape(rows,cols)[:,c], coords[:,1].reshape(rows,cols)[:,c])
            plt.plot(idx[0], idx[1], lw=0.4, color='red')
        plt.title('Neuron lattice overlay (PC1 vs PC2)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{title_prefix}_lattice_overlay.png', dpi=300)
        plt.close()
    except ImportError:
        print("sklearn not installed – skipping PC overlay figure.")

    print(f'All figures saved to “{save_dir}”.')


def plot_training_curves(som, alpha_s, alpha_f, lambda_s, lambda_f, epochs,
                         save_to='figures/qe_adj_alpha_lambda.png'):

    import matplotlib.pyplot as plt
    import numpy as np

    frac   = np.arange(epochs) / (epochs - 1)
    alpha  = alpha_s  * (alpha_f  / alpha_s)  ** frac
    lam    = lambda_s * (lambda_f / lambda_s) ** frac

    fig, ax1 = plt.subplots(figsize=(6,4))

    ax1.plot(som.quant_err, color='tab:orange',  label='avg. distance')
    ax1.plot(som.avg_adj,  color='tab:blue',     label='avg. adjust')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('QE / adjustment')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(alpha, color='gray',        lw=1.2,  label='alpha decay')
    ax2.plot(lam,   color='gray', ls='--', lw=1.2, label='lambda decay')
    ax2.set_ylabel('α, λ  (normalised)')
    ax2.set_ylim(0, 1.05)

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    fig.tight_layout()
    plt.savefig(save_to, dpi=300)
    plt.close()
