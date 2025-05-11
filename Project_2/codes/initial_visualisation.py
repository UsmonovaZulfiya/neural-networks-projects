import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. Load Seeds data ----------
df = pd.read_csv('seeds.txt', delim_whitespace=True, header=None)   # adjust path if needed
X  = df.iloc[:, :7].values          # (210, 7)

# ---------- 2. Z-score standardise ----------
X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

# ---------- 3. PCA (retain first 2 PCs) ----------
cov = np.cov(X_z, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvecs = eigvecs[:, eigvals.argsort()[::-1]]      # sort by eigen-value
PC = X_z @ eigvecs[:, :2]                          # (210, 2)

# ---------- 4. Build an 18 × 12 conceptual lattice ----------
n_rows, n_cols = 15, 15
row_coords = np.linspace(0, 1, n_rows)
col_coords = np.linspace(0, 1, n_cols)
grid_points = np.array([(r, c) for r in row_coords for c in col_coords])

# scale the unit rectangle so it spans the data extents in PC space
pc_min, pc_max = PC.min(axis=0), PC.max(axis=0)
grid_scaled = grid_points.copy()
grid_scaled[:, 0] = pc_min[0] + grid_points[:, 0] * (pc_max[0] - pc_min[0])   # PC1 axis
grid_scaled[:, 1] = pc_min[1] + grid_points[:, 1] * (pc_max[1] - pc_min[1])   # PC2 axis
grid_scaled = grid_scaled.reshape(n_rows, n_cols, 2)  # for easy plotting

# ---------- 5. Plot ----------
plt.figure(figsize=(6, 6))
plt.scatter(PC[:, 0], PC[:, 1], s=20, color='gray', alpha=0.6, label='Samples')

# lattice: red horizontal & vertical lines
for r in range(n_rows):
    plt.plot(grid_scaled[r, :, 0], grid_scaled[r, :, 1], color='red', lw=0.6)
for c in range(n_cols):
    plt.plot(grid_scaled[:, c, 0], grid_scaled[:, c, 1], color='red', lw=0.6)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Seeds data in PC space with 18×12 lattice overlay')
plt.legend()
plt.tight_layout()
plt.show()
