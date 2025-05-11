import numpy as np
import pandas as pd

df = pd.read_csv('seeds.txt', delim_whitespace=True, header=None)   # path relative to script

X = df.iloc[:, :7].values
X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

cov = np.cov(X_z, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = eigvals[::-1]
first, second = eigvals[:2]
ratio = first / second

print(f"Eigenvalue-1 = {first:.3f}")
print(f"Eigenvalue-2 = {second:.3f}")
print(f"Variance ratio (λ₁/λ₂) = {ratio:.2f}")

if ratio < 1.5:
    print("≈ square spread ➜ choose square grid (e.g., 15×15).")
else:
    print("elongated spread ➜ stretch map along first PC, e.g., 18×12.")
