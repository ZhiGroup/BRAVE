#!/usr/bin/env python3
"""
Check correlation matrix from JAGWAS residuals_cor.txt for singularity issues.

Usage:
    python check_correlation_matrix.py <path_to_residuals_cor.txt>

Example:
    python check_correlation_matrix.py /path/to/Left_Putamen/residuals_cor.txt
"""

import sys
import numpy as np

def check_correlation_matrix(filepath):
    # Load the correlation matrix
    R = np.loadtxt(filepath)
    n = R.shape[0]

    print(f"File: {filepath}")
    print(f"Matrix shape: {R.shape}")
    print(f"{'='*60}")

    # 1. Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # descending

    print(f"\n--- Eigenvalue Analysis ---")
    print(f"  Min eigenvalue:       {eigenvalues_sorted[-1]:.6e}")
    print(f"  Max eigenvalue:       {eigenvalues_sorted[0]:.6e}")
    print(f"  Condition number:     {eigenvalues_sorted[0] / max(eigenvalues_sorted[-1], 1e-15):.6e}")
    print(f"  Num eigenvalues < 0:  {np.sum(eigenvalues < 0)}")
    print(f"  Num eigenvalues < 1e-10: {np.sum(eigenvalues < 1e-10)}")
    print(f"  Num eigenvalues < 1e-6:  {np.sum(eigenvalues < 1e-6)}")
    print(f"  Num eigenvalues < 1e-3:  {np.sum(eigenvalues < 1e-3)}")
    print(f"  Effective rank (eig > 1e-6): {np.sum(eigenvalues > 1e-6)}/{n}")

    # 2. Determinant (log scale to avoid overflow/underflow)
    sign, logdet = np.linalg.slogdet(R)
    print(f"\n--- Determinant ---")
    print(f"  Sign: {sign}")
    print(f"  Log|det|: {logdet:.4f}")

    # 3. Diagonal check (should be ~1.0 for correlation matrix)
    diag = np.diag(R)
    print(f"\n--- Diagonal Check ---")
    print(f"  Min diagonal: {diag.min():.6f}")
    print(f"  Max diagonal: {diag.max():.6f}")
    print(f"  All ~1.0:     {np.allclose(diag, 1.0, atol=0.01)}")

    # 4. Off-diagonal correlations
    mask = ~np.eye(n, dtype=bool)
    off_diag = R[mask]
    print(f"\n--- Off-diagonal Correlations ---")
    print(f"  Min:    {off_diag.min():.6f}")
    print(f"  Max:    {off_diag.max():.6f}")
    print(f"  Mean:   {off_diag.mean():.6f}")
    print(f"  Median: {np.median(off_diag):.6f}")
    print(f"  |r| > 0.9: {np.sum(np.abs(off_diag) > 0.9)}")
    print(f"  |r| > 0.95: {np.sum(np.abs(off_diag) > 0.95)}")
    print(f"  |r| > 0.99: {np.sum(np.abs(off_diag) > 0.99)}")

    # 5. Symmetry check
    print(f"\n--- Symmetry Check ---")
    print(f"  Symmetric: {np.allclose(R, R.T)}")
    print(f"  Max asymmetry: {np.max(np.abs(R - R.T)):.2e}")

    # 6. Verdict
    print(f"\n{'='*60}")
    print(f"VERDICT:")

    issues = []
    if eigenvalues_sorted[-1] < 1e-10:
        issues.append("SINGULAR: min eigenvalue < 1e-10")
    elif eigenvalues_sorted[-1] < 1e-6:
        issues.append("NEAR-SINGULAR: min eigenvalue < 1e-6")
    elif eigenvalues_sorted[-1] < 1e-3:
        issues.append("WARNING: min eigenvalue < 1e-3 (ill-conditioned)")

    cond = eigenvalues_sorted[0] / max(eigenvalues_sorted[-1], 1e-15)
    if cond > 1e12:
        issues.append(f"SEVERE: condition number = {cond:.2e} (>1e12)")
    elif cond > 1e6:
        issues.append(f"WARNING: condition number = {cond:.2e} (>1e6)")

    if np.sum(eigenvalues < 0) > 0:
        issues.append(f"NOT POSITIVE DEFINITE: {np.sum(eigenvalues < 0)} negative eigenvalues")

    if np.sum(np.abs(off_diag) > 0.99) > 0:
        issues.append(f"COLLINEARITY: {np.sum(np.abs(off_diag) > 0.99)} pairs with |r| > 0.99")

    eff_rank = np.sum(eigenvalues > 1e-6)
    if eff_rank < n * 0.5:
        issues.append(f"LOW EFFECTIVE RANK: {eff_rank}/{n} ({100*eff_rank/n:.0f}%)")

    if not issues:
        print("  OK - No singularity or conditioning issues detected.")
    else:
        for issue in issues:
            print(f"  {issue}")

    print(f"{'='*60}\n")
    return len(issues) == 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_residuals_cor.txt>")
        sys.exit(1)

    filepath = sys.argv[1]
    ok = check_correlation_matrix(filepath)
    sys.exit(0 if ok else 1)
