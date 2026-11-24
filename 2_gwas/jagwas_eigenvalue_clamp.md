# JAGWAS residual-correlation eigenvalue clamp

## Problem

JAGWAS inverts the residual correlation matrix R (128x128, one entry per BRE
dimension) to combine the per-dimension association statistics into a single
joint test. Some regions produce an ill-conditioned R. The worst case is the
thalamus, where the condition number reaches about 6e6. A near-singular R makes
the inverse blow up along its smallest eigen-directions, and that noise feeds
straight into the joint statistic. The result is an inflated joint test for the
affected regions.

## Fix

Invert R through its eigendecomposition instead of a plain matrix inverse, and
clamp the small eigenvalues before inverting. The threshold is

    tol = max_eigval * 1e-5

Any eigenvalue below `tol` is inverted as `1/tol` rather than `1/eigval`. This
caps the effective condition number at 1e5. The clamp targets the problematic
tail (eigenvalues around 1e-5) and leaves the bulk (about 0.01 to 80) untouched.

Clamping rather than zeroing keeps every eigen-direction. When clamping is active, however,
the joint statistic is a weighted sum of chi-squared components rather than exactly
chi-squared(d): each clamped direction contributes 1/tol instead of 1/eigval, which lowers the
null mean of the statistic below d. P-values are still evaluated against chi-squared(d) for
the retained dimensionality d, so the test is conservative when clamping is active.

```cpp
// Regularized inversion via eigendecomposition with Tikhonov-style clamping.
// Strategy: clamp eigenvalues below tol = max_eigval * 1e-5 before inverting.
// This caps the effective condition number at 1e5 (original can be ~6e6 for thalamus).
// Clamping (not zeroing) keeps all directions; with clamping active the statistic is a
// weighted sum of chi-squared terms, so testing against chi-squared(d) is conservative.
arma::vec eigval;
arma::mat eigvec;
arma::eig_sym(eigval, eigvec, cor);
double max_eigval = eigval.max();
double tol = max_eigval * 1e-5;  // clamp threshold: condition number cap ~1e5
int n_clamped = 0;
for (arma::uword k = 0; k < eigval.n_elem; k++) {
    if (eigval(k) < tol) {
        eigval(k) = 1.0 / tol;  // clamp: invert at floor, not zero
        n_clamped++;
    } else {
        eigval(k) = 1.0 / eigval(k);
    }
}
arma::mat cor_i = eigvec * arma::diagmat(eigval) * eigvec.t();
```

The binary writes `max_eigval`, `tol`, and `n_clamped` to stderr per run so you
can see how many directions were clamped for each region.

## Binary

The modified JAGWAS binary is set via `cfg.tools.jagwas` (see
`config/paths.yaml`).
