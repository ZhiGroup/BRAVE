# JAGWAS with the eigenvalue clamp

The joint test was run with JAGWAS (https://github.com/hanchenlab/JAGWAS) at upstream commit
`cd74c1d5255c3f98400de9fa07e1745f79940a4c`, with one local change: the residual correlation matrix R is inverted through its
eigendecomposition with small eigenvalues clamped (see `../jagwas_eigenvalue_clamp.md`).

```bash
git clone https://github.com/hanchenlab/JAGWAS.git && cd JAGWAS
git checkout cd74c1d
git apply /path/to/BRAVE/2_gwas/jagwas_patch/JAGWAS_eigenvalue_clamp.patch
cd src && make   # edit the makefile include/library paths for Armadillo, R headers and LAPACK/BLAS
```

The patch changes only `src/JAGWAS.cpp`. Our build linked LAPACK/BLAS from a conda R environment
(RcppArmadillo headers); adapt the makefile paths to your system.
