# Kernel-based Dimensionality Reduction

# Kernel PCA

## Motivation from PCA

The problem of PCA is that we only consider linear subspace.

Non-linear transformation of input space to bigger feature space.

Kernals provide an easier way to **compute dot products** in very high-dimensional feature space.

Kernel PCA is a nonlinear extension of MDS...

Problem

## Choice of Kernel

How to choose a kernel?

In practice, we don't engineer $\phi(\cdot)$, but direcly think about $k(\cdot, \codt)$. $\phi$ may be in infinite dimension.

## Learning
pg 38 summary.

```{margin} c.f. PCA

PCA work with $\boldsymbol{X} ^\top \boldsymbol{X}$, Kernal PCA work with $\boldsymbol{X} \boldsymbol{X} ^\top$.
```

## Pros and Cons


computation issue.

# Relation

## Graph-based Methods

Both are motivates as extensions of MDS
