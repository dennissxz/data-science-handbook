# Multivariate Notations

In machine learning models, we often deal with more than one variables at a time. Below are the notations for multivariate case, and their properties.


## Mean Vector

The mean vector of a random vector $\boldsymbol{x}$ is defined as

$$
E(\boldsymbol{x})=\left[\begin{array}{c}
E\left(x_{1}\right) \\
\vdots \\
E\left(x_{p}\right)
\end{array}\right]=\left[\begin{array}{c}
\mu_{1} \\
\vdots \\
\mu_{p}
\end{array}\right]=\boldsymbol{\mu}
$$

**Properties**

1. $E\left(\boldsymbol{a}^{\boldsymbol{\top}} \boldsymbol{x}\right)=\boldsymbol{a}^{\boldsymbol{\top}} \boldsymbol{\mu}$

1. $E(\boldsymbol{A x})=\boldsymbol{A} \boldsymbol{\mu}$

## Covariance Matrix

Aka variance-covariance matrix.

Covariance matrix of a random vector $\boldsymbol{x}$ summarizes pairwise covariance,

$$
\operatorname{Var}(\boldsymbol{x}) \text { or } \boldsymbol{\Sigma}=\left[\begin{array}{cccc}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1 p} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2 p} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{p 1} & \sigma_{p 2} & \cdots & \sigma_{p p}
\end{array}\right]
$$

where
- $\sigma_{ii} = \operatorname{Cov}\left( x_i, x_j \right) = \operatorname{Var}\left( x_i \right)$
- $\sigma_{ij} = \operatorname{Cov}\left( x_i, x_j \right) = \sigma_{ji}$

In matrix form,

$$
\begin{align}
\operatorname{Var}(\boldsymbol{x})
&=E\left[(\boldsymbol{x}-E(\boldsymbol{x}))(\boldsymbol{x}-E(\boldsymbol{x}))^{\top}\right] \\
&=E\left(\boldsymbol{x} \boldsymbol{x}^{\top}\right)-E(\boldsymbol{x}) E(\boldsymbol{x})
\end{align}
$$

**Properties**

1. $\operatorname{Var}\left( \boldsymbol{a}^\top \boldsymbol{x} + b \right)$

1. $\operatorname{Var}\left( \boldsymbol{a}^\top \boldsymbol{x} + b \right)$


## Covariance Matrix of Two Vectors


### Population Correlation Matrix

## Sample Statistics

### Data Matrix

### Mean
