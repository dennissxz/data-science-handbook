# Multivariate Notations

In machine learning models, we often deal with more than one variables at a time. Below are the notations for multivariate case, and their properties.

## Data Matrix

Suppose there are $p$ random variables $X_1, X_2, \ldots, X_p$ and we have $n$ observed values for each of them. The data matrix is

$$
\boldsymbol{X}=\left(x_{i j}\right)_{n \times p}=\left[\begin{array}{c}
\boldsymbol{x}_{1}^{\top} \\
\vdots \\
\boldsymbol{x}_{i}^{\top} \\
\vdots \\
\boldsymbol{x}_{n}^{\top}
\end{array}\right]=\left[\begin{array}{ccccc}
x_{11} & \cdots & x_{1 j} & \cdots & x_{1 p} \\
\vdots & & \vdots & & \vdots \\
x_{i 1} & \cdots & x_{i j} & \cdots & x_{i p} \\
\vdots & & \vdots & & \vdots \\
x_{n 1} & \cdots & x_{n j} & \cdots & x_{n p}
\end{array}\right]
$$

where
- Column $j$ contains observations of variable $j$.
- Row $i$ is an observed vector $\boldsymbol{x}_i$

## Mean Vector

### Population Mean Vector

The mean vector of a random vector $\boldsymbol{x}$ is defined as

$$
\operatorname{\mathbb{E}}(\boldsymbol{x})=\left[\begin{array}{c}
\operatorname{\mathbb{E}}\left(x_{1}\right) \\
\vdots \\
\operatorname{\mathbb{E}}\left(x_{p}\right)
\end{array}\right]=\left[\begin{array}{c}
\mu_{1} \\
\vdots \\
\mu_{p}
\end{array}\right]=\boldsymbol{\mu}
$$

**Properties**

1. $\operatorname{\mathbb{E}}\left( \boldsymbol{a}^{\boldsymbol{\top}} \boldsymbol{x} \right)=\boldsymbol{a}^{\boldsymbol{\top}} \boldsymbol{\mu}$
1. $\operatorname{\mathbb{E}}\left( \boldsymbol{A x} \right)=\boldsymbol{A} \boldsymbol{\mu}$

### Sample Mean Vector

Let $\bar{x}_i$ be the sample mean of variable $X_i$. The sample mean vector is

$$
\overline{\boldsymbol{x}}_{p \times 1}=\left[\begin{array}{c}
\bar{x}_{1} \\
\vdots \\
\bar{x}_{p}
\end{array}\right]=\frac{1}{n} \boldsymbol{X}^{\top} \boldsymbol{1}
$$



## Covariance Matrix

(covariance-matrix)=
### Population Covariance Matrix

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
&=\operatorname{\mathbb{E}}\left[(\boldsymbol{x}-\operatorname{\mathbb{E}}(\boldsymbol{x}))(\boldsymbol{x}-\operatorname{\mathbb{E}}(\boldsymbol{x}))^{\top}\right] \\
&=\operatorname{\mathbb{E}}\left(\boldsymbol{x} \boldsymbol{x}^{\top}\right)-\operatorname{\mathbb{E}}(\boldsymbol{x}) \operatorname{\mathbb{E}}(\boldsymbol{x}) \\
&=\operatorname{\mathbb{E}}\left(\boldsymbol{x} \boldsymbol{x}^{\top}\right)-\mu \mu ^\top
\end{align}
$$

**Properties**

1. $\boldsymbol{\Sigma}$ is positive definite, and hence $\boldsymbol{\Sigma} ^{-1}$ exists

    This holds unless $x_1, x_2, \ldots, x_p$ is linearly related, in which case we say that $\boldsymbol{x}$ is a **degenerated** random vector, i.e. its effective dimension is less than $p$; in other words, its joint distribution is concentrated in a subspace of lower dimension.

1. $\operatorname{Var}\left( \boldsymbol{a}^\top \boldsymbol{x} + b \right) = \operatorname{Var}\left( \boldsymbol{a} ^\top \boldsymbol{x} \right) = \boldsymbol{a}^\top \boldsymbol{\Sigma} \boldsymbol{a} \ge 0$

    The equality holds iff $\boldsymbol{a} ^\top \boldsymbol{x}
  \ne c$, a constant.

1. $\operatorname{Var}\left( \boldsymbol{A} \boldsymbol{x} + b \right) = \boldsymbol{A} \boldsymbol{\Sigma} \boldsymbol{A} ^\top$

1. The determinant of the covariance matrix $\left\vert \boldsymbol{\boldsymbol{\Sigma}}  \right\vert = \operatorname{det} (\boldsymbol{\boldsymbol{\Sigma}} )$ is called the generalized variance. It changes for scaling of variables like the case of univariate variance. Suppose $\boldsymbol{x}$ follows [multivariate Gaussian](multi-gaussian) $\boldsymbol{x} \sim \mathcal{N}_p(\boldsymbol{\mu} , \boldsymbol{\Sigma})$, then we have the following interpretation for $\operatorname{det} (\boldsymbol{\Sigma})$:
   - $\operatorname{det}(\boldsymbol{\Sigma})$ is a (indirect) measure of the entropy of the Gaussian density

    $$
    H(\mathcal{N} _p)=\frac{p}{2}(1+\ln (2 \pi))+\frac{1}{2} \ln |\Sigma|
    $$

   - $\operatorname{det} (\boldsymbol{\Sigma})$ is proportional to the squared of the [volume](ellipsoid) of the ellipsoid $E(\boldsymbol{\mu} , \boldsymbol{\Sigma}, c) = \left\{\boldsymbol{x} \in \mathbb{R} ^p: (\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \le c \right\}$ which measures the disperse of the "data cloud", i.e. uncertainty.

   The interpretation for other distributions is analogous.

### Sample Covariance Matrix and CSSP

The following sample covariance matrix $\boldsymbol{S}$ is an unbiased estimate of the population covariance matrix $\boldsymbol{\Sigma}$

$$
\boldsymbol{S}_{p \times p}=\left[\begin{array}{cccc}
s_{11} & s_{12} & \cdots & s_{1 p} \\
s_{21} & s_{22} & \cdots & s_{2 p} \\
\vdots & \vdots & \ddots & \vdots \\
s_{p 1} & s_{p 2} & \cdots & s_{p p}
\end{array}\right]
$$

where
- $s_{jj} = s_j ^2$ is the sample variance of $x_j$
- $s_{kj} = \frac{1}{n-1} \sum_{i=1}^{n}\left(x_{i k}-\bar{x}_{k}\right)\left(x_{i j}-\bar{x}_{j}\right)$ is the sample covariance between $x_k$ and $x_j$

In matrix form,

$$
\begin{aligned}
\boldsymbol{S} &=\frac{1}{n-1}\left[\boldsymbol{X}^{\top} \boldsymbol{X}-n \overline{\boldsymbol{x}} \overline{\boldsymbol{x}}^{\top}\right] \\
&=\frac{1}{n-1} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{i}-\overline{\boldsymbol{x}}\right)^{\top} \\
&=\frac{1}{n-1} \boldsymbol{W}
\end{aligned}
$$

where

$$
\boldsymbol{W}_{p \times p}=\boldsymbol{X}^{\top} \boldsymbol{X}-n \overline{\boldsymbol{x}} \overline{\boldsymbol{x}}^{\top}=\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{i}-\overline{\boldsymbol{x}}\right)^{\top}
$$

is called the corrected (centered) sums of squares and products matrix (CSSP). One can view it as a multivariate generalization of the corrected (centered) sum of squares $\sum_i \left( x_i - \bar{x} \right)^2$ in the univariate case.

The determinant of the sample covariance $\left\vert \boldsymbol{S}  \right\vert = \operatorname{det} (\boldsymbol{S} )$ is called the generalized sample variance. It changes for scaling of variables like the case of univariate sample variance. Since $\boldsymbol{S}$ is an estimator for $\boldsymbol{\Sigma}$, the interpretations of $\operatorname{det}(\boldsymbol{S} )$ and $\operatorname{det}(\boldsymbol{\Sigma} )$ are similar. See the above section for $\operatorname{det} (\boldsymbol{\Sigma})$.


(prob-covariance-matrix-of-two-vectors)=
### Covariance Matrix of Two Vectors

The covariance matrix of two random vectors $\boldsymbol{x} _{p\times 1}, \boldsymbol{y} _{q \times 1}$ is defined as

$$
\operatorname{Cov}\left( \boldsymbol{x} _{p \times 1}, \boldsymbol{y} _ {q \times 1} \right) = \operatorname{\mathbb{E}}\left[(\boldsymbol{x}-\boldsymbol{\mu} _x)(\boldsymbol{y}-\boldsymbol{\mu} _y)^{\top}\right]_{p\times q}
$$

Note that the shape is $p \times q$, which implies the non-symmetry of covariance matrix

$$
\operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{y}  \right) \ne \operatorname{Cov}\left( \boldsymbol{y} , \boldsymbol{x}  \right)
$$

**Properties**

1. $\operatorname{Var}\left( \boldsymbol{x}  \right) = \operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{x}  \right)$
1. If $\boldsymbol{x} _1, \boldsymbol{x} _2, \boldsymbol{y}$ are $p \times 1$ vectors, then $\operatorname{Var}\left( \boldsymbol{x} + \boldsymbol{y} \right) = \operatorname{Cov}\left( \boldsymbol{x} _1, \boldsymbol{y}  \right) + \operatorname{Cov}\left( \boldsymbol{x} _2 + \boldsymbol{y} \right)$
1. If $\boldsymbol{x}$ and $\boldsymbol{y}$ are $p \times 1$ vectors, then $\operatorname{Var}\left( \boldsymbol{x} +\boldsymbol{y}  \right) = \operatorname{Var}\left( \boldsymbol{x}  \right) + \operatorname{Var}\left( y \right) + \operatorname{Cov}\left( \boldsymbol{y} ,\boldsymbol{x} \right) + \operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{y} \right)$
1. $\operatorname{Cov}\left( \boldsymbol{A} \boldsymbol{x} , \boldsymbol{B} \boldsymbol{y} \right) = \boldsymbol{A} \operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{y} \right) \boldsymbol{B} ^\top$
1. If $\boldsymbol{x}$ and $\boldsymbol{y}$ are independent, then $\operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{y} \right)$. Note that the converse is not always true.


## Correlation Matrix


### Population Correlation Matrix

The correlation matrix of $p$ random variables $x_1, x_2, \ldots, x_p$ is

$$
\boldsymbol{\rho}=\left[\begin{array}{cccc}
1 & \rho_{12} & \cdots & \rho_{1 p} \\
\rho_{21} & 1 & \cdots & \rho_{2 p} \\
\vdots & & \ddots & \vdots \\
\rho_{p 1} & \rho_{p 2} & \cdots & 1
\end{array}\right]
$$

where $\rho_{ij} = \operatorname{Corr}\left( x_i, x_j \right) = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii} \sigma_{jj}}}$

In matrix form,

$$
\begin{align}
\boldsymbol{\rho} &=\left[\begin{array}{cccc}
\frac{1}{\sqrt{\sigma_{11}}}  & 1 & \cdots & 0 \\
0 & \frac{1}{\sqrt{\sigma_{22}}}  & \cdots & \vdots \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & \frac{1}{\sqrt{\sigma_{pp}}}
\end{array}\right]\left[\begin{array}{cccc}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1 p} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2 p} \\
\vdots & & \ddots & \vdots \\
\sigma_{p 1} & \sigma_{p 2} & \cdots & \sigma_{p p}
\end{array}\right]\left[\begin{array}{cccc}
\frac{1}{\sqrt{\sigma}_{11}}  & 1 & \cdots & 0 \\
0 & \frac{1}{\sqrt{\sigma_{22}}}  & \cdots & \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & \frac{1}{\sqrt{\sigma_{pp}}}
\end{array}\right] \\
&= \boldsymbol{D} ^{-1} \boldsymbol{\Sigma} \boldsymbol{D} ^{-1}
\end{align}
$$

where

$$
\boldsymbol{D} = \left[\begin{array}{cccc}
\sqrt{\sigma}_{11} & 0 & \cdots & 0 \\
0 & \sqrt{\sigma}_{22} & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & \sqrt{\sigma}_{p p}
\end{array}\right] = \left( \operatorname{diag}\left( \boldsymbol{\Sigma}  \right) \right) ^{\frac{1}{2}}
$$

In short we will write $D ^{-1} = \left( \operatorname{diag}\left( \boldsymbol{\Sigma}  \right) \right) ^{-\frac{1}{2}}$.

**Properties**

- $\rho_{ii} = 1$. $\rho_{ij} = \rho_{ji}$. $\rho_{ij} = 0$ iff $\sigma_{ij} = 0$
- Each $\rho_{ij}$ does not change under re-location or rescaling of $x_i$ and $x_j$
- $\operatorname{det}(\boldsymbol{\rho} ) \in [0, 1]$: it is 1 if all variables are independent, and 0 if at least one variable is degenerate $\sigma _{ii} = 0$. The larger the value, higher level of independence, and higher level of uncertainty.
- $\operatorname{det}(\boldsymbol{\rho} ) = \operatorname{det} (\boldsymbol{D} ^{-1} \boldsymbol{\Sigma} \boldsymbol{D} ^{-1} ) = \operatorname{det}(\boldsymbol{D} ^{-1}) \operatorname{det} (\boldsymbol{\Sigma} ) \operatorname{det} (\boldsymbol{D} ^{-1} ) = \operatorname{det} (\boldsymbol{\Sigma}) \prod_i {\sigma}_{ii}$

### Sample Correlation Matrix

From the sample covariance matrix, we can obtain the sample correlation matrix as an estimate of the population correlation matrix

$$
\boldsymbol{R}_{p \times p}=\left[\begin{array}{cccc}
1 & r_{12} & \cdots & r_{1 p} \\
r_{21} & 1 & \cdots & r_{2 p} \\
\vdots & & \ddots & \vdots \\
r_{p 1} & r_{p 2} & \cdots & 1
\end{array}\right]
$$

where $r_{kj} = \frac{s_{kj}}{\sqrt{s_{kk} s_{jj}}}$, which is the sample correlation coefficient between $x_k$ and $x_j$.

In matrix form,

$$
\boldsymbol{R} = \boldsymbol{D} ^{-1}  \boldsymbol{S} \boldsymbol{D} ^{-1}    
$$

where

$$
\boldsymbol{D}=\left[\begin{array}{cccc}
\sqrt{s}_{11} & 0 & \cdots & 0 \\
0 & \sqrt{s}_{22} & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
0 & 0 & \cdots & \sqrt{s}_{p p}
\end{array}\right]
$$


```{note}
The transform from
covariance matrix to correlation matrix

$$\boldsymbol{\rho} = \left( \operatorname{diag}\left( \boldsymbol{\Sigma}  \right) \right) ^{-\frac{1}{2}}  \boldsymbol{\boldsymbol{\Sigma} }  \left( \operatorname{diag}\left( \boldsymbol{\Sigma}  \right) \right) ^{-\frac{1}{2}}
$$

is just a particular application of standardizing a positive definite matrix into a standard form where the diagonal elements all equal to **one**. The original positive definite matrix can be
recovered by means of the the diagonal elements and the standardized matrix.
```
