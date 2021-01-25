# Exponential Families


## Multivariate Gaussian


### Definition

Definition
:  A random vector x is said to have a multivariate normal distribution (multinormal distribution) if **every** linear combination of its components has a univariate normal distribution.


For a multivariate normal distribution $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu} , \boldsymbol{\Sigma} )$, the probability density function is


$$
\begin{equation}
f(\boldsymbol{x} ;\boldsymbol{\mu}, \mathbf{\Sigma})=\frac{1}{(2 \pi)^{p / 2}|\mathbf{\Sigma}|^{1 / 2}} \exp \left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
\end{equation}
$$


where
- $\boldsymbol{\mu}$ is the mean vector
- $\boldsymbol{\Sigma}$ is the covariance matrix, with $\sigma_{ij} = \operatorname{Cov}\left( x_i, x_j \right)$
- The fraction $\frac{1}{(2 \pi)^{p / 2}|\mathbf{\Sigma}|^{1 / 2}}$ is a normalizing constant.

### Properties

More
- Since $\boldsymbol{\Sigma} = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^\top$, then $\left\vert \boldsymbol{\Sigma}  \right\vert = \left\vert \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^\top  \right\vert = \left\vert \boldsymbol{\Lambda}  \right\vert$
- For every multivariate Gaussian $\boldsymbol{x} \sim N(\boldsymbol{\mu} , \boldsymbol{\Sigma} )$ with $\Sigma
  = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^\top$, there exists a transformation $\boldsymbol{x} ^\prime = \boldsymbol{U} ^\top \boldsymbol{x}, \boldsymbol{\mu} ^\prime= \boldsymbol{U} ^\top \boldsymbol{\mu}$ such that

  $$
  f(\boldsymbol{x} ; \boldsymbol{\mu}, \boldsymbol{\Sigma} ) = f(\boldsymbol{x} ^\prime; \boldsymbol{\mu} ^\prime, \boldsymbol{\Lambda})
  $$

  where

  $$\begin{align}
  f(\boldsymbol{x} ^\prime; \boldsymbol{\mu} ^\prime, \boldsymbol{\Lambda})
  &= \frac{1}{(2 \pi)^{p / 2}|\boldsymbol{\Lambda}|^{1 / 2}} \exp \left(-\frac{1}{2}(\mathbf{x}^\prime-\boldsymbol{\mu} ^\prime)^{\top} \boldsymbol{\Lambda} ^{-1}(
    \boldsymbol{x} ^\prime -\boldsymbol{\mu} ^\prime)\right) \\
  &=  \frac{1}{(2 \pi)^{p / 2}\Pi_{i=1}^p \lambda_i} \exp \left(-\sum_{i=1}^p\frac{1}{2\sigma^2}(x_i^\prime-\mu_i ^\prime)^2\right) \\
  &= \prod_{i=1}^{p} \frac{1}{(2 \pi)^{1 / 2} \lambda_{i}} \exp \left(-\frac{1}{2 \lambda_{i}^{2}}\left(x_{i}^{\prime}-\mu_{i}^{\prime}\right)^{2}\right)
  \end{align}$$

  which is a product of PDFs of univariate Gaussians, since $\boldsymbol{\Lambda}$ is diagonal. Geometrically, $U$ rotate the axes of the distribution but keep the function value intact.


- Marginal is also Gaussian
- Conditional (slice) is algo Gaussian

:::{figure,myclass} gaussian-marginal-conditional [Shi 2021]
<img src="../imgs/gaussian-marginal-conditional.png" width = "80%" alt=""/>

Marginal Gaussian and conditional Gaussian are also Gaussians [Shi 2020]
:::

### Visualization

All $\boldsymbol{x}$ satisfy the equality below is a contour. This contour is an ellipsoid.

$$
c = f(\boldsymbol{x} ;\boldsymbol{\mu}, \mathbf{\Sigma})
$$


- $\boldsymbol{\mu}$ determines the center of the ellipsoid.
- $\boldsymbol{U}$ determines the rotation angle of the ellipsoid. The vectors $\boldsymbol{u} _i$ are the directions of the axes of the ellipsoid.
- $\boldsymbol{\Lambda}$ determines the lengths of the axes. The length should be proportional to $\sqrt{\lambda_i}$. If all eigenvalues are the same, the ellipsoid reduces to a ball.
- As shown above, the transformation $\boldsymbol{x} ^\prime = \boldsymbol{U} ^\top \boldsymbol{x}, \boldsymbol{\mu} ^\prime = \boldsymbol{U} ^\top \boldsymbol{\mu}, \boldsymbol{\Sigma} ^\prime = \boldsymbol{\Lambda}$ will transform the distribution will change the center, align the ellipsoid axes to the coordinate axes (so the variables becomes independent and the joint PDF factorizes to univariate PDF), while keep the axes lengths intact (rotation preserve lengths and angles).

In the 2-d case, an ellipsoid reduces to an ellipse.

:::{figure,myclass} gaussian-contour-2d
<img src="../imgs/gaussian-2d.png" width = "50%" alt=""/>

Gaussian ellipse in 2-d plane [Shi 2020]
:::


### Pros and Cons

Pros
- Related to CLM
- Evaluation is convenient
- Marginal/conditional also Gaussian
- Can be convert to product of univariate Gaussians in some rotated space
- Mixtures of Gaussian are sufficient to approximate a wide range of distributions

Cons
- $\boldsymbol{\Sigma}$ is hard and expensive to estimate in high dimensions
  - sol: assume special structure: diagonal, spherical
