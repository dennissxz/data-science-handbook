# Factor Analysis

Goal: Model $p$ random, observable variables as linear combinations of very **few** underlying variables commonly called **factors**.
  - Factors are unobserved, hidden, sometimes not-easy-to measure, latent random variables, e.g. intelligence.
  - Hopefully the factors have reasonable interpretations in terms of the
subject matter.

It focus the covariance structure of the $d$ variables.


## Model

We introduce the orthogonal factor model, which assumes that the factors are uncorrelated.

The orthogonal factor model formulates a $p$-variate random vector $\boldsymbol{x}$ of mean $\boldsymbol{\mu}$ as a **linear** model of $k \le d$ underlying factors $\boldsymbol{f} = [F_1, F_2, \ldots, F_k]$.

$$
\boldsymbol{x}_{d \times 1}=\boldsymbol{\mu}_{d \times 1}+\boldsymbol{L} _{d \times k} \boldsymbol{f}_{k \times 1}+\boldsymbol{\varepsilon} _{d \times 1}
$$

- $\boldsymbol{x}$ is a random vector of $p$ random variables with mean $\boldsymbol{\mu}$
- $\boldsymbol{f}$ is the model-assumed, unobservable vector of $m$ random components called **common factors**. The factors are assumed to be uncorrelated (strong assumption), usually normalized,

  $$
  \mathbb{E} [\boldsymbol{f} ] = \boldsymbol{0} _k, \quad \operatorname{Cov}\left( \boldsymbol{f}  \right) = \mathbb{E} [\boldsymbol{f} \boldsymbol{f} ^{\top} ]  = \boldsymbol{I} _k
  $$

  - If we further assume $\boldsymbol{f} \sim \mathcal{N}$, then the common factors are then independent.
  - If we drop the uncorrelated assumption, and impose some covariance structure of the factors, we have more general factor models.

- $\boldsymbol{L}$ is called the **loading matrix**, to be estimated. $\ell_{ij}$ is the loading of the $i$-th variable $X_i$ on the $j$-th common factor $F_j$.
  - It is easy to see $\operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{f}  \right) = \boldsymbol{L}$.
  - $h_i ^2 = \ell_{i1}^2 + \ldots + \ell_{ik}^2 = \operatorname{diag}\left( \boldsymbol{L} \boldsymbol{L} ^{\top} \right)_i$ is called the $i$-th **communality**, which is the portion in $\operatorname{Var}\left( X_i \right)$ contributed by the common factors.
- $\boldsymbol{\varepsilon} = \left(\epsilon_{1}, \cdots, \epsilon_{d}\right)^{\top}$ are unobservable errors, aka **specific factors**, assumed to be independent,

  $$
  \mathbb{E} [\boldsymbol{\varepsilon} ] = \boldsymbol{0} , \quad \operatorname{Cov}\left( \boldsymbol{\varepsilon} \right) = \boldsymbol{\Psi} = \operatorname{diag}(\Psi_1, \ldots, \Psi_d)
  $$

  - $\Psi_i$ is called the **specific variance** or **uniqueness** of variable $X_i$.
  - $\boldsymbol{\varepsilon}$ is independent with $\boldsymbol{f}$.


By the above assumptions, we have the following relations in covariance structure
- $\boldsymbol{\Sigma} = \boldsymbol{L} \boldsymbol{L} ^{\top} + \boldsymbol{\Psi}$
- $\operatorname{Var}\left( X_i \right) = h_i^2 + \Psi_i = \operatorname{rowsumsq}_i (\boldsymbol{L}) + \Psi_i$
- variance = communality + uniqueness
- The proportion of the total variance due to the $j$-th common factor is $\frac{\ell_{1 j}^{2}+\cdots+\ell_{p j}^{2}}{\sigma_{11}+\cdots+\sigma_{p p}} = \frac{\operatorname{colsumsq}_j (\boldsymbol{L})}{\operatorname{tr}\left( \boldsymbol{\Sigma} \right)}$.

One observation:

## Estimation

Given the covariance matrix $\boldsymbol{\Sigma}$ of $\boldsymbol{X}$, recall the decomposition

$$\boldsymbol{\Sigma} = \boldsymbol{L} \boldsymbol{L} ^{\prime}+\boldsymbol{\Psi}$$

How do we estimate $\boldsymbol{L}$ and $\boldsymbol{\Psi}$?

### Principal Component Method

The principal component method is easy to implement, thus commonly used in preliminary estimation of factor loadings. Consider EVD of $\boldsymbol{\Sigma}$:

$$\boldsymbol{\Sigma} = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^{\top} = (\boldsymbol{U} \sqrt{\boldsymbol{\Lambda}})(\boldsymbol{U} \sqrt{\boldsymbol{\Lambda}}) ^{\top}$$

Hence we can choose $\boldsymbol{L}_d = (\boldsymbol{U} \sqrt{\boldsymbol{\Lambda}})$. Instead using all of them, we can use the first $k < d$ columns. Then, $\boldsymbol{L} _k \boldsymbol{L} _k ^\prime \ne \boldsymbol{L} _d \boldsymbol{L} _d$. The difference in the diagonal entires is then interpreted as the specific factors $\boldsymbol{\Psi}$.


$$\begin{aligned}
\boldsymbol{\Psi} &= \operatorname{diag}(\boldsymbol{\Sigma} - \boldsymbol{L} _k \boldsymbol{L} _k ^{\top} ) \\
\psi_{i}&=\sigma_{i i}-\left(\ell_{i 1}^{2}+\cdots+\ell_{i k}^{2}\right) \\
&=0 \quad \text{if } k=d
\end{aligned}$$


In sample data, we replace $\boldsymbol{\Sigma}$ by $\boldsymbol{S}$. The **residual matrix** is defined as $\boldsymbol{S} - (\hat{\boldsymbol{L}} _k \hat{\boldsymbol{L}} ^{\top}  _k + \hat{\boldsymbol{\Psi}})$, where the diagonal entries are 0, and the sum of squared entires is bounded by the sum remaining eigenvalues of $\boldsymbol{S}$

$$
\left\| \boldsymbol{S} - (\hat{\boldsymbol{L}} _k \hat{\boldsymbol{L}} ^{\top}  _k + \hat{\boldsymbol{\Psi}}) \right\| _F^2 \le \hat{\lambda}_{k+1}^2 + \ldots + \hat{\lambda}_d^2
$$

Hence, a way to select $k$ is that $\sum_{i=k+1}^d \lambda_i^2 \approx 0$.

The proportion of the total sample variance due to the $j$-th common factor is $\frac{\operatorname{colsumsq}_j (\boldsymbol{L}_k)}{\operatorname{tr}\left( \boldsymbol{\boldsymbol{S}} \right)} = \frac{\hat{\lambda}_j}{\sum_{i=1}^d \hat{\lambda}_i}$.

### Maximum Likelihood Method

Assume the common factors $\boldsymbol{F}$ and the specific factors $\boldsymbol{\varepsilon}$ are assumed to be normally distributed.

Smaller RSS.

J&W section 9A on Pg. 527.

## Relation to

PCA

- same: both for dimension reduction and easy interpretation.
- distinct: While PCA is generally used as a mathematical technique, factor analysis is a statistical model.

Others

Factor model is a type of latent variable model, which has been the origin of the development of many popular statistical models, including Structured Equation Models, Independent Component Analysis, and Probability Principal Component Analysis.
