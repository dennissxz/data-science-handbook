# Factor Analysis

Goal: Model $p$ random, observable variables as linear combinations of very **few** underlying variables commonly called **factors**.
  - Factors are unobserved, hidden, sometimes not-easy-to measure, latent random variables, e.g. intelligence.
  - Hopefully the factors have reasonable interpretations in terms of the
subject matter.

It focus the covariance structure of the $p$ variables.


## Model

We introduce the orthogonal factor model, which assumes that the factors are uncorrelated.

The orthogonal factor model formulates a $p$-variate random vector $\boldsymbol{x}$ of mean $\boldsymbol{\mu}$ as a **linear** model of $m \le p$ underlying factors $\boldsymbol{f} = [F_1, F_2, \ldots, F_m]$.


$$
\boldsymbol{x}_{p \times 1}=\boldsymbol{\mu}_{p \times 1}+L_{p \times m} \boldsymbol{f}_{m \times 1}+\varepsilon_{p \times 1}
$$


- $\boldsymbol{x}$ is the observed vector of $p$ random variables
- $\boldsymbol{f}$ is the model-assumed, unobservable vector of $m$ random components called **common factors**. The factors are assumed to be uncorrelated (strong assumption), usually normalized

  $$
  \mathbb{E} [\boldsymbol{f} ] = \boldsymbol{0} _m, \quad \operatorname{Cov}\left( \boldsymbol{f}  \right) = \mathbb{E} [\boldsymbol{f} \boldsymbol{f} ^{\top} ]  = \boldsymbol{I} _m
  $$

  - If we further assume $\boldsymbol{f} \sim \mathcal{N}$, then the common factors are then independent.
  - If we drop the uncorrelated assumption, and impose some covariance structure of the factors, we have more general factor models.

- loading matrix
- **specific factors**
  - $\operatorname{Cov}\left( \boldsymbol{\varepsilon} \right) = \boldsymbol{\Psi}$

  independent with $\boldsymbol{f}$.

By the above assumptions, we have relations in covariance structure
- $\boldsymbol{\Sigma} = \boldsymbol{L} \boldsymbol{L} ^{\top} + \boldsymbol{\Psi}$
- $\operatorname{Cov}\left( \boldsymbol{x} , \boldsymbol{f}  \right) = \boldsymbol{L}$

One observation:




## Relation to

PCA

- same: both for dimension reduction and easy interpretation.
- distinct: While PCA is generally used as a mathematical technique, factor analysis is a statistical model.

Others

Factor model is a type of latent variable model, which has been the origin of the development of many popular statistical models, including Structured Equation Models, Independent Component Analysis, and Probability Principal Component Analysis.
