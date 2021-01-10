# Genearlized Linear Models


## Model

### Structural Form

A genearlized linear model has the form


$$g \left( \mathrm{E}\left( Y_{i} \right) \right) = \boldsymbol{x_i} ^\top \boldsymbol{\beta} $$

where

- $Y_{i}$ is a random component
- $\boldsymbol{x}_i ^\top \boldsymbol{\beta}$ is a linear predictor. As in ordinary least squared models, $\boldsymbol{x}_{i}$ can include interactions, non-linear transformations of the observed covariates and the constant term.
- $g(\cdot)$ is a link function, which connects the linear predictor with the random components.
  - For linear models, $g(\mu)=\mu$, such that $\mathrm{E}\left( Y_i \right) = \boldsymbol{x} _i ^\top \boldsymbol{\beta}$
  - For logistics regression, $g(\mu) = \ln \frac{\mu}{1-\mu}$, such that $\ln \frac{\mathrm{P}\left( Y_i=1\vert \boldsymbol{x} _i \right) }{\mathrm{P}\left( Y_i=0 \vert \boldsymbol{x} _i\right) } = \boldsymbol{x} _i^\top \boldsymbol{\beta}$
  - For poisson regression, $g(\mu) = \ln \mu$, such that $\ln \mathrm{E}\left( Y_i \right)  = \boldsymbol{x} _i^\top \boldsymbol{\beta}$

### Assumptions

- We assume $Y_i \overset{  \text{iid}}{\sim} F$ where $F$ is some distribution, such as normal, binomial, poisson. Thus, we generalize the response $y_i$ from continuous real values in ordinary linear models, to binary response, counts, categories etc. Usually F is from an exponential family.

## Estimation

### Objective Function

### Metrics

## Prediction
