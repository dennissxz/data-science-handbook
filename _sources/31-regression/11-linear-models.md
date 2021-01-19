

# Linear Models

- Problem: regression
- Objective: minimize MSE
- Solved by: closed form, gradient descent

## Model

### Structural Form
$$y_i  = \boldsymbol{x}_i ^\top \boldsymbol{\beta}  + \varepsilon_i $$

### Assumptions

- $\boldsymbol{x}_i$ is known and fixed
- $\epsilon_i \overset{\text{iid}}{\sim}   N(0,\sigma^2)$


## Estimation

### Objective Function

We can estimate the parameter $\hat{\boldsymbol{\beta}}$ by minimizing the MSE

$$\hat{\boldsymbol{\beta}} =  \underset{\boldsymbol{\beta} }{\mathrm{argmin}} \, \left\Vert \boldsymbol{y}  - \boldsymbol{X}  \boldsymbol{\beta}  \right\Vert ^2$$

The closed form solution by differenciation of MSE is
$$\hat{\boldsymbol{\beta}} = \left( \boldsymbol{X} ^\top \boldsymbol{X}   \right)^{-1}\boldsymbol{X} \boldsymbol{y}  $$


### Metrics

- MSE
$$\mathrm{MSE}(\boldsymbol{\beta}) = \left\Vert \boldsymbol{y}  - \boldsymbol{X}  \boldsymbol{\beta}  \right\Vert ^2$$

- $R$-squared

R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Whereas correlation explains the strength of the relationship between an independent and dependent variable, R-squared explains to what extent the variance of one variable explains the variance of the second variable.

relation with $\beta$ in simple linear models:

if no intercept

## Prediction

For a new observation $\boldsymbol{x}_i$, the predicted value $\hat{y}_i$ is

$$\hat{y}_i = \boldsymbol{x}_i^\top \boldsymbol{\beta}$$





## What is ANOVA?
The Analysis Of Variance, popularly known as the ANOVA, can be used in cases where there are more than two groups.

https://www.1point3acres.com/bbs/thread-703302-1-1.html
