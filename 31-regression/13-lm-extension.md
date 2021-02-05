---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Linear Regression - Extension

No models are perfect. In this section we introduce what happen when our model is misspecified or when some assumptions fail. We will also introduce some alternative models, e.g. Lasso, ridge regression, etc.

(lm-omit-variable)=
## Omit a Variable

Suppose the true model is

$$
\boldsymbol{y} = \boldsymbol{X}_{n \times p} \boldsymbol{\beta} + \boldsymbol{\varepsilon}  
$$

And we omit one explanatory variable $X_j$. Thus, our new design matrix has size $n \times (p-1)$, denoted by $\boldsymbol{X}_{-j}$. Without loss of generality, let it be in the last column of the original design matrix, i.e. $\boldsymbol{X} = \left[ \boldsymbol{X} _{-j} \quad \boldsymbol{x}_j \right]$. The new estimated coefficients vector is denoted by $\hat{\boldsymbol{\beta}}_{-j}$. The coefficient for $\boldsymbol{x}_j$ in the true model is denoted by $\beta_j$, and the vector of coefficients for other explanatory variables is denoted by $\boldsymbol{\beta} _{-j}$. Hence, $\boldsymbol{\beta} ^\top = \left[ \boldsymbol{\beta} _{-j} \quad \beta_j \right] ^\top$.

``` {margin}
Though the common focus is on bias, omitting a variable probably decreases variance. See the relevant section [below](lm-include-variable), or the variance expression [above](lm-inference-variance).
```

*Question: Is $\hat{\boldsymbol{\beta}}_{-j}$ unbised for $\boldsymbol{\beta}_{-j}$?*

*Answer: No. Omitting a relevant variable increases bias. There is a deterministic identity for the bias.*

We will see the meaning of “relevant” later.

We first find the expression of the new estimator $\hat{\boldsymbol{\beta}}_{-j}$

$$\begin{align}
 \hat{\boldsymbol{\beta} }_{-j}
&= \left( \boldsymbol{X} ^\top _{-j} \boldsymbol{X}  _{-j} \right) ^{-1} \boldsymbol{X} ^\top _{-j} \boldsymbol{y} \\
&= \left( \boldsymbol{X} ^\top _{-j} \boldsymbol{X}  _{-j} \right) ^{-1} \boldsymbol{X} ^\top _{-j} \left\{ \left[ \boldsymbol{X} _{-j} \quad \boldsymbol{x}_j \right]\left[\begin{array}{l}
\boldsymbol{\beta} _{-j}  \\
\beta _j
\end{array}\right] + \boldsymbol{\varepsilon}  \right\}\\
&= \left( \boldsymbol{X} ^\top _{-j} \boldsymbol{X}  _{-j} \right) ^{-1} \boldsymbol{X} ^\top _{-j} \left( \boldsymbol{X} _{-j} \boldsymbol{\beta} _{-j} +  \boldsymbol{x}_j \beta _j + \boldsymbol{\varepsilon}  \right) \\
&=  \boldsymbol{\beta} _{-j} + \left[ \left( \boldsymbol{X} ^\top _{-j} \boldsymbol{X}  _{-j} \right) ^{-1} \boldsymbol{X} ^\top _{-j} \right]\left(  \boldsymbol{x}_j \beta _j+ \boldsymbol{\varepsilon}  \right)\\
\end{align}$$

The expectation, therefore, is

$$
\operatorname{E}\left( \hat{\boldsymbol{\beta} }_{-j} \right) =  \boldsymbol{\beta} _{-j} + \left[ \left( \boldsymbol{X} ^\top _{-j} \boldsymbol{X}  _{-j} \right) ^{-1} \boldsymbol{X} ^\top _{-j} \boldsymbol{x}_j \right]\beta _j\\
$$

What is $\left( \boldsymbol{X} ^\top _{-j} \boldsymbol{X} _{-j} \right) ^{-1} \boldsymbol{X} ^\top _{-j} \boldsymbol{x}_j$? You may recognize this form. It is actually the vector of estimated coefficients when we regress the omitted variable $X_j$ on all other explanatory variables $\boldsymbol{X} _{-j}$. Let it be $\boldsymbol{\alpha}_{(p-1) \times 1}$.

Therefore, we have, for the $k$-th explanatory variable in the new model,

$$
\operatorname{E}\left( \hat{\beta} _{-j,k} \right) = \beta_{k} + \alpha_k \beta_j
$$


So the bias is $\alpha_k \beta_j$. The sign can be positive or negative.

This identity can be converted to the following diagram. The explanatory variable $X_k$ is associated with the response $Y$ in two ways. First is directly by itself with strength is $\beta_k$, and second is through the omitted variable $X_j$, with a “compound” strength $\alpha_k \beta_j$.

$$
X_k \quad \overset{\quad \beta_{k} \quad }{\longrightarrow} \quad Y
$$

$$
\alpha_k \searrow \qquad \nearrow \beta_j
$$

$$
X_j
$$

When will the bias be zero?

-   If $\alpha_k = 0$, that is, the omitted variable $X_j$ and the concerned explanatory variable $X_k$ is uncorrelated, i.e., $\boldsymbol{x}_j ^\top \boldsymbol{x}_k = 0$ in the design matrix.
-   If $\beta_j = 0$, that is, the omitted variable $X_j$ and the response $Y$ is uncorrelated, i.e., $\boldsymbol{x}_j ^\top \boldsymbol{y} = 0$.

```{margin}
The takeaway here is that we should include all relevant omitted factors to reduce bias. But in practice, we can never know what all relevant factors are, and rarely can we measure all relevant factors.
```

That’s how we define “relevant”.

What is the relation between the sample estimates? The relation has a similar form.

$$
\hat{\beta }_{-j,k} =  \hat{\beta}_k + \hat{\alpha}_k\hat{\beta}_j
$$

Proof: TBD. Need linear algebra about inverse.

Verify:

```{code-cell}
import numpy as np
from sklearn.linear_model import LinearRegression

n = 1000
b0 = np.ones(n)
x1 = np.random.normal(0,1,n)
x2 = np.random.normal(0,1,n)
rho = 0.5
x3 = rho * x2 + np.sqrt(1-rho**2) * np.random.normal(0,1,n)
e = np.random.normal(0,1,n)*0.1
y = 1 + 1* x1 + 2*x2 + 3*x3 + e
y = y.reshape((-1,1))
X = np.vstack([b0,x1,x2,x3]).transpose()

lm = LinearRegression(fit_intercept=False).fit(X, y)
print("coefficients in y ~ x1 + x2 + x3 :", lm.coef_)
r = y - lm.predict(X)

lmo = LinearRegression(fit_intercept=False).fit(X[:, :-1], y)
print("coefficients in y ~ x1 + x2 :", lmo.coef_)
ro = y - lmo.predict(X[:, :-1])

lmx = LinearRegression(fit_intercept=False).fit(X[:, :-1], X[:, [-1]])
print("coefficients in x3 ~ x1 + x2 :", lmx.coef_)
rx = y - lmx.predict(X[:, :-1])

print("reconstruction difference of b0, b1, b2 :", lm.coef_[0,:3] + lmx.coef_[0] * lm.coef_[0, -1] - lmo.coef_[0])
```



(lm-include-variable)=
## Include a Variable

What if we add a new variable $X_j$? What will happen to the existing estimator $\hat\beta_k$?

Increase

$$\operatorname{Var}\left(\hat{\beta}_{k}\right)=\sigma^{2} \frac{1}{1-R_k^{2}} \frac{1}{\sum_{i}\left(x_{i k}-\bar{x}_{k}\right)^{2}}$$

if $R_{k}^2$ increases. When will $R^2_{k}$ be unchanged? When the new variable $X_j$ has no explanatory power to $X_k$. See the [section](lm-rss-nonincreasing).

In terms of bias, if we say the model with $X_p$ is "true", then $\operatorname{E}\left( \hat{\beta}_k \right)$ is probably closer to $\beta_k$ according to the equation described in the above [section](lm-omit-variable).


## Special Observations

### Outliers

What if an outlier exists?

- If outlier is a mistake (typo) you can drop it (or correct it)

- If outlier is valid but unusual, look for robustness – does dropping it change answer?

- If it does change answer, report both versions – and argue for the approach you think more appropriate

## Multicollinearity

Definition (Multicollinearity)  
Multicollinearity measure the extent of pairwise correlation of variables in the design matrix.

```{margin} Multicollinearity in computation
From numerical algebra's perspective, the extent of correlation of variables in the design matrix $\boldsymbol{X}$ determines the condition number of $\boldsymbol{X} ^\top \boldsymbol{X}$. As the correlation increases, its inverse becomes unstable. When perfect linear relation exists, then $\boldsymbol{X} ^\top \boldsymbol{X}$ is not of full rank, and thus no inverse exists.
```

Definition (Perfect multicollinearity)  
A set of variables is perfectly multicollinear if a variable does not vary, or if there is an exact linear relationship between a set of variables:

$$
X_{j}=\delta_{0}+\delta_{1} X_{1}+\cdots+\delta_{j-1} X_{j-1}+\delta_{i+1} X_{i+1}+\cdots+\delta_{k} X_{k}
$$

As long as the variables in the design matrix are not uncorrelated, then multicollinearity exists.

### Diagnosis

Some common symptoms include
- Large standard error $\operatorname{se}(\beta_j)$
- Overall $F$-test is significant, $R^2$ is good, but individual $t$-tests are not significant due to large standard errors.

We can measure the extent of multicollinearity by **variance inflation factor** (VIF) for each explanatory variable.

$$
\operatorname{VIF}_j = \frac{1}{1-R_j^2}
$$

where $R_j^2$ is the value of $R^2$ when we regress $X_j$ over all other explanatory variables excluding $X_j$. The value of $\operatorname{VIF}_j$ can be interpreted as: the standard error $\operatorname{se}(\beta)$ is $\sqrt{\operatorname{VIF}_j}$ times larger than it would have been without multicollinearity.

A second way of measurement is the **condition number** of $\boldsymbol{X} ^\top \boldsymbol{X}$. If it is greater than $30$, then we can conclude that the multicollinearity problem cannot be ignored.

$$
\kappa_2 \left( \boldsymbol{X} ^\top \boldsymbol{X}  \right) = \sqrt{\frac{\lambda_1 (\boldsymbol{X} ^\top \boldsymbol{X} )}{\lambda_p (\boldsymbol{X} ^\top \boldsymbol{X} )} }
$$

Finally, **correlation matrix** can also be used to measure multicollinearity since it is closely related to the condition number $\kappa_2 \left( \boldsymbol{X} ^\top \boldsymbol{X} \right)$.

### Consequences

1.  It inflates $\operatorname{Var}\left( \hat{\beta}_j \right)$.

    $$\begin{align}
     \operatorname{Var}\left( \hat{\beta}_j \right)
     &= \sigma^2 \frac{1}{1- R^2_{j}} \frac{1}{\sum_i (x_{ij} - \bar{x}_j)^2}  \\
     &=  \sigma^2 \frac{\operatorname{VIF}_j}{\operatorname{Var}\left( X_j \right)}  
     \end{align}$$

    When perfect multicollinearity exists, the variance goes to infinity since $R^2_{j} = 1$.

2.  $t$-tests fail to reveal significant predictors, due to 1.

3.  Estimated coefficients are sensitive to randomness in $Y$, i.e. unreliable. If you run the experiment again, the coefficients can change dramatically, which is measured by $\operatorname{Var}\left( \hat{\boldsymbol{\beta} } \right)$.

4.  If $\operatorname{Corr}\left( X_1, X_2 \right)$ is large, then we expect to have large $\operatorname{Var}\left( \hat{\beta}_1 \right), \operatorname{Var}\left( \hat{\beta}_2 \right), \operatorname{Var}\left( \hat{\beta}_1, \hat{\beta}_2 \right)$, but $\operatorname{Var}\left( \hat{\beta}_1 + \hat{\beta}_2 \right)$ can be small. This means we cannot distinguish the effect of $X_1 + X_2$ on $Y$ is from $X_1$ or $X_2$, i.e. **non-identifiable**.

    ```{dropdown} *Proof*
    By the fact that, for symmetric positive definite matrix $\boldsymbol{S}$, if

    $$
     \boldsymbol{a} ^\top \boldsymbol{S} \boldsymbol{a}  = \boldsymbol{a} \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^\top \boldsymbol{a} = \boldsymbol{b} ^\top \boldsymbol{\Lambda} \boldsymbol{b} = \sum \lambda_i b_i ^2
     $$

    then

    $$
     \boldsymbol{a} ^\top \boldsymbol{S} ^{-1}  \boldsymbol{a}  = \boldsymbol{a} \boldsymbol{U} \boldsymbol{\Lambda} ^{-1}  \boldsymbol{U} ^\top \boldsymbol{a} = \boldsymbol{b} ^\top \boldsymbol{\Lambda} ^{-1}  \boldsymbol{b} = \sum \frac{1}{\lambda_i}  b_i ^2
     $$

    we have:

    If

    $$
     \left( \boldsymbol{x}_1 - \boldsymbol{x}_2 \right) ^\top \left( \boldsymbol{x}_1 - \boldsymbol{x}_2 \right)  = \left( \boldsymbol{e}_1 - \boldsymbol{e}_2   \right) ^\top \boldsymbol{X} ^\top \boldsymbol{X} \left( \boldsymbol{e}_1 - \boldsymbol{e} _2   \right) \approx 0
     $$

    then

    $$
     \operatorname{Var}\left( \hat{\beta}_1 - \hat{\beta}_2 \right)  = \sigma^2  \left( \boldsymbol{e}_1 - \boldsymbol{e}_2   \right) ^\top \left( \boldsymbol{X} ^\top \boldsymbol{X} \right) ^{-1}  \left( \boldsymbol{e}_1 - \boldsymbol{e} _2   \right) \approx \infty
     $$

    If

    $$
     \left( \boldsymbol{x}_1 + \boldsymbol{x}_2 \right) ^\top \left( \boldsymbol{x}_1 + \boldsymbol{x}_2 \right)  = \left( \boldsymbol{e}_1 + \boldsymbol{e}_2   \right) ^\top \boldsymbol{X} ^\top \boldsymbol{X} \left( \boldsymbol{e}_1 + \boldsymbol{e} _2   \right) \approx \text{constant}
     $$

    then

    $$
     \operatorname{Var}\left( \hat{\beta}_1 + \hat{\beta}_2 \right)  = \sigma^2  \left( \boldsymbol{e}_1 + \boldsymbol{e}_2   \right) ^\top \left( \boldsymbol{X} ^\top \boldsymbol{X} \right) ^{-1}  \left( \boldsymbol{e}_1 + \boldsymbol{e} _2   \right) \approx \text{constant}
     $$
    ```

### Implications

If $X_1$ and $X_2$ show high correlation, then

1.  $X_1$ may be a proxy of $X_2$.
2.  $X_1 - X_2$ may just be noise.
3.  If $X_2$ is removed, $X_1$ may still be good for prediction.

## Heteroscedasticity

TBD

## Categorical $X$

dummy variables $X_ij$

when $c = 2$,

interpretation
- $\hat{\beta_1}$: difference in means between the group with $X=1$ and $X=0$.
- $\hat{\beta_0}$: mean of the group with $X=0$.

TBD

https://www.1point3acres.com/bbs/thread-703302-1-1.html
