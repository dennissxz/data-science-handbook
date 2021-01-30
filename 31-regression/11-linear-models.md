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

# Linear Models

In this section we introduce linear models from a statistics’ perspective. The introduction from econometrics’ perspective or social science’s perspective may be different. In short, the statistics’ perspective focuses on general multivariate cases and heavily rely on linear algebra for derivation, while the econometrics’ or the social science’s perspective prefers to introduce models in univariate cases by basic arithmetics (whose form can be complicated without linear algebra notations) and extend the intuitions and conclusions into multivariate cases.

<!---
My handwritten notes for the graduate level course STAT 343 offered by UChicago statistics department can be found [here](../imgs/lm-notes-applied-stat.pdf).
-->

Personally, I involved in four courses that introduced linear models, i.e. at undergrad/grad level offered by stat/social science department. The style of the two courses offered by the stat departments were quite alike while the graduate level one covered more topics. In both undergrad/grad level courses offered by the social science departments, sometimes I got confused by the course materials that were contradictory to my statistics training , but the instructors had no clear response or even no response at all...

In sum, to fully understand the most fundamental and widely used statistical model, I highly suggest to take a linear algebra course first and take the regression course offered by math/stat department.

## Objective

Linear models aim to model the relationship between a scalar response and one or more explanatory variables in a linear format:

$$Y_i  = \beta_0 + \beta_1 x_{i,1} + \ldots + \beta_{p-1} x_{i,p-1}  + \varepsilon_i $$

for observations $i=1, 2, \ldots, n$.

In matrix form,

$$
\boldsymbol{y} = \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}.
$$

where
- $\boldsymbol{X}_{n\times p}$ is called the design matrix. The first column is usually set to be $\boldsymbol{1}$, i.e., intercept. The remaining $p-1$ columns are designed values $x_{ij}$ where $i = 1, 2, \ldots, n$ and $j=1, \ldots, p-1$. These $p-1$ columns are called explanatory/independent variables, or covariates.
- $\boldsymbol{y}_{n \times 1}$ is a vector of response/dependent variables $Y_1, Y_2, \ldots, Y_n$.
- $\boldsymbol{\beta}_{p \times 1}$ is a vector of coefficients to be estimated.
- $\boldsymbol{\varepsilon}_{n \times 1}$ is a vector of unobserved random errors, which includes everything that we have not measured and included in the model.

When $p=2$, we have

$$
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

which is called **simple linear regression**.

When $p>2$, it is called **multiple linear regression**. For instance, when $p=3$

$$
Y_i = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon_i
$$

When there are multiple dependent variables, we call it **multivariate regression**, which will be introduced in another section.

When $p=1$,

- if we include intercept, then the regression model $y_i = \beta_0$ means that we use a single constant to predict $y_i$. The estimator, $\hat{\beta}_0$, by ordinary least square, should be the sample mean $\hat{y}$.
- if we do not include intercept, then the regression model $y_i = \beta x_i$ means that we expect that $y$ is proportional to $x$. See [here](lm-proportional-model) for details.

:::{admonition,dropdown,note} Fixed or random $\boldsymbol{X}$?
In natural science, researchers design $n\times p$ values in the design matrix $\boldsymbol{X}$ and run experiments to obtain the response $y_i$. We call this kind of data **experimental data**. In this sense, the explanatory variables $x_{ij}$’s are designed before the experiment, so they are also constants. The coefficients $\beta_j$’s are unknown constants. The error term $\varepsilon_i$ is random. The response variable $Y_i$ on the left hand side is random due to the randomness in the error term $\varepsilon_i$.

In social science, most of data is **observational data**. That is, researchers obtain the values of many variables at the same time, and choose one of interest to be the response variable $y_i$ and some others to be the explanatory variables $\boldsymbol{x}_i$. In this case, $\boldsymbol{X}$ is viewed as a data set, and we can talk about descriptive statistics, such as variance of each explanatory variable, or covariance between pair of explanatory variables. This is valid since we often view the columns of a data set as random variables.

However, the inference methods of the coefficients $\boldsymbol{\beta}$ are developed based on the natural science setting, i.e., the values of explanatory variables are pre-designed constants. Many social science courses frequently use descriptive statistics of the explanatory variables which assumes they are random, and apply inference methods which assumes they are constant. This is quite confusing for beginners to linear models.

To be clear, we stick to the natural science setting and make the second assumption below. We use subscript $i$ in every $y_i, x_i, \varepsilon_i$ instead of $y, x, \varepsilon$ which gives a sense that $x$ is random. And we use descriptive statistics for the explanatory variables only when necessary.
:::

## Assumptions

Basic assumptions

1.  $\operatorname{E}\left( y_i \right) = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$ is **linear** in covariates $X_j$.

2.  The values of explanatory variables $\boldsymbol{x}_i$ are known and fixed. Randomness only comes from $\varepsilon_i$.

3.  No $X_j$ is constant for all observations. No exact linear relationships among the explanatory variables (aka no perfect multicollinearity, or the design matrix $\boldsymbol{X}$ is of full rank).

4.  The error terms are uncorrelated $\operatorname{Cov}\left( \varepsilon_i, \varepsilon_j \right)= 0$, with common mean $\operatorname{E}\left( \varepsilon_i \right) = 0$ and variance $\operatorname{Var}\left( \varepsilon_i \right) = \sigma^2$ (homoskedasticity).

    As a result, $\operatorname{E}\left( \boldsymbol{y} \mid \boldsymbol{X} \right) = \boldsymbol{X} \boldsymbol{\beta}$, or $\operatorname{E}\left( y_i \mid x_i \right) = \beta_0 + \beta_1 x_i$ when $p=2$, which can be illustrated by the plots below.

    :::{figure} lm-distribution-of-y-given-x
    <img src="../imgs/lm_cond_distribution.png" width = "40%" alt=""/>

    Distributions of $y$ given $x$ \[Meyer 2021\]
    :::

    :::{figure} lm-observation-of-y-given-x
    <img src="../imgs/lm_xyplane_dots.png" width = "50%" alt=""/>

    Observations of $y$ given $x$ \[Meyer 2021\]
    :::

    To predict $\hat{y}_i$, we just use $\hat{y}_i = \boldsymbol{x}_i ^\top \hat{\boldsymbol{\beta}}$ .

5.  The error terms are independent and follow Gaussian distribution $\varepsilon_i \overset{\text{iid}}{\sim}N(0, \sigma^2)$, or $\boldsymbol{\varepsilon} \sim N_n (\boldsymbol{0} , \sigma^2 \boldsymbol{I} _n)$.

    As a result, we have $Y_i \sim N(\boldsymbol{x}_i ^\top \boldsymbol{\beta} , \sigma^2 )$ or $\boldsymbol{y} \sim N_n(\boldsymbol{X} \boldsymbol{\beta} , \sigma^2 \boldsymbol{I} _n)$

These assumptions are used for different objectives. The first 3 assumptions are the base, and in additiona to them,
- derivation of $\hat{\boldsymbol{\beta}}$ by least squares uses no more assumptions.
- derivation of $\hat{\boldsymbol{\beta}}$ by maximal likelihood uses assumptions 4 and 5.
- derivation of $\operatorname{E}\left( \hat{\boldsymbol{\beta}} \right)$ uses $\operatorname{E}\left( \varepsilon_i \right) = 0$ in 4.
- derivation of $\operatorname{Var}\left( \hat{\boldsymbol{\beta}} \right)$ uses 1, 2, $\operatorname{Cov}\left( \varepsilon_i, \varepsilon_j \right) = 0$ and $\operatorname{Var}\left( \epsilon_i \right) = \sigma^2$ in 4.
- proof of Gaussian-Markov Theorem (BLUE) uses 4.
- derivation of the distribution of $\hat{\boldsymbol{\beta} }$ uses 4 and 5.

:::{admonition,dropdown,note} Zero conditional mean assumption
In some social science or econometrics courses, they follow the “Gauss-Markov assumptions” that are roughly the same to the assumptions, but in different formats. One of them is zero conditional mean assumption.

In general, it says

$$
\operatorname{E}\left( \varepsilon \mid x_1, x_2, \ldots, x_p\right) = 0
$$

For $p=2$, it is

$$\operatorname{E}\left( \varepsilon \mid x  \right) = 0$$

which (in their setting) implies

$$\begin{align}
\operatorname{E}\left( \varepsilon \right)
&= \operatorname{E}\left( \operatorname{E}\left( \varepsilon \mid x \right) \right)\\
&= 0\\
\operatorname{Cov}\left( \varepsilon, x \right)
&= \operatorname{E}\left( \varepsilon x \right) - \operatorname{E}\left( \varepsilon \right)\operatorname{E}\left( x \right)\\
&= \operatorname{E}\left( \operatorname{E}\left( \varepsilon x \mid x \right) \right)- 0 \times \operatorname{E}\left( x \right)\\
&= \operatorname{E}\left( x \operatorname{E}\left( \varepsilon \mid x \right) \right) \\
&= 0
\end{align}$$

Then they these two corollaries are used for [estimation](lm-estimation-by-assumpation).

As discussed above, in their setting $x$ is random (at this stage), so they use notations such as $\operatorname{E}\left( \varepsilon \mid x \right)$ and $\operatorname{Cov}\left( x, \varepsilon \right)$. It also seems that they view $\varepsilon$ as an “overall” measure of random error, instead of $\varepsilon_i$ for specific $i$ in the natural science setting. But they can mean so by using the conditional notation $\operatorname{E}\left( \varepsilon \mid x \right)$.
:::

## Estimation (Learning)

We introduce various methods to estimate the parameters $\boldsymbol{\beta}$ and $\sigma^2$.

### Ordinary Least Squares

The most common way is to estimate the parameter $\hat{\boldsymbol{\beta}}$ by minimizing the sum of squared errors $\sum_i(y_i-\hat{y}_i)^2$.

```{margin} A note on substitution
We substitute the predicted $\hat{\boldsymbol{y} }$ by $\boldsymbol{X} \boldsymbol{\beta}$. The $\boldsymbol{\beta}$ here just means a variable in the optimization problem, not the unknown constant coefficients in our model.
```

$$\begin{align}
\hat{\boldsymbol{\beta}} &= \underset{\boldsymbol{\beta} }{\mathrm{argmin}} \, \left\Vert \boldsymbol{y}  - \hat{\boldsymbol{y}}  \right\Vert ^2 \\
&= \underset{\boldsymbol{\beta} }{\mathrm{argmin}} \, \left\Vert \boldsymbol{y}  - \boldsymbol{X}  \boldsymbol{\beta}  \right\Vert ^2 \\
\end{align}$$

The gradient w.r.t. $\boldsymbol{\beta}$ is

$$\begin{align}
\nabla_{\boldsymbol{\beta}} &= -2 \boldsymbol{X}  ^\top (\boldsymbol{y} - \boldsymbol{X} \boldsymbol{\beta} )  \\
&\overset{\text{set}}{=} \boldsymbol{0}
\end{align}$$

Hence, we have

$$
\boldsymbol{X} ^\top \boldsymbol{X} \boldsymbol{\beta} = \boldsymbol{X} ^\top \boldsymbol{y}
$$

This linear system is called the **normal equation**.

The closed form solution is

$$\hat{\boldsymbol{\beta}} = \left( \boldsymbol{X} ^\top \boldsymbol{X}   \right)^{-1}\boldsymbol{X} ^\top  \boldsymbol{y}  $$


::::{admonition,dropdown,tip} View least squares as projection

Substitute the solve $\hat{\boldsymbol{\beta}}$ into the prediction $\hat{\boldsymbol{y}}$ we have


$$
\hat{\boldsymbol{y}} = \boldsymbol{X} \hat{\boldsymbol{\beta}} = \underbrace {\boldsymbol{X} (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top}_{\boldsymbol{H}} \boldsymbol{y}
$$

Here $\boldsymbol{H}$ is a projection matrix onto the column space (image) of $\boldsymbol{X}$. Recall that a projection matrix onto the column span of a matrix $\boldsymbol{X}$ has the form $\boldsymbol{P} _{\operatorname{col}(\boldsymbol{X} )} = \boldsymbol{X} \boldsymbol{X} ^\dagger$ where $\boldsymbol{X} ^\dagger =  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top$ is the pseudo inverse of $\boldsymbol{X}$.

Essentially, we are trying to find a vector $\hat{\boldsymbol{y}}$ in the column space of the data matrix $\boldsymbol{X}$ that is as close to $\boldsymbol{y}$ as possible, and the closest one is just the projection of $\boldsymbol{y}$ onto $\operatorname{col}(\boldsymbol{X})$, which is $\boldsymbol{H}\boldsymbol{y}$. The distance is measured by the norm $\left\| \boldsymbol{y} - \hat{\boldsymbol{y}}  \right\|$, which is the squared root of sum of squared errors. Note that $\boldsymbol{y} - \hat{\boldsymbol{y}} = (\boldsymbol{I} - \boldsymbol{H} ) \boldsymbol{y} \in \operatorname{col}(\boldsymbol{X}) ^ \bot$ since $\boldsymbol{I} - \boldsymbol{H} = \boldsymbol{I}  - \boldsymbol{P}_{\operatorname{col}(\boldsymbol{X}) } = \boldsymbol{P}_{\operatorname{col}(\boldsymbol{X}) ^ \bot}$ is the projection matrix onto the orthogonal complement $\operatorname{col}(\boldsymbol{X}) ^ \bot$.

:::{figure} lm-projection
<img src="../imgs/lm-projection.png" width = "50%" alt=""/>

Least squares as a projection [[Gold 2017]](https://waterprogramming.wordpress.com/2017/05/12/an-introduction-to-econometrics-part-1-classical-ordinary-least-squares-regression/)
:::

::::


:::{admonition,dropdown,note} Solving the linear system by software
Computing software use specific functions to solve the normal equation $\boldsymbol{X} ^\top \boldsymbol{X} \boldsymbol{\beta} = \boldsymbol{X} ^\top \boldsymbol{y}$ for $\boldsymbol{\beta}$, instead of using the inverse $(\boldsymbol{X} ^\top \boldsymbol{X}) ^{-1}$ directly which can be slow and numerically unstable. For instance, one can use QR factorization of $X$,

$$
\boldsymbol{X} = \boldsymbol{Q} \left[\begin{array}{l}
\boldsymbol{R}_{p \times p}  \\
\boldsymbol{0}_{(n-p) \times p}
\end{array}\right]
$$

Hence,

$$
\begin{aligned}
\| \boldsymbol{y} - \boldsymbol{X}  \boldsymbol{\beta}  \|^{2}
&=\left\|\boldsymbol{Q} ^{\top} \boldsymbol{y}  - \boldsymbol{Q} ^\top \boldsymbol{X} \boldsymbol{\beta}  \right\|^{2} \\
&=\left\|\left(\begin{array}{c}
\boldsymbol{f}  \\
\boldsymbol{r}
\end{array}\right)-\left(\begin{array}{c}
\boldsymbol{R} \boldsymbol{\beta}  \\
\boldsymbol{0}
\end{array}\right)\right\|^{2} \\
&=\|\boldsymbol{f} - \boldsymbol{R} \boldsymbol{\beta} \|^{2}+\|\boldsymbol{r} \|^{2}
\end{aligned}
$$

Finally

$$
\boldsymbol{\beta} = \boldsymbol{R} ^{-1} \boldsymbol{f}
$$
:::

An unbiased estimator of the error variance $\sigma^2 = \operatorname{Var}\left( \varepsilon \right)$ is (to be discussed \[later\])

$$
\hat{\sigma}^2 = \frac{\left\Vert \boldsymbol{y} - \boldsymbol{X} \hat{\boldsymbol{\beta}} \right\Vert ^2}{n-p}
$$

When $p=2$, we have

$$\hat{\beta}_0, \hat{\beta}_1 =  \underset{\beta_0, \beta_1 }{\mathrm{argmin}} \, \sum_i \left( y_i - \beta_0 - \beta_1 x_i \right)^2$$

Differentiation w.r.t. $\beta_1$ gives

$$
- 2\sum_i (y_i - \beta_0 - \beta_1 x_i) x_i = 0
$$

Differentiation w.r.t. $\beta_0$ gives

$$
- 2\sum_i (y_i - \beta_0 - \beta_1 x_i) = 0
$$

Solve the system of the equations, we have

$$\begin{align}
\hat{\beta}_{1} &=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \\
\hat{\beta}_{0} &=\bar{y}-\hat{\beta}_{1} \bar{x}
\end{align}$$

The expression for $\hat{\beta}_0$ implies that the fitted line cross the sample mean point $(\bar{x}, \bar{y})$.

Moreover,

$$
\hat{\sigma}^2 = \frac{1}{n-2} \sum_i \hat\varepsilon_i^2
$$

where $\hat\varepsilon_i = y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i$.

:::{admonition,note} Minimizing mean squared error
The objective function, **sum of squared errors**,

$$
\left\Vert \boldsymbol{y}  - \boldsymbol{X}  \boldsymbol{\beta}  \right\Vert ^2 = \sum_i \left( y_i - \boldsymbol{x}_i ^\top \boldsymbol{\beta} \right)^2
$$

can be replaced by **mean squared error**,

$$
\frac{1}{n} \sum_i \left( y_i - \boldsymbol{x}_i ^\top \boldsymbol{\beta} \right)^2
$$

and the results are the same.
:::




(lm-estimation-by-assumpation)=
### By Assumptions

In some social science courses, the estimation is done by using the assumptions

- $\operatorname{E}\left( \varepsilon \right) = 0$
- $\operatorname{E}\left( \varepsilon \mid X \right) = 0$

The first one gives

$$
\frac{1}{n}  \sum_{i=1}^{n}\left(y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{i}\right)=0
$$

The second one gives

$$\begin{align}
\operatorname{Cov}\left( X, \varepsilon \right)
&= \operatorname{E}\left( X \varepsilon \right) - \operatorname{E}\left( X \right) \operatorname{E}\left( \varepsilon \right) \\
&= \operatorname{E}\left[ \operatorname{E}\left( X \varepsilon \mid X \right) \right] - \operatorname{E}\left(  X \right)\operatorname{E}\left[ \operatorname{E}\left( \varepsilon \mid X\right) \right]\\
&= \operatorname{E}\left[ X \operatorname{E}\left( \varepsilon \mid X \right) \right] - \operatorname{E}\left(  X \right)\operatorname{E}\left[ \operatorname{E}\left( \varepsilon \mid X\right) \right]\\
&= 0
\end{align}$$

which gives

$$
\frac{1}{n}  \sum_{i=1}^{n} x_{i}\left(y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{i}\right)=0
$$

Therefore, we have the same normal equations to solve for $\hat{\beta}_0$ and $\hat{\beta}_1$.



:::{admonition,warning} Warning

Estimation by the two assumptions derived from the zero conditional mean assumption can be problematic. Consider a model without intercept $y_i = \beta x_i + \varepsilon_i$. Fitting by OLS, we have only ONE first order condition

$$
\sum_{i=1}^{n} x_{i}\left(y_{i}-\hat{\beta}_{1} x_{i}\right)=0
$$

If we fit by assumptions, then in addition to the condition above, the first assumption $\operatorname{E}\left( \varepsilon \right) = 0$ also gives

$$
 \sum_{i=1}^{n}\left(y_{i}-\hat{\beta}_{1} x_{i}\right)=0
$$

These two conditions may not hold at the same time.
:::


### Maximum Likelihood

biased. TBD.

### Gradient Descent

TBD.

## Properties

We describe the properties of OLS estimator $\hat{\boldsymbol{\beta}}$ and the corresponding residuals $\hat{\boldsymbol{\varepsilon} }$.

Note that $\hat{\boldsymbol{\beta}}=(\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X}^\top \boldsymbol{y}$ is a random variable, since it is a linear combination of the random vector $\boldsymbol{y}$. This means that, keeping $\boldsymbol{X}$ fixed, repeat the experiment, we will probably get different response values $\boldsymbol{y}$, and hence different $\hat{\boldsymbol{\beta}}$. As a result, there is a sampling distribution of $\hat{\boldsymbol{\beta}}$, and we can find its mean, variance, and conduct hypothesis testing.

### Coefficients

#### Unbiasedness

The OLS estimators are unbiased since

$$\begin{align}
\operatorname{E}\left( \hat{\boldsymbol{\beta} } \right) &= \operatorname{E}\left( (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X}^\top  \boldsymbol{y}  \right) \\
&=  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top  \operatorname{E}\left( \boldsymbol{y} \right) \\
&=  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top  \boldsymbol{X} \boldsymbol{\beta} \\
&= \boldsymbol{\beta}
\end{align}$$

when $p=2$,

$$\begin{align}
\hat{\beta}_{1}
&=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \\
\end{align}$$

To prove unbiasedness, using the fact that for any constant $c$,

$$
\sum_i (x_i - \bar{x})c = 0
$$

Then, the numerator becomes

$$\begin{align}
\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)
&=\sum\left(x_{i}-\bar{x}\right)\left(\beta_{0}+\beta_{1} x_{i}+u_{i}\right) \\
&=\sum\left(x_{i}-\bar{x}\right) \beta_{0}+\sum\left(x_{i}-\bar{x}\right) \beta_{1} x_{i} +\sum\left(x_{i}-\bar{x}\right) u_{i} \\
&=\beta_{0} \sum\left(x_{i}-\bar{x}\right)+\beta_{1} \sum\left(x_{i}-\bar{x}\right) x_{i} +\sum\left(x_{i}-\bar{x}\right) u_{i} \\
&=\beta_{1} \sum\left(x_{i}-\bar{x}\right)^2 +\sum\left(x_{i}-\bar{x}\right) u_{i} \\
\end{align}$$

Hence

$$
\begin{equation}
\hat{\beta}_{1}=\beta_{1}+\frac{\sum\left(x_{i}-\bar{x}\right) u_{i}}{\sum \left(x_{i}-\bar{x}\right)^{2}}
\end{equation}
$$

and

$$
\operatorname{E}\left( \hat{\beta}_1 \right) = \beta_1
$$

(lm-inference-variance)=
#### Variance

The variance (covariance matrix) of the coefficients is

$$\begin{align}
\operatorname{Var}\left( \boldsymbol{\beta}  \right) &= \operatorname{Var}\left(  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X}^\top  \boldsymbol{y}  \right)  \\
&=   (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X}^\top \operatorname{Var}\left( \boldsymbol{y}  \right)  \boldsymbol{X}  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \\
&= \sigma^2 (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1}\\
\end{align}$$

:::{admonition} Note
More specifically, for the $j$-th coefficient estimator $\hat{\beta}_j$, its variance is,

$$\begin{align}
\operatorname{Var}\left( \hat{\beta}_j \right)
&= \sigma^2 \left[ (\boldsymbol{X} ^\top \boldsymbol{X} )^{-1} \right]_{[j,j]} \\
&= \sigma^2 \frac{1}{1- R^2_{j}} \frac{1}{\sum_i (x_{ij} - \bar{x}_j)^2}
\end{align}$$

where $R_j^2$ is the value of [$R$-squared](lm-rsquared) when we regress $X_j$ over all other explanatory variables.

Note that the value of $R^2$ when we regressing $X_1$ to an constant intercept is 0. So we have the particular result below.
:::


When $p=2$, the variance of $\hat{\beta}_1$ is

$$\begin{align}
\operatorname{Var}\left( \hat{\beta}_1 \right)
&= \operatorname{Var}\left( \beta_{1}+\frac{\sum\left(x_{i}-\bar{x}\right) u_{i}}{\sum \left(x_{i}-\bar{x}\right)^{2}} \right)\\
&= \frac{\operatorname{Var}\left( \sum\left(x_{i}-\bar{x}\right) u_{i} \right)}{\left[ \sum \left(x_{i}-\bar{x}\right)^{2} \right]^2}\\
&= \frac{\sum\left(x_{i}-\bar{x}\right)^2 \operatorname{Var}\left( u_{i} \right)}{\left[ \sum \left(x_{i}-\bar{x}\right)^{2} \right]^2}\\
&= \sigma^2 \frac{\sum\left(x_{i}-\bar{x}\right)^2 }{\left[ \sum \left(x_{i}-\bar{x}\right)^{2} \right]^2}\\
&= \frac{\sigma^2}{\sum_{i=1}^n \left(x_{i}-\bar{x}\right)^{2}}\\
\end{align}$$

We conclude that

-   The larger the error variance, $\sigma^2$, the larger the variance of the coefficient estimates.
-   The larger the variability in the $x_i$, the smaller the variance.
-   A larger sample size should decrease the variance.
-   In multiple regression, reduce the relation between $X_j$ and other covariates (e.g. by orthogonal design) can decreases $R^2_{j}$, and hence decrease the variance.

A problem is that the error $\sigma^2$ variance is **unknown**. In practice, we can estimate $\sigma^2$ by its unbiased estimator $\hat{\sigma}^2=\frac{\sum_i (x_i - \bar{x})}{n-2}$ (to be shown \[link\]), and substitute it into $\operatorname{Var}\left( \hat{\beta}_1 \right)$. Since the error variance $\hat{\sigma}^2$ is estimated, the slope variance $\operatorname{Var}\left( \hat{\beta}_1 \right)$ is estimated too, and hence the square root is called standard error of $\hat{\beta}$, instead of standard deviation.

$$\begin{align}
\operatorname{se}\left(\hat{\beta}_{1}\right)
&= \sqrt{\widehat{\operatorname{Var}}\left( \hat{\beta}_1 \right)}\\
&= \frac{\hat{\sigma}}{\sqrt{\sum \left(x_{i}-\bar{x}\right)^{2}}}
\end{align}$$

#### BLUE

Theorem (Gauss–Markov)  
: The ordinary least squares (OLS) estimator has the **lowest** sampling variance within the class of linear unbiased estimators, if the errors in the linear regression model are uncorrelated, have equal variances and expectation value of zero. In abbreviation, the OLS estimator is BLUE: Best (lowest variance) Linear Unbiased Estimator.

:::{admonition,dropdown,seealso} *Proof*

Let $\tilde{\boldsymbol{\beta}} = \boldsymbol{C} \boldsymbol{y}$ be another linear estimator of $\boldsymbol{\beta}$. We can write $\boldsymbol{C} = \left( \boldsymbol{X} ^\top \boldsymbol{X} \right)^{-1} \boldsymbol{X} ^\top + \boldsymbol{D}$ where $\boldsymbol{D} \ne \boldsymbol{0}$. Then

$$\begin{align}
  \operatorname{E}\left( \tilde{\boldsymbol{\beta} } \right)
  &= \operatorname{E}\left( \boldsymbol{C} \boldsymbol{y}   \right)\\
  &= \boldsymbol{C} \operatorname{E}\left( \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}  \right)\\
  &= \boldsymbol{\beta} + \boldsymbol{D} \boldsymbol{X} \boldsymbol{\beta} \\
  \end{align}$$

Hence, $\tilde{\boldsymbol{\beta}}$ is unbiased iff $\boldsymbol{D} \boldsymbol{X} = 0$.

The variance is

$$\begin{align}
  \operatorname{Var}\left( \tilde{\boldsymbol{\beta} } \right)
  &= \boldsymbol{C}\operatorname{Var}\left( \boldsymbol{y}  \right) \boldsymbol{C} ^\top \\
  &= \sigma^2 \boldsymbol{C} \boldsymbol{C} ^\top \\
  &= \sigma^2 \left[ \left( \boldsymbol{X} ^\top \boldsymbol{X}  \right) ^{-1}  + (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top \boldsymbol{D} ^\top + \boldsymbol{D} \boldsymbol{X} \left( \boldsymbol{X} ^\top \boldsymbol{X}  \right) ^\top  + \boldsymbol{D} \boldsymbol{D} ^\top  \right]\\
  &= \sigma^2 \left[ \left( \boldsymbol{X} ^\top \boldsymbol{X}  \right) ^{-1} + \boldsymbol{D} \boldsymbol{D} ^\top  \right]\\
  &= \operatorname{Var}\left( \hat{\boldsymbol{\beta} } \right) + \sigma^2 \boldsymbol{D} \boldsymbol{D} ^\top \\
  \end{align}$$

Since $\sigma^2 \boldsymbol{D} \boldsymbol{D} ^\top \in \mathrm{PSD}$, we have

$$
  \operatorname{Var}\left( \tilde{\boldsymbol{\beta} } \right) \succeq \operatorname{Var}\left( \hat{\boldsymbol{\beta} } \right)
  $$

The equality holds iff $\boldsymbol{D} ^\top \boldsymbol{D} = 0$, which implies that $\operatorname{tr}\left( \boldsymbol{D} \boldsymbol{D} ^\top \right) = 0$, then $\left\Vert \boldsymbol{D} \right\Vert _F^2 = 0$, then $\boldsymbol{D} = 0$, i.e. $\tilde{\boldsymbol{\beta} } = \hat{\boldsymbol{\beta} }$. Therefore, BLUE is unique.
:::

If error term is normally distributed, then OLS is most efficient among all consistent estimators (not just linear ones).

When the distribution of error term is non-normal, other estimators may have lower variance than OLS such as least absolute deviation (median regression).

#### Distribution

If we assume $\varepsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2)$, or $\boldsymbol{\varepsilon} \sim N_n(\boldsymbol{0} , \boldsymbol{I} _n)$, then

$$
\boldsymbol{y} \sim N(\boldsymbol{X} \boldsymbol{\beta} , \sigma^2 \boldsymbol{I} )
$$

Hence, the distribution of the coefficients estimator is


$$\begin{aligned}
\hat{\boldsymbol{\beta}}
&= (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top \boldsymbol{y}   \\
&\sim  N(\boldsymbol{\beta} , (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} \operatorname{Var}\left( \boldsymbol{y}  \right)) \boldsymbol{X} ^\top (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \\
&\sim N(\boldsymbol{\beta} , \sigma^2 (\boldsymbol{X} ^\top \boldsymbol{X} )^{-1} ) \\
\end{aligned}$$



### Residuals and Error Variance

#### Residuals

Definition  
: The residual is defined as the difference between the true response value $y$ and our fitted response value $\hat{y}$.

  $$\hat\varepsilon_i = y_i - \hat{y}_i = y_i - \boldsymbol{x}_i ^\top \hat{\boldsymbol{\beta}}$$

  It is an estimate of the error term $\varepsilon_i$.


**Properties**

1. The sum of the residual is zero: $\sum_i \hat{\varepsilon}_i = 0$

1. The sum of the product of residual and any covariate is zero, or they are "uncorrelated": $\sum_i x_{ij} \hat{\varepsilon}_i = 0$ for all $j$.

1. The sum of squared residuals: $\left\| \boldsymbol{\hat{\varepsilon}}  \right\|^2   = \left\| \boldsymbol{y} - \boldsymbol{H} \boldsymbol{y}   \right\|^2  = \boldsymbol{y} ^\top (\boldsymbol{I} - \boldsymbol{H} ) \boldsymbol{y}$

:::{admonition,dropdown,seealso} *Proof*
Recall the normal equation

$$
\boldsymbol{X} ^\top (\boldsymbol{y} - \boldsymbol{X} \hat{\boldsymbol{\beta} }) = \boldsymbol{0}
$$

We obtain

$$
\boldsymbol{X} ^\top \boldsymbol{\hat{\varepsilon}}  = \boldsymbol{0}
$$

Since the first column of $\boldsymbol{X}$ is $\boldsymbol{1}$ , we have

$$\begin{align}
\sum_i \hat{\varepsilon}_i  
&= \sum_i(y_i - \hat{y}_i)  \\
&= \sum_i(y_i - \boldsymbol{x}_i ^\top \hat{\boldsymbol{\beta} }_i)  \\
&= 0
\end{align}$$

For other columns $\boldsymbol{x}_j$ in $\boldsymbol{X}$, we have

$$
\boldsymbol{x}_j ^\top \boldsymbol{\hat{\varepsilon}}  = \boldsymbol{0}
$$

The 3rd equality holds since $\boldsymbol{I} - \boldsymbol{H}$ is a projection matrix if $\boldsymbol{H}$ is a projection matrix, i.e.,

$$
(\boldsymbol{I} - \boldsymbol{H}) (\boldsymbol{I} - \boldsymbol{H} ) = \boldsymbol{I} -\boldsymbol{H}
$$

:::

#### Estimation of Error Variance

In the estimation section we mentioned the estimator for the error variance $\sigma^2$ is

$$
\hat{\sigma}^{2}=\frac{\|\boldsymbol{y}-\boldsymbol{X} \hat{\boldsymbol{\beta}}\|^{2}}{n-p}
$$

This is because


$$
\left\| \hat{\boldsymbol{\varepsilon}}  \right\|  ^2 \sim \sigma^2\chi ^2 _{n-p}  \\



\Rightarrow  \quad \sigma^2 =\operatorname{E}\left( \frac{\|\boldsymbol{y}-\boldsymbol{X} \hat{\boldsymbol{\beta}}\|^{2}}{n-p} \right)
$$

and we used the method of moment estimator. The derivation of the above expectation is a little involved.

:::{admonition,dropdown,seealso} *Derivation*

Let

- $\boldsymbol{U} = [\boldsymbol{U} _ \boldsymbol{X} , \boldsymbol{U} _\bot]$ be an orthogonal basis of $\mathbb{R} ^{n\times n}$ where
- $\boldsymbol{U} _ \boldsymbol{X}  = [\boldsymbol{u} _1, \ldots, \boldsymbol{u} _p]$ is an orthogonal basis of the column space (image) of $\boldsymbol{X}$, denoted $\operatorname{col}(\boldsymbol{X} )$
- $\boldsymbol{U} _ \bot  = [\boldsymbol{u} _{p+1}, \ldots, \boldsymbol{u} _n]$ is an orthogonal basis of the orthogonal complement of the column space (kernel) of $\boldsymbol{X}$, , denoted $\operatorname{col}(\boldsymbol{X} ) ^\bot$.

Recall

$$ \hat{\boldsymbol{\varepsilon}}  = \boldsymbol{y} - \hat{\boldsymbol{y}} = (\boldsymbol{I} - \boldsymbol{H} ) \boldsymbol{y} \in \operatorname{col}(\boldsymbol{X} ) ^\bot$$

which is

$$\begin{aligned}
\left\| \hat{\boldsymbol{\varepsilon}}  \right\|  ^2
&= \left\| \boldsymbol{P} _{\boldsymbol{U} _\bot} \boldsymbol{y}   \right\|  \\
&= \left\| \boldsymbol{U} _\bot \boldsymbol{U} _\bot ^\top \boldsymbol{y}  \right\|^2  \\
&= \left\| \boldsymbol{U} _\bot ^\top \boldsymbol{y}  \right\|^2  \\
&= \left\| \boldsymbol{U} _\bot ^\top (\boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon})    \right\|^2  \\
&= \left\| \boldsymbol{U} _\bot ^\top  \boldsymbol{\varepsilon}    \right\|^2  \quad \because \boldsymbol{U} _\bot ^\top\boldsymbol{X} = \boldsymbol{0} \\
\end{aligned}$$

Note that assuming $\boldsymbol{\varepsilon} \sim N(\boldsymbol{0} , \sigma^2 \boldsymbol{I} )$, we have


$$\begin{aligned}
\boldsymbol{U} _\bot ^\top \boldsymbol{\varepsilon}
&\sim N_{n-p}(\boldsymbol{0} , \boldsymbol{U} _\bot ^\top \sigma^2 \boldsymbol{I}_n \boldsymbol{U} _\bot) \\
&\sim N_{n-p}(\boldsymbol{0} , \sigma^2 \boldsymbol{I}_{n-p}) \\
\end{aligned}$$

and hence the sum of squared normal variables follows

$$
\left\| \boldsymbol{U} _\bot ^\top \boldsymbol{\varepsilon} \right\|  ^2 \sim \sigma^2 \chi ^2 _{n-p}  
$$

Thus,

$$
\left\| \hat{\boldsymbol{\varepsilon}}  \right\|  ^2 \sim \sigma^2 \chi ^2 _{n-p}  
$$

The first moment is

$$
\operatorname{E}\left( \left\| \hat{\boldsymbol{\varepsilon}} \right\|^2    \right) = \sigma^2 (n-p)
$$

or equivalently

$$
\sigma^2 = \frac{\operatorname{E}\left( \left\| \hat{\boldsymbol{\varepsilon}} \right\|^2\right)}{n-p}
$$

Therefore, the method of moment estimator for $\sigma^2$ is

$$
\hat{\sigma}^2 = \frac{\left\| \hat{\boldsymbol{\varepsilon}}  \right\|^2  }{n-p}
$$

which is unbiased.

:::


Can we find $\operatorname{Var}\left( \hat{\sigma}^2  \right)$ like we did for $\operatorname{Var}\left( \hat{\boldsymbol{\beta}}  \right)$? No, unless we assume higher order moments of $\varepsilon_i$.


(lm-independent-beta-sigma)=
### Independence of $\hat{\boldsymbol{\beta}}$ and $\hat{\sigma}^2$

To prove the independence between the coefficients estimator $\hat{\boldsymbol{\beta} }$ and the error variance estiamtor $\hat{\sigma}^2$, we need the Lemma below.


Lemma
: Suppose a random vector $\boldsymbol{y}$ follows multivariate normal distribution $\boldsymbol{y} \sim N_m(\boldsymbol{\mu} , \sigma^2 I_m)$ and $S, T$ are orthogonal subspaces of $\mathbb{R} ^m$, then the two projected random vectors are independent

$$
\boldsymbol{P}_S (\boldsymbol{y}) \perp\!\!\!\perp  \boldsymbol{P}_T (\boldsymbol{y})
$$

:::{admonition,dropdown,seealso} *Proof*

Let $\boldsymbol{z} \sim N(\boldsymbol{0} , \boldsymbol{I} _m)$ be a standard multivariate normal random vector, and $\boldsymbol{U} = [\boldsymbol{U} _S, \boldsymbol{U} _T]$ be orthogonal basis of $\mathbb{R} ^n$. Then

$$\begin{aligned}
&&\boldsymbol{U} ^\top \boldsymbol{z} &\sim N(\boldsymbol{0} , \boldsymbol{I} _m) \\
&\Rightarrow& \quad \left[\begin{array}{l}
\boldsymbol{U}_S ^\top \boldsymbol{z} \\
\boldsymbol{U}_T ^\top \boldsymbol{z} \\
\end{array}\right]&\sim N(\boldsymbol{0} , \boldsymbol{I} _m) \\
&\Rightarrow& \quad \boldsymbol{U} ^\top _S \boldsymbol{z}  &\perp\!\!\!\perp \boldsymbol{U} ^\top _T \boldsymbol{z}  \\
&\Rightarrow& \quad  f(\boldsymbol{U} ^\top _S \boldsymbol{z})  &\perp\!\!\!\perp f(\boldsymbol{U} ^\top _T \boldsymbol{z})  \\
\end{aligned}$$

Let $\boldsymbol{y} = \boldsymbol{\mu} + \sigma \boldsymbol{z}$, then  

$$
\boldsymbol{P} _S(\boldsymbol{y} ) = \boldsymbol{U} _S \boldsymbol{U} _S ^\top (\boldsymbol{\mu} + \sigma \boldsymbol{z} ) \perp\!\!\!\perp \boldsymbol{U} _T \boldsymbol{U} _T ^\top (\boldsymbol{\mu} + \sigma \boldsymbol{z} ) = \boldsymbol{P} _T(\boldsymbol{y} )
$$

$\square$

:::

Note that $\hat{\boldsymbol{\beta}}$ is a function of $\boldsymbol{P}  _{\operatorname{im}(\boldsymbol{X}) } (\boldsymbol{y})$ since


$$\begin{aligned}
\hat{\boldsymbol{\beta}}
&= (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top \boldsymbol{y} \\
&= (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top \boldsymbol{X} (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top \boldsymbol{y} \\
&= (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X} ^\top \boldsymbol{P}_{\operatorname{im}(\boldsymbol{X} ) } (\boldsymbol{y})  \\
\end{aligned}$$

and note that $\hat{\sigma}^2$ is a function of $\boldsymbol{P}  _{\operatorname{im}(\boldsymbol{X} )^\bot } (\boldsymbol{y})$ since


$$
\hat{\sigma}^{2}=\frac{\|\hat{\boldsymbol{\varepsilon}} \|^{2}}{n-p} =
\frac{\left\| \boldsymbol{P}  _{\operatorname{im}(\boldsymbol{X})^\bot } (\boldsymbol{y}) \right\| ^2  }{n-p}
$$

Therefore, by the lemma, they are independent. As a result, we can then perform $t$-test of the coefficients.


### Sum of Squares

We can think of each observation as being made up of an explained part, and an unexplained part.

-   Total sum of squares: $TSS = \sum\left(y_{i}-\bar{y}\right)^{2}$
-   Explained sum of squares: $ESS = \sum\left(\hat{y}_{i}-\bar{y}\right)^{2}$
-   Residual sum of squares: $RSS = \sum (y_i - \hat{y}_i)^2$


(lm-tss-identity)=
#### Decomposition of TSS

We have the decomposition identity

$$\begin{align}
TSS
&=\sum\left(y_{i}-\bar{y}\right)^{2} \\
&=\sum\left[\left(y_{i}-\hat{y}_{i}\right)+\left(\hat{y}_{i}-\bar{y}\right)\right]^{2} \\
&=\sum\left[\hat{\varepsilon}_{i}+\left(\hat{y}_{i}-\bar{y}\right)\right]^{2} \\
&=\sum \hat{\varepsilon}_{i}^{2}+2 \sum \hat{\varepsilon}_{i}\left(\hat{y}_{i}-\bar{y}\right)+\sum\left(\hat{y}_{i}-\bar{y}\right)^{2} \\
&= RSS + 2  \sum \hat{\varepsilon}_{i}\left(\hat{\beta}_0 + \hat{\beta}_1 x_{i}-\bar{y}\right)+ ESS \\
&= RSS + ESS
\end{align}$$

where use the fact that $\sum_i \varepsilon_i = 0$ and $\sum_i \varepsilon_i x_i = 0$ shown \[above\].

```{warning}
Some courses use the letters $R$ and $E$ to denote the opposite quantity in statistics courses.

- Sum of squares due to regression: $SSR = \sum\left(\hat{y}_{i}-\bar{y}\right)^{2}$
- Sum of squared errors: $SSE = \sum (y_i - \hat{y}_i)^2$
```

From linear algebra’s perspective, the identity is equivalent to

$$
\left\Vert \boldsymbol{y} - \bar{y} \boldsymbol{1} _n  \right\Vert ^2 = \left\Vert \boldsymbol{y} - \hat{\boldsymbol{y} }  \right\Vert ^2 + \left\Vert \hat{\boldsymbol{y} } - \bar{y} \boldsymbol{1} _n\right\Vert ^2
$$

which holds because the LHS vector $\boldsymbol{y} - \bar{y}\boldsymbol{1} _n$ is the the sum of the two RHS vectors, and the two vectors are orthogonal


$$\begin{aligned}
\boldsymbol{y}  - \bar{y} \boldsymbol{1} _n = (\boldsymbol{y} - \hat{\boldsymbol{y} }) &+ (\hat{\boldsymbol{y} } - \bar{y} \boldsymbol{1} _n) \\
\boldsymbol{y} - \hat{\boldsymbol{y} } &\perp \hat{\boldsymbol{y} } - \bar{y} \boldsymbol{1} _n
\end{aligned}$$

More specifically, they are orthogonal because

$$
\boldsymbol{y} - \hat{\boldsymbol{y} } \in \operatorname{col}(\boldsymbol{X} )^\perp \quad  \hat{\boldsymbol{y} } - \bar{y} \boldsymbol{1} _n \in \operatorname{col}(\boldsymbol{X} )
$$

since $\boldsymbol{1} _n \in \operatorname{col}(\boldsymbol{X})$, if an intercept term is included in the model.

drawing \[here\]

(lm-rss-nonincreasing)=
#### Non-increasing RSS

```{margin}
This is equivalent to say [$R$-squared](lm-rsquared) is always increasing or unchanged, if an intercept term in included in the model.
```

Given a data set, when we add an new explanatory variable into a regression model, $RSS$ is non-decreasing.

Since we are comparing two nested minimization problems

$$\begin{aligned}
&\text{Problem 1: with } x_{p}  \ &\min &\ \left\Vert  \boldsymbol{y} - \boldsymbol{X} _{n \times p} \boldsymbol{\beta} _{p\times 1} \right\Vert ^2   = \min \ RSS_1 \\
&\text{Problem 2: without } x_{p} \ &\min &\ \left\Vert  \boldsymbol{y} - \boldsymbol{X} _{n \times p} \boldsymbol{\beta} _{p\times 1} \right\Vert ^2  = \min \ RSS_2 \\
&&\text{s.t.}  &\ \beta_{p} = 0
\end{aligned}$$

As a result, the minimum value of the Problem 1 should be no larger than the minimum value of the Problem 2, i.e. $RSS_1^* \le RSS_2^*$ , When will they be equal? Only when the solution $\hat{\beta}_{p}=0$ in Problame 1.

When will $\hat{\beta}_{p}=0$ in problem 1? No clear condition.

- If $\boldsymbol{x}_i$'s are orthogonal such that $\boldsymbol{X} ^\top \boldsymbol{X} = I_{p}$, then

  $$
  \boldsymbol{x}_{p} ^\top \boldsymbol{y} = 0 \Leftrightarrow \hat{\beta}_{p}=0
  $$

- Note that in general, $\not\Leftarrow$. An simple example can be a data set of two points $(1,0), (1,1)$. The fitted line is $y=0.5$.

- Also, in general, $\not\Rightarrow$. An example is shown below where $\hat{\beta}_{2} \ne 0$.

```{code-cell}
import numpy as np
y = np.array([[1,2,3]]).T
x0 = np.array([[1,1,1]])
x1 = np.array([[1,2,5]])
x2 = np.array([[1,-2,1]])
print(x2 @ y)
X = np.vstack((x0, x1, x2)).T
XXinv = np.linalg.inv(np.dot(X.T, X))
b = np.dot(XXinv, np.dot(X.T, y))
print(b)
```

## Interpretation

### Coefficients

#### Value

$\beta_j$ is the expected change in the value of the response variable $y$ if the value of the covariate $x_j$ increases by 1, holding other covariates fixed.

$\beta_0$ is the expected value of the response variable $y$ if all covariates have values of zero.

If the response is in log format, i.e. $\log(Y)$, then the $\beta_j$ can be interpreted as the percentage change in $Y$ associated with one unit increase of $X_j$.


``` {warning}
Linear regression models only reveal linear associations between the response variable and the independent variables. But association does not imply causation. Simple example: in SLR, regress $X$ over $Y$, the coefficient has same sign and significance??, but causation cannot be reversed.

Only when the data is from a randomized controlled trial, correlation will imply causation.
```

#### Partialling Out

We can interpret the coefficients in multiple linear regression from “partialling out” perspective.

When $p=3$, i.e.,

$$
\hat{y}=\hat{\beta}_{0}+\hat{\beta}_{1} x_{1}+\hat{\beta}_{2} x_{2}
$$

We can obtain $\hat{\beta}_1$ by the following three steps

1.  regress $x_1$ over $x_2$ and obtain

    $$\hat{x}_{1}=\hat{\gamma}_{0}+\hat{\gamma}_{1} x_{2}$$

2.  compute the residuals $\hat{u}_{1}$ in the above regression

    $$
     \hat{u}_{i} = x_{1i} - \hat{x}_{1i}
     $$

3.  regress $y$ on the the residuals $\hat{u}_{1}$, and the estimated coefficient equals the required coefficient.

    $$\begin{align}
     \hat{y}
     &=\hat{\alpha}_{0}+\hat{\alpha}_{1} \hat{u} \\
     \hat{\alpha}_{1}
     &= \frac{\sum (\hat{u}_i - \bar{\hat{u}}_i)(y_i - \bar{y})}{\sum (\hat{u}_i - \bar{\hat{u}}_i)^2} \\
     &= \frac{\sum \hat{u}_{i}y_i}{\sum \hat{u}_{i}^2} \qquad \because \bar{\hat{u}}_i = 0\\
     &\overset{\text{claimed}}{=} \hat{\beta}_1
     \end{align}$$

In this approach, $\hat{u}$ is interpreted as the part in $x_1$ that cannot be predicted by $x_2$, or is uncorrelated with $x_2$. We then regress $y$ on $\hat{u}$, to get the effect of $x_1$ on $y$ after $x_2$ has been “partialled out”.



#### Hypothesis Testing

## Model Selection

(lm-rsquared)=
### $R$-squared and Adjusted $R$-squared

Assuming the [decomposition identity](lm-tss-identity) of $TSS$ holds, we can define $R$-squared.

Defintion  
$R$-squared is a statistical measure that represents the **proportion of the variance** for a dependent variable that’s **explained** by an independent variable or variables in a regression model.

$$
  R^2 = \frac{SSR}{SST}  = 1 - \frac{SSE}{SST} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  $$

**Properties**

1. $R$-squared can never decrease when an additional explanatory variable is added to the model.

    As long as $Cov(Y, X_j) \ne 0$, then $X_j$ has some explanatory power to $Y$, and thus $RSS$ decreases, See the [section](lm-rss-nonincreasing) of $RSS$ for details. As a result, $R$-squared  is not a good measure for model selection, which can cause overfitting.

1. $R$-squared equals the squared correlation coefficient between the actual value of the response and the fitted value $\operatorname{Corr}\left( Y, \hat{Y} \right)^2$.

    In particular, in simple linear regression, $R^2 = \rho_{X,Y}^2$.

    :::{admonition,dropdown,seealso} *Proof*
    By the definition of correlation,

    $$\begin{align}
    \operatorname{Corr}\left( y, \hat{y} \right)^2
    &= \frac{\operatorname{Cov}\left( y, \hat{y} \right)^2}{\operatorname{Var}\left( y \right)\operatorname{Var}\left( \hat{y} \right)} \\
    &= \frac{\operatorname{Cov}\left( \hat{y} + \hat{\varepsilon}, \hat{y} \right)^2}{\operatorname{Var}\left( y \right)\operatorname{Var}\left( \hat{y} \right)} \\
    &= \frac{\operatorname{Cov}\left( \hat{y} , \hat{y} \right)^2}{\operatorname{Var}\left( y \right)\operatorname{Var}\left( \hat{y} \right)} \\
    &= \frac{\operatorname{Var}\left( \hat{y} \right)^2}{\operatorname{Var}\left( y \right)\operatorname{Var}\left( \hat{y} \right)} \\
    &= \frac{\operatorname{Var}\left( \hat{y} \right)}{\operatorname{Var}\left( y \right)} \\
    &= \frac{SSR}{SST} \\
    &= R^2 \\
    \end{align}$$

    The third equality holds since

    $$
    \operatorname{Cov}\left( \hat{\varepsilon}, \hat{y} \right) = \operatorname{Cov}\left( \hat{\varepsilon}, \sum_j x_j \hat{\beta}_j  \right) = \sum_j \hat{\beta}_j \operatorname{Cov}\left( \hat{\varepsilon},  x_j \right) = 0
    $$

    When $p=2$, since $\hat{y_i} = \hat{\beta}_0 + \hat{\beta}_1 x$, we have

    $$
    R^2 = \operatorname{Corr}\left(y, \hat{y} \right)^2 = \operatorname{Corr}\left(y, x \right)^2
    $$
    :::


When there is **no** intercept, then $\bar{y} \boldsymbol{1} _n \notin \operatorname{col}(X)$ and hence $\hat{\boldsymbol{y} } - \bar{y} \boldsymbol{1} _n \notin \operatorname{col}(X)$. The decomposition identity may not hold. Thus, the value of $R$-squared may no longer be in $[0,1]$, and its interpretation is no longer valid. What actually happen to the value $R$-squared depends on whether we define it using $TSS$ with $RSS$ or $ESS$.

If we define $R^2 = \frac{ESS}{TSS}$, then when

$$
\sqrt{ESS} = \left\Vert \hat{\boldsymbol{y} } - \bar{y}\boldsymbol{1} _n \right\Vert> \left\Vert \boldsymbol{y} - \bar{y}\boldsymbol{1} _n \right\Vert = \sqrt{TSS}
$$

we will have $ESS > TSS$, i.e., $R^2 > 1$.

On the other hand, if we define $R^2 = 1 - \frac{RSS}{TSS}$, then when

$$
\sqrt{RSS} = \left\Vert \hat{\boldsymbol{y} } -  \boldsymbol{y} \right\Vert> \left\Vert \boldsymbol{y} - \bar{y}\boldsymbol{1} _n \right\Vert = \sqrt{TSS}
$$

we will have $RSS > TSS$, i.e. $R^2 < 0$.



### Adjusted $R$-squared


Due to the non-decrease property of $R$-squared, we define adjusted $R$-squared which is a better measure of goodness of fitting.

Definition  
: Adjusted $R$-squared, denoted by $\bar{R}^2$, is defined as

$$
  \bar{R}^2 = 1-\frac{RSS / (n-p)}{ TSS / (n-1)}
  $$

Properties  
-   $\bar{R}^2$ can increase or decrease. When a new variable is included, $RSS$ decreases, but $(n-p)$ also decreases.
    -   Relation to $R$-squared is

$$
\bar{R}^2 = 1-\frac{n-1}{ n-p}(1 - R^2) < R^2
$$

-   Relation to estimated variance of random error and variance of response

    $$
    \bar{R}^2 = 1-\frac{\hat{\sigma}^2}{\operatorname{Var}\left( y \right)}
    $$

-   Can be negative when

    $$
    R^2 < \frac{p-1}{n-p}
    $$

    If $p > \frac{n+1}{2}$ then the above inequality always hold, and adjusted $R$-squared is always negative.



### $t$-test

We can use $t$-test to test a null hypothesis on $\boldsymbol{\beta}$, which has a general form

$$
\boldsymbol{v} ^\top \boldsymbol{\beta}_{\text{null}} = c
$$

Usually $c=0$.

- If $c=0, \boldsymbol{v} = \boldsymbol{e} _j$ then this is equivalent to test $\beta_j=0$, i.e. the variable $X_i$ has no effect on $Y$ given all other variabels.
- If $c=0, v_i=1, v_j=-1$ and $v_k=0, k\ne i, j$ then this is equivalent to test $\beta_i = \beta_j$, i.e. the two variables $X_i$ and $X_j$ has the same effect on $Y$ given all other variables.

First, we need to find the distribution of $\boldsymbol{v} ^\top \hat{\boldsymbol{\beta}}$. Recall that

$$
\hat{\boldsymbol{\beta}} \sim N(\boldsymbol{\beta}_{\text{null}} , \sigma^2 (\boldsymbol{X} ^\top \boldsymbol{X} )^{-1})
$$

Hence,


$$
\boldsymbol{v} ^\top \hat{\boldsymbol{\beta}}  \sim N(\boldsymbol{v} ^\top \boldsymbol{\beta}_{\text{null}} , \sigma^2 \boldsymbol{v} ^\top (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{v} )
$$

Also recall that the RSS has the distribution

$$
(n-p)\frac{\hat{\sigma}^2}{\sigma^2 } \sim \sim \chi ^2 _{n-p}  
$$

and the two quantities $\boldsymbol{v} ^\top \hat{\boldsymbol{\beta}}$ and $(n-p)\frac{\hat{\sigma}^2}{\sigma^2 }$ are [independent](lm-independent-beta-sigma). Therefore, we can construct a test statistic


$$
\frac{\boldsymbol{v} ^\top \hat{\boldsymbol{\beta}} - \boldsymbol{v} ^\top \boldsymbol{\beta}_{\text{null}}}{\sigma\sqrt{\boldsymbol{v} ^\top (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{v} }} / \sqrt{\frac{(n-p)\hat{\sigma}^2 }{\sigma^2 } / (n-p)} \sim t_{n-p}
$$

i.e.,


$$
\frac{\boldsymbol{v} ^\top \hat{\boldsymbol{\beta}} - \boldsymbol{v} ^\top \boldsymbol{\beta}_{\text{null}}}{\hat{\sigma}\sqrt{\boldsymbol{v} ^\top (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{v} }} \sim t_{n-p}
$$

In particular, when $p=2$, to test $\beta_1 = c$, we use

$$
\frac{\hat{\beta}_1 - c}{\hat{\sigma}/ \sqrt{\operatorname{Var}\left( X \right)}}  \sim t_{n-2}
$$




### ANOVA?

The Analysis Of Variance, popularly known as the ANOVA, can be used in cases where there are more than two groups.

TBD

### Stepwise

TBD

## Special Cases

No models are perfect. In this section we introduce what happen when our model is misspecified or when some assumptions fail.

(lm-omit-variable)=
### Omit a Variable

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
### Include a Variable

What if we add a new variable $x_p$? What will happen to the existing estimator $\hat\beta_k$?

Increase $\operatorname{Var}\left(\hat{\beta}_{k}\right)=\sigma^{2} \frac{1}{1-R_j^{2}} \frac{1}{\sum_{i}\left(x_{i k}-\bar{x}_{k}\right)^{2}}$ if $R_{-k}^2$ increases. When will $R^2_{-k}$ be unchanged? When the new variable $x_p$ has no explanatory power to $x_k$. See the [exercise](lm-rsq-non-decreasing).

In terms of bias, if we say the model with $x_p$ is "true", then $\operatorname{E}\left( \hat{\beta}_k \right)$ is probably closer to $\beta_k$ according to the equation described in the above [section](lm-omit-variable).


### Multicollinearity

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

#### Diagnosis

Some common symptoms include
- $F$-test is significant, $R^2$ is good, but $t$-test is not significant.
- Large magnitude of $\hat{\beta}_j$
- Large standard error $\operatorname{se}(\beta_j)$

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

#### Consequences

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

#### Implications

If $X_1$ and $X_2$ show high correlation, then

1.  $X_1$ may be a proxy of $X_2$.
2.  $X_1 - X_2$ may just be noise.
3.  If $X_2$ is removed, $X_1$ may still be good for prediction.

### Heteroscedasticity

TBD

### Categorical $X$

dummy variables $X_ij$

when $c = 2$,

interpretation
- $\hat{\beta_1}$: difference in means between the group with $X=1$ and $X=0$.
- $\hat{\beta_0}$: mean of the group with $X=0$.

TBD

https://www.1point3acres.com/bbs/thread-703302-1-1.html

## Exercise

### Slope vs Correlation

When $p=2$, we can see from the solution

$$\begin{align}
\hat{\beta}_{1} &=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}
\end{align}$$

that

$$\begin{align}
\hat{\beta}_1 &= \frac{\widehat{\operatorname{Cov}}\left( Y, X \right)}{\widehat{\operatorname{Var}}\left( X \right)}  \\
&= r_{X,Y} \frac{s_Y}{s_X}
\end{align}$$

Thus, the slope has the same sign with the correlation $r_{X,Y}$, and equals to the correlation times a ratio of the sample standard deviations of the dependent variable over the independent variable.

Once can see that the magnitude of $\hat\beta_1$ increases with the magnitude of $\rho_{X,Y}$ and $s_Y$, and decreases with $s_X$, holding others fixed.

### Fitted Line Passes Sample Mean

Since $\hat{\beta}_{0} =\bar{y}-\hat{\beta}_{1} \bar{x}$, we have $\bar{y} = \hat{\beta}_{0} + \hat{\beta}_{1} \bar{x}$, i.e. the regression line always goes through the mean $(\bar{x}, \bar{y})$ of the sample.

This also hold for multiple regression, by the first order condition w.r.t. $\beta_0$.

### Non-zero Mean of Error Term

*What if the mean of the error term is not zero?*

If $\operatorname{E}\left( \varepsilon \right) = \mu_\varepsilon \ne 0$, we can just denote $\varepsilon = \mu_\varepsilon + v$, where $v$ is a new error term with zero mean. Our model becomes

$$
y_i = (\beta_0 + \mu_\varepsilon) + \beta_1 x_1 + v
$$

where $(\beta_0 + \mu_\varepsilon)$ is the new intercept. We can still apply the methods above to conduct estimation and inference.

(lm-proportional-model)=
### No Intercept

*Assume the intercept $\beta_0$ in the model $y=\beta_0 + \beta_1 x + \varepsilon$ is zero. Find the OLS estimate for $\beta_1$, denoted $\tilde{\beta}$. Find its mean, variance, and compare them with those of the OLS estimate for $\beta_1$ when there is an intercept term.*

If there is no intercept, consider a simple case

$$
y = \beta x + \varepsilon
$$

Then by minimizing sum of squared errors

$$
\min \sum_i (y_i - \beta x)^2
$$

we have

$$
-2 \sum_i (y_i - \beta x) x = 0
$$

and hence,

$$\begin{align}
\tilde{\beta}
&= \frac{\sum_i x_i y_i}{\sum_i x_i^2} \\
&= \frac{\sum_i x_i (\beta x_i + \varepsilon_i)}{\sum_i x_i^2}\\
&= \beta + \frac{\sum x_i \varepsilon_i}{\sum_i x_i^2}
\end{align}$$

Therefore, $\tilde{\beta}$ is still an unbiased estimator of $\beta$, while its variance is smaller than the variance calculated assuming the intercept is non-zero.

$$
\operatorname{Var}\left( \tilde{\beta} \right) = \frac{\sigma^2}{\sum x_i^2} \le  \frac{\sigma^2}{\sum (x_i - \bar{x})^2} = \operatorname{Var}\left( \hat{\beta}  \right)
$$

Hence, if the intercept is known to be zero, better use $\tilde\beta$ instead of $\hat\beta$, since the standard error of the $\tilde\beta$ is smaller, and both are unbiased.

If the true model has a non-zero intercept, then $\tilde\beta$ is biased for $\beta$, but it has a smaller variance, which brings a tradeoff of bias vs variance.

### Transformation of Variables

[insert] summary table.

First, we take simple linear regression as an example.

If $X ^\prime = aX + b$, then the new slope estimate is

$$\begin{align}
\tilde{\beta}_1 &= \frac{\widehat{\operatorname{Cov}}\left( Y, X ^\prime \right)}{\widehat{\operatorname{Var}}\left( X ^\prime \right)}  \\
&= \frac{\widehat{\operatorname{Cov}}\left( Y, aX + b  \right)}{\widehat{\operatorname{Var}}\left( aX+b \right)}  \\
&= \frac{a\widehat{\operatorname{Cov}}\left( Y, X \right)}{a^2\widehat{\operatorname{Var}}\left( X \right)}  \\
&= \frac{1}{a} \hat\beta_1 \\
\end{align}$$

and the new intercept is

$$\begin{align}
\tilde\beta_0
&= \bar{y} - \tilde\beta_1 \bar{x} ^\prime \\
&= \bar{y} - \hat\beta_1 \frac{1}{a}  (a\bar{x}+b) \\
&= \hat\beta_0 - \hat\beta_1 \frac{b}{a} \\
\end{align}$$

If $Y ^\prime = cY + d$ then

$$\begin{align}
\tilde{\beta}_1 &= \frac{\widehat{\operatorname{Cov}}\left( Y ^\prime, X ^\prime \right)}{\widehat{\operatorname{Var}}\left( X ^\prime \right)}  \\
&= \frac{\widehat{\operatorname{Cov}}\left( cY+d, X  \right)}{\widehat{\operatorname{Var}}\left( X \right)}  \\
&= \frac{c\widehat{\operatorname{Cov}}\left( Y, X \right)}{c\widehat{\operatorname{Var}}\left( X \right)}  \\
&= c \hat\beta_1 \\
\end{align}$$

and

$$\begin{align}
\tilde\beta_0
&= \bar{y}^\prime - \tilde\beta_1 \bar{x} \\
&= (c\bar{y}+d) - c\hat\beta_1 \bar{x} \\
&= c\hat\beta_0 + d\\
\end{align}$$


Can the conclusions be extended to multiple regression?

TBD.

### Exchange $X$ and $Y$

TBD.

### Covariance, $R$-squared, and $\beta_j$

In multiple regression, if $\operatorname{Cov}\left( Y, X_j \right) = 0$ then $\beta_j= 0$?

Is it possible that $\operatorname{Cov}\left( X_j, X_k \right) \ne 0, \operatorname{Cov}\left( Y, X_k \right) \ne 0$ but $\operatorname{Cov}\left( Y, X_j \right) = 0$?

TBD.

### Increase Estimation Precision

TBD.

-   The larger the error variance, $\sigma^2$, the larger the variance of the coefficient estimates.
-   The larger the variability in the $x_i$, the smaller the variance.
-   A larger sample size should decrease the variance.
-   In multiple regression, reduce the relation between $X_j$ and other covariates (e.g. by orthogonal design) can decreases $R^2_{j}$, and hence decrease the variance.


### Partialling Out in General Cases

TBD.

### Causal?

313.qz1.q2

TBD.


(lm-rsq-non-decreasing)=

### Add/Remove a Variable/Observation

TBD

Table summary.

Rows: E(b), Var(b), RSS, TSS, R^2
