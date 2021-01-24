# Linear Models

In this section we introduce linear models from a statistics' perspective. The introduction from econometrics' perspective or social science's perspective may be different. In short, the statistics' perspective focuses on general multivariate cases and heavily rely on linear algebra for derivation, while the econometrics' or social science's perspective prefers to introduce models in univariate cases by basic arithmetics (whose form can be complicated without linear algebra notations) and extend the intuitions and conclusions into multivariate cases.

Personally, I involved in four courses that introduced linear models, i.e. at undergrad/grad level offered by stat/social science department. The style of the two courses offered by the stat departments were quite alike while the graduate level one covered more topics. In both undergrad/grad level courses offered by the social science departments, sometimes I got confused by the course materials that was contradictory to my statistics training , but the instructors did not have a clear response...In sum, to fully understand the fundamental and most widely used statistical model, I highly suggest to take a linear algebra course first and take the regression course offered by math/stat department.

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


```{note}
In natural science, researchers design $n\times p$ values in the design matrix $\boldsymbol{X}$ and run experiments to obtain the response $y_i$. We call this kind of data **experimental data**. In this sense, the explanatory variables $x_{ij}$'s are designed before the experiment, so they are also constants. The coefficients $\beta_j$'s are unknown constants. The error term $\varepsilon_i$ is random. The response variable $Y_i$ on the left hand side is random due to the randomness in the error term $\varepsilon_i$.  

In social science, most of data is **observational data**. That is, researchers obtain the values of many variables at the same time, and they choose one of interest to be the response variable $y_i$ and some others to be the explanatory variables $\boldsymbol{x}_i$. In this case, as a data set, we can talk about descriptive statistics, such as variance of each explanatory variable, or covariance between pair of explanatory variables. This is valid since we often view the columns of a data set as random variables.

However, the inference methods of the coefficients $\boldsymbol{\beta}$ are developed based on the natural science setting, i.e. the values of explanatory variables are pre-designed constants. Many social science courses frequently use descriptive statistics of the explanatory variables assuming they are random, and apply inference methods assuming they are constant. This is quite confusing for beginners to linear models.

To sum up, we stick to the natural science setting. We use subscript $i$ in every $y_i, x_i, \varepsilon_i$ instead of $y, x, \varepsilon$ which gives a sense that $x$ is random. And we use descriptive statistics for the explanatory variables only when necessary.
```

## Assumptions

Basic assumptions

1. $\operatorname{E}\left( y_i \right) = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$ is linear in covariates $X_j$.

1. $\boldsymbol{x}_i$ is known and fixed.

1. The error terms are uncorrelated $\operatorname{Cov}\left( \varepsilon_i, \varepsilon_j \right)= 0$, with common mean $\operatorname{E}\left( \varepsilon_i \right) = 0$ and variance $\operatorname{Var}\left( \varepsilon_i \right) = \sigma^2$.

    As a result, $\operatorname{E}\left( \boldsymbol{y}  \mid \boldsymbol{X} \right) = \boldsymbol{X} \boldsymbol{\beta}$, or $\operatorname{E}\left( y \mid x \right) = \beta_0 + \beta_1 x$ when $p=2$, which can be illustrated by the plots below.

    :::{figure} lm-distribution-of-y-given-x
    <img src="../imgs/lm_cond_distribution.png" width = "50%" alt=""/>

    Distributions of $y$ given $x$ [Meyer 2021]
    :::

    :::{figure} lm-observation-of-y-given-x
    <img src="../imgs/lm_xyplane_dots.png" width = "50%" alt=""/>

    Observations of $y$ given $x$ [Meyer 2021]
    :::

    To predict $\hat{y}_i$, we just use $\hat{y}_i = \boldsymbol{x}_i ^\top \hat{\boldsymbol{\beta}}$ .


1. The error terms are independent and follow Gaussian distribution $\varepsilon_i \overset{\text{iid}}{\sim}N(0, \sigma^2)$, or $\boldsymbol{\varepsilon} \sim N_n (\boldsymbol{0} , \sigma^2 \boldsymbol{I} _n)$.

    As a result, we have $Y_i \sim N(\boldsymbol{x}_i ^\top \boldsymbol{\beta} , \sigma^2 )$ or $\boldsymbol{y} \sim N_n(\boldsymbol{X} \boldsymbol{\beta} , \sigma^2 \boldsymbol{I} _n)$


These assumptions are used for different objectives.
- derivation of $\hat{\boldsymbol{\beta}}$ by least squares uses assumptions 1, 2.
- derivation of $\operatorname{E}\left( \hat{\boldsymbol{\beta}} \right)$ uses assumptions 1, 2, $\operatorname{E}\left( \varepsilon_i \right) = 0$ in 3.
- derivation of $\operatorname{Var}\left( \hat{\boldsymbol{\beta}} \right)$ uses assumptions 1, 2, $\operatorname{Cov}\left( \varepsilon_i, \varepsilon_j \right) = 0$ and $\operatorname{Var}\left( \epsilon_i \right) = \sigma^2$ in 3.
- derivation of the distribution of $\hat{\boldsymbol{\beta} }$ uses 1, 2 and 3.
- derivation of $\hat{\boldsymbol{\beta}}$ by maximal likelihood uses assumptions 1, 2, 3.


```{dropdown, note}
In some social science or econometrics courses, there is an Zero Conditional Mean assumption. For $p=2$, it is

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

As discussed above, in their setting $x$ is random (at this stage), so they use notations such as $\operatorname{E}\left( \varepsilon \mid x \right)$ and $\operatorname{Cov}\left( x, \varepsilon \right)$. It also seems that they view $\varepsilon$ as an "overall" measure of random error, instead of $\varepsilon_i$ for specific $i$ in the natural science setting. But they can mean so by using the conditional notation $\operatorname{E}\left( \varepsilon \mid x \right)$.

I suppose they introduce these two identities purely to interpret the two normal equations (introduced below) from the minimizing sum of squared errors. Then one day, someone found that there is a beautiful equation $\operatorname{E}\left( \varepsilon \mid x \right)=0$ that can summarize these two identities. So it is added to the assumptions.
```

## Estimation (Learning)

We introduce various methods to estimate the parameters $\boldsymbol{\beta}$ and $\sigma^2$.

### Least Squares

We can estimate the parameter $\hat{\boldsymbol{\beta}}$ by minimizing the sum of squared errors.


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

$$\hat{\boldsymbol{\beta}} = \left( \boldsymbol{X} ^\top \boldsymbol{X}   \right)^{-1}\boldsymbol{X} \boldsymbol{y}  $$

```{note}
Computing software use specific functions to solve the normal equation $\boldsymbol{X} ^\top \boldsymbol{X} \boldsymbol{\beta} = \boldsymbol{X} ^\top \boldsymbol{y}$ for $\boldsymbol{\beta}$, instead of using the inverse $(\boldsymbol{X} ^\top \boldsymbol{X}) ^{-1}$ directly which can be slow and unstable.
```

An unbiased estimator of the error variance $\sigma^2 = \operatorname{Var}\left( \varepsilon \right)$ is (to be discussed [later])

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


```{note}
The objective function, **sum of squared errors**,

$$
\left\Vert \boldsymbol{y}  - \boldsymbol{X}  \boldsymbol{\beta}  \right\Vert ^2 = \sum_i \left( y_i - \boldsymbol{x}_i ^\top \boldsymbol{\beta} \right)^2
$$

can be replaced by **mean squared error**,


$$
\frac{1}{n} \sum_i \left( y_i - \boldsymbol{x}_i ^\top \boldsymbol{\beta} \right)^2
$$

and the results are the same.
```

(lm-estimation-by-assumpation)
### By Assumptions

In some social science courses, the estimation is done by using the assumptions
- $\operatorname{E}\left( \varepsilon \right) = 0$
- $\operatorname{E}\left( \varepsilon \mid X \right) = 0$

The first one gives

$$
\begin{equation}
\frac{1}{n}  \sum_{i=1}^{n}\left(y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{i}\right)=0
\end{equation}
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


### Maximum Likelihood

biased. TBD.

### Gradient Descent

TBD.

## Properties

We describe the properties of OLS estimator $\hat{\boldsymbol{\beta}}$ and the corresponding residuals $\hat{\boldsymbol{\varepsilon} }$.

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
#### \end{align}$$

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

#### Variance

The variance (matrix) is


$$\begin{align}
\operatorname{Var}\left( \boldsymbol{\beta}  \right) &= \operatorname{Var}\left(  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X}^\top  \boldsymbol{y}  \right)  \\
&=   (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \boldsymbol{X}^\top \operatorname{Var}\left( \boldsymbol{y}  \right)  \boldsymbol{X}  (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1} \\
&= \sigma^2 (\boldsymbol{X} ^\top \boldsymbol{X} ) ^{-1}\\
\end{align}$$

when $p=2$, it is


$$\begin{align}
\operatorname{Var}\left( \hat{\beta}_1 \right)
&= \operatorname{Var}\left( \beta_{1}+\frac{\sum\left(x_{i}-\bar{x}\right) u_{i}}{\sum \left(x_{i}-\bar{x}\right)^{2}} \right)\\
&= \frac{\operatorname{Var}\left( \sum\left(x_{i}-\bar{x}\right) u_{i} \right)}{\left[ \sum \left(x_{i}-\bar{x}\right)^{2} \right]^2}\\
&= \frac{\sum\left(x_{i}-\bar{x}\right)^2 \operatorname{Var}\left( u_{i} \right)}{\left[ \sum \left(x_{i}-\bar{x}\right)^{2} \right]^2}\\
&= \sigma^2 \frac{\sum\left(x_{i}-\bar{x}\right)^2 }{\left[ \sum \left(x_{i}-\bar{x}\right)^{2} \right]^2}\\
&= \frac{\sigma^2}{\sum \left(x_{i}-\bar{x}\right)^{2}}\\
&= \frac{\sigma^2}{(n-1)s_X^2}\\
\end{align}$$

We conclude that

- The larger the error variance, $\sigma^2$, the larger the variance of the slope estimator
- The larger the variability in the $x_i$, the smaller the variance of the slope estimator
- A larger sample size should decrease the variance of the slope estimator
- A problem is that the error $\sigma^2$ variance is **unknown**

We can estimate $\sigma^2$ by its unbiased estimator $\hat{\sigma}^2=\frac{\sum_i (x_i - \bar{x})}{n-2}$ (to be shown [below]), and substitute it into $\operatorname{Var}\left( \hat{\beta}_1 \right)$. Since the error variance $\hat{\sigma}^2$ is estimated, the slope variance $\operatorname{Var}\left( \hat{\beta}_1 \right)$ is estimated too, and hence the square root is called standard error of $\hat{\beta}$, instead of standard deviation.


$$\begin{align}
\operatorname{se}\left(\hat{\beta}_{1}\right)
&= \sqrt{\widehat{\operatorname{Var}}\left( \hat{\beta}_1 \right)}\\
&= \frac{\hat{\sigma}}{\sqrt{\sum \left(x_{i}-\bar{x}\right)^{2}}}
\end{align}$$

Theorem (Gauss–Markov)
: The ordinary least squares (OLS) estimator has the lowest sampling variance within the class of linear unbiased estimators, if the errors in the linear regression model are uncorrelated, have equal variances and expectation value of zero.

#### Hypothesis Testing

### Residuals

Definition
: The residual is defined as the difference between the true response value $y$ and our fitted response value $\hat{y}$.

    $$\hat\varepsilon_i = y_i - \hat{y}_i = y_i - \boldsymbol{x}_i ^\top \hat{\boldsymbol{\beta}}$$

    It is an estimate of the error term $\varepsilon_i$

Properties
: - The sum of the residual is zero $\sum_i \hat{\varepsilon}_i  = 0$

Proof

Recall the normal equation

$$
\boldsymbol{X} ^\top (\boldsymbol{y} - \boldsymbol{X} \hat{\boldsymbol{\beta} }) = \boldsymbol{0}
$$

we get

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


: - The sum of the product of residual and any covariate: $\sum_i x_{ij} \hat{\varepsilon}_i = 0$ for all $j$.

Proof

A corollary of

$$
\boldsymbol{X} ^\top \boldsymbol{\hat{\varepsilon}}  = \boldsymbol{0}
$$


```{note}
Some social science courses describe this property as "the residual and the independent variables are uncorrelated".
```


### Independence of $\hat{\boldsymbol{\beta}}$ and $\hat{\sigma}^2$

TBD



### Decomposition of Total Sum of Squares

We can think of each observation as being made up of an explained part, and an unexplained part.

- Total sum of squares: $SST = \sum\left(y_{i}-\bar{y}\right)^{2}$
- Sum of squares due to regression : $SSR = \sum\left(\hat{y}_{i}-\bar{y}\right)^{2}$
- Sum of squared errors: $SSE = \sum (y_i - \hat{y}_i)^2$

Then

$$\begin{align}
SST
&=\sum\left(y_{i}-\bar{y}\right)^{2} \\
&=\sum\left[\left(y_{i}-\hat{y}_{i}\right)+\left(\hat{y}_{i}-\bar{y}\right)\right]^{2} \\
&=\sum\left[\hat{\varepsilon}_{i}+\left(\hat{y}_{i}-\bar{y}\right)\right]^{2} \\
&=\sum \hat{\varepsilon}_{i}^{2}+2 \sum \hat{\varepsilon}_{i}\left(\hat{y}_{i}-\bar{y}\right)+\sum\left(\hat{y}_{i}-\bar{y}\right)^{2} \\
&=\mathrm{SSR}+2 \sum \hat{\varepsilon}_{i}\left(\hat{\beta}_0 + \hat{\beta}_1 x_{i}-\bar{y}\right)+SSE \\
&= SSR + SSE
\end{align}$$

where use the fact that $\sum_i \varepsilon_i = 0$ and $\sum_i \varepsilon_i x_i = 0$ shown [above].


```{warning}
Some social science courses use $SSR$ and $SSE$ to denote the opposite quantity in statistics courses.

- Explained sum of squares: $SSE = \sum\left(\hat{y}_{i}-\bar{y}\right)^{2}$
- Residual sum of squares: $SSR = \sum (y_i - \hat{y}_i)^2$
```



## Interpretation

### Coefficients

$\beta_j$ is the expected change in the value of the response variable $y$ if the value of the covariate $x_j$ increases by 1, holding other covariates fixed.

$\beta_0$ is the expected value of the response variable $y$ if all covariates have values of zero.


```{warning}
Linear regression models only reveal linear associations between the response variable and the independent variables. But association does not imply causation.

Only when the data is from a randomized controlled trial, correlation will imply causation,
```

### Partialling Out


When $p=3$, i.e.,

$$
\hat{y}=\hat{\beta}_{0}+\hat{\beta}_{1} x_{1}+\hat{\beta}_{2} x_{2}
$$

We can obtain $\hat{\beta}_1$ by the following three steps

1. regress $x_1$ over $x_2$ and obtain

    $$\hat{x}_{1}=\hat{\gamma}_{0}+\hat{\gamma}_{1} x_{2}$$

1. compute the residuals $\hat{u}_{1}$ in the above regression

    $$
    \hat{u}_{i} = x_{1i} - \hat{x}_{1i}
    $$

1. regress $y$ on the the residuals $\hat{u}_{1}$, and the estimated coefficient equals the required coefficient.


    $$\begin{align}
    \hat{y}
    &=\hat{\alpha}_{0}+\hat{\alpha}_{1} \hat{u} \\
    \hat{\alpha}_{1}
    &= \frac{\sum (\hat{u}_i - \bar{\hat{u}}_i)(y_i - \bar{y})}{\sum (\hat{u}_i - \bar{\hat{u}}_i)^2} \\
    &= \frac{\sum \hat{u}_{i}y_i}{\sum \hat{u}_{i}^2} \qquad \because \bar{\hat{u}}_i = 0\\
    &\overset{\text{claimed}}{=} \hat{\beta}_1
    \end{align}$$


In this approach, $\hat{u}$ is interpreted as the part in $x_1$ that cannot be predicted by $x_2$, or is uncorrelated with $x_2$. We then regress $y$ on $\hat{u}$, to get the effect of $x_1$ on $y$ after $x_2$ has been "partialled out".




## Model Selection


### $R$-squared

$R$-squared ($R$^2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.

Whereas correlation explains the strength of the relationship between an independent and dependent variable, $R$-squared explains to what extent the variance of one variable explains the variance of the second variable.

$$
R^2 = \frac{SSR}{SST}  = 1 - \frac{SSE}{SST}
$$

relation with $\beta$ in simple linear models:


### What is ANOVA?
The Analysis Of Variance, popularly known as the ANOVA, can be used in cases where there are more than two groups.


### Stepwise

## Special Cases

### Omitted Variable Bias




For instance, suppose the true model (p=3) is

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon
$$




### Heteroscedasticity

### Multicollinearity

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

Since $\hat{\beta}_{0} =\bar{y}-\hat{\beta}_{1} \bar{x}$, we have $\bar{y} = \hat{\beta}_{0} + \hat{\beta}_{1} \bar{x}$, i.e. the regression line always goes through the mean $(\bar{x}, \bar{y})$ of the sample.

This also hold for multiple regression, by the first order condition w.r.t. $\beta_0$.

### Non-zero Mean of Error Term

*What if the mean of the error term is not zero?*

If $\operatorname{E}\left( \varepsilon \right) = \mu_\varepsilon \ne 0$, we can just denote $\varepsilon = \mu_\varepsilon + v$, where $v$ is a new error term with zero mean. Our model becomes

$$
y_i = (\beta_0 + \mu_\varepsilon) + \beta_1 x_1 + v
$$

where $(\beta_0 + \mu_\varepsilon)$ is the new intercept. We can still apply the methods above to conduct estimation and inference.

(lm-proportional-model)
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

### Exchange $X$ and $Y$


## Question

When will

$$\operatorname{E}\left( Y \mid X \right) = 0 ?$$

## Claim

If

$$\operatorname{E}\left( v \right) = 0$$

and

$$\operatorname{Cov}\left( v, edu \right) = 0$$

then

$$\operatorname{E}\left( v \mid edu \right) = 0$$

## Counterexample

Generate random values $edu, v$ by

$$\begin{align}

edu &\sim \mathrm{Uniform}(0,1) \\
v &= |edu - 0.5| \\
\end{align}$$

then

$$
E(v) = 0 \\
\operatorname{Cov}\left( v, edu \right) = 0
$$

but

$$
\operatorname{E}\left( v \mid edu \right) = \operatorname{E}\left( |edu - 0.5| \mid edu \right) = |edu - 0.5|  \ne 0
$$
