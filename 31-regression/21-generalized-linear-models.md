# Generalized Linear Models


## From LM to GLM

### One-parameter Exponential Family

#### Definition

Consider a random variable $Y$ with probability density function parameterized by $\theta \in \mathbb{R}$. If its PDF can be written in the form

$$f(y;\theta) = e^{y\theta - b(\theta)} f_0 (y)$$

where

- $b(\theta)$ is some function of $\theta$
- $f_0(y)$ involves only $y$, no $\theta$

then we call there PDF from one-parameter exponential family, where "one" means $\theta \in \mathbb{R} ^1$.

Some examples include

-  Normal with known variance $\sigma^2$

  \[
  f(y)=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp(-\frac{1}{2\sigma^{2}}(y^{2}-2\mu y+\mu^{2}))=\underbrace{\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left(-\frac{y^{2}}{2\sigma^{2}}\right)}_{\theta}{f_{0}(y)}\exp\left(y\underbrace{\frac{\mu}{\sigma^{2}}}_{\theta}-\frac{\mu^{2}}{2\sigma^{2}}\right)
  \]

-  Bernoulli

  \[
  P(y)=p^{y}(1-p)^{1-y}=\exp(y\underbrace{\ln\frac{p}{1-p}}_{\theta}+\ln(1-p))
  \]

-  Binomial

  \[
  P(y)=\left(\begin{array}{c}
  n\\
  y
  \end{array}\right)p^{y}(1-p)^{n-y}=\left(\begin{array}{c}
  n\\
  y
  \end{array}\right)\exp(y\underbrace{\ln\frac{p}{1-p}}_{\theta}+n\ln(1-p))
  \]

-  Poisson

  \[
  P(y)=\frac{e^{-\mu}\mu^{y}}{y!}=\frac{1}{y!}\exp(y\underbrace{\log\mu}_{\theta}-\mu)
  \]

Moreover, we call

- $y$: sufficient statistics
- $b(\theta)$: normalizing or cumulant function


#### Moments Relations

Distributions in one-parameter exponential family has some nice properties

- $\mu = \operatorname{E}\left( Y \right) = b ^\prime (\theta)$

- $\operatorname{Var}\left( Y \right) = b ^{\prime\prime}  (\theta) = v(\mu)$

- $\frac{\partial \mu}{\partial \theta} = b ^{\prime\prime}  (\theta) = \operatorname{Var}\left( Y \right) > 0$.

:::{admonition,dropdown,seealso} *Proof*


$$\begin{aligned}
1 &= \int f(y; \theta) \boldsymbol{~d}y \\
&= e ^{-b(\theta)} \int e^{y\theta} f_0(y) \boldsymbol{~d} y\\
\Rightarrow \quad e ^{b(\theta)}&=  \int e^{y\theta} f_0(y) \boldsymbol{~d} y \\
\end{aligned}$$

Taking derivative w.r.t. $\theta$ on both sides, we have


$$\begin{aligned}
b ^\prime (\theta) e ^{b(\theta)}
&=  \int y e^{y\theta} f_0(y) \boldsymbol{~d} y \\
&= e ^{b(\theta)} \int y e^{y\theta - b(\theta)} f_0(y) \boldsymbol{~d} y \\
&= e ^{b(\theta)} \int y f(y;\theta) \boldsymbol{~d} y \\
&= e ^{b(\theta)} \operatorname{E}\left( Y \right)\\
\Rightarrow \quad b ^\prime (\theta) &= \operatorname{E}\left( Y \right) \\
\end{aligned}$$

With a similar approach we can find $b ^{\prime \prime }(\theta) = \operatorname{Var}\left( Y \right)$

:::



#### Likelihood

Consider observations $y_1, y_2, \ldots, y_n$, each from a one-parameter exponential distribution parameterized by $\theta_i$. The log-likelihood of $\theta_1, \theta_2, \ldots \theta_1_n$ is


$$\begin{aligned}
\ell(\theta)
&= \log \prod_{i=1}^n f(y_i ;\theta)\\
&= \sum_{i=1}^n \left\{ y_i \theta_i - b(\theta_i) + \ln f_0 (y_i) \right\}\\
\end{aligned}$$

### Limitation of Linear Models

Recall a linear model is

$$Y_i = \boldsymbol{x} _i^\top \boldsymbol{\beta} + \varepsilon$$

and the linear predictor $\boldsymbol{x}_i ^\top \boldsymbol{\beta}$ aimes to predict the mean response

$$\operatorname{E}\left( Y_i \right) = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$$

Clearly, the range of RHS is $\mathbb{R}$, while our LHS response may not be so. It can be binary, discrete non-negatives, or other form. Thus, we need more capable models to model these particular data types.


### Model Form

A generalized linear model has the form

$$g \left( \boldsymbol{E}\left( Y_{i} \right) \right) = \boldsymbol{x_i} ^\top \boldsymbol{\beta} $$

where

- $Y_{i}$ is response, aka random component.

  We assume $Y_i \overset{  \text{iid}}{\sim} F$ where $F$ is some distribution, such as normal, binomial, poisson. Thus, we generalize the response $y_i$ from continuous real values in ordinary linear models, to binary response, counts, categories etc. Usually $F$ is from an exponential family.

- $\boldsymbol{x}_i ^\top \boldsymbol{\beta}$ is a linear predictor, often denoted $\eta_i$.

  As in ordinary least squared models, $\boldsymbol{x}_{i}$ can include interactions, non-linear transformations of the observed covariates and the constant term.

- $g(\cdot)$ is a link function

  Unlike linear model which use a linear predictor $\boldsymbol{x}_i ^\top \boldsymbol{\beta}$ to predict the mean response $\operatorname{E}\left( Y_i \right)$, GLM use the linear predictor to predict a function $g(\cdot)$ of the mean response. The link function connects the linear predictor with the random components: $g(\mu_i) = \eta_i$.

There are many ways to choose the link function. Note that if $Y_i \sim f(y_i; \theta)$ then $b ^\prime (\theta) = \operatorname{E}\left( Y \right) = \mu$, i.e. $\theta$ can be rewritten as a function of $\mu$: $\theta = (b ^\prime ) ^{-1} (\mu)$ (this form can usually be found in the PDF). If we choose this $(b ^\prime ) ^{-1}$ as the link function $g = (b ^\prime ) ^{-1}$, then it is called the **canonical link function**, and we will have $\theta_i = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$.

Some examples at a glance

- For linear models, $g(\mu)=\mu$, such that $\boldsymbol{E}\left( Y_i \right) = \boldsymbol{x} _i ^\top \boldsymbol{\beta}$
- For logistics regression, $Y_i \in \operatorname{Ber}(p)$ where $p=\mu$, then $\theta = \ln \frac{\mu}{1-\mu}$, hence $g(\mu) = \ln \frac{\mu}{1-\mu}$, and $\ln \frac{\boldsymbol{P}\left( Y_i=1\vert \boldsymbol{x} _i \right) }{\boldsymbol{P}\left( Y_i=0 \vert \boldsymbol{x} _i\right) } = \boldsymbol{x} _i^\top \boldsymbol{\beta}$
- For poisson regression, $\theta = \ln(\mu)$, hence $g(\mu) = \ln \mu$, and $\ln \boldsymbol{E}\left( Y_i \right)  = \boldsymbol{x} _i^\top \boldsymbol{\beta}$


## Estimation

Usually, we estimate the parameters by maximum likelihood.

First, we consider using the canonical link, such that $\theta_i = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$.

### Score Function

#### Canonical Link

As discussed above, the log-likelihood is

$$\begin{aligned}
\ell(\theta)
&= \log \prod_{i=1}^n f(y_i ;\theta)\\
&= \sum_{i=1}^n \left\{ y_i \theta_i - b(\theta_i) + \ln f_0 (y_i) \right\}\\
&= \sum_{i=1}^n \left\{ y_i \boldsymbol{x}_i ^\top \boldsymbol{\beta}  - b(\boldsymbol{x}_i ^\top \boldsymbol{\beta} ) \right\} + c\\
\end{aligned}$$

Taking derivative w.r.t. to $\beta_j$ gives


$$\begin{aligned}
\frac{\partial \ell(\boldsymbol{\beta})}{\partial \beta_j}
&= \sum_{i=1}^n \left\{ y_i x_{ij}  - b ^\prime (\boldsymbol{x}_i ^\top \boldsymbol{\beta} ) x_{ij}  \right\} \\
&= \sum_{i=1}^n \left\{ y_i x_{ij}  - b ^\prime (\theta)   x_{ij}\right\} \\
&= \sum_{i=1}^n (y_i -\mu_i)x_{ij} \\
\end{aligned}$$

In matrix, form, we have

$$
\frac{\partial \ell(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = \boldsymbol{X} ^\top (\boldsymbol{y} - \boldsymbol{\mu} )
$$

The score equation is, therefore,

$$\boldsymbol{X}  ^\top (\boldsymbol{y} - \boldsymbol{\mu} ) = \boldsymbol{0}$$


:::{admonition,note} Note


- Note that

  \[
  \frac{\partial^{2}\ell(\boldsymbol{\beta})}{\partial\beta_{h}\partial\beta_{j}}=-\sum_{i}b^{\prime\prime}\left(\boldsymbol{x}_{i}^\top\boldsymbol{\beta}\right)x_{ih}x_{ij}=-\text{Var}(Y_{i})x_{ih}x_{ij}
  \]

  So the Hessian is

  \[
  \frac{\partial^{2}\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^{\top}}=-\boldsymbol{X} ^\top \boldsymbol{V} \boldsymbol{X} \preceq \boldsymbol{0}
  \]

  where
  \[
  \boldsymbol{V} =\text{diag}(\operatorname{Var}\left(Y_{i}\right))
  \]

  The equality holds if $X$ has full rank.

  That is, the likelihood function $\ell(\boldsymbol{\beta})$ is concave in $\boldsymbol{\beta}$.

- There is no randomness in the Hessian matrix since it does not involve $y$. This holds for all exponential families

- For the log-likelihood, when $\boldsymbol{\beta}$ is the parameter, we see the
part involving both data and parameter is

  \[
  \sum_{j}\left(\sum_{i}y_{i}x_{ij}\right)\beta_{j}
  \]

  So the sufficient statistics for $\beta_{j}$ is $\sum_{i}y_{i}x_{ij}=\mathbf{x}_{j}^\top\boldsymbol{y}$.

  The overall score equations $\boldsymbol{X} ^\top\boldsymbol{y}= \boldsymbol{X} ^\top\boldsymbol{\mu}$ equate the sufficient statistics to their expected values.

:::

#### General Link


```{margin}
The likelihood \ell(\boldsymbol{\beta}) may not be concave if we use non-canoinical link.
```

If we use general link, then the score function is

\[
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}}=\boldsymbol{X} \boldsymbol{D} \boldsymbol{V} ^{-1}(\boldsymbol{y}-\boldsymbol{\mu})=0
\]

where

\[
\boldsymbol{D} =\text{diag}\left(\frac{\partial\mu_{i}}{\partial\eta_{i}}\right)
\]

\[
\boldsymbol{V} =\text{diag}(\operatorname{Var}\left(Y_{i}\right))
\]


:::{admonition,dropdown,seealso} *Derivation*

By the chain rule, we have

\[
\frac{\partial\ell_{i}}{\partial\beta_{j}}=\frac{\partial\ell_{i}}{\partial\theta_{i}}\frac{\partial\theta_{i}}{\partial\mu_{i}}\frac{\partial\mu_{i}}{\partial\eta_{i}}\frac{\partial\eta_{i}}{\partial\beta_{j}}
\]

Since

$$
\begin{align*}
\frac{\partial\ell_{i}}{\partial\theta_{i}} & =y_{i}-b^{\prime}\left(\theta_{i}\right)=y_{i}-\mu_{i}\\
\frac{\partial\theta_{i}}{\partial\mu_{i}} & =\frac{1}{b^{\prime\prime}\left(\theta_{i}\right)}=\frac{1}{\operatorname{Var}\left(Y_{i}\right)}\\
\frac{\partial\eta_{i}}{\partial\beta_{j}} & =x_{ij}
\end{align*}
$$


Then

\[
\frac{\partial\ell_{i}}{\partial\beta_{j}}=\frac{\left(y_{i}-\mu_{i}\right)x_{ij}}{\operatorname{Var}\left(Y_{i}\right)}\frac{\partial\mu_{i}}{\partial\eta_{i}}
\]

where
\[
\frac{\partial\mu_{i}}{\partial\eta_{i}}{=\frac{\partial\mu_{i}}{\partial g\left(\mu_{i}\right)}=\frac{1}{g^{\prime}\left(\mu_{i}\right)}}
\]

Hence in matrix form, the score equations are

\[
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\beta_{j}}=\sum_{i}\frac{\partial\ell_{i}}{\partial\beta_{j}}=\sum_{i}\frac{\left(y_{i}-\mu_{i}\right)x_{ij}}{\operatorname{Var}\left(y_{i}\right)}\frac{\partial\mu_{i}}{\partial\eta_{i}}
\]

and

\[
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}}=\boldsymbol{X} \boldsymbol{D} \boldsymbol{V} ^{-1}(\boldsymbol{y}-\boldsymbol{\mu})=0
\]

where

\[
\boldsymbol{D} =\text{diag}\left(\frac{\partial\mu_{i}}{\partial\eta_{i}}\right)
\]

\[
\boldsymbol{V} =\text{diag}(\operatorname{Var}\left(Y_{i}\right))
\]

Note that if $g$ is the canonical link then

\[
\frac{\partial\mu_{i}}{\partial\eta_{i}}=\frac{\partial\mu_{i}}{\partial\theta_{i}}=b^{\prime\prime}(\theta_{i})=\operatorname{Var}\left(y_{i}\right)
\]

hence $\boldsymbol{D} =\boldsymbol{V}$ and the score equations reduce to

\[
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}}=X(\boldsymbol{y}-\boldsymbol{\mu})=0
\]

:::

:::{admonition,note} Note

That is, the variance is a function of $\mu$. The relation $v(\cdot)$ uniquely characterized an exponential family distribution. For instance $v(\mu_{i})=\mu$ for Poisson and $v(\mu_{i})=\sigma^{2}$ (constant) for the normal. If we do not assume Y_{i} is from the exponential families, can we specify our own mean-variance relation? We will see the answer in Quasi-Likelihood section.

:::

### Computation

There are several ways to solve the score equations for $\widehat{\boldsymbol{\beta}} _{ML}$: Newton's Method, Fisher Scoring, Iterative Reweighted Least Squares.

## Asymptotic Distribution of MLE

## Hypothesis Testing




### Wald Test

### Likelihood Ratio Test

### Score Test
