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

  $$
  f(y)=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp(-\frac{1}{2\sigma^{2}}(y^{2}-2\mu y+\mu^{2}))=\underbrace{\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left(-\frac{y^{2}}{2\sigma^{2}}\right)}_{\theta}{f_{0}(y)}\exp\left(y\underbrace{\frac{\mu}{\sigma^{2}}}_{\theta}-\frac{\mu^{2}}{2\sigma^{2}}\right)
  $$

-  Bernoulli

  $$
  P(y)=p^{y}(1-p)^{1-y}=\exp(y\underbrace{\ln\frac{p}{1-p}}_{\theta}+\ln(1-p))
  $$

-  Binomial

  $$
  P(y)=\left(\begin{array}{c}
  n\\
  y
  \end{array}\right)p^{y}(1-p)^{n-y}=\left(\begin{array}{c}
  n\\
  y
  \end{array}\right)\exp(y\underbrace{\ln\frac{p}{1-p}}_{\theta}+n\ln(1-p))
  $$

-  Poisson

  $$
  P(y)=\frac{e^{-\mu}\mu^{y}}{y!}=\frac{1}{y!}\exp(y\underbrace{\ln\mu}_{\theta}-\mu)
  $$

Moreover, we call

- $y$: sufficient statistics
- $b(\theta)$: normalizing or cumulant function


#### Moments Relations

Distributions in one-parameter exponential family has some nice properties

- $\mu = \operatorname{E}\left( Y \right) = b ^\prime (\theta)$

- $\operatorname{Var}\left( Y \right) = b ^{\prime\prime}  (\theta) = v(\mu)$

  This variance-mean relation uniquely characterize a distribution class (normal/binomial//Poisson) from exponential family.

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

Consider observations $y_1, y_2, \ldots, y_n$, each from a one-parameter exponential distribution parameterized by $\theta_i$. The log-likelihood of $\theta_1, \theta_2, \ldots \theta_n$ is

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

$$
g\left( \operatorname{E}\left( Y_{i} \right) \right) =\boldsymbol{x}_i ^\top \boldsymbol{\beta}
$$

where

- $Y_{i}$ is response, aka random component.

  We assume $Y_i \overset{  \text{iid}}{\sim} F$ where $F$ is some distribution, such as normal, binomial, poisson. Thus, we generalize the response $y_i$ from continuous real values in ordinary linear models, to binary response, counts, categories etc. Usually $F$ is from an exponential family.

- $\boldsymbol{x}_i ^\top \boldsymbol{\beta}$ is a linear predictor, often denoted $\eta_i$.

  As in ordinary least squared models, $\boldsymbol{x}_{i}$ can include interactions, non-linear transformations of the observed covariates and the constant term.

- $g(\cdot)$ is a link function

  Unlike linear model which use a linear predictor $\boldsymbol{x}_i ^\top \boldsymbol{\beta}$ to predict the mean response $\operatorname{E}\left( Y_i \right)$, GLM use the linear predictor to predict a function $g(\cdot)$ of the mean response. The link function connects the linear predictor with the random components: $g(\mu_i) = \eta_i$.

There are many ways to choose the link function. Note that if $Y_i \sim f(y_i; \theta)$ then $b ^\prime (\theta) = \operatorname{E}\left( Y \right) = \mu$, i.e. $\theta$ can be rewritten as a function of $\mu$: $\theta = (b ^\prime ) ^{-1} (\mu)$ (this form can usually be found in the PDF). If we choose this $(b ^\prime ) ^{-1}$ as the link function $g = (b ^\prime ) ^{-1}$, then it is called the **canonical link function**, and we will have $\theta_i = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$.

Some examples at a glance

- For linear models, $g(\mu)=\mu$, such that $\operatorname{E}\left( Y_i \right) = \boldsymbol{x} _i ^\top \boldsymbol{\beta}$
- For logistics regression, $Y_i \in \operatorname{Ber}(p)$ where $p=\mu$, then $\theta = \ln \frac{\mu}{1-\mu}$, hence $g(\mu) = \ln \frac{\mu}{1-\mu}$, and $\ln \frac{\boldsymbol{P}\left( Y_i=1\vert \boldsymbol{x} _i \right) }{\boldsymbol{P}\left( Y_i=0 \vert \boldsymbol{x} _i\right) } = \boldsymbol{x} _i^\top \boldsymbol{\beta}$
- For poisson regression, $\theta = \ln(\mu)$, hence $g(\mu) = \ln \mu$, and $\ln \operatorname{E}\left( Y_i \right)  = \boldsymbol{x} _i^\top \boldsymbol{\beta}$


## Estimation

Usually, we estimate the parameters by maximum likelihood.

First, we consider using the canonical link, such that $\theta_i = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$.

### Score Function

#### Canonical Link

As discussed above, the log-likelihood of parameter $\theta$ is

$$\begin{aligned}
\ell(\theta_1, \theta_2, \ldots, \theta_n)
&= \log \prod_{i=1}^n f(y_i ;\theta_i)\\
&= \sum_{i=1}^n \left\{ y_i \theta_i - b(\theta_i) + \ln f_0 (y_i) \right\}\\
\end{aligned}$$

Since $\theta = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$ and $\boldsymbol{x} _i$'s are known, the log-likelihood of coefficients $\boldsymbol{\beta}$ can be written in the same way by substituting $\theta_i = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$.

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left\{ y_i \boldsymbol{x}_i ^\top \boldsymbol{\beta}  - b(\boldsymbol{x}_i ^\top \boldsymbol{\beta} ) \right\} + c\\$$

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

  $$
  \frac{\partial^{2}\ell(\boldsymbol{\beta})}{\partial\beta_{h}\partial\beta_{j}}=-\sum_{i}b^{\prime\prime}\left(\boldsymbol{x}_{i}^\top\boldsymbol{\beta}\right)x_{ih}x_{ij}=-\text{Var}(Y_{i})x_{ih}x_{ij}
  $$

  So the Hessian is

  $$
  \frac{\partial^{2}\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^{\top}}=-\boldsymbol{X} ^\top \boldsymbol{V} \boldsymbol{X} \preceq \boldsymbol{0} \quad \text{where }  \boldsymbol{V} =\text{diag}(\operatorname{Var}\left(Y_{i}\right))
  $$

  with equality iff $X$ has full rank.

  That is, the likelihood function $\ell(\boldsymbol{\beta})$ is concave in $\boldsymbol{\beta}$.

- There is no randomness in the Hessian matrix since it does not involve $y$. This holds for all exponential families

- For the log-likelihood, when $\boldsymbol{\beta}$ is the parameter, we see the
part involving both data and parameter is

  $$
  \sum_{j}\left(\sum_{i}y_{i}x_{ij}\right)\beta_{j}
  $$

  So the sufficient statistics for $\beta_{j}$ is $\sum_{i}y_{i}x_{ij}=\boldsymbol{x}_{j}^\top\boldsymbol{y}$.

  The overall score equations $\boldsymbol{X} ^\top\boldsymbol{y}= \boldsymbol{X} ^\top\boldsymbol{\mu}$ equate the sufficient statistics to their expected values.

:::

#### General Link


```{margin}
The likelihood \ell(\boldsymbol{\beta}) may not be concave if we use non-canonical link.
```

If we use general link, then the score function is

$$
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}}=\boldsymbol{X} \boldsymbol{D} \boldsymbol{V} ^{-1}(\boldsymbol{y}-\boldsymbol{\mu})=0
$$

where

$$\begin{aligned}
\boldsymbol{D} &=\operatorname{diag}\left(\frac{\partial\mu_{i}}{\partial\eta_{i}}\right) \\
\boldsymbol{V} &=\operatorname{diag}(\operatorname{Var}\left(Y_{i}\right)) \\
\end{aligned}$$


:::{admonition,dropdown,seealso} *Derivation*

By the chain rule, we have

$$
\frac{\partial\ell_{i}}{\partial\beta_{j}}=\frac{\partial\ell_{i}}{\partial\theta_{i}}\frac{\partial\theta_{i}}{\partial\mu_{i}}\frac{\partial\mu_{i}}{\partial\eta_{i}}\frac{\partial\eta_{i}}{\partial\beta_{j}}
$$

Since

$$
\begin{align*}
\frac{\partial\ell_{i}}{\partial\theta_{i}} & =y_{i}-b^{\prime}\left(\theta_{i}\right)=y_{i}-\mu_{i}\\
\frac{\partial\theta_{i}}{\partial\mu_{i}} & =\frac{1}{b^{\prime\prime}\left(\theta_{i}\right)}=\frac{1}{\operatorname{Var}\left(Y_{i}\right)}\\
\frac{\partial\eta_{i}}{\partial\beta_{j}} & =x_{ij}
\end{align*}
$$


Then

$$
\frac{\partial\ell_{i}}{\partial\beta_{j}}=\frac{\left(y_{i}-\mu_{i}\right)x_{ij}}{\operatorname{Var}\left(Y_{i}\right)}\frac{\partial\mu_{i}}{\partial\eta_{i}}
$$

where
$$
\frac{\partial\mu_{i}}{\partial\eta_{i}}{=\frac{\partial\mu_{i}}{\partial g\left(\mu_{i}\right)}=\frac{1}{g^{\prime}\left(\mu_{i}\right)}}
$$

Hence in matrix form, the score equations are

$$
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\beta_{j}}=\sum_{i}\frac{\partial\ell_{i}}{\partial\beta_{j}}=\sum_{i}\frac{\left(y_{i}-\mu_{i}\right)x_{ij}}{\operatorname{Var}\left(y_{i}\right)}\frac{\partial\mu_{i}}{\partial\eta_{i}}
$$

and

$$
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}}=\boldsymbol{X} \boldsymbol{D} \boldsymbol{V} ^{-1}(\boldsymbol{y}-\boldsymbol{\mu})=0
$$

where

$$
\boldsymbol{D} =\text{diag}\left(\frac{\partial\mu_{i}}{\partial\eta_{i}}\right)
$$

$$
\boldsymbol{V} =\text{diag}(\operatorname{Var}\left(Y_{i}\right))
$$

Note that if $g$ is the canonical link then

$$
\frac{\partial\mu_{i}}{\partial\eta_{i}}=\frac{\partial\mu_{i}}{\partial\theta_{i}}=b^{\prime\prime}(\theta_{i})=\operatorname{Var}\left(y_{i}\right)
$$

hence $\boldsymbol{D} =\boldsymbol{V}$ and the score equations reduce to

$$
\frac{\partial\ell(\boldsymbol{\beta})}{\partial\boldsymbol{\beta}}=X(\boldsymbol{y}-\boldsymbol{\mu})=0
$$

:::

:::{admonition,note} More on $\operatorname{Var}\left( Y_i \right)$

That is, the variance is a function of $\mu$. The relation $v(\cdot)$ uniquely characterized an exponential family distribution. For instance $v(\mu_{i})=\mu$ for Poisson and $v(\mu_{i})=\sigma^{2}$ (constant) for the normal. If we do not assume $Y_{i}$ is from the exponential families, can we specify our own mean-variance relation? We will see the answer in Quasi-Likelihood section.

:::

### Computation

There are several ways to solve the score equations for $\widehat{\boldsymbol{\beta}} _{ML}$: Newton's Method, Fisher Scoring, Iterative Reweighted Least Squares.

## Inference

### Asymptotic Distribution of MLE

### Test $\boldsymbol{\beta} = \boldsymbol{\beta} _0$

#### Wald Test

#### Likelihood Ratio Test

#### Score Test


### Deviance

#### Saturated Models

First we define saturated models.

Definition (Saturated models)
: For a particular GLM with three components specified, we want to
fit means $\mu_{i}$. A saturated model fits $\mu_{i}$ by the
observation itself, i.e.,

  $$\tilde{\mu}_{i}=y_{i}$$

  which yields a perfect fit without parsimony. The number of parameter is
$p=n$. It is often use as a baseline for comparison with other models.


:::{admonition,note} Note

In principle, if two observations $i_{1}$ and $i_{2}$ share the same
covariates values $\boldsymbol{x}_{i_{1}}=\boldsymbol{x}_{i_{2}}$ then their
prediction value $\hat{y}_{i_{1}},\hat{y}_{i_{2}}$ should be the
same. But in the saturated model for GLM, we allow them to be different, i.e. $\hat{y}_{i_{1}} = y_{i_1},\hat{y}_{i_{2}} = y_{i_{2}}$.

:::

Recall that $g(\mu_i) = \boldsymbol{x}_i ^\top \boldsymbol{\beta}$. If we write the log-likelihood in terms of $\boldsymbol{\mu}$, we
get

$$
\ell(\boldsymbol{\beta}\vert\boldsymbol{y})=\ell(\boldsymbol{\mu}\vert\boldsymbol{y})
$$

In a saturated model, the value $\boldsymbol{\mu}$ is estimated by data $\tilde{\boldsymbol{\mu} } = \boldsymbol{y}$. The likelihood for the saturated model is

$$\begin{aligned}
\ell(\tilde{\boldsymbol{\mu} } \vert\boldsymbol{y}) & =\sum_{i}\ell(\tilde{\mu}_i \vert y_{i})\\
 & =\sum_{i}\left[y_{i}\tilde{\theta}_{i}-b(\tilde{\theta}_{i})+\log f_{0}(y_{i})\right]
\end{aligned}$$


where $\tilde{\theta}_{i}$ is a function of $\tilde{\mu}_{i}$.

The saturated model achieves the maximum achievable log-likelihood.
So for any other fit $\hat{\boldsymbol{\mu}}$, we have

$$
\ell(\tilde{\boldsymbol{\mu} } \vert\boldsymbol{y})\ge\ell( \hat{\boldsymbol{\mu}}\vert\boldsymbol{y})
$$


#### Goodness-of-fit by Deviance

As its name suggests, goodness-of-fit measures the goodness of fit
of a chosen model. In the test, the null hypothesis is

$$
H_{0}:\text{ the chosen model truly holds}
$$

The alternative hypothesis is

$$
H_{1}:\text{ other models are better}
$$

Defined in this way, sometimes it's called **global alternatives**.

Definition (Deviance)
: We want to measure the difference between a chosen model with fit
$\hat{\boldsymbol{\mu}}$, and the sacturated model with fit $\tilde{\boldsymbol{\mu}}=\boldsymbol{y}$.
Deviance is a statistic defined as

  $$
  \begin{aligned}
  D(\tilde{\boldsymbol{\mu}},\hat{\boldsymbol{\mu}}) & =-2\left[\ell(\hat{\boldsymbol{\mu}}\vert\boldsymbol{y})-\ell(\tilde{\boldsymbol{\mu}}\vert\boldsymbol{y})\right]\\
   & =-2\sum_{i}\log\frac{f(y_{i}\vert\hat{\mu})}{f(y_{i}\vert\tilde{\mu}_{i})}\\
   & \sim\chi_{n-p}^2
  \end{aligned}$$

  where $p$ is the number of parameters in the fit $\hat{\boldsymbol{\mu}}$.
  Note that in gerneral $D(\boldsymbol{\mu}_{1},\boldsymbol{\mu}_{2})\ne D(\boldsymbol{\mu}_{2},\boldsymbol{\mu}_{1})$.

We can denote deviance by $G^{2}$. In this course, it has the form

$$
G^2 := {D(\boldsymbol{y},\hat{\boldsymbol{\mu}})={\color{red}2}\sum}\text{observed}{\times\log \left( \frac{\text{observed}}{\text{fitted}} \right)}
$$

Since the saturated model is the perfect fit, the **larger** the deviance
$D(\boldsymbol{y},\hat{\boldsymbol{\mu}})$, the **poor** the fit $\hat{\boldsymbol{\mu}}$.

To maximize log-likelihood $\ell(\hat{\boldsymbol{\beta}})$ is equivalent to minimize deviance $D(\boldsymbol{y},\hat{\boldsymbol{\mu}})$.

Definition (Null deviance)
: Null deviance is defined as

  $$
  D(\tilde{\boldsymbol{u}},\bar{\boldsymbol{y}})=-2\left[\ell(\bar{\boldsymbol{y}}\vert\boldsymbol{y})-\ell(\tilde{\boldsymbol{\mu} }\vert\boldsymbol{y})\right]
  $$

  That is, all mean estimates has the same value $\hat{\mu}_i = \bar{y}$.

In sum, deviance is the $2$ times the distance from log-likelihood of some estimate $\hat{\boldsymbol{\mu}}$ to the log-likelihood of the saturated estimate $\tilde{\boldsymbol{\mu} } = \boldsymbol{y}$, as illustrated below.

:::{figure} glm-deviance-likelihood
<img src="../imgs/glm-deviance-likelihood.png" width = "50%" alt=""/>

Likelihood curve and deviance computation for saturated model $\boldsymbol{\mu}_{\text{saturated} } = \tilde{\boldsymbol{\mu}} = \boldsymbol{y}$, some model $\hat{\boldsymbol{\mu} }$, and null model $\boldsymbol{\mu}_{\text{null} } = \bar{\boldsymbol{y}}$
:::



Example (Deviance for Normal with $\sigma=1$)
: The deviance for normal distribution is

  $$
  D(\boldsymbol{\mu}_{1},\boldsymbol{\mu}_{2})=\sum_{i}(\mu_{1i}-\mu_{2i})^{2}
  $$

  So for an OLS model, the deviance is

  $$
  D(\boldsymbol{y}, \hat{\boldsymbol{\mu}})=\sum_{i}(y_{i}-\hat{\mu}_{i})^{2}=\sum_{i}(y_{i}-\hat{y})^{2}=RSS
  $$

  Thus we see deviance can be interpreted as a **generalization of sum of squared residuals** in OLS.

  Moreover, the null deviance is

  $$
  D(\boldsymbol{y},\bar{\boldsymbol{y} })=\sum_{i}(y_{i}-\bar{y}_{i})^{2}=TSS
  $$

As suggested by the example, we can define the generalized $R^{2}$ in GLM as

$$
R^{2}=1-\frac{RSS}{TSS}=1-\frac{D(\boldsymbol{y},\hat{\boldsymbol{\mu}})}{D(\boldsymbol{y},\bar{\boldsymbol{y}})}
$$


#### Compare Two Nested Models by Deviance

To compare two nested models $M_{\text{reduced} }:\boldsymbol{\beta}=\boldsymbol{\beta}_{\text{reduced} }$
v.s. $M_{\text{full} }:\boldsymbol{\beta}=\boldsymbol{\beta}_{\text{full} }$, in linear models we use $F$-test, in GLM we can can compare their deviances.

Suppose the corresponding fits are $\hat{\boldsymbol{\mu}}_{r}$ and $\hat{\boldsymbol{\mu}}_{f}$, and the numbers of parameters are $p_{r}$ and $p_{f}$, then the test statistic is essentially a likelihood-ratio statistic, defined as

$$\begin{aligned}
G^{2}(M_{r}\vert M_f) & =D(\boldsymbol{y},\hat{\boldsymbol{\mu}}_{r})-D(\boldsymbol{y},\hat{\boldsymbol{\mu}}_f)\\
 & =-2\left[\ell(\hat{\boldsymbol{\mu}}_r\vert\boldsymbol{y})-\ell(\hat{\boldsymbol{\mu}}_f\vert\boldsymbol{y})\right]\\
 & =-2\sum_{i}\log\frac{f(y_{i}\vert\hat{\mu}_{r ,i})}{f(y_{i}\vert\hat{\mu}_{f,i})}\\
 & \sim\chi_{p_f-p_r}^2
\end{aligned}$$

Note the **reduced** model has a **larger** deviance so $D(\boldsymbol{y},\hat{\boldsymbol{\mu}}_{r})>D(\boldsymbol{y},\hat{\boldsymbol{\mu}}_{f})$. This is analogous to the fact that a reduced model has a larger $RSS$ in [linear models](lm-rss-nonincreasing).

:::{figure} glm-deviance-test
<img src="../imgs/glm-deviance-test.png" width = "50%" alt=""/>

The likelihood test statistic $G^{2}(M_{r}\vert M_f)$ is the difference in log-likelihood, or deviance.
:::



When we want to compare multiple models, in linear models we use ANOVA. In GLM, we can use deviance analysis table.

### Generalized Pearson Statistic

In addition to deviance, generalized Pearson statistic is another quantity used to check goodness-of-fit.

Definition (Generalized Pearson statistic)
: A useful statistic to check goodness-of-fit is the generalized Pearson
statistic, which is defined as

  $$X^{2}:=\sum_{i}\frac{(y_{i}-\hat{\mu}_{i})^{2}}{v(\hat{\mu}_{i})}$$

  In some GLM,  it has the form

  $$
  X^2 = \sum\frac{(\text{observed}-\text{fitted})^{2}}{\text{fitted}}
  $$

  It is an alternative to the deviance for testing the fit of certain
GLMs.

```{margin}
In the book, this generalized Pearson statistic is often called *Pearson statistic* in short, which is quite confusing with *Pearson's Chi-square statistic* for testing independence of categorical variables in contingency tables.
```


### Residuals

To detect a model's lack of fit, any particular type of residual $\hat{\boldsymbol{\varepsilon}}$ below
can be plotted against the fitted values $\hat{\boldsymbol{\mu}}$
and against each explanatory variables.

#### Asymptotically Uncorrelated with $\hat{\boldsymbol{\mu} }$

In LM, residuals $\hat{\boldsymbol{\varepsilon}}  =\boldsymbol{y}-\hat{\boldsymbol{y}}$ and fitted values $\hat{\boldsymbol{y}}$ are orthogonal, or uncorrelated, regardless of sample size $n$.

But in GLM, $\hat{\boldsymbol{\varepsilon}} = \boldsymbol{y}-\hat{\boldsymbol{\boldsymbol{\mu}}}$ and
$\hat{\boldsymbol{\boldsymbol{\mu}}}$ are **asymptotically** uncorrelated. See Section 4.4.5 for details.


#### Pearson Residuals

The Pearson residual for observation $i$ is defined as

$$e_{i}=\frac{y_{i}-\hat{\mu}_{i}}{\sqrt{v(\hat{\mu}_{i})}}$$

Their squared values sum to the generalized Pearson statistic.


#### Deviance Residual

Let $D(\boldsymbol{y},\boldsymbol{\hat{\boldsymbol{\mu}}})=\sum_{i}d_{i}$. The deviance residual for observation $i$ is defined as

$$e_{i}=\sqrt{d_{i}} \operatorname{sign} (y_{i}-\hat{\mu}_{i})$$

Their squared values sum to the deviance.

#### Standardized Residuals

The standardized residual divides each raw residual $(y_{i}-\hat{\mu}_{i})$
by its standard error. Let $h_{ii}$ be the diagonal element of the
generalized hat matrix for observation $i$, which is called its *leverage*.
Then, the standardized residual for observation $i$ is defined as

$$
r_{i}=\frac{y_{i}-\hat{\mu}_{i}}{\sqrt{v(\hat{\mu}_{i})(1-\hat{h}_{ii})}}=\frac{e_{i}}{\sqrt{1-\hat{h}_{ii}}}
$$

Their squared values sum to the deviance.
