# Multinomial Logistic Regression

Aka multinomial GLM, or multinomial logit in short.

A generalization of binomial GLM is multinomial GLM, where we have $c$ categories. In particular,

- if the categories have internal order, we use [ordinal](24-ordinal-logistic) logistic regression
- if the categories are nested (train/car/bus and then red/blue bus), we use nested logit models.

## Data Structure

In this chapter we express models in terms of such ungrouped data. As with binary data, however, with discrete explanatory variables it is better to group the $N$ observations according to their multicategory trial indices $\left\{ n_{i}\right\}$ before forming the deviance and other goodness-of-fit statistics and residuals.

For subject $i$, let $\pi_{ij}$ denote the probability of response in category $j$, with ${\sum_{j=1}^{c}\pi_{ij}=1}$. The observation is a vector of binary entries ${\boldsymbol{y}_{i}=\left(y_{i1},\ldots,y_{ic}\right)}$, where $y_{ij}=1$ when the response is in category $j$ and $y_{ij}=0$ otherwise, then ${\sum_{j}y_{ij}=1}$. The probability distribution is

$$
{p\left(y_{i1},\ldots, y_{ic}\right)=\pi_{i1}^{y_{i1}}\cdots\pi_{ic}^{y_{i}}}
$$

If there is no order in the categories, the response is called nominal response. Otherwise, it's called ordinal response.

## Baseline-Category Logit Model

### Link Function

We construct a multinomial logistic model by pairing each response category with a baseline category, such as the last category $c$. Like in logistic regression We use a linear predictor to model the log odds:

$$
g_{j}(\boldsymbol{\pi}_{i})=\log\frac{\pi_{ij}}{\pi_{ic}}=\boldsymbol{x}_i ^\top \boldsymbol{\beta} _{j}\quad j=1,2,\ldots,{c-1}
$$

Note for each category $j$, there is a separate $\boldsymbol{\beta}_{j}$, so there are $(c-1)\times p$ parameters in total.

Suppose $\boldsymbol{\beta} _c = \boldsymbol{0}$. We can obtain

$$
\pi_{ij}=\frac{\exp(\boldsymbol{x} ^\top \boldsymbol{\beta} _{j})}{\sum_{h=1}^{c}\exp(\boldsymbol{x}_{i}^{\top}\boldsymbol{\beta}_{h})}=\frac{\exp(\boldsymbol{x} ^\top \boldsymbol{\beta} _{j})}{1+\sum_{h=1}^{c-1}\exp(\boldsymbol{x}_{i}^{\top}\boldsymbol{\beta}_{h})},\ j=1,2,\ldots,{c-1}
$$

If $c=2$, this formula simplifies to the probability for logistic regression.

By convention $\boldsymbol{\beta}_{c}=\boldsymbol{0}$ and $\pi_{ic}=\frac{1}{1+\sum_{h=1}^{c-1}\exp(\boldsymbol{x}_{i}^{\top}\boldsymbol{\beta}_{h})}$.

### Interpretation

Recall there are $(c-1)\times p$ number of $\beta$'s. Interpretation of the overall effects for $\boldsymbol{\beta}_{j}$ is not simple, since in the above formula we see $\pi_{ij}$ also depends on other $\boldsymbol{\beta}_{h}$. The derivative of $\pi_{ij}$ w.r.t. the $k$-th covariate $x_{ik}$ is

$$
{\frac{\partial\pi_{ij}}{\partial x_{ik}}=\pi_{ij}\left(\beta_{jk}-\sum_{h}\pi_{ih}\beta_{hk}\right)}
$$

which depends on $\beta_{hk}$ in other $\boldsymbol{\beta}_{h}$. So the derivative need not have the same sign as $\beta_{jk}$.

Note that $\pi_{ij}$ is a function of $\boldsymbol{x}_{i}$. In fact, the derivative may change sign as $x_{ik}$ increases (Exercise 6.4).

However, we can interpret the effect of $\beta_{jk}$ on the odds $\frac{\pi_{ij}}{\pi_{ic}}$ for setting $i$. By the link function,

$$
\frac{\pi_{ij}}{\pi_{ic}}=\exp\left(\boldsymbol{x}_i ^\top \boldsymbol{\beta} _{j}\right)
$$

If $x_{ik}$ increases 1 unit, the odds is multiplied by $\exp(\beta_{jk})$, similar to the meaning in logistic regression.

### Multivariate Exponential Families

Multivariate here means $\boldsymbol{y}$ is a vector. The PDF is

$$
f(\boldsymbol{y}\vert\boldsymbol{\theta})=\exp(\boldsymbol{y}^{\top}\boldsymbol{\theta}-b(\boldsymbol{\theta}))f_{0}(\boldsymbol{y})
$$

Let $\boldsymbol{g}$ be a vector of link functions, the multivariate GLM has the form

$$
\boldsymbol{g}(\boldsymbol{\mu}_{i})= \boldsymbol{X} _{i}^{\top}\boldsymbol{\beta}
$$

or equivalently,

$$
\left(\begin{array}{c}
g_{1}(\boldsymbol{\mu}_{i})\\
g_{2}(\boldsymbol{\mu}_{i})\\
\vdots\\
g_{c-1}(\boldsymbol{\mu}_{i})
\end{array}\right)=\left(\begin{array}{cccc}
\boldsymbol{x}_{i}^{\top} & \boldsymbol{0} & \dots & \boldsymbol{0}\\
\boldsymbol{0} & \boldsymbol{x}_{i}^{\top} & \dots & \boldsymbol{0}\\
\vdots & \vdots & \ddots & \vdots\\
\boldsymbol{0} & \boldsymbol{0} & \dots & \boldsymbol{x}_{i}^{\top}
\end{array}\right)\left(\begin{array}{c}
\boldsymbol{\beta}_{1}\\
\boldsymbol{\beta}_{2}\\
\vdots\\
\boldsymbol{\beta}_{c-1}
\end{array}\right)_{(c-1)p\times1}
$$

Note the $c$-th category is redundant.

The baseline-category logit model is a multivariate GLM with

$$
\boldsymbol{\mu}_{i}={\left(\pi_{i1},\dots,\pi_{i,c-1}\right)}^{\top}
$$

and **canonical** link functions

$$
g_{j}(\boldsymbol{\mu}_{i})=\log\left(\frac{\pi_{ij}}{\pi_{ic}}\right),\ j=1,2,\ldots,{c-1}
$$


## Estimation

### Score Equations

The log-likelihood for subject $i$ is

$$
{\begin{aligned}\log\left(\prod_{j=1}^{c}\pi_{ij}^{y_{ij}}\right) & =\sum_{j=1}^{c-1}y_{ij}\log\pi_{ij}+\left(1-\sum_{j=1}^{c-1}y_{ij}\right)\log\pi_{ic}\\
 & =\sum_{j=1}^{c-1}y_{ij}\log\frac{\pi_{ij}}{\pi_{ic}}+\log\pi_{ic}
\end{aligned}
}
$$

Substituting $\log\left(\frac{\pi_{ij}}{\pi_{ic}}\right)=\boldsymbol{x}_i ^\top \boldsymbol{\beta} _{j}$, the total log-likelihood is

$$
\begin{aligned}\ell(\boldsymbol{\beta};\boldsymbol{y}) & =\sum_{i=1}^{N}\left\{ \sum_{j=1}^{c-1}y_{ij}\left(\boldsymbol{x}_i ^\top \boldsymbol{\beta} _{j}\right)-\log\left[1+\sum_{j=1}^{c-1}\exp\left(\boldsymbol{x} _i ^{\top} \boldsymbol{\beta} _{j}\right)\right]\right\} \\
 & =\sum_{j=1}^{c-1}\left[\sum_{k=1}^{p}\beta_{jk}\left(\sum_{i=1}^{N}x_{ik}y_{ij}\right)\right]-\sum_{i=1}^{N}\log\left[1+\sum_{j=1}^{c-1}\exp\left(\boldsymbol{x} ^\top \boldsymbol{\beta} _{j}\right)\right]
\end{aligned}
$$

The sufficient statistic for $\beta_{jk}$ is $\sum_{i=1}^{N}x_{ik}y_{ij}$. In particular, for the intercept $\beta_{j1}$, it is $\sum_{i=1}^{N}y_{ij}$, which is the total number of observations in category $j$.

The partial derivatives are

$$
{\frac{\partial\ell(\boldsymbol{\beta},\boldsymbol{y})}{\partial\beta_{jk}}=\sum_{i=1}^{N}x_{ik}y_{ij}-\sum_{i=1}^{N}\left[\frac{x_{ik}\exp\left(\boldsymbol{x}_i ^\top \boldsymbol{\beta} _{j}\right)}{1+\sum_{h=1}^{c-1}\exp\left(\boldsymbol{x} _i^\top \boldsymbol{\beta} _{h}\right)}\right]=\sum_{i=1}^{N}x_{ik}\left(y_{ij}-\pi_{ij}\right)}
$$

for $j=1, \ldots, c-1$ and $k=1\ldots,p$

So the score equations are

$$
\sum_{i=1}^{N}x_{ik}y_{ij}=\sum_{i=1}^{N}x_{ik}\pi_{ij}
$$

for $j=1, \ldots, c-1$ and $k=1\ldots,p$.

Again, the score equations equate the sufficient statistics to their expected value.

### Computation

We derive the Hessian matrix.

Note that for two coefficients $\beta_{jk}$ and $\beta_{jh}$ in $\boldsymbol{\beta}_{j}$,

$$
{\frac{\partial^{2}\ell(\boldsymbol{\beta},\boldsymbol{y})}{\partial\beta_{jk}\partial\beta_{jl}}=-\sum_{i=1}^{N}x_{ik}x_{il}\pi_{ij}\left(1-\pi_{ij}\right)}
$$

for categories $j\ne h$,

$$
{\frac{\partial^{2}\ell(\boldsymbol{\beta},\boldsymbol{y})}{\partial\beta_{jk}\partial\beta_{hl}}=\sum_{i=1}^{N}x_{ik}x_{il}\pi_{ij}\pi_{ih}}
$$

So the Fisher Information matrix consists of $(c-1)^{2}$ blocks of size $p\times p$,

$$
{-\frac{\partial^{2}\ell(\boldsymbol{\beta},\boldsymbol{y})}{\partial\boldsymbol{\beta}_{j}\partial\boldsymbol{\beta}_{h}^{\mathrm{\top}}}=\sum_{i=1}^{N}\pi_{ij}\left[\mathbb{I}\left\{ j=h\right\} -\pi_{ij^{\prime}}\right]\boldsymbol{x}_{i}^{\top}\boldsymbol{x}_{i}}
$$

The Hessian is negative-definite, so the log-likelihood function is concave and has a unique maximum. The observed and expected information are identical, so the Newton-Raphson method is equivalent to Fisher scoring for finding the ML parameter estimates, a consequence of the link function being the canonical one. Convergence is usually fast unless at least one estimate is infinite or does not exist (see Note 6.2).

## Hypothesis Testing

### Test the Effect of $x_{k}$

To test the effect of the $k$-th covariate is to test all the coefficients
of it, namely $\beta_{jk}$ for category $j=1,2,\ldots,{c-1}$. There are $k$ constraints

$$
{H_{0}:\beta_{1k}=\beta_{2k}=\cdots=\beta_{c-1,k}=0}
$$

The likelihood ratio test can be applied, with $df=c-1$. The likelihood-ratio
test statistic equals the difference in the deviance values for comparing
the models.

### Deviance

We will use grouped data for summary of goodness-of-fit. Now $\boldsymbol{y}_{i}\sim\frac{1}{n_{i}} \operatorname{Multinomial} (n_{i},\boldsymbol{\pi}_{i})$. That is, $y_{ij}$ is the proportion of the observations in category $j$.

The deviance compares log-likelihood of a fit $\left\{ \hat{\pi}_{ij}\right\}$
and the saturated model $\left\{ \tilde{\pi}_{ij}=y_{ij}\right\}$,
which is

$$
{D(\boldsymbol{y},\hat{\boldsymbol{\mu}})=2\sum_{i=1}^{N}\sum_{j=1}^{c}n_{i}y_{ij}\log\frac{n_{i}y_{ij}}{n_{i}\hat{\pi}_{ij}}}\sim\chi^2_{df}
$$

where $df=\#\text{probabilities}-\#\text{parameters}$. Usually $\#\text{probabilities}=N(c-1)$ and $\#\text{parameters}=p(c-1)$.

It satisfies the general form

$$
{D(\boldsymbol{y},\hat{\boldsymbol{\mu}})=2\sum}\text{observed}\times{\log\left(\frac{\text{ observed }}{\text{ fitted }}\right)}
$$

For ungrouped data, the above formula remains valid and is used to
compare nested unsaturated models.

### Pearson Statistic

The Pearson statistic is

$$
{X^{2}=\sum_{i=1}^{N}\sum_{j=1}^{c}\frac{\left(n_{i}y_{ij}-n_{i}\hat{\pi}_{ij}\right)^{2}}{n_{i}\hat{\pi}_{ij}}}\sim\chi^2_{df}
$$

where $df=\#\text{probabilities}-\#\text{parameters}$. Normally $\#\text{probabilities}=N(c-1)$
and $\#\text{parameters}=p(c-1)$.

It satisfies the general form

$$
{X^{2}=\sum\frac{(\text{ observed }-\text{ fitted })^{2}}{\text{ fitted }}}
$$


:::{admonition,note} Note

For both deviance and Pearson statistic

- they have approximate chi-squared null distributions when the expected cell counts mostly exceed about 5.

- the sums are taken over all observed counts ${\left\{ n_{i}y_{ij}\right\} }$. This explains why the sums are taken over $2N$ success and failures in the deviance and Pearson statistic for logistic regression.

:::
