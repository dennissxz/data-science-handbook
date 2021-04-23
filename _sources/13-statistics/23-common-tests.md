# Common Tests

<!-- ## Median test

Mood's median test is a special case of Pearson's chi-squared test. It is a nonparametric test that tests the null hypothesis that the medians of the populations from which two or more samples are drawn are identical.

## $z$-test

## $t$-test

## $F$-test

$F$ -distribution with degrees of freedom $\left(d_{1}, d_{2}\right)$, denoted as $F_{d_{1}, d_{2}}$, has the form
$$
\frac{Y_{1} / d_{1}}{Y_{2} / d_{2}}
$$
with $Y_{1} \sim \chi_{d_{1}}^{2}, \quad Y_{1} \sim \chi_{d_{2}}^{2}$ and $Y_{1} \Perp Y_{2}$.

## $\chi^2$-test -->

## Two-sample Mean Tests

Suppose we have two samples of data $\left\{x_{1}, \cdots, x_{n}\right\}$ and $\left\{y_{1}, \cdots, y_{m}\right\}$.

A question of interest: Did the two samples come from the same distribution, as opposed to, one sample having larger values than the other on average?

To model them, we assume

- $X_{i}, \cdots, X_{n}$ i.i.d. sampled from a distribution with mean with mean $\mu_X$ and variance $\sigma^2 _X$

- $Y_{i}, \cdots, Y_{m}$ i.i.d. sampled from a distribution with mean with mean $\mu_Y$ and variance $\sigma^2 _Y$

We are interested in comparing the two samples, often by comparing the two means $\mu_X$ and $\mu_Y$.

An unbiased estimator for the difference in mean $\mu_X - \mu_Y$ is $\bar{X} - \bar{Y}$. To provide standard error to this estimator, $\operatorname{Var}\left( \bar{X} - \bar{Y} \right)$ need to be estimated, contingent on sample properties.


### Paired

In many studies, the $i$-th measurements in the two samples $x_i$ and $y_i$ actually are related, such as measurements before and after a treatment from the same subject.

When $m = n$ and $\operatorname{Corr}\left( X_i,Y_i \right) = \rho \ne 0$, very often $\rho >0$. Then,

$$
\begin{aligned}
\operatorname{Var}(\bar{X}-\bar{Y}) &=\operatorname{Var}(\bar{X})+\operatorname{Var}(\bar{Y})-2 \times \operatorname{Cov}(\bar{X}, \bar{Y}) \\
& \leq \operatorname{Var}(\bar{X})+\operatorname{Var}(\bar{Y}) \quad \text{when }\rho >0
\end{aligned}
$$

To have a more precise variance estimate, it is appropriate to consider pairing $X_i$ and $Y_i$ by investigating their difference:

$$
D_{i}=X_{i}-Y_{i}, \quad i=1, \cdots, n
$$

Note that $D_i$'s are i.i.d. under the i.i.d. assumption of each of the two samples, with

$$
\mu_{D}=\mu_{X}-\mu_{Y}, \quad \sigma_{D}^{2}=\sigma_{X}^{2}+\sigma_{Y}^{2}-2 \rho \sigma_{X} \sigma_{Y}
$$

Then essentially, we changed the inference problem into that of a **one-sample case**.

An unbiased estimator for the variance of difference $\sigma^2_D$ is the sample variance.

$$
S_{D}^{2}=\frac{1}{n-1} \sum_{i=1}^{n}\left(D_{i}-\bar{D}\right)^{2}=\widehat{\operatorname{Var}}\left(D_{i}\right)=\hat{\sigma}_{D}^{2}
$$

#### Normal

If $X \sim \mathcal{N} \left(\mu_{X}, \sigma_{X}^{2}\right), Y \sim \mathcal{N} \left(\mu_{Y}, \sigma_{Y}^{2}\right)$, then we have the distributions for the sample estimators

$$
\frac{\bar{D}-\mu_{D}}{\sigma_{D} / \sqrt{n}} \sim \mathcal{N}(0,1), \quad \frac{(n-1) S_{D}^{2}}{\sigma_{D}^{2}} \sim \chi_{n-1}^{2}
$$

In addition, they are independent. By the definition of $t$-distribution, the test statistic

$$
\frac{\bar{D}-\mu_{D}}{S_{D} / \sqrt{n}} \sim t_{n-1}
$$

is a pivot quantity not depending on the parameters if we are testing a hypothesis on $\mu_D$. For instance,

$$
H_{0}: \mu_{D}=0
$$

A $(1-\alpha)\%$ confidence interval for the true difference in mean $\mu_X - \mu_Y$ is


$$
\bar{X}-\bar{Y} \pm t_{n-1}^{(1-\alpha / 2)} \frac{S_{D}}{\sqrt{n}}
$$

#### Non-normal

When the samples are not normally distributed, $t_{n-1}$ can be used as an approximate distribution.

When n is large, we may use the Central Limit Theorem,

$$
\frac{\bar{D}-\mu_{D}}{S_{D} / \sqrt{n}} \overset{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)
$$

The asymptotic pivotal property can be used to conduct hypothesis tests.

A $(1-\alpha)\%$ confidence interval for the true difference in mean $\mu_X - \mu_Y$ is

$$
\bar{X}-\bar{Y} \pm z^{(1-\alpha / 2)} \frac{S_{D}}{\sqrt{n}}
$$

### Independent

Now we consider $X_i$ and $Y_j$ are independent.

#### Equal Variance

If $\sigma^2 _X = \sigma^2 _Y$, then

$$
\operatorname{Var}(\bar{X}-\bar{Y})=\frac{\sigma_{X}^{2}}{n}+\frac{\sigma_{Y}^{2}}{m}=\sigma^{2}\left(\frac{1}{n}+\frac{1}{m}\right)
$$

and both sample variances $S_X^2$ and $S_Y^2$ are unbiased estimators of $\sigma^2$

A better unbiased estimator is the **pooled sample variance**

$$
S_{p}^{2}=S_{\text {pooled }}^{2}=\frac{(n-1) S_{X}^{2}+(m-1) S_{Y}^{2}}{n+m-2}
$$

which has a **larger testing power** than the two sample variances.

##### Normal

If both $X$ and $Y$ are of normal distributions, then the test statistic is

$$
\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}} \sim t_{n+m-2}
$$

A $(1-\alpha)\%$ confidence interval for the true difference in mean $\mu_X - \mu_Y$ is

$$
\bar{X}-\bar{Y} \pm t_{n+m-2,1-\alpha / 2} S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}
$$


##### Non-normal

When the samples are not normally distributed, $t_{n+m-2}$ distribution can be used as an approximation.

When **both** $n$ and $m$ are large, we may apply the Central Limit Theorem,

$$
\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}} \overset{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)
$$


:::{admonition,note} Pooling vs paring

Consider the case n = m with equal variance.

- Under assumption that the two samples are independent, the variance is

    $$
    \operatorname{Var}(\bar{X}-\bar{Y})= \frac{2 \sigma^{2}}{n}
    $$

    The pooled sample variance is the appropriate estimator to be used.

- If the two samples were correlated with $Corr(X_i, X_j) = \rho > 0$, the variance becomes smaller

    $$
    \operatorname{Var} (\bar{X}-\bar{Y})=(1-\rho) \frac{2 \sigma^{2}}{n}< \frac{2 \sigma^{2}}{n}
    $$

As a result, when correlation exists, the smaller paired sample variance is the appropriate one to use, since the test statistic using it has a **larger power**.

On the other hand, if the correlation is substantial and we fail to take it into consideration, the pooled sample variance estimator likely will overestimate the variance, and the estimate could be too large to be useful.

:::

#### Unequal Variance

If $\sigma_{X}^{2} \neq \sigma_{Y}^{2}$, the variance we are interested to estimate has the form

$$
\operatorname{Var}(\bar{X}-\bar{Y})=\frac{\sigma_{X}^{2}}{n}+\frac{\sigma_{Y}^{2}}{m}
$$

which can be estimated by the unbiased estimator

$$
\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}
$$

It is complicated to construct a pivot quantity like we did for the previous cases. Consider

$$
T=\frac{\bar{X}-\bar{Y}-\left(\mu_{X}-\mu_{Y}\right)}{\sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}}
$$

When both $X$ and $Y$ are of normal distributions, we have

$$
(n-1) S_{X}^{2} / \sigma_{X}^{2} \sim \chi_{n-1}^{2}, \quad(m-1) S_{Y}^{2} / \sigma_{Y}^{2} \sim \chi_{m-1}^{2}
$$

But since $\sigma_{X}^{2} \neq \sigma_{Y}^{2}$, the summation

$$
\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m} \sim \frac{\sigma_{X}^{2}}{n(n-1)} x_{n-1}^{2}+\frac{\sigma_{Y}^{2}}{m(m-1)} X_{m-1}^{2}
$$

is not a multiple of a $\chi^2$ distribution.

Hence, $T$ is not $t$-distributed.

If $n$ and $m$ are both large, we can resort to Central Limit Theorem as usual,

$$
T=\frac{\left( \bar{X}-\bar{Y} \right)-\left(\mu_{X}-\mu_{Y}\right)}{\sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}} \stackrel{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)
$$

The asymptotic approximation lead to a $(1-\alpha)\%$ confidence interval for the true difference in mean $\mu_X - \mu_Y$

$$
\bar{X}_{i}-\bar{Y} \pm z_{1-\alpha / 2} \sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}
$$

However, there is a better approximation using $t_v$ distribution than the normal approximation.

$$
T=\frac{\bar{X}-\bar{Y}-\left(\mu_{X}-\mu_{Y}\right)}{\sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}} \stackrel{\mathcal{D}}{\longrightarrow} t_v
$$

But the degree of freedom $v$ is involved. It is estimated by  Welch-Satterthwaite approximation,


$$
\nu \approx \frac{\left(\frac{S_{X}^{2}}{n}+\frac{S_{T}^{2}}{m}\right)^{2}}{\left(\frac{S_{x}^{2}}{n}\right)^{2} /(n-1)+\left(\frac{S_{Y}^{2}}{m}\right)^{2} /(m-1)}
$$

The asymptotic approximation lead to a $(1-\alpha)\%$ confidence interval for the true difference in mean $\mu_X - \mu_Y$,

$$
\bar{X}_{i}-\bar{Y} \pm t_{\nu}^{1-\alpha / 2} \sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}
$$

### Summary

The analysis for the above cases are summarized into the table below. In general, if $X$ and $Y$ are of normal distributions, the pivot quantity follows a known distribution. If not, we use CLT to obtain an approximate distribution, which requires **large** $n$ and $m$.

$$
H_0: \mu_X - \mu_Y = 0
$$


| Dependency | Test statistic | Normal | Non-normal, large $n, m$ |
| - | - | - | - |
| Paired (reduced to a univariate test)| $\frac{\bar{D}-\mu_{D}}{S_{D} / \sqrt{n}}$ | $\sim t_{n-1}$ | $\stackrel{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)$ |
| Independent with equal variance | $\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}}$ | $\sim t_{n+m-2}$ | $\stackrel{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)$ |
| Independent with unequal variance | $\frac{\left( \bar{X}-\bar{Y} \right)-\left(\mu_{X}-\mu_{Y}\right)}{\sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}}$ | / | $\stackrel{\mathcal{D}}{\longrightarrow} t_v$ |


## ANOVA

Analysis of variance is used to compare several univariate sample means. For instance, in the plot below, we have five levels and observed the response $y$. We are interested in whether the five means are equal.

:::{figure} test-one-way
<img src="../imgs/test-one-way.png" width = "70%" alt=""/>

One-way Layout
:::

### Model

There are $n_\ell$ observations from population or treatment group $\ell = 1, 2, \ldots, g$

$$
X_{\ell j}=\mu+\tau_{\ell}+e_{\ell j}, \quad \ell=1, \cdots, g, \quad j=1, \cdots, n_{\ell}
$$

where
- $\mu$ is the **overall mean** parameter.
- $\tau_{\ell}$ is the **treatment effect** parameter of the $\ell$ th population or $\ell$ th treatment group.
- $e_{\ell j} \sim N\left(0, \sigma^{2}\right)$ is individual specific homogenous noise.
- Parameter constraints: There should be constraints on the parameters such as $\sum_{\ell=1}^{g} n_{\ell} \tau_{\ell}=0$, to avoid redundancy or unidentifiability.


To detect differences in treatment effects among the groups,
the first test of interest is

$$
\left\{\begin{array}{ll}
H_{0}: & \tau_{1}=\cdots=\tau_{g}=0 \\
H_{1}: & \tau_{\ell} \neq 0, \text { for some } \ell=1, \cdots, g
\end{array}\right.
$$

### Test Statistic

First, we decompose the observations as

$$\begin{aligned}
x_{\ell j} &= \bar{x} + \left(\bar{x}_{\ell}-\bar{x}\right)+\left(x_{\ell j}-\bar{x}_{\ell}\right) \\
&= \hat{\mu} + \hat{\tau}_\ell + \hat{e}_{\ell j}\\
\end{aligned}$$


where $\bar{x}$ is the overall mean, $\bar{x}_\ell$ is the $\ell$-th group mean. Hence, the observation can be seen as estimated overall mean + estimated treatment effect + estimated noise. Equivalently,

$$
\left(x_{\ell j}-\bar{x}\right)=\left(\bar{x}_{\ell}-\bar{x}\right)+\left(x_{\ell j}-\bar{x}_{\ell}\right)
$$

Summing up all **squared** terms, noticing that $\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\bar{x}_{\ell}-\bar{x}\right)\left(x_{\ell j}-\bar{x}_{\ell}\right)=0$, we have

$$
\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(x_{\ell j}-\bar{x}\right)^{2}=\sum_{\ell=1}^{g} n_{\ell}\left(\bar{x}_{\ell}-\bar{x}\right)^{2}+\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(x_{\ell j}-\bar{x}_{\ell}\right)^{2}
$$

The decomposition can be stated as

$$
\sum(\text { total variation })^{2}=\sum\left(\begin{array}{c}
\text { between-group } \\
\text { treatment variation }
\end{array}\right)^{2}+\sum\left(\begin{array}{c}
\text { within-group } \\
\text { residual variation }
\end{array}\right)^{2}
$$

The corresponding numbers of independent quantities of each term, i.e. the degrees of freedom, have the relation

$$
\sum_{\ell=1}^{g} n_{\ell}-1=(g-1)+\sum_{\ell=1}^{g}\left(n_{\ell}-1\right)
$$

Therefore, we obtain the analysis of variance table

$$
\begin{array}{c|c|c|c}
\hline \begin{array}{c}
\text { Source } \\
\text { of variation }
\end{array} & \text { SS (sum of squares) } & \text { d.f. } & \begin{array}{c}
F \text {-value } \\
\text { (variance ratio) }
\end{array} \\
\hline \text { Treatments } & S S_{t r t}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\bar{x}_{\ell}-\bar{x}\right)^{2} & g-1 & S S_{t r t} /(g-1) \\
\text { Residuals } & S S_{\text {res }}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(x_{\ell j}-\bar{x}_{\ell}\right)^{2} & \sum_{\ell=1}^{g} n_{\ell}-g & \\
\hline \text { Total } & S S_{t o t}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(x_{\ell j}-\bar{x}\right)^{2} & \sum_{\ell=1}^{g} n_{\ell}-1 & \\
\hline
\end{array}
$$

At test level $\alpha$, the null $H_0: \tau_{1}=\cdots=\tau_{g}=0$ is rejected if


$$
\frac{S S_{t r t} /(g-1)}{S S_{\text {res }} /\left(\sum_{\ell=1}^{g} n_{\ell}-g\right)}>F_{g-1, \sum_{\ell=1}^{g} n_{\ell}-g}(\alpha)
$$


:::{admonition,warning} Warning

In `manova()` function in R, if the group variable $\ell$ is in numerical value, it is necessary to coerce it as data type `factor` by calling `as.factor()`.

:::


## Multivariate Settings

### Hotelling's $T^2$ Distribution

Definition (Hotelling's $T^2$ Distribution )
: Suppose random zero-mean Gaussian $\boldsymbol{x} \sim \mathcal{N} _p(\boldsymbol{0} , \boldsymbol{\Sigma} )$ and Wishart random matrix $\boldsymbol{V} \sim W_p(k, \boldsymbol{\Sigma} )$ are independent. Define

  $$
  T^2 = k \boldsymbol{x} ^{\top} \boldsymbol{V} ^{-1} \boldsymbol{x}
  $$

  Then $T^2$ is said to follow a Hotelling's $T^2$ distribution with parameter $p$ and $k$, denoted as $T^2(p, k)$.

  In univariate sense, the Hotelling’s $T^2$ statistic can be reduced to the **squared** $t$-statistic. Hence it can be seen as multivariate generalization of (squared) $t$-distribution.

Properties
: - If $\bar{\boldsymbol{x}}$ and $\boldsymbol{S}$ are respectively the sample mean vector and sample covariance matrix of a random sample of size $n$ taken from $\mathcal{N} _p(\boldsymbol{\mu} , \boldsymbol{\Sigma} )$, then

    $$n(\bar{\boldsymbol{x}}-\boldsymbol{\mu})^{\top} \boldsymbol{S}^{-1}(\bar{\boldsymbol{x}}-\boldsymbol{\mu}) \sim T^{2}(p, n-1)$$

  - The distribution of the quadratic form under non-normality is reasonably robust as long as the underlying multivariate distribution has pdf contours close to elliptical shape, but $T^2$ is sensitive to the departure from such elliptical symmetry of the distribution.
  - Invariant under transformation of $\boldsymbol{x}$:$\boldsymbol{C} \boldsymbol{x} + \boldsymbol{d}$, where $\boldsymbol{C}$ is non-singular.

  - Related to other distribution:
    - $T^{2}(p, k)=\frac{k p}{k-p+1} F(p, k-p+1)$, usually used to find quantile $T^2(\alpha)$.
    - $T^{2}(1, k)=t^{2}(k)=F(1, k)$
    - $T^{2}(p, \infty) \rightarrow \chi ^2 _p$ by multivariate [CLT](CLT), **without** assuming normality of the distribution of $\boldsymbol{x}$
  - Related to Mahalanobis distance: $T^{2}=n D_{\boldsymbol{S}}^{2}(\bar{\boldsymbol{x}}, \boldsymbol{\mu})$


### One-sample Mean

Assume $\boldsymbol{x} \sim \mathcal{N} _p(\boldsymbol{0} , \boldsymbol{\Sigma} )$, want to test

$$
H_{0}: \boldsymbol{\mu}=\boldsymbol{\mu}_{0} \operatorname{vs } H_{1}: \boldsymbol{\mu} \neq \boldsymbol{\mu}_{0}
$$

Test statistic under $H_0$
: - $\boldsymbol{\Sigma}$ is known:   

    $$T^{2}=n\left(\bar{\boldsymbol{x}}-\boldsymbol{\mu}_{0}\right)^{\top} \boldsymbol{\Sigma}^{-1}\left(\bar{\boldsymbol{x}}-\boldsymbol{\mu}_{0}\right) \sim \chi^{2}(p)$$

  - $\boldsymbol{\Sigma}$ is unknown, estimated by $\boldsymbol{S}$:


    $$\begin{aligned}
    T^{2}=n\left(\bar{\boldsymbol{x}}-\boldsymbol{\mu}_{0}\right)^{\top} \boldsymbol{S}^{-1}\left(\bar{\boldsymbol{x}}-\boldsymbol{\mu}_{0}\right) &\sim T^{2}(p, n-1) \\
    &\sim \frac{(n-1) p}{n-p} F(p, n-p) \\
    & \rightarrow \chi ^2 _p \quad \text{as } n \rightarrow \infty  
    \end{aligned}$$

  - Analogously, in univariate case,

    $$
    \left\{\begin{array}{l}
    \frac{\sqrt{n}\left(\bar{x}-\mu_{0}\right)}{\sigma} \sim N(0,1) \text { if } \sigma^{2} \text { is known } \\
    \frac{\sqrt{n}\left(\bar{x}-\mu_{0}\right)}{s} \sim t(n-1) \text { if } \sigma^{2} \text { is unknown. }
    \end{array}\right.
    $$


Confidence Region
: - A $(1-\alpha)100\%$ confidence region for $\boldsymbol{\mu}$ is a $p$-dimensional ellipsoid centered at $\bar{\boldsymbol{x}}$, i.e. a collection of all those $\boldsymbol{\mu}$ which will not be rejected by the above $T^2$ test at significance level $\alpha$.

    $$
    \left\{\boldsymbol{\mu}: n(\bar{\boldsymbol{x}}-\boldsymbol{\mu})^{\top} \boldsymbol{S}^{-1}(\bar{\boldsymbol{x}}-\boldsymbol{\mu}) \leq T_{\alpha}^{2}(p, n-1)=c_{\alpha}\right\}
    $$

    :::{figure} test-ellipsoid
    <img src="../imgs/test-ellipsoid.png" width = "50%" alt=""/>

    Confidence region
    :::

  - This confidence ellipsoid above is the most precise confidence region of the vector $\boldsymbol{\mu}$, in the sense that any other form of confidence region for $\boldsymbol{\mu}$ with the same confidence level $(1-\alpha)$ will have **larger volume** in the $p$-dimensional space of $\boldsymbol{\mu}$ and hence less precise.

Simultaneous confidence intervals for each component
: - Individual CIs: Sometimes people get used to confidence intervals for individual components, such as

    $$
    \bar{x}_{j}-t^{\alpha / 2}_{n-1} \frac{s_j}{\sqrt{n}} <\mu_{j}<\bar{x}_{j}+t^{\alpha / 2}_{n-1} \frac{s_j}{\sqrt{n}}
    $$

    where $\bar{x}_j$ and $s_j$ are respectively the sample mean and standard deviation of the $j$-th variate that has mean $\mu_j$. But there are [multiple testing](multiple-testing) issues. We can then use Bonferroni or Scheffe simultaneous C.I.s to correct this.

  - The $(1-\alpha)100\%$ Bonferroni simultaneous C.I.s for $m$ **pre-determined** linear components of means, $\boldsymbol{a}_{i}^{\top} \boldsymbol{\mu}(i=1, \ldots, m)$, are given by

    $$
    \boldsymbol{a}_{i}^{\top} \bar{\boldsymbol{x}} \pm t ^{\alpha/(2m)}_{n-1} \sqrt{\frac{\boldsymbol{a}_{i}^{\top} \boldsymbol{S} \boldsymbol{a}_{i}}{n}}
    $$

  ```{margin}
  Scheffe simultaneous C.I. works like a guarantee for any 'data snooping' linear combinations in exploratory data analysis. Besides, it is related to [union intersection test](UIT).
  ```

  - The $(1-\alpha)100\%$ Scheffe simultaneous C.I.s for **all possible** linear combinations of means $\boldsymbol{a} ^{\top} \boldsymbol{\mu}$ are given by

    $$
    \boldsymbol{a}^{\top} \bar{\boldsymbol{x}} \pm \sqrt{T_{\alpha}^{2}(p, n-1)} \sqrt{\frac{\boldsymbol{a}^{\top} \boldsymbol{S a}}{n}}
    $$

  - Pros: Compared with the advantages of ellipsoidal confidence regions, these hyper-rectangles (orthotopes) are easier to form and to compute.

  - Cons: Both Bonferroni and Scheffé intervals are **wider** (hence less accurate) than the ordinary confidence intervals which are constructed with separate confidence level of $(1-\alpha)$.

    ```{margin}
    Are Scheffe interval boundaries tangent to the ellipsoid confidence region??
    ```

    :::{figure} test-multi-Bon-Sch
    <img src="../imgs/test-multi-Bon-Sch.png" width = "100%" alt=""/>

    Bonferroni (left) and Scheffe (right) simultaneous C.I.s.
    :::

  - If we just want to conduct univariate tests of means $H_0: \mu_k = 0$ for each $k = 1, 2, \ldots, p$, i.e. $\boldsymbol{a} _k = \boldsymbol{e} _k$, then the C.I. has the general form $\bar{x}_{k} \pm c_{n, p, \alpha} \sqrt{\frac{s_{k k}}{n}}$ for some multiplier $c_{n, p, \alpha}$ depending on $n,p,\alpha$. The above methods can be summarized as follows
    - marginal C.I. using $t$ statistics (ignoring dependence among components): $t_{n-1}^{\alpha/2}$
    - Bonferroni simultaneous C.I. using $t$ statistics: $t_{n-1}^{\alpha/(2p)}$
    - Scheffe simultaneous C.I.: $\sqrt{T^2_\alpha (p, n-1)}$
    - Asymptotic simultaneous C.I. using $\chi ^2$ statistic as $n$ is large: $\sqrt{\chi ^2 _p (\alpha)}$

### Two-sample Means

Given two samples of $p$-variates, we are interest in whether their means are equal.

$$
H_0: \boldsymbol{\mu} _1 = \boldsymbol{\mu} _2,\quad H_1: \text{otherwise}
$$

#### Paired

First, we consider paired comparison for two dependent samples.

This is easy, we just define $\bar{\boldsymbol{d}} = \bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2$, and apply the above one-sample mean method to test $H_0: \boldsymbol{d} = \boldsymbol{0}$.

More precisely, if $\boldsymbol{d} \sim \mathcal{N} _p (\boldsymbol{\delta}, \boldsymbol{\Sigma} _d)$

$$
T^{2}=n(\bar{\boldsymbol{d} }-\boldsymbol{\delta} ) ^{\top}  \boldsymbol{S} _d^{-1}(\bar{\boldsymbol{d} }-\boldsymbol{\delta} ) \sim T^2(p, n-1) \sim \frac{(n-1) p}{n-p} F_{p, n-p}
$$


#### Two Independent Samples

We assume equal variance $\boldsymbol{\Sigma} _1 = \boldsymbol{\Sigma} _2 = \boldsymbol{\Sigma}$. The pooled sample covariance matrix is an unbiased estimator of it

$$
\boldsymbol{S}_{\text {pool }}=\frac{\left(n_{1}-1\right) \boldsymbol{S}_{1}+\left(n_{2}-1\right) \boldsymbol{S}_{2}}{n_{1}+n_{2}-2}, \quad \mathbb{E}\left(\boldsymbol{S}_{\text {pool }}\right)=\boldsymbol{\Sigma}
$$

By the independence between the two samples, the covariance of sample difference is

$$
\operatorname{Cov}\left(\bar{\boldsymbol{x}}_{1}-\bar{\boldsymbol{x}}_{2}\right)=\operatorname{Cov}\left(\bar{\boldsymbol{x}}_{1}\right)+\operatorname{Cov}\left(\bar{\boldsymbol{x}}_{2}\right)=\frac{1}{n_{1}} \boldsymbol{\Sigma} +\frac{1}{n_{2}} \boldsymbol{\Sigma}
$$

which can be estimated by $\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right) \boldsymbol{S}_{\text {pool }}$ since

$$
\mathbb{E}\left[\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right) \boldsymbol{S}_{\text {pool }}\right]=\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right) \boldsymbol{\Sigma} =\operatorname{Cov}\left(\bar{\boldsymbol{x}}_{1}-\bar{\boldsymbol{x}}_{2}\right)
$$

Assume $\boldsymbol{x} _1 \sim \mathcal{N} _p (\boldsymbol{\mu} _1, \boldsymbol{\Sigma} ), \boldsymbol{x} _2 \sim \mathcal{N} _p (\boldsymbol{\mu} _2, \boldsymbol{\Sigma} )$, then under $H_0: \boldsymbol{\mu} _1 = \boldsymbol{\mu} _2$,

$$\begin{aligned}
T^{2} &\sim T^{2}\left(p, n_{1}+n_{2}-2\right) \\
&\sim \frac{\left(n_{1}+n_{2}-2\right) p}{n_{1}+n_{2}-p-1} F_{p, n_{1}+n_{2}-p-1}
\end{aligned}$$

where

$$
T^{2}=\left[\left(\bar{\boldsymbol{x}}_{1}-\bar{\boldsymbol{x}}_{2}\right)-\left(\boldsymbol{\mu} _{1}-\boldsymbol{\mu} _{2}\right)\right]^{\top}\left[\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right) \boldsymbol{S}_{\text {pool }}\right]^{-1}\left[\left(\bar{\boldsymbol{x}}_{1}-\bar{\boldsymbol{x}}_{2}\right)-\left(\boldsymbol{\mu} _{1}-\boldsymbol{\mu} _{2}\right)\right]
$$

The $(1-\alpha)\%$ Bonferroni simultaneous confidence interval for the difference of the $j$-th component means $\mu_{1j} - \mu_{2j}$ is

$$
\bar{x}_{1 j}-\bar{x}_{2 j} \pm t_{n_{1}+n_{2}-2}^{\alpha / (2 p)} \sqrt{\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right) s_{j j, \text{pool} }}, \quad j=1, \cdots, p
$$

More generally,

- $(1-\alpha)\%$ Bonferroni intervals for pre-determined $\boldsymbol{a}_{i}^{\top}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right), i=1, \ldots, k$ is

  $$
  \boldsymbol{a}_{i}^{\top}\left(\bar{\boldsymbol{x}}_{1}- \bar{\boldsymbol{x}}_{2}\right) \pm t_{n_{1}+n_{2}-2}^{\alpha / (2 k)} \sqrt{\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right)\boldsymbol{a}_{i}^{\top} \boldsymbol{S}_{\text{pool} } \boldsymbol{a}_{i}}
  $$

- $(1-\alpha)\%$ Scheffe simultaneous intervals for  $\boldsymbol{a}^{\top}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right)$ for all $\boldsymbol{a}$ is

  $$
  \boldsymbol{a}^{\top}\left(\bar{\boldsymbol{x}}_{1}- \bar{\boldsymbol{x}}_{2}\right) \pm \sqrt{T_{\alpha}^{2}\left(p, n_{1}+n_{2}-2\right)} \sqrt{\left(\frac{1}{n_{1}}+\frac{1}{n_{2}}\right)\boldsymbol{a}^{\top} \boldsymbol{S}_{\text{pool} } \boldsymbol{a}}
  $$

(manova)=
### MANOVA

If there are multiple samples of multivariate observations, we use Multivariate Analysis of Variance (MANOVA). The data are treated as $g$ sample groups of observed sample values, each sample group is from one of $g$ populations.

#### Model

The MANOVA model, generalized from ANOVA, becomes

$$
\boldsymbol{X}_{\ell j}=\boldsymbol{\mu}+\boldsymbol{\tau}_{\ell}+\boldsymbol{e}_{\ell j}, \quad j=1, \cdots, n_{\ell}, \quad \ell=1, \cdots, g
$$

- $\boldsymbol{\mu}$ is the **overall mean** vector,
- $\boldsymbol{\tau} _{\ell}$ is the **treatment effect** vector of the $\ell$ th population or treatment group,
- $\boldsymbol{e}_{\ell j} \sim N_{p}(0, \Sigma)$ is individual specific homogenous noise.
- The parameter constraint here is $\sum_{\ell=1}^{g} n_{\ell} \boldsymbol{\tau}_{\ell}=0$.


#### Test Statistic

Analogous to the univariate case, the test of interest is

$$
\left\{\begin{array}{ll}
H_{0}: & \boldsymbol{\tau}_{1}=\cdots=\boldsymbol{\tau}_{g}=\boldsymbol{0}_{p} \\
H_{1}: & \boldsymbol{\tau}_{\ell} \neq 0_{p}, \text { for some } \ell=1, \cdots, g .
\end{array}\right.
$$

The data can be decomposed similarly,

$$
\boldsymbol{x}_{\ell j}=\bar{\boldsymbol{x}}+\left(\bar{\boldsymbol{x}}_{\ell}-\bar{\boldsymbol{x}}\right)+\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}_{\ell}\right)
$$

or

$$
(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}})=\left(\bar{\boldsymbol{x}}_{\ell}-\bar{\boldsymbol{x}}\right)+\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}_{\ell}\right)
$$

Then

$$
\begin{aligned}
\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}\right)^{\prime}=&\left(\bar{\boldsymbol{x}}_{\ell}-\bar{\boldsymbol{x}}\right)\left(\bar{\boldsymbol{x}}_{\ell}-\bar{\boldsymbol{x}}\right)^{\prime}+\left(\bar{\boldsymbol{x}}_{\ell}-\bar{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}_{\ell}\right)^{\prime} \\
&+\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}_{\ell}\right)\left(\bar{\boldsymbol{x}}_{\ell}-\bar{\boldsymbol{x}}\right)^{\prime}+\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}_{\ell}\right)\left(\boldsymbol{x}_{\ell j}-\bar{\boldsymbol{x}}_{\ell}\right)^{\prime}
\end{aligned}
$$

Summing up, we have

$$
\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}\right)^{\prime}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\overline{\boldsymbol{x}}_{\ell}-\overline{\boldsymbol{x}}\right)\left(\overline{\boldsymbol{x}}_{\ell}-\overline{\boldsymbol{x}}\right)^{\prime}+\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\mathbf{x}}_{\ell}\right)\left(\boldsymbol{x}_{\ell_{j}}-\overline{\boldsymbol{x}}_{\ell}\right)^{\prime}
$$

since

$$
\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\overline{\boldsymbol{x}}_{\ell}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}_{\ell}\right)^{\prime}=\boldsymbol{0}_{p \times p}, \quad \sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}_{\ell}\right)\left(\overline{\boldsymbol{x}}_{\ell}-\overline{\boldsymbol{x}}\right)^{\prime}=\boldsymbol{0}_{p \times p}
$$

The decomposition can be stated as

$$
\sum(\text { total variation })^{2}=\sum\left(\begin{array}{c}
\text { "between-group" } \\
\text { treatment variation }
\end{array}\right)^{2}+\sum\left(\begin{array}{c}
\text { "within-group" } \\
\text { residual variation }
\end{array}\right)^{2}
$$

with corresponding degrees of freedom

$$
\sum_{\ell=1}^{g} n_{\ell}-1=(g-1)+\sum_{\ell=1}^{g}\left(n_{\ell}-1\right)
$$

Now we analyze the test statistic

Denote the between group (or between population) sum of squares and
cross products matrix as

$$
\boldsymbol{B}=\sum_{\ell=1}^{g} n_{\ell}\left(\overline{\boldsymbol{x}}_{\ell}-\overline{\boldsymbol{x}}\right)\left(\overline{\boldsymbol{x}}_{\ell}-\overline{\boldsymbol{x}}\right)^{\prime}
$$

and the within group sum of squares and cross products matrix as

$$
\boldsymbol{W}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}_{\ell}\right)\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}_{\ell}\right)^{\prime}=\left(n_{1}-1\right) \boldsymbol{S}_{1}+\cdots+\left(n_{g}-1\right) \boldsymbol{S}_{g}
$$

In fact $\boldsymbol{W}$ is related to the pooled covariance matrix,


$$
\boldsymbol{S}_{\text {pool }}=\frac{1}{\sum_{\ell=1}^{g}\left(n_{\ell}-1\right)}\left[\left(n_{1}-1\right) \boldsymbol{S}_{1}+\cdots+\left(n_{g}-1\right) \boldsymbol{S}_{g}\right]=\frac{1}{n-\mathrm{g}} \boldsymbol{W}
$$

The MANOVA table is then

$$
\begin{array}{c|c|c}
\hline \begin{array}{c}
\text { Source } \\
\text { of variation }
\end{array} & \begin{array}{c}
\text { Matrix of sum of squares } \\
\text { and cross-products }
\end{array} & \begin{array}{c}
\text { Degrees } \\
\text { of freedom }
\end{array} \\
\hline \text { Treatments } & \boldsymbol{B}=\sum_{\ell=1}^{g} n_{\ell}\left(\overline{\boldsymbol{x}}_{\ell}-\bar{x}\right)\left(\overline{\boldsymbol{x}}_{\ell}-\bar{x}\right)^{\prime} & g-1 \\
\text { Residuals } & \boldsymbol{W}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}_{\ell}\right)\left(\boldsymbol{x}_{\ell j}-\overline{\mathbf{x}}_{\ell}\right)^{\prime} & \sum_{\ell=1}^{g} n_{\ell}-\mathrm{g} \\
\hline \text { Total } & \boldsymbol{B}+\boldsymbol{W}=\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}\right)^{\prime} & \sum_{\ell=1}^{g} n_{\ell}-1 \\
\hline
\end{array}
$$

Since $\boldsymbol{B}$ and $\boldsymbol{W}$ are $p \times p$ covariance matrices, the test statistic uses their determinants, or the generalized variances.

We introduce Wilks' Lambda

$$
\Lambda^{*}=\frac{\operatorname{det}(\boldsymbol{W})}{\operatorname{det}(\boldsymbol{B}+\boldsymbol{W})}=\frac{\left|\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell j}-\overline{\boldsymbol{x}}_{\ell}\right)\left(\boldsymbol{x}_{\ell j}-\overline{\mathbf{x}}_{\ell}\right)^{\prime}\right|}{\left|\sum_{\ell=1}^{g} \sum_{j=1}^{n_{\ell}}\left(\boldsymbol{x}_{\ell_{j}}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{\ell_{j}}-\overline{\boldsymbol{x}}\right)^{\prime}\right|}
$$

which is the ratio of generalized variance of residual / generalized variance of total.

The distribution of $\Lambda^{*}$ depends on $p, g, n_\ell$ and is related to $F$ distribution. When $n = \sum n_\ell$ is large, Bartlett gives a simple chi-square approximation

$$
-\left(n-1-\frac{p+g}{2}\right) \ln \Lambda^{*} \sim \chi_{p(g-1)}^{2}
$$

Even for moderate sample size, it is good practice to check and compare both tests.

Note

- We reject the null hypothesis that all group means are equal if the value of $\Lambda^{*}$ is too “small”.
- Wilks’ lambda is equivalent to the likelihood ratio test statistic.
- Under the null Wilks’ lambda is of its own $\Lambda^{*}$-distribution, which is derived from the ratio of two random matrices $\boldsymbol{W}$ and $\boldsymbol{B} + \boldsymbol{W}$, each is of [Wishart distribution](wishart-distribution)
- We can express Wilks' lambda by eigenvalues of $\boldsymbol{B} \boldsymbol{W} ^{-1}$, which can be seen as signal-noise ratio. If it is large, then $\lambda$ is large, and $\Lambda^{*}$ is small.

  $$
  \Lambda^{*}=\frac{|\boldsymbol{W}|}{|\boldsymbol{B}+\boldsymbol{W}|}=\frac{1}{\left|\boldsymbol{W}^{-1} \boldsymbol{B}+\boldsymbol{l}\right|}=\prod_{k=1}^{p} \frac{1}{1+\lambda_{k}}
  $$

Other test statistics using the eigenvalues of $\boldsymbol{B} \boldsymbol{W} ^{-1}$ include
- Hotelling-Lawley’s Trace: $\operatorname{trace}\left(\mathbf{B W}^{-1}\right)=\sum_{k=1}^{p} \lambda_{k}$
- Pillai’s Trace:  $\operatorname{trace}\left(\boldsymbol{B}(\boldsymbol{B}+\boldsymbol{W})^{-1}\right)=\operatorname{trace}\left(\boldsymbol{B} \boldsymbol{W}^{-1}\left(\boldsymbol{B} \boldsymbol{W}^{-1}+I\right)^{-1}\right)=\sum_{k=1}^{p} \frac{\lambda_{k}}{1+\lambda_{k}}$
- Roy's Largest Root: $\max _{k}\left\{\lambda_{k}\right\}=\left\|\boldsymbol{B} \boldsymbol{W}^{-1}\right\|_{\infty}$ which gives an upper bound

#### C.I. for Difference in Two Means

If the null hypothesis of MANOVA is rejected, a natural question is, **which** treatments have significant effects?

To compare the effect of treatment $k$ and treatment $\ell$, the quantity of interests is the difference of the vectors $\boldsymbol{\tau}_k - \boldsymbol{\tau}_\ell$ which is the same as $\boldsymbol{\mu} _k - \boldsymbol{\mu} _\ell$. For two fixed $k, \ell$, there are $p$ variables to compare. For each variable $i$, we want a confidence interval for $\tau_{ki} - \tau_{\ell i}$, which have the form

$$
\hat{\tau}_{k i}-\hat{\tau}_{\ell i} \pm c \times \sqrt{\widehat{\operatorname{Var}}\left(\hat{\tau}_{k i}-\hat{\tau}_{\ell i}\right)}
$$

where the multiplier $c$ depends on the level and the type of the confidence interval.

Assuming mutual independence and equal variance $\boldsymbol{\Sigma}$ among the $g$ samples, using Bonferroni correction, we have
- $c = t_{n-g}^{\alpha/2m}$, where $m= p \binom{g}{2}$ is the number of simultaneous confidence intervals.
- $\widehat{\operatorname{Var}}\left(\hat{\tau}_{k i}-\hat{\tau}_{\ell i}\right)=\frac{w_{i i}}{n-g}\left(\frac{1}{n_{k}}+\frac{1}{n_{\ell}}\right)$ where $w_{ii}$ is the diagonal entry of $\boldsymbol{W}$.

Note that
- The Bonferroni method often gives confidence intervals too wide to be practical even for moderate $p$ and $g$.
- The equal variance can be tested.

#### Test for Equal Covariance

Are the variables in the $g$ population groups sharing the same
covariance structure?

$$
\left\{\begin{array}{ll}
H_{0}: & \boldsymbol{\Sigma}_{1}=\boldsymbol{\Sigma}_{2}=\cdots=\boldsymbol{\Sigma}_{g}=\boldsymbol{\Sigma} \\
H_{1}: & \boldsymbol{\Sigma}_{i} \neq \boldsymbol{\Sigma}_{j} \quad \text { for some } i \neq j
\end{array}\right.
$$

Box’s $M$-test for equal covariance structure is a likelihood-ratio type of test. Denote

$$
\Lambda=\prod_{\ell=1}^{g}\left(\frac{\left|\boldsymbol{S}_{\ell}\right|}{\left|\boldsymbol{S}_{\text {pool }}\right|}\right)^{\left(n_{\ell}-1\right) / 2}
$$

where

$$
\boldsymbol{S}_{\text {pool }}=\frac{1}{\sum_{\ell=1}^{g}\left(n_{\ell}-1\right)}\left[\left(n_{1}-1\right) \boldsymbol{S}_{1}+\cdots+\left(n_{g}-1\right) \boldsymbol{S}_{g}\right]=\frac{1}{n-\mathrm{g}} \boldsymbol{W}
$$

Box’s test is based on an approximation that the sampling distribution of $\ln \Lambda$ is approximately of $\chi ^2$ distribution under the equal covariance matrix hypothesis. Box's $M$ is defined as

$$
\begin{aligned}
M &=-2 \ln \Lambda \\
&=(n-g) \ln \left|\boldsymbol{S}_{\text {pool }}\right|-\sum_{\ell=1}^{g}\left[\left(n_{\ell}-1\right) \ln \left|\boldsymbol{S}_{\ell}\right|\right]
\end{aligned}
$$

Under the hypothesis $H_0$ of equal covariance, approximately

$$
(1-u) M \sim \chi_{v}^{2}
$$

where
- $u = \left(\sum_{\ell=1}^{g} \frac{1}{n_{\ell}-1}-\frac{1}{n-g}\right) \frac{2 p^{2}+3 p-1}{6(p+1)(g-1)}$
- $v = p(p+1)(g-1)/2$

$H_0$ is rejected if $(1-u) M > \chi_{v}^{2}(\alpha)$. Box’s M-test works well for small $p$ and $g$ $(\le 5)$ and moderate to large $n_\ell$ $(\ge 20)$.
