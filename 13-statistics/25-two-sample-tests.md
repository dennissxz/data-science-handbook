# Two Sample Mean Tests


Suppose we have two samples of data $\left\{x_{1}, \cdots, x_{n}\right\}$ and $\left\{y_{1}, \cdots, y_{m}\right\}$.

A question of interest: Did the two samples come from the same distribution, as opposed to, one sample having larger values than the other on average?

To model them, we assume

- $X_{i}, \cdots, X_{n}$ i.i.d. sampled from a distribution with mean with mean $\mu_X$ and variance $\sigma^2 _X$

- $Y_{i}, \cdots, Y_{m}$ i.i.d. sampled from a distribution with mean with mean $\mu_Y$ and variance $\sigma^2 _Y$

We are interested in comparing the two samples, often by comparing the two means $\mu_X$ and $\mu_Y$.

An unbiased estimator for the difference in mean $\mu_X - \mu_Y$ is $\bar{X} - \bar{Y}$. To provide standard error to this estimator, $\operatorname{Var}\left( \bar{X} - \bar{Y} \right)$ need to be estimated, contingent on sample properties.


## Paired

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

### Normal

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

### Non-normal

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

## Independent

Now we consider $X_i$ and $Y_j$ are independent.

### Equal Variance

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

#### Normal

If both $X$ and $Y$ are of normal distributions, then the test statistic is

$$
\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}} \sim t_{n+m-2}
$$

A $(1-\alpha)\%$ confidence interval for the true difference in mean $\mu_X - \mu_Y$ is

$$
\bar{X}-\bar{Y} \pm t_{n+m-2,1-\alpha / 2} S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}
$$


#### Non-normal

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

### Unequal Variance

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

## Summary

The analysis for the above cases are summarized into the table below. In general, if $X$ and $Y$ are of normal distributions, the pivot quantity follows a known distribution. If not, we use CLT to obtain an approximate distribution, which requires **large** $n$ and $m$.

$$
H_0: \mu_X - \mu_Y = 0
$$


| Dependency | Test statistic | Normal | Non-normal, large $n, m$ |
| - | - | - | - |
| Paired (reduced to a univariate test)| $\frac{\bar{D}-\mu_{D}}{S_{D} / \sqrt{n}}$ | $\sim t_{n-1}$ | $\stackrel{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)$ |
| Independent with equal variance | $\frac{(\bar{X}-\bar{Y})-\left(\mu_{X}-\mu_{Y}\right)}{S_{p} \sqrt{\frac{1}{n}+\frac{1}{m}}}$ | $\sim t_{n+m-2}$ | $\stackrel{\mathcal{D}}{\longrightarrow} \mathcal{N}(0,1)$ |
| Independent with unequal variance | $\frac{\left( \bar{X}-\bar{Y} \right)-\left(\mu_{X}-\mu_{Y}\right)}{\sqrt{\frac{S_{X}^{2}}{n}+\frac{S_{Y}^{2}}{m}}}$ | / | $t_v$ |
