# Hypothesis Testing

*Explain hypothesis testing related concepts*

<!-- TOC -->

- [Hypothesis Testing](#hypothesis-testing)
  - [$p$-value](#p-value)
  - [Type I error and Type II error](#type-i-error-and-type-ii-error)
    - [Definitions](#definitions)
    - [Error Control](#error-control)
  - [Confidence Interval](#confidence-interval)
  - [Credible Interval](#credible-interval)
  - [Bonferroni Correction](#bonferroni-correction)

<!-- /TOC -->


When we investigate a question of interest, we first formulate a null hypothesis and an alternative hypothesis, denoted as $H_0$ and $H_1$ respectively. For instance,

$$H_0: \theta = \theta_0\quad \text{vs} \quad H_1: \theta > \theta_0$$

To conduct a hypothesis test, there are three components

1. Null and alternative hypothesis $H_0,H_1$
2. Test statistic $T$
3. Rejection rule, e.g. $T>c$

Give the null hypothesis $H_0$, we collect data and compute some test statistic $T$, and use its value in some rejection rule $(T>c)$ to see whether we reject the null hypothesis or not.

There are many other components that analysis the goodness of a test,

- $p$-value
- significance level
- Type I error, Type II error
- Power of a test

We will introduce them one by one.


:::{admonition,note} Fail to reject $H_0$

Remember that failing to reject a null hypothesis does not necessarily mean that the null hypothesis is true. So we don’t say “accept the null”, instead we say "fail to reject the null" or "$H_0$ is not rejected".

:::


## $p$-value

After we collected data and find the estimate, we want to know whether the estimate prefer $H_0$ or $H_1$. The $p$-value describe that how likely it is to observe more extreme cases than your current estimate, under the null hypothesis.

$$p\text{-value}=P(\text{more extreme cases of your estimate}\,\vert\, H_0)$$

If the $p$-value is small, then it means your current estimate and the more extreme cases are unlikely to be observed under the null. But you do observed it, which implies the null may not hold. Hence, when $p$-value is too small, we reject $H_0$. That is, comparing $p$-value and some threshold can be one of the rejection rules.

The threshold is called significance level, which is the type I error we want to control, and set before the test.


## Type I error and Type II error

### Definitions

**Type I error**, aka **size** of a test, denoted $\alpha$, is defined as

$$\alpha = \mathrm{P}\left( H_0 \text{ is rejected} \,\vert\, H_0 \text{ is true}\right) $$

**Type II error**, $\beta$, is defined as

$$\beta = \mathrm{P}\left( H_0 \text{ is not rejected} \,\vert\, H_0 \text{ is false}\right) $$

**Power** of a test is defined as

$$\begin{aligned}
\text{power}
&= \mathrm{P}\left( H_0 \text{ is rejected} \,\vert\, H_0 \text{ is false}\right)\\
&= 1 - \beta\\
\end{aligned}$$

$\beta$ and power depends on the distribution of the parameter and the rejection rule (e.g., $\left\vert T \right\vert) > t_{df}^{(1-\alpha/2)}$). If $H_0$ is false, the distribution of the test statistic depends on the true parameter value, so do $\beta$ and power. Since the true parameter value is unknown, there is no easy formula for $\beta$ and power.

### Error Control

Both type I error and type II error are important. We want small $\alpha$ and small $\beta$ (large power). Though $\alpha$ can be pre-set, $\beta$ is hard to control. Recall the power is

$$\begin{aligned}
\text{power}
&= \mathrm{P}\left(H_{0} \text { is rejected } \mid H_{1} \text { is true }\right) \\
&= \mathrm{P}\left( T > c\mid H_{1} \text { is true }\right) \\
\end{aligned}$$

Usually the constant $c$ in the rejection rule $T>c$ involves $\alpha$. The distribution of $T$ under $H_1$ depends on the true parameter in $H_1$, and probably sample size $n$. To see their effect, we look at the following examples of $t$-test in simple linear regression.

Example (Linear regression $t$-test with different $H_1$)
: In linear regression, $t$-test can be used to test parameter value. Suppose $H_0: \ \beta_1=0$, we see how the true distribution of the test statistic varies with varying true value of $\beta_1$ and sample size $n$.

  The null distribution $t_{n-2}$ (red curve) does not change with true $\beta_1$, which is unknown. Suppose the type I error is fixed at $\alpha$. Recall that the rejection rule is $\left\vert T \right\vert > t_{n-2}^{{1-\alpha/2}}$, which is roughly $2$. The power is then

  $$\begin{aligned}
  \text{power}
  &= \mathrm{P}\left(H_{0} \text { is rejected } \mid H_{1} \text { is true }\right) \\
  &= \mathrm{P}\left( \left\vert T \right\vert > t_{n-2}^{{1-\alpha/2}}\mid H_{1} \text { is true }\right) \\
  &= \text{yellow area to the right of } 2 \text{ and to the left of } -2
  \end{aligned}$$

  The power increases with larger sample size and farther true $\beta_1$ from null $0$.

  :::{figure} test-power-phase1
  <img src="../imgs/test-power-phase1.png" width = "80%" alt=""/>

  Comparison of null and true distribution in $t$-test [Meyer 2021]
  :::

Example (Linear regression $t$-test with different $\alpha$)
: From the above example we can see power also depends on $\alpha$. We change $\alpha$ and see its effect on power.

  The observation is, **larger $\alpha$ leads to larger power (smaller $\beta$)**. So there is a tradeoff between type I error and type two error.

  :::{figure} test-power-phase2
  <img src="../imgs/test-power-phase2.png" width = "80%" alt=""/>

  How power changes with $\alpha,n$ and $H_1$ in $t$-test [Meyer 2021]
  :::


In practice, to control both types of errors, usually we **fixed** on type I error $\alpha$ (the value is called significance level), and try to minimize type II error $\beta$ by some methods, such as increasing the sample size $n$.



## Confidence Interval

A $(1-\alpha)$-confidence interval is an interval such that when you repeat the experiments many times, there is $(1-\alpha)$ of the times that the estimate falls into the interval.

A $(1-\alpha)$-confidence interval can be constructed with an estimate and its standard error.

$$\hat \theta \pm c_\alpha \cdot \mathrm{se}(\hat \theta)$$

where $c_\alpha$ is a coefficient that depends on $\alpha$ such that the interval cover $(1-\alpha)$ of the cases.

## Credible Interval

In Bayesian statistics, a credible interval is constructed from the posterior distribution of the parameter of interest. The are various methods to choose which $(1-\alpha)$ interval to be the credible interval, e.g. equal tail $\alpha/2$, horizontal cutoff of the PDF, etc.

## Bonferroni Correction
