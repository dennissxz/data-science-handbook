# Hypothesis Testing

*Explain hypothesis testing related concepts*

<!-- TOC -->

- [Hypothesis Testing](#hypothesis-testing)
  - [Hypothesis testing](#hypothesis-testing)
  - [$p$-value](#p-value)
  - [Type I error and Type II error](#type-i-error-and-type-ii-error)
  - [Confidence Interval](#confidence-interval)
  - [Credible Interval](#credible-interval)
  - [Bonferroni Correction](#bonferroni-correction)

<!-- /TOC -->

## Hypothesis testing

When we investigate a question of interest, we first formulate a null hypothesis and an alternative hypothesis, denoted as $H_0$ and $H_1$ respectively. For instance,

$$H_0: \theta = \theta_0\quad \text{vs} \quad H_1: \theta > \theta_0$$

## $p$-value

After we collected data and find the estimate, we want to know whether the estimate prefer $H_0$ or $H_1$. The $p$-value describe that how likely it is to observe more extreme cases than your current estimate, under the null hypothesis.

$$p\text{-value}=P(\text{more extreme cases of your estimate}\,\vert\, H_0)$$

If the $p$-value is small, then it means your current estimate and the more extreme cases are unlikely to be observed under the null. But you do observed it, which implies the null may not hold. Hence, when $p$-value is too small, we reject $H_0$.


## Type I error and Type II error

Type I error ($\alpha$) is defined as

$$\alpha = \mathrm{P}\left( H_0 \text{ is rejected} \,\vert\, H_0 \text{ is true}\right) $$

Type II error ($\beta$) is defined as

$$\beta = \mathrm{P}\left( H_0 \text{ is not rejected} \,\vert\, H_0 \text{ is false}\right) $$

Power of a test is defined as

$$\text{power} = 1 - \beta$$

## Confidence Interval

A $(1-\alpha)$-confidence interval is an interval such that when you repeat the experiments many times, there is $(1-\alpha)$ of the times that the estimate falls into the interval.

A $(1-\alpha)$-confidence interval can be constructed with an estimate and its standard error.

$$\hat \theta \pm c_\alpha \cdot \mathrm{se}(\hat \theta)$$

where $c_\alpha$ is a coefficient that depends on $\alpha$ such that the interval cover $(1-\alpha)$ of the cases.

## Credible Interval

In Bayesian statistics, a credible interval is constructed from the posterior distribution of the parameter of interest. The are various methods to choose which $(1-\alpha)$ interval to be the credible interval, e.g. equal tail $\alpha/2$, horizontal cutoff of the PDF, etc.

## Bonferroni Correction
