# Estimators Evaluation

There are many ways to evaluate the performance of an estimator. The three commonly used metrics are

- Unbiasedness

- Efficiency

- Consistency


## Consistency

Definition (Consistent estimator)

: We say $\hat{\theta}$ is a consistent estimator of $\theta$ if for every $\varepsilon > 0$, as $n\rightarrow \infty$.


  $$
  \operatorname{\mathbb{P}}\left(\lim_{n \rightarrow \infty} \left\vert \hat{\theta}_n - \theta \right\vert > \varepsilon \right) \rightarrow 0
  $$

  or equivalently,

  $$
  \hat{\theta}_n  \overset{\mathcal{P}}{\longrightarrow} \theta
  $$

It can be interpreted as the distribution of the estimator $\hat{\theta}$ collapses to the true parameter value $\theta$.



:::{admonition,note} Comparison of unbiasedness and consistent estimators.

Note that unbiased estimators aren't necessarily consistent. For instance,

- an estimator that always use the first fixed $m$ observations, or
- an estimator of $\mu=0.5$ in $\mathcal{U}(0,1)$ that only takes $0$ or $1$ value.

If the variance of an unbiased estimator shrinks to 0 as $n\rightarrow \infty$, then it is consistent.

Vice versa, consistent estimators are not necessarily unbiased, like many maximum likelihood estimators.

:::
