# Estimators Evaluation


## Consistency

Definition (Consistent estimator)

: We say $\hat{\theta}$ is a consistent estimator of $\theta$ if for every $\varepsilon > 0$, as $n\rightarrow \infty$.


  $$
  \operatorname{P}\left(\lim_{n \rightarrow \infty} \left\vert \hat{\theta}_n - \theta \right\vert > \varepsilon \right) \rightarrow 0
  $$

  or equivalently,

  $$
  \hat{\theta}_n  \stackrel{P}{\rightarrow} \theta
  $$

It can be interpreted as the distribution of the estimator $\hat{\theta}$ collapses to the true parameter value $\theta$.

Note that unbiased estimators aren't necessarily consistent. For instance,

- an estimator that always use the first fixed $m$ observations, or
- an estimator of $\mu=0.5$ in $U(0,1)$ that only takes $0$ or $1$ value.

If the variance of an unbiased estimator shrinks to 0 as $n\righthand \infty$, then it is consistent.

Vice versa, consistent estimators are not necessarily unbiased, like many maximum likelihood estimators.
