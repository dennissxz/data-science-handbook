# Maximum Likelihood Estimator

We consider a distribution function $p(x; \theta)$ of $x$ parameterized by $\theta$. Our goal is to construct an estimator for the parameter. Maximum likelihood estimator, aka. MLE, as it name suggests, is an estimator for the parameter that is constructed by maximizing the likelihood function.

## Likelihood Function

Definition (Likelihood Function)
: Given $n$ observations $\boldsymbol{x} = (x_1, x_2, \ldots, x_n)$, the likelihood function for parameter $\theta$, denoted $L(\theta ; \boldsymbol{x})$, is defined as

  $$
  L(\theta ; \boldsymbol{x})=p(\boldsymbol{x} ; \theta)=\prod_{x \in \boldsymbol{x}} p(x ; \theta)
  $$

Definition (Maximum Likelihood Estimator)
: The MLE for $\theta$, denoted $\theta _ {MLE}$, is defined as

  $$
  \begin{aligned}
  \theta_{M L} &=\underset{\theta}{\arg \max } L(\theta ; \boldsymbol{x}) \\
  &=\underset{\theta}{\arg \max } \prod_{x \in \mathrm{X}} p(x ; \theta)
  \end{aligned}
  $$


However, it is typically hard to compute the derivative of a product. We instead maximize the log-likelihood.

Definition (Log-likelihood)
: Given $n$ observations $\boldsymbol{x} = (x_1, x_2, \ldots, x_n)$, the log-likelihood function for parameter $\theta$, denoted $\ell(\theta ; \boldsymbol{x})$, is defined as

  $$
  \ell(\theta ; \boldsymbol{x})=\log \left( L(\theta ; \boldsymbol{x}) \right)
  $$

Therefore, the MLE can be defined as

$$
\begin{aligned}
\theta_{MLE} &=\underset{\theta}{\arg \max } \,\ell(\theta ; \boldsymbol{x}) \\
&=\underset{\theta}{\arg \max } \log \left( \prod_{x \in \mathrm{X}} p(x ; \theta) \right)
 \\
&=\underset{\theta}{\arg \max } \sum_{x \in \mathrm{X}} \log p(x ; \theta)
\end{aligned}
$$

Equating the derivative of $\theta$ to zero, we have

$$
\frac{\partial \ell(\theta ; \boldsymbol{x})}{\partial \theta}=\sum_{x \in \boldsymbol{x}} \frac{\partial \log p(x ; \theta)}{\partial \theta} \overset{\text{set}}{=}0
$$

and we can solve for $\theta_{MLE}$.
