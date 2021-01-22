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

## Examples

### Gaussian

The log-likelihood function for multivariate Gaussian is


$$
\begin{equation}
\log \mathcal{N}(\mathbf{X} ; \boldsymbol{\mu}, \boldsymbol{\Sigma})=-\frac{n}{2} \log |\boldsymbol{\Sigma}|-\frac{1}{2} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\top} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)+ \text{constant}
\end{equation}
$$

The MLE for $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ are


$$
\begin{equation}
\begin{array}{l}
\widehat{\boldsymbol{\mu}}_{MLE}=\frac{1}{n} \sum_{i=1}^{n} \boldsymbol{x}_{i} \\
\widehat{\boldsymbol{\Sigma}}_{MLE}=\frac{1}{n} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\hat{\boldsymbol{\mu}}_{MLE}\right)\left(\boldsymbol{x}_{i}-\hat{\boldsymbol{\mu}}_{MLE}\right)^{\top}
\end{array}
\end{equation}
$$
