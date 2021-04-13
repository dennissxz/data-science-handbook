# Maximum Likelihood Estimator

We consider a distribution function $p(x; \theta)$ of $x$ parameterized by $\theta$. Our goal is to construct an estimator for the parameter. Maximum likelihood estimator, aka. MLE, as it name suggests, is an estimator for the parameter that is constructed by maximizing the likelihood function.


Good example of likelihood principle: 245.ps1.q3

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

## Properties

- Invariance: if MLE of $\theta$ is $\hat{\theta}$, then MLE of $\phi=h(\theta)$ is $\hat{\phi} = h(\hat{\theta})$, provided that $h(\cdot)$ is a one-to-one function.

## Examples

(MLE-Gaussian-derivation)=
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


:::{admonition,dropdown,seealso} *Derivation*

$$

\begin{aligned}
L(\boldsymbol{\mu}, \boldsymbol{\Sigma}) &=f\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{n}\right) \\
&=f\left(\boldsymbol{x}_{1}\right) f\left(\boldsymbol{x}_{2}\right) \cdots f\left(\boldsymbol{x}_{n}\right) \quad \boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{n} \text { are independent } \\
&=\prod_{i=1}^{n} f\left(\boldsymbol{x}_{i}\right) \\
&=\prod_{i=1}^{n} \frac{1}{(2 \pi)^{\frac{p}{2}}|\boldsymbol{\Sigma}|^{\frac{1}{2}}} \exp \left\{-\frac{1}{2}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)\right\} \\
&=\frac{1}{(2 \pi)^{\frac{n p}{2}}|\boldsymbol{\Sigma}|^{\frac{n}{2}}} \exp \left\{-\frac{1}{2} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)\right\}
\end{aligned}

$$

$$
\begin{aligned}
\ell(\boldsymbol{\mu}, \boldsymbol{\Sigma}) &=\log L(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \\
&=-\frac{n p}{2} \log (2 \pi)-\frac{n}{2} \log |\boldsymbol{\Sigma}|-\frac{1}{2} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right) \\
&=-\frac{n p}{2} \log (2 \pi)-\frac{n}{2} \log |\boldsymbol{\Sigma}|-\frac{1}{2} \sum_{i=1}^{n} \operatorname{tr}\left[\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)\right] \\
&=-\frac{n p}{2} \log (2 \pi)-\frac{n}{2} \log |\boldsymbol{\Sigma}|-\frac{1}{2} \operatorname{tr}\left[\boldsymbol{\Sigma}^{-1} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\prime}\right]
\end{aligned}
$$

By the method of completing squares,

$$
\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)\left(\boldsymbol{x}_{i}-\boldsymbol{\mu}\right)^{\prime}=\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_{i}-\overline{\boldsymbol{x}}\right)^{\prime}+n(\overline{\boldsymbol{x}}-\boldsymbol{\mu})(\overline{\boldsymbol{x}}-\boldsymbol{\mu})^{\prime}=\boldsymbol{W}+n(\overline{\boldsymbol{x}}-\boldsymbol{\mu})(\overline{\boldsymbol{x}}-\boldsymbol{\mu}) ^{\top}
$$

We therefore have

$$
\ell(\boldsymbol{\mu}, \mathbf{\Sigma})=-\frac{n p}{2} \log (2 \pi)-\frac{n}{2}\left(\log |\boldsymbol{\Sigma}|+\operatorname{tr}\left(\boldsymbol{\Sigma}^{-1} \frac{\boldsymbol{W}}{n}\right)\right)-\frac{n}{2}(\overline{\boldsymbol{x}}-\boldsymbol{\mu})^{\prime} \boldsymbol{\Sigma}^{-1}(\overline{\boldsymbol{x}}-\boldsymbol{\mu})
$$

Now only the last tern involves $\boldsymbol{\mu}$. Since $(\overline{\boldsymbol{x}}-\boldsymbol{\mu})^{\prime} \boldsymbol{\Sigma}^{-1}(\overline{\boldsymbol{x}}-\boldsymbol{\mu}) \geq 0$ with equality iff $\boldsymbol{\mu} = \bar{\boldsymbol{x}}$, we have $\hat{\boldsymbol{\mu}} = \bar{\boldsymbol{x}}$. Now

$$
\ell(\overline{\boldsymbol{x}}, \mathbf{\Sigma})=-\frac{n p}{2} \log (2 \pi)-\frac{n}{2}\underbrace{\left(\log |\mathbf{\Sigma}|+\operatorname{tr}\left(\boldsymbol{\Sigma}^{-1} \frac{\boldsymbol{W}}{n}\right)\right)}_{g(\boldsymbol{\Sigma})}
$$

Since $\boldsymbol{\Sigma}$ and $\boldsymbol{W} /n$ are p.d., the function $g(\boldsymbol{\Sigma})$ attains its minimum at $\boldsymbol{\Sigma} = \boldsymbol{W} /n$.

:::
