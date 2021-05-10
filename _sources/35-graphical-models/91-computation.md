# Computation Issues

## Problem

Given a probability distribution $p(x_1, \ldots, x_n)$, two natural questions are to
- compute marginal distributions $p(x_i)$ (inference).
- compute mean $\mathbb{E} [\boldsymbol{x}]$.

Both computation involves integration in high dimensions. For arbitrary distribution $p$, theses are difficult. MCMC is an approximation method.

Alternatively, we can convert these integration problem to some optimization problem, and then solve by optimization algorithms. Instead of doing integration where we 'enumerate' every single possible $x$, after converting it to an optimization problem, we can have some efficient greedy or dynamic-programming algorithm.

## Exponential Random Graphs

In exponential random graphs, the joint probability is from [exponential families](one-dim-exponential)

$$
p_\theta (x_1, \ldots, x_n) = \exp \left\{ \boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x} ) - \boldsymbol{b}(\boldsymbol{\theta}) \right\}
$$

where
- $\boldsymbol{\theta}$ are the parameters of the distribution
- $\boldsymbol{\phi} (\boldsymbol{x})$ is a function of $\boldsymbol{x} = (x_1, \ldots, x_n)$. Its entries can be some functions of $(x_1, \ldots, x_n)$, e.g.

  $$
  \boldsymbol{\phi} (\boldsymbol{x} ) = [f_1(x_1), f_{23}(x_2, x_3), \ldots]
  $$

- $b(\boldsymbol{\theta})$ is a normalizing constant/function such that

  $$
  \int p(\boldsymbol{x})\mathrm{~d} \boldsymbol{x}= \frac{1}{\exp(b(\boldsymbol{\theta}))} \int \exp (\boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x} )) \mathrm{~d} \boldsymbol{x} = 1
  $$

  To compute it, we need integration, which is sometimes intractable. The denominator $\exp(b(\boldsymbol{\theta}))$ is also called the partition function.

Examples

- Multivariate Gaussian $\boldsymbol{x} \in \mathbb{R} ^{n}$

  $$
  p _\theta (\boldsymbol{x} ) \propto \exp (- \boldsymbol{x} ^{\top} \Theta \boldsymbol{x})
  $$

- 1-d Bernoulli $x \in \left\{ 0, 1 \right\}$ parameterized by $\mu$.

  $$
  p_{\theta }(x) = \exp ( x\underbrace{\ln \frac{\mu}{1-\mu}}_{\theta}+\ln (1-\mu)) = \frac{1}{\exp (b(\theta ))} \exp (\theta x)
  $$

  - The mean is $\mu_\theta=\frac{e^\theta}{e^\theta + 1}$.
  - The normalizing function is $b(\theta) = -\ln (1-\mu) = \ln ( 1 + e ^\theta)$.

For other distributions, computing mean via computing $b(\boldsymbol{\theta})$ is difficult. Can we turn computing $b(\boldsymbol{\theta})$ to an optimization problem?

## Conjugate Functions

Definition (conjugate function)
: The conjugate function $f^* (\lambda)$ of a function $f(\theta)$ is defined as

  $$
  f^*(\lambda) = \sup_\theta \left\{ \langle \theta, \lambda \rangle - f(\theta) \right\}
  $$

  where $\lambda \in \mathbb{R}$.

Properties
: - $f^*(\lambda)$ is convex.

    :::{admonition,dropdown,seealso} *Proof*

    $$\begin{aligned}
    f^*(t \lambda_1 + (1-t) \lambda _2)
    &= \sup_\theta \left\{ t \langle \theta, \lambda_1 \rangle  - t f(\theta) + (1 - t) \langle \theta, \lambda_2 \rangle - (1-t) f(\theta) \right\}\\
    &\le t \sup_\theta \left\{ \langle \theta, \lambda_1 \rangle - f(\theta) \right\} + (1-t) \sup_\theta \left\{ \langle \theta, \lambda_2 \rangle - f(\theta) \right\}\\
    &= t f ^* (\lambda_1) + (1 - t)f ^* (\lambda_2)\\
    \end{aligned}$$
    :::

  - If $f$ is convex, then $f=(f^*)^*$. In other words,

    $$
    f(\theta) = \sup_\lambda \left\{ \langle \theta, \lambda \rangle - f^* (\lambda)\right\}
    $$

It can be shown that the normalizing function $b(\boldsymbol{\theta})$ is convex. Hence,

$$
b(\boldsymbol{\theta}) = \sup_{\boldsymbol{\lambda}} \left\{ \boldsymbol{\theta} , \boldsymbol{\lambda} \rangle - b^* (\boldsymbol{\lambda})\right\}
$$

Thus, we turn computing $b(\boldsymbol{\theta})$ to an optimization problem over $\boldsymbol{\lambda}$.

:::{admonition,note} Example: Bernoulli

In Bernoulli case, the normalizing function is $b(\theta) = \ln ( 1 + e ^\theta)$, then its conjugate function is

$$
b^* (\lambda) = \sup_\theta \left\{ \lambda \theta - \ln (1 + e^\theta) \right\}
$$

Taking derivative and set to 0 gives

$$
\frac{\partial b^* (\lambda)}{\partial \theta} = 0 \Rightarrow \lambda = \frac{e^\theta}{ 1 + e ^\theta}  \quad \text{or} \quad  \theta^+ = \ln \frac{\lambda}{1 - \lambda}  
$$

Substituting the optimizer $\theta^+$ gives the conjugate function

  $$
  b^* (\lambda) = \lambda \log \lambda + (1 - \lambda) \log (1 - \lambda),  \quad\text { for } \lambda \in (0, 1)
  $$


On the other hand, since $b(\theta)$ is convex, we have $b = (b^*)^*$, i.e.

$$
b(\theta) = \max_{\lambda \in (0, 1)} \langle \lambda \theta - b^* (\lambda) \rangle
$$

Taking derivative and set to 0 gives

$$\theta = \nabla_\lambda b^* (\lambda) = \ln \frac{\lambda}{1 - \lambda} \quad \text{or} \quad \lambda^+ = \frac{e^\theta}{ 1 + e ^\theta}  $$

Substituting the optimizer gives $b(\theta) = \ln ( 1 + e^\theta)$.

The above prove-by-example process show how to compute $b(\theta)$ as an optimization problem. We can also find that

- The optimizer $\lambda ^+= \frac{e ^\theta}{1 + e^\theta}$ is the mean of the Bernoulli distribution
- The conjugate function $b^*(\lambda)$ is the negative entropy of the Bernoulli distribution

  $$b^*(\lambda) = -\operatorname{H} (p_\lambda) = - \sum_x p_\lambda(x) \ln p_\lambda(x)$$

In general, these two statements always hold.

:::

In general,

- The optimizer $\lambda ^+$ is the mean of the distribution

  $$
  \lambda^+ = \mathbb{E} [\boldsymbol{x}]
  $$

- The conjugate function $b^*(\lambda)$ is the negative entropy of the distribution

  $$b^*(\lambda) = -\operatorname{H} (p_\lambda)$$
