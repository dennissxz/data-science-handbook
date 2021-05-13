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
- $\boldsymbol{\theta}$ are the parameters of the distribution, in parameter space $\Omega$
- $\boldsymbol{\phi} (\boldsymbol{x})$ is a function of $\boldsymbol{x} = (x_1, \ldots, x_n)$. Its entries can be some functions of $(x_1, \ldots, x_n)$, e.g.

  $$
  \boldsymbol{\phi} (\boldsymbol{x} ) = [f_1(x_1), f_{23}(x_2, x_3), \ldots]
  $$

  usually $f$ corresponds to some objects in graph, e.g. $f_1$ for node-level, $f_{23}$ for edge-level, and so on.

- $b(\boldsymbol{\theta})$ is a normalizing constant/function such that

  $$
  \int p_{\theta}(\boldsymbol{x})\mathrm{~d} \boldsymbol{x}= \frac{1}{\exp(b(\boldsymbol{\theta}))} \int \exp (\boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x} )) \mathrm{~d} \boldsymbol{x} = 1
  $$

  The denominator $\exp(b(\boldsymbol{\theta}))$ is also called the partition function, and $b(\boldsymbol{\theta})$ is often called the log-partition function.

  To compute $b(\boldsymbol{\theta})$, we need integration, which is sometimes intractable.


Equivalently, we can write the distribution $p_\theta(\boldsymbol{x})$ as

$$p_{\theta}(\boldsymbol{x}) = \frac{ \exp (\boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x} )) }{\int\exp (\boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x} ))\mathrm{~d} \boldsymbol{x}} $$



Examples

- Multivariate Gaussian $\boldsymbol{x} \in \mathbb{R} ^{n}$

  $$
  p _\theta (\boldsymbol{x} ) \propto \exp (- \boldsymbol{x} ^{\top} \Theta \boldsymbol{x})
  $$

- 1-d Bernoulli $x \in \left\{ 0, 1 \right\}$ parameterized by $\lambda$.

  $$
  p_{\theta }(x) = \exp ( x\underbrace{\ln \frac{\lambda}{1-\lambda}}_{\theta}+\ln (1-\lambda)) = \frac{1}{\exp (b(\theta ))} \exp (\theta x)
  $$

  - The mean is $\lambda_\theta=\frac{e^\theta}{e^\theta + 1}$.
  - The normalizing function is $b(\theta) = -\ln (1-\lambda) = \ln ( 1 + e ^\theta)$.

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
b^* (\lambda) = \max_{\theta \in \Omega} \left\{ \lambda \theta - \ln (1 + e^\theta) \right\}
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


:::{admonition,dropdown,seealso} *Proof*

Recall the definition

$$
b^* (\boldsymbol{\lambda}) = \max_{\boldsymbol{\theta} \in \Omega} \left\{ \boldsymbol{\lambda} \boldsymbol{\theta}  - b(\boldsymbol{\theta}) \right\}
$$

Hence setting the first order derivative to 0 gives

$$\begin{aligned}
\boldsymbol{\lambda}
&= \nabla _\theta b(\boldsymbol{\theta})\\
&= \nabla _\theta \ln \int \exp( \boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x})) \mathrm{~d} \boldsymbol{x}\\
&= \frac{\int \boldsymbol{\phi} (\boldsymbol{x} ) \exp( \boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x}))) \mathrm{~d} \boldsymbol{x}}{\int \exp( \boldsymbol{\theta} ^{\top} \boldsymbol{\phi} (\boldsymbol{x}))) \mathrm{~d} \boldsymbol{x}} \\
&= \int \boldsymbol{\phi} (\boldsymbol{x}) p_\theta(\boldsymbol{x}) \mathrm{~d} \boldsymbol{x} \\
&= \mathbb{E}_{p_\theta} [\boldsymbol{\phi} (\boldsymbol{x} )]
\end{aligned}$$

Hence $\boldsymbol{\lambda}$ is the mean of $\boldsymbol{\phi} (\boldsymbol{x})$ w.r.t. distribution $\boldsymbol{x} \sim p_{\boldsymbol{\theta}}$. This equation gives a relation between $\boldsymbol{\lambda}$ and $\boldsymbol{\theta}$.

Suppose (and indeed) this relation is one-one relation, then we can solve for $\boldsymbol{\theta} = \boldsymbol{\theta} (\boldsymbol{\lambda})$. Then the maximum is


$$\begin{aligned}
b^*(\boldsymbol{\lambda})
&= \langle \boldsymbol{\lambda}, \boldsymbol{\theta} (\boldsymbol{\lambda} )  \rangle - b(\boldsymbol{\theta} (\boldsymbol{\lambda} ))\\
&= \langle \mathbb{E}_{p_\theta} [\boldsymbol{\phi} (\boldsymbol{x} )], \boldsymbol{\theta} (\boldsymbol{\lambda} )  \rangle - b(\boldsymbol{\theta} (\boldsymbol{\lambda} ))\\
&= \mathbb{E}_{p_\theta} [ \langle  \boldsymbol{\phi} (\boldsymbol{x} ), \boldsymbol{\theta} (\boldsymbol{\lambda} )  \rangle - b(\boldsymbol{\theta} (\boldsymbol{\lambda} ))]\\
&= \mathbb{E}_{p_\theta} [ \ln p_{\boldsymbol{\theta} (\boldsymbol{\lambda})}]\\
&= - \operatorname{H} (p_{\boldsymbol{\theta} (\boldsymbol{\lambda})}) \\
\end{aligned}$$

which is negative likelihood.


:::


Then substituting this result to the optimization problem of $b(\boldsymbol{\theta})$

$$
b(\boldsymbol{\theta}) = \sup_{\boldsymbol{\lambda}} \left\{\langle \boldsymbol{\theta} , \boldsymbol{\lambda} \rangle - b^* (\boldsymbol{\lambda})\right\} = \sup_{\boldsymbol{\lambda}} \left\{ \langle\boldsymbol{\theta} , \boldsymbol{\lambda} \rangle + \operatorname{H} (p_{\boldsymbol{\theta} (\boldsymbol{\lambda})}) \right\}
$$

Claim:

$$
b(\boldsymbol{\theta}) = \sup_{p \in \mathcal{P}} \left\{ \langle \boldsymbol{\theta} , \boldsymbol{\lambda}_p \rangle + \operatorname{H} (p) \right\}
$$

where $\mathcal{P}$ is the set of $n$-dim distribution and $\boldsymbol{\lambda} = \mathbb{E}_p [\boldsymbol{\phi} (\boldsymbol{\theta})]$.

To claims to justify.

- $\boldsymbol{\lambda} = \mathbb{E} _{\boldsymbol{\theta}}[\boldsymbol{\phi} (\boldsymbol{x} )]$ defines a 1-1 map between $\boldsymbol{\theta}$ and $\boldsymbol{\lambda}$ $\Leftrightarrow$ Exponential family is minimal
- The image of $\boldsymbol{\lambda} = \mathbb{E} _{p _\theta}[\boldsymbol{\phi} (\boldsymbol{x})]$ for $p_{\theta} = e$ is the interior of $M:=\left\{ \boldsymbol{\lambda} \mid \boldsymbol{\lambda} = \mathbb{E} _p[ \boldsymbol{\phi} (\boldsymbol{x} )], p \in \mathcal{P}    \right\}$

Proposition :

$$
b(\boldsymbol{\theta}) \ge \langle \boldsymbol{\theta} , \boldsymbol{\lambda} \rangle - b^*(\boldsymbol{\lambda})
$$

for all $\boldsymbol{\lambda} \in \operatorname{Int}(M)$, with equality when $\boldsymbol{\lambda} = \mathbb{E}_{\theta} [\boldsymbol{\phi} (\boldsymbol{x} )]$

Key: when $f$ is strictly convex, then $y = \partial f(x) \Leftrightarrow x = \partial f ^* (y)$. Thus,

$$\boldsymbol{\lambda} = \nabla b(\boldsymbol{\theta}) \Leftrightarrow \boldsymbol{\theta} = \nabla b^* (\boldsymbol{\lambda})$$

Any mean parameter $\mathbb{E}_{p} [\boldsymbol{\phi} (\boldsymbol{x} )]$ for $p \in \mathcal{P}$ gives a lower-bound of $b(\boldsymbol{\theta})$. But $\mathcal{P}$ has exponential size. Can we find a smaller set $\tilde{\mathcal{P}} \subset \mathcal{P}$?

Meanfield approximation

Assume mutual independence among the variables, then

$$
p(x_1, \ldots, x_n) = p_1 (x_1) p_2 (x_2) \cdots p_n (x_n)
$$

Thus,


$$\begin{aligned}
- \operatorname{H}(p)
&= \int \log \left\{ p_1 (x_1) p_2 (x_2) \cdots p_n (x_n) \right\} p_1 (x_1) p_2 (x_2) \cdots p_n (x_n) \mathrm{~d} \boldsymbol{x} \\
&= \sum_i \int p_i (x_i) \log p_i (x_i) \mathrm{~d} x_i\\
&= - \sum_i \operatorname{H} (p_i)\\
\end{aligned}$$

Consider an Ising model

$$
\langle \boldsymbol{\theta} , \boldsymbol{\phi} (\boldsymbol{x}) \rangle = \sum_i \theta_i x_i + \sum_{i,j} \theta_{ij} x_i x_j
$$

where $x_i \in \left\{ 0,1 \right\}$. We have

$$
\mathbb{E}_p [X_i] = \mathbb{E}_{p_i} [X_i]  = \mu_i
$$

Hence

$$
b(\boldsymbol{\theta} ) = \max_{\mu_i \in [0,1]^n}  \sum_i \theta_i \mu_i + \sum_{(i,j) \in E} \theta_{ij} \mu_i \mu_j + \sum_i \operatorname{H}(p_i)  
$$

where $\sum_i \operatorname{H} (p_i) = \sum \mu _i \log \mu _i + (1 - \mu_i) \log (1- \mu _i)$.

Let $AMF$ be the objective function, then

$$
\frac{\partial AMF}{\mu_i} = 0 \Leftrightarrow \mu_i = \sigma \left( \theta_i + \sum_{j \in \mathscr{N} _i} \theta_{ij} \mu_j \right)
$$

where $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is the sigmoid function.

Repeat until convergence.
