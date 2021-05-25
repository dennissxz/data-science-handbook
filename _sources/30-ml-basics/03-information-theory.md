# Information Theory

## Definitions

First we introduce the definitions of some fundamental concepts in informaition theory summarized in the table below.

$$\begin{align}
& \text{Name} & & \text{Definition}  & & \text{Remark}  \\
& \text{Entropy} & \operatorname{H}(Y)&=-\sum_{\mathcal{Y}}p(y)\log p(y) & & \text{Non-negative} \\
& \text{Differntial Entropy} & h(Y)&=-\int_{\mathcal{Y}}f(y)\log f(y)\mathrm{~d}y & & \text{Can be negative} \\
& \text{Joint Entropy} & \operatorname{H}(X,Y)&=-\mathbb{E}_{X,Y\sim p(x,y)}\log\operatorname{P}(X,Y) & & \\
& \text{Conditional Entropy} & \operatorname{H}(Y\vert X)&=-\mathbb{E}_{X,Y\sim p(x,y)}\log \operatorname{P}(Y\mid X) & & \text{Sampling from } p(x,y) \\
& \text{Cross Entropy} & \operatorname{H}(P,Q)&=-\mathbb{E}_{Y\sim P}\log Q(Y) & & \text{Asymmetric} \\
& \text{KL Divergence} & \operatorname{KL}(P,Q)&=\mathbb{E}_{Y\sim P}\log\frac{P(Y)}{Q(Y)} & & \text{Non-negative. Asymmetric} \\
& \text{Mutual Information} & \operatorname{I}(X,Y)&=\operatorname{KL}\left(P_{X,Y},P_{X}P_{Y}\right) && \text{Non-negative. Symmetric}
\end{align}$$

### Shannon Entropy

Definition
: The Shannon Entropy of a discrete distribution $p\left( y \right)$ is defined by


$$\begin{align}
\operatorname{H}\left( Y \right)
&= \mathbb{E}_{Y\sim p(y)}\left[ - \log \operatorname{P}\left( Y \right) \right]\\
&= -\sum_{i=1}^{n}p(y_i)\log p(y_i)
\end{align}$$

This quantity measures the average level of “information”, or “uncertainty” inherent in the variables' possible outcomes.

```{seealso}
Entropy has similarity with variance. Both are non-negative, measure uncertainty/information. But variance depends on the observed value $y$ of the random variable $Y$, while entropy only depends on the probability $p(y)$.
```

Properties
: $\operatorname{H}\left( Y \right) \ge 0$ with equalit iff $p(y)=1$ for some $y$, i.e. no uncertainty.

(differential-entropy)=
### Differential Entropy

Definition
: Let $Y$ be a random variable with probability density function $f$ whose support is $\mathcal{Y}$. The differential entropy of $Y$, denoted $h(Y)$, is defined as

  $$
  h(Y)=-\int_{\mathcal{Y}}f(y)\log f(y)\mathrm{~d}y
  $$


```{note}
Differential Entropy began as an attempt by Shannon to extend the idea of Shannon entropy to continuous PDF. Actually, the correct extension of Shannon entropy to continuous distributions is called limiting density of discrete points (LDDP), while differential entropy, aka continuous entropy, is a limiting case of the LDDP.
```

Properties
: - Can be negative

    Differential entropy does not have all properties of discrete entropy. For instance, discrete entropy is always greater than 0, while differential entropy can be less than 0, or diverges to $-\infty$.

  - Sensitive to units

    The value of differential entropy is sensitive to the choice of units.

  - Transformation

    The entropy of processed information $h\left(z(Y)\right)$ can be larger **or** smaller than the original entropy $h(Y)$. Just consider $z=ay$ for $a>1$ and $a<1$.


Examples
: - Consider a uniform distribution over an interval on the real line of width $\Delta$, then $p(y) = \frac{1}{\Delta}$, and hence the entropy is

    $$\begin{align}
    h(Y) &= \mathbb{E}_{y\sim p(\cdot)} \left[ - \ln \frac{1}{\Delta} \right]\\
    &= \ln \Delta \\
    \end{align}$$

    When $\Delta < 1$, we have $h\rightarrow < 0$.

  - The entropy for a Gaussian variable with density $\mathcal{N}(\mu, \sigma^2)$ is

    $$
    h \left( \mathcal{N}(\mu, \sigma^2) \right) = \ln(\sigma \sqrt{2 \pi e})
    $$

    :::{admonition,dropdown,seealso} *Proof*

    Let $\phi(y)$ be the density of $\mathcal{N} (\mu, \sigma^2)$

    $$\begin{aligned}
    h(Y)
    &= -\int_{\mathcal{Y}}\phi(y)\ln \phi(y)\mathrm{~d}y\\
    &= - \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{(y-\mu)^2}{\sigma^2} \right) \left[ - \ln(\sqrt{2 \pi} \sigma)  -\frac{(y-\mu)^2}{2\sigma^2} \right] \mathrm{~d} y\\
    &=  \ln(\sqrt{2 \pi} \sigma) + \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{(y-\mu)^2}{\sigma^2} \right) \left[\frac{(y-\mu)^2}{2\sigma^2} \right] \mathrm{~d} y\\
    &= \ln(\sqrt{2 \pi} \sigma) +  \mathbb{E} \left[\frac{(y-\mu)^2}{2\sigma^2} \right] \\
    &= \ln(\sqrt{2 \pi} \sigma) +  \frac{\mathbb{E} [X^2] - 2\mu \mathbb{E} [X] + \mu^2  } {2\sigma^2} \\
    &= \ln(\sqrt{2 \pi} \sigma) +  \frac{1}{2}  \\
    &= \ln(\sqrt{2 \pi e} \sigma)\\
    \end{aligned}$$

    :::

    - Obviously, the entropy does not depend on location $\mu$ but only depends on scale $\sigma$.
    - When $\sigma < \frac{1}{\sqrt{2 \pi e}}$, we have $h< 0$.
    - Among all continuous distribution $f(y)$ with mean $\mu$, variance $\sigma^2$, normal distribution has the largest differential entropy, i.e. most randomness.

      $$h(Y) \le \ln(\sigma \sqrt{2 \pi e})\quad \forall f(y): \mathbb{E} [Y] = \mu, \operatorname{Var}\left( Y \right) = \sigma^2$$

      :::{admonition,dropdown,seealso} *Proof*

      Lemma: Let $\phi(y)$ be the density of $\mathcal{N} (\mu, \sigma^2)$, then note that

      $$\begin{aligned}
      -\int_{\mathcal{Y}}f(y)\ln \phi(y)\mathrm{~d}y
      &= - \int f(y) \left[ - \ln(\sqrt{2 \pi} \sigma)  -\frac{(y-\mu)^2}{2\sigma^2} \right] \mathrm{~d} y\\
      &=  \ln(\sqrt{2 \pi} \sigma) + \int f(y) \left[\frac{(y-\mu)^2}{2\sigma^2} \right] \mathrm{~d} y\\
      &= \ln(\sqrt{2 \pi} \sigma) +  \mathbb{E} \left[\frac{(y-\mu)^2}{2\sigma^2} \right] \\
      &= \ln(\sqrt{2 \pi} \sigma) +  \frac{\mathbb{E} [Y^2] - 2\mu \mathbb{E} [Y] + \mu^2  } {2\sigma^2} \\
      &= \ln(\sqrt{2 \pi} \sigma) +  \frac{1}{2}  \\
      &= \ln(\sqrt{2 \pi e} \sigma)\\
      \end{aligned}$$

      Hence,

      $$\begin{aligned}
      h(Y)
      &= - \int f(y) \ln f(y) \mathrm{~d} y\\
      &= - \int f(y) \ln \left( \frac{f(y)}{\phi(y)} \phi(y) \right)  \mathrm{~d} y\\
      &= - \int f(y) \ln \frac{f(y)}{\phi(y)}  \mathrm{~d} y - \int f(y) \ln \phi (y)  \mathrm{~d} y\\
      &= - \mathbb{E} _f\left[- \ln \frac{\phi(y)}{f (y)}  \right] +  \ln(\sqrt{2 \pi e} \sigma) \quad \because \text{Lemma} \\
      &\le \ln \mathbb{E} _f\left[ \frac{\phi(y)}{f (y)}  \right] +  \ln(\sqrt{2 \pi e} \sigma) \quad \because \text{Jensen's inequality}  \\
      &= \ln \int f(y) \frac{\phi(y)}{f(y)}  \mathrm{~d} y+  \ln(\sqrt{2 \pi e} \sigma) \\
      &= \ln(\sqrt{2 \pi e} \sigma)\\
      \end{aligned}$$

      The equality hold if and only if $\frac{\phi(y)}{f(y)}$ is a constant for all $y$, i.e. $Y$ is of normal distribution.

      :::

  - Consider the uniform distribution above. Suppose the unit is meter and the interval is $(0, 1000)$. Note if we change the unit to kilometer, then the interval changes to $(0, 1)$, and the entropy decreases from $\ln 1000$ to $\ln 1$.


    ```{note}
    The right way to think about differential entropy is that, it is actually **infinite**. Recall entropy measures uncertainty. An actual real number carries an infinite number of bits, i.e. infinite amount of information. A meaningful convention is the $h(f)=+\infty$ for any continuous density $p(y)$.
    ```

Definition (NegEntropy)
: Short for Negative Entropy, it is a non-Gaussian-ness measure, a measure of distance to normality. The negEntropy for a random variable $X$ is

  $$
  J(X) = \operatorname{H} (Z) - \operatorname{H} (X)
  $$

  where $\operatorname{H} (Z)$ is the differential entropy of the Gaussian density with the same mean and variance as $X$.

NegEntropy is used for its convenience in computation and approximation. A common approximation (supposedly from Jones 1987)

$$
J(X) \approx \frac{1}{12} \mathbb{E} [X^{3}]^{2}+\frac{1}{48} \kappa(X)^{2}
$$

where $\kappa(X)$ is the excess kurtosis of the distribution of $X$.

### Joint Entropy

Definition
: The joint Shannon entropy of two discrete random variables $X$ and $Y$ with joint distribution $p(x,y)$ is defined as

  $$
  \begin{align}
  \operatorname{H}(X,Y)
  & = \mathbb{E}_{X,Y\sim p(x, y)}\left[ - \log \operatorname{P}(X,Y) \right]\\
  & = - \sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log_{2}[p(x,y)]
  \end{align}
  $$

  More generally, for multiple random variables, we can define

  $$
  \operatorname{H}\left(X_{1},\ldots,X_{n}\right)=-\sum_{x_{1}\in\mathcal{X}_{1}}\ldots\sum_{x_{n}\in\mathcal{X}_{n}}p\left(x_{1},\ldots,x_{n}\right)\log_{2}\left[p\left(x_{1},\ldots,x_{n}\right)\right]
  $$

  Likewise, we can define joint differential entropy for two continuous random variables with joint density $f(x,y)$ as

  $$
  h(X,Y)=-\int_{\mathcal{X},\mathcal{Y}}f(x,y)\log f(x,y) \mathrm{~d}x\mathrm{~d}y
  $$

  and

  $$
  h\left(X_{1},\ldots,X_{n}\right)=-\int f\left(x_{1},\ldots,x_{n}\right)\log f\left(x_{1},\ldots,x_{n}\right)\mathrm{~d}x_{1}\ldots \mathrm{~d}x_{n}
  $$

Properties of Discrete Joint Entropy
: - Nonnegativeity: $\operatorname{H}(X,Y)\ge0$
  - Greater than or equal to individual entropies: $\operatorname{H}(X,Y)\ge\max\left(\operatorname{H}(X),\operatorname{H}(Y)\right)$
  - Less than or equal to the sum of individual entropies $\operatorname{H}(X,Y)\le \operatorname{H}(X)+\operatorname{H}(Y)$. Equality holds iff $X$ and $Y$ are independent.



### Conditional Entropy

Definition
: For two random variables $X, Y$ with density function $p(x,y)$, the conditional entropy of $Y$ given $X$ is defined as


  $$\begin{align}
  \operatorname{H}(Y\mid X)
  & = \mathbb{E}_{X,Y\sim p(x, y)}\left[ - \log \operatorname{P}(Y \mid X) \right]\\
  &=-\sum_{x\in\mathcal{X},y\in\mathcal{Y}}p(x,y)\log p(y\vert x) \\
  h(Y\mid X)	&=-\int_{\mathcal{X},\mathcal{Y}}f(x,y)\log f(y\vert x)\mathrm{~d}x\mathrm{~d}y
  \end{align}$$

  It also quantifies the amount of information needed to describe the outcome of a random variable $Y$ given the value of another random variable $X$.

```{note}
One can compare entropy vs. conditional entropy, and probability vs conditional probability. Both are defined by the expectation of negative log probability.
  - the entropy of $Y$ is the expectation of negative unconditional log probability $\log p(y)$
  - the conditional entropy of $Y$ given $X$ is the expectation of the negative conditional log probability $\log p(y \mid x)$

```


```{warning}
Conditional entropy samples $X,Y$ from the joint distribution $p(x,y)$. This is different from conditional expectation, which samples $Y$ from the conditional distribution $Y \sim p_{Y\mid X}(y\mid x)$.
```

To understand it, first we consider the entropy of $Y$ given $X$ takes a certain value $x$. Recall the unconditional entropy of $Y$ is

$$
\operatorname{H}(Y)=-\sum_{y\in\mathcal{Y}}\operatorname{P}(Y=y)\log_{2}\operatorname{P}(Y=y)
$$

Now $X$ is known to be $x$, then the distribution of $Y$ becomes conditional on $X=x$. We just replace $\operatorname{P}(Y=y)$ by $\operatorname{P}(Y=y\mid X=x)$. Hence,


$$
\operatorname{H}(Y\vert X=x)=-\sum_{y\in\mathcal{Y}}\operatorname{P}(Y=y\mid X=x)\log_{2}\operatorname{P}(Y=y\mid X=x)
$$

Finally, the conditional entropy $\operatorname{H}(Y\vert X)$ is defined as the sum of $\operatorname{H}(Y\vert X=x)$ weighted on $\operatorname{P}(X=x)$, i.e.,

$$
\begin{aligned}\operatorname{H}(Y\vert X) & \equiv\sum_{x\in\mathcal{X}}p(x)\operatorname{H}(Y\vert X=x)\\
 & =-\sum_{x\in\mathcal{X}}p(x)\sum_{y\in\mathcal{Y}}p(y\vert x)\log p(y\vert x)\\
 & =-\sum_{x\in\mathcal{X}}\sum_{y\in\mathcal{Y}}p(x,y)\log p(y\vert x)\\
 & =-\sum_{x\in\mathcal{X},y\in\mathcal{Y}}p(x,y)\log\frac{p(x,y)}{p(x)}
\end{aligned}
$$

Properties
: - $\operatorname{H}(Y\vert X)=0$ iff $Y$ is completely determined by the value of $X$

  - $\operatorname{H}(Y\vert X)=\operatorname{H}(Y)$ iff $Y$ and $X$ are independent


### Cross Entropy


Definition
: The cross entropy of two distributions $P$ and $Q$ on the same
support $\mathcal{Y}$ is defined as

$$
\begin{align}
\operatorname{H}\left( P, Q \right) & =\mathbb{E}_{Y\sim p(y)}\left[-\ln\left(Q(Y)\right)\right]\\
 & =\int_{\mathcal{Y}}f_{P}(y)\left[-\ln\left(f_{Q}(y)\right)\right]\text{d}y\quad\text{for continuous}\ P,Q\\
 & =-\sum_{y\in\mathcal{Y}}P(y)\log Q(y)\qquad\qquad\text{for discrete}\ P,Q
\end{align}
$$

Cross entropy can be interpreted as the **expected message-length** per datum when a wrong distribution $Q$ is assumed while the data actually follows a distribution $P$. If we use $\log_{2}$, the quantity $\operatorname{H}\left( P, Q \right)$ can also be interpreted
as 1.44 times the number of bits used to code draws from $P$ when
using the imperfect code defined by $Q$.

```{note}
- The cross entropy $\operatorname{H}(P,Q)$ of **two** distributions $P(y), Q(y)$ is different from the joint entropy $\operatorname{H}(X,Y)$ of **one** joint distribution $p(x,y)$ of **two** random variables $X, Y$.
- "Cross" means we sample $y$ from one distribution $P(y)$, and then compute the negative log probability with the other distribution $-\ln(Q(y))$.
```


Properties
: - asymmetric: $\operatorname{H}\left( P, Q \right)\ne \operatorname{H}(Q,P)$
  - larger than entropy: $\operatorname{H}\left( P, Q \right)\ge \operatorname{H}(P)$ with equality iff $Q=P$



Cross entropy is tightly related to KL divergence.

(kl-divergence)=
### Kullback-Leibler Divergence

Definition
: The KL Divergence of two distributions $P$ and $Q$ on the \textbf{same}
support $\mathcal{Y}$ is defined as

  $$
  \begin{aligned}
  \operatorname{KL}\left( P, Q \right) & =\mathbb{E}_{Y\sim p(y)}\left[\ln\frac{P(Y)}{Q(Y)}\right]
  \end{aligned}
  $$

Kullback--Leibler divergence (also called relative entropy) is a  measure of how one probability distribution is different from a second, reference probability distribution, i.e. the **distance** between two distributions on the same support.

Properties
: - $\operatorname{KL}\left( P, Q \right) \ge 0$, with equality iff the two distributions are identical.

    :::{admonition,dropdown,seealso} *Proof*

    KL divergence is non-negative by Jensen's inequality of convex functions

    $$
    \begin{aligned}\operatorname{KL}\left( P, Q \right) & =\mathbb{E}_{y\sim p(\cdot)}\left[-\ln\frac{Q(y)}{P(y)}\right]\\
     & \geq-\ln \mathbb{E}_{y\sim p(\cdot)}\frac{Q(y)}{P(y)}\\
     & =-\ln\sum_{y}P(y)\frac{Q(y)}{P(y)}\\
     & =-\ln\sum_{y}Q(y)\\
     & =0
    \end{aligned}
    $$

    For two continuous distributions,

    $$\begin{align}
    \operatorname{KL}\left( P, Q \right) & =\int_{-\infty}^{\infty}p(y)\log\left(\frac{p(y)}{q(y)}\right)\mathrm{~d}y\\
     & \ge0
    \end{align}$$

    Equality holds iff $\forall y\ P(y)=Q(y)$, i.e. two distributions
    are identical.

    :::

  - It is a distribution-wise **asymmetric** measure and thus does not qualify as a statistical **metric** of spread - it also does not satisfy the triangle inequality.

  - Relation to cross entropy:

    $$
    \operatorname{KL}\left( P, Q \right)=\operatorname{H}\left( P, Q \right)-\operatorname{H}(P)
    $$


  - If $P$ is some fixed distribution, then $\operatorname{H}(P)$ is a constant. Hence,

    $$
    \operatorname{argmax}_{Q}\ \operatorname{KL}\left( P, Q \right)=\operatorname{argmax}_{Q}\operatorname{H}\left( P, Q \right)
    $$

    i.e. minimizing/maximizing the KL divergence $\operatorname{KL}\left( P, Q \right)$ of two distributions is equivalent to minimizing/maximizing their cross entropy $\operatorname{H}\left( P, Q \right)$, given $P$ is a fixed distribution.

Example
: The KL-Divergence between two multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}_{p},\boldsymbol{\Sigma}_{p})$
and $\mathcal{N}(\boldsymbol{\mu}_{q},\boldsymbol{\Sigma}\boldsymbol{\Sigma}_{q})$ is

  $$
  \operatorname{KL}\left( P, Q \right)=\frac{1}{2}\left[\log\frac{\left|\boldsymbol{\Sigma}_{q}\right|}{\left|\boldsymbol{\Sigma}_{p}\right|}-k+\left(\boldsymbol{\mu}_{p}-\boldsymbol{\mu}_{q}\right)^{\top}\boldsymbol{\Sigma}_{q}^{-1}\left(\boldsymbol{\mu}_{p}-\boldsymbol{\mu}_{q}\right)+\operatorname{tr} (\boldsymbol{\Sigma}_{q}^{-1}\boldsymbol{\Sigma}_{p}) \right]
  $$

  In particular, if the reference distribution $Q$ is $\mathcal{N}(0,I)$ then we get

  $$
  \operatorname{KL}\left( P, Q \right)=\frac{1}{2}\left[\Vert\boldsymbol{\mu}_{p}\Vert^{2}+\operatorname{tr}(\boldsymbol{\Sigma}_{p}) -k-\log\left|\boldsymbol{\Sigma}_{p}\right|\right]
  $$


(mutual-information)=
### Mutual Information

Aka information gain.

Definition
: Let $\left(X,Y\right)$ be a pair of random variables with values over the space $\mathcal{X}\times\mathcal{Y}$. Suppose their joint distribution is $\operatorname{P}_{X,Y}$ and the marginal distributions are $\operatorname{P}_{X}$ and $\operatorname{P}_{Y}$. The mutual information of $X, Y$ is defined by a KL divergence

  $$\begin{align}
  \operatorname{I}\left(X, Y \right) & = \operatorname{KL}\left(\operatorname{P}_{X,Y},\operatorname{P}_{X}\operatorname{P}_{Y}\right)\\
   & = \mathbb{E}_{X,Y}\left[ \ln\frac{\operatorname{P}_{X,Y}(X,Y)}{\operatorname{P}_X(X)\operatorname{P}_Y(Y)} \right]\\
  \end{align}$$

  For discrete case,

  $$
  \operatorname{I}\left(X, Y \right)=\sum_{y\in\mathcal{Y}}\sum_{x\in\mathcal{X}}p_{X,Y}(x,y)\log\left(\frac{p_{(X,Y)}(x,y)}{p_{X}(x)p_{Y}(y)}\right)
  $$

  and for continuous case,

  $$
  \operatorname{I}\left(X, Y \right)=\int_{\mathcal{Y} }\int_{\mathcal{X}}p_{X,Y}(x,y)\log\left(\frac{p_{(X,Y)}(x,y)}{p_{X}(x)p_{Y}(y)}\right)
  $$

Properties
: - $\operatorname{I}(X,Y) =\operatorname{H}(X)+\operatorname{H}(Y)-\operatorname{H}\left(X, Y \right)$
  - $\operatorname{I}(X,Y) \ge 0$, with equality iff $P_{X,Y}=P_{X}P_{Y}$, i.e. when $X$ and $Y$ are independent, and hence there is no mutual dependence.

Mutual information is a measure of the mutual **dependence** between the two variables. More specifically, it quantifies the amount of information (in units such as shannons, commonly called bits) obtained about one random variable through observing the other random variable.


```{seealso}
Similar to absolute correlation $\left\vert \rho \right\vert$,  both are
- non-negative,
- larger when $X$ and $Y$ are dependent,
- 0 when independent.

But $\vert\rho\vert=0\ \not{\Rightarrow}\ X\perp Y$ while $\operatorname{I}\left(X, Y \right)=0\ \Leftrightarrow\ X\perp Y$.
```

(wasserstein-distance)=
### Wasserstein Distance

#### Definition

KL divergence can measure the 'distance' between two distributions, but it is asymmetric. Wasserstein distance of two distributions, is symmetric, and it is actually a metric.

Definition
: Given two distributions $P$ and $Q$ over a region $D$ in $\mathbb{R} ^{d}$, let $\mathcal{\Pi}$ be all joint distributions $\pi(X, Y)$ over $D \times D$ that have marginals $P$ and $Q$. Then the $p$-Wasserstein distance between $P$ and $Q$ is defined as

$$
W_{p}(P, Q)=\left(\inf _{\pi \in \mathcal{\Pi} (P, Q)} \int\|x-y\|^{p} \pi(x, y) \mathrm{~d}x \mathrm{~d} y \right)^{1 / p}
$$

Specifically, $\mathcal{\Pi} (P, Q)$ represents the collection of all binary functions $\pi$ satisfying
- $\int \pi(x,y) \mathrm{~d}y = P(x)$
- $\int \pi(x,y) \mathrm{~d}x = Q(y)$
- $\pi(x,y) \ge 0$

It can be shown that $W_p$ satisfies all the axioms of a metric.

Usually, $p=1$ is used. $1$-Wasserstein distance is also known as Earth mover's distance (EMD) in computer science. Informally, if the distributions are interpreted as two different ways of piling up a certain amount of dirt over the region $D$, the EMD is the minimum cost of turning one pile into the other.

(wasserstein-dual)=
#### Dual Form

It can be shown that the dual formulation is

$$
W_p ^p(P, Q) =\sup _{\psi, \phi} \int \psi(y)  Q(y) \mathrm{~d}y-\int \phi(x)  P(x) \mathrm{~d}x
$$

where $\psi(y) - \phi(x) \le \left\| x - y \right\| ^p$.

In special case where $p=1$ we have the very simple representation

$$\begin{aligned}
\sup_{f \text{ is 1-Lipschitz} } && \int f (x) P(x) \mathrm{~d} x -  \int f (y) Q(y) \mathrm{~d} y\\
\end{aligned}$$

:::{admonition,dropdown,seealso} *Proof*

When $p=1$, consider the above three constarints, the Lagragean is

$$\begin{aligned}
\mathcal{L} (\pi, f_1, f_2, M)
&= \int \left\| x - y \right\| \pi(x, y) \mathrm{~d}x \mathrm{~d} y   \\
&- \langle f_1, \int \pi(\cdot,y) \mathrm{~d}y - p(\cdot) \rangle \\
&- \langle f_2, \int \pi(x,\cdot) \mathrm{~d}y - p_T(\cdot) \rangle \\
&- \langle M, \pi \rangle \\
\end{aligned}$$

where $f_1 (\cdot), f_2(\cdot)$ and $M(\cdot, \cdot)$ are dual functions, and $\langle \cdot, \cdot \rangle$ is the inner product of two functions: $\langle f, g \rangle = \int f(x)g(x) \mathrm{~d}x$.

Setting the first order derivative to 0 gives

$$
\frac{\partial \mathcal{L}}{\partial \pi} = 0  \Leftrightarrow \left\| x - y \right\| - f_1 (x) - f_2 (y) = M \ge 0
$$

Substituting this equality, the dual problem is then

$$\begin{aligned}
\sup_{f_1, f_2} && \int f_1 (x) P(x) \mathrm{~d} x +  \int f_2 (y) Q(y) \mathrm{~d} y\\
\mathrm{s.t.}
&& \left\| x- y \right\|  \ge f_1(x) + f_2(y)\\
\end{aligned}$$

Lemma: For $W_1$, we have a relation of two maximizers $f_2 = - f_1$.

Hence, the problem becomes.

$$\begin{aligned}
\sup_{f_1} && \int f_1 (x) P(x) \mathrm{~d} x -  \int f_1 (y) Q(y) \mathrm{~d} y\\
\mathrm{s.t.}
&& \left\| x- y \right\|  \ge f_1(x) - f_1(y)\\
\end{aligned}$$

Equivalently,

$$\begin{aligned}
\sup_{f \text{ is 1-Lipshitz} } && \int f (x) P(x) \mathrm{~d} x -  \int f (y) Q(y) \mathrm{~d} y\\
\end{aligned}$$

:::

#### Examples

- In particular, when $d=1$, the distance has a closed form

  $$
  W_{p}(P, Q)=\left(\int_{0}^{1}\left|F^{-1}(z)-G^{-1}(z)\right|^{p}\right)^{1 / p}
  $$

  where $F$ and $G$ are the CDF's of $P$ and $Q$.

  If $P$ is the empirical distribution of a dataset $x_1, x_2, \ldots, x_n$ and $Q$ is the empirical distribution of another dataset $y_1, y_2, \ldots, y_n$ of the same size, then the distance takes a very simple function of the order statistics:

  $$
  W_{p}(P, Q)=\left(\sum_{i=1}^{n}\left\|x_{(i)}-y_{(i)}\right\|^{p}\right)^{1 / p}
  $$

- An interesting special case occurs for normal distributions. If $P=\mathcal{N}\left(\boldsymbol{\mu}_{1}, \boldsymbol{\Sigma}_{1}\right)$ and $Q=\mathcal{N}\left(\boldsymbol{\mu}_{2}, \boldsymbol{\Sigma}_{2}\right)$ then

  $$
  W^{2}(P, Q)=\left\|\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{2}\right\|^{2}+B^{2}\left(\boldsymbol{\Sigma}_{1}, \boldsymbol{\Sigma} _2\right)
  $$

  where

  $$
  B^{2}\left(\boldsymbol{\Sigma}_{1}, \boldsymbol{\Sigma} _2\right)=\operatorname{tr}\left(\boldsymbol{\Sigma}_{1}\right)+\operatorname{tr}\left(\boldsymbol{\Sigma}_{2}\right)-2 \operatorname{tr}\left[\left(\boldsymbol{\Sigma}_{1}^{1 / 2} \boldsymbol{\Sigma}_{2} \boldsymbol{\Sigma}_{1}^{1 / 2}\right)^{1 / 2}\right]
  $$

Reference
- [Larry's notes](https://www.stat.cmu.edu/~larry/=sml/Opt.pdf)

## Identities

### Chain Rule for Conditional Entropy

Analogous to $p(y\vert x)=\frac{p(x,y)}{p(x)}$ in probability, we have an equation that connects entropy, joint entropy and conditional entropy. Instead of division, we use subtraction,

$$
\operatorname{H}(Y\vert X)=\operatorname{H}(X,Y)-\operatorname{H}(X)
$$

:::{admonition,dropdown,seealso} *Proof*

$$
\begin{aligned}\operatorname{H}(Y\vert X)
& = -\sum_{x\in\mathcal{X},y\in\mathcal{Y}}p(x,y)\log\frac{p(x)}{p(x,y)}\\
 & =- \sum_{x\in\mathcal{X},y\in\mathcal{Y}}p(x,y)(\log p(x)-\log p(x,y))\\
 & =-\sum_{x\in\mathcal{X},y\in\mathcal{Y}}p(x,y)\log p(x,y)+\sum_{x\in\mathcal{X},y\in\mathcal{Y}}p(x,y)\log p(x)\\
 & =\operatorname{H}(X,Y)+\sum_{x\in\mathcal{X}}p(x)\log p(x)\\
 & =\operatorname{H}(X,Y)-\operatorname{H}(X)
\end{aligned}
$$

:::

Interpretation

- $\operatorname{H}(X,Y)$ measures the bits of information on average to describe the state of the combined system $(X,Y)$

- $\operatorname{H}(X)$ measures the bits of information we have about $\operatorname{H}(X)$

- $\operatorname{H}(Y\vert X)$ measures the **additional** information required to describe $(X,Y)$, given $X$. Note that $\operatorname{H}(Y\vert X)\le \operatorname{H}(Y)$ with equality iff $X\bot Y$.

The general form for multiple random variables is

$$ \operatorname{H}\left(X_{1},X_{2},\ldots,X_{n}\right)=\sum_{i=1}^{n}\operatorname{H}\left(X_{i}\vert X_{1},\ldots,X_{i-1}\right)
$$

which has a similar form to chain rule in probability theory, except that here is addition $\sum_{i=1}^{n}$ instead of multiplication $\Pi_{i=1}^{n}$.



### Bayes' Rule for Conditional Entropy

Analogous to the Bayes rule $p(y\vert x)=\frac{p(x\vert y)p(y)}{p(x)}$ in probability, we have an equation to link $\operatorname{H}(Y\vert X)$ and $\operatorname{H}(X\vert Y)$, which is simply a result from the chain rule. Instead of division for the Bayes' rule in probability, we use subtraction

$$
\operatorname{H}(Y\vert X)=\operatorname{H}(X\vert Y)-\operatorname{H}(X)+\operatorname{H}(Y)
$$

### Others

$$
\begin{aligned}\operatorname{H}(X,Y) & =\operatorname{H}(X\vert Y)+\operatorname{H}(Y\vert X)+\operatorname{I}\left(X, Y \right)\\
\operatorname{H}(X,Y) & =\operatorname{H}(X)+\operatorname{H}(Y)-\operatorname{I}\left(X, Y \right)\\
\operatorname{I}\left(X, Y \right) & \leq \operatorname{H}(X)
\end{aligned}
$$

## Inequalities

### Data Processing Inequality

The data processing inequality says post-processing cannot increase information.

Let three random variables form the Markov Chain $X\rightarrow Y\rightarrow Z$, implying that the conditional distribution of $Z$ depends only on $Y$ and is conditionally independent of $X$. The joint PMF can be written as

$$p(x,y,z)=p(x)p(y\mid x)p(z\mid y)$$

In this setting, no processing $Z(Y)$ of $Y$, deterministic or random, can increase the information that $Y$ contains about $X$.

$$\operatorname{I}\left(X, Y \right)\ge \operatorname{I} (X,Z)$$

with the quality iff $Z$ and $Y$ contain the same information about $X$.

Note that $\operatorname{H}(Z)\le \operatorname{H}(Y)$ but $h(Z)$ can be larger or smaller than $h(Y)$.
