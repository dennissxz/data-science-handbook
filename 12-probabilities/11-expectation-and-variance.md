# Expectation and Variance

## Definitions

We quickly review the definitions of expectation, variance and covariance.


$$\begin{align}
&&&\text{Population} && \text{Sample} \\
& \text{Mean} & \mu &= \sum_{i=1}^n x_i p(x_i) \text{ or } \int_{\mathcal{X}} x f(x) \mathrm{~d}x &  \bar x &= \frac{1}{n}\sum_i x_i  \\
& \text{Variance} & \sigma^2 &= \operatorname{\mathbb{E}}\left[ \left( X-\mu \right)^2 \right]  & s^2 &= \frac{1}{n}\sum_i(x_i - \bar x)^2\\
& \text{Standard deviation}  & \sigma &= \sqrt{\operatorname{\mathbb{E}}\left[ \left( X-\mu \right)^2 \right]}  & s &= \sqrt{\frac{1}{n}\sum_i(x_i - \bar x)^2} \\
& \text{Covariance}  & \sigma_{X,Y} &= \operatorname{\mathbb{E}}\left[ (X-\mu_X)(Y-\mu_Y) \right] & s_{X,Y} &= \frac{1}{n}\sum_i \left[ (x_i - \bar x)(y_i - \bar y) \right]
\end{align}$$


Also recall the definitions of conditional expectation and conditional variance:

$$\begin{align}
 \operatorname{\mathbb{E}}(X \mid Y=y)
 &= \sum_{x} x P(X=x \mid Y=y) \\
 &\text{or} \int_{-\infty}^{\infty} x f_{X \mid Y}(x, y) \mathrm{~d} x \\
\operatorname{Var}\left( X \mid Y=y \right)
 &= \operatorname{\mathbb{E}}\left[ (X-\mu_{X\mid Y=y})^{2} \mid Y=y \right] \\
\end{align}$$


:::{admonition,note} Notations
- The notation $X \mid Y=y$ means that $Y=y$ is observed. In this case, the conditional expectation (variance) is a function of the observed value $y$, i.e., $\operatorname{\mathbb{E}}(X \mid Y=y) = g(y)$, which itself is a constant.
- The notation $X \mid Y$ means that $Y$ is a random variable and has not been observed yet. In this case, the conditional expectation (variance) is a function of the random variable $Y$, i.e., $\operatorname{\mathbb{E}}(X \mid Y) = g(Y)$, which itself is a random variable.
:::


## Identities

### Basics

In general, we have

$$\begin{align}
\operatorname{\mathbb{E}}\left( aX + bY \right) &= a \operatorname{\mathbb{E}}\left( X \right) + b \operatorname{\mathbb{E}}\left( Y \right) \\
\operatorname{Var}\left( aX + bY \right) &= a^2\operatorname{Var}\left( X \right) + b^2\operatorname{Var}\left( Y \right) + 2ab\operatorname{Cov}\left( X, Y \right) \\
\operatorname{Var}\left( X \right) &= \operatorname{\mathbb{E}}\left( X^2 \right) - \left[ \operatorname{\mathbb{E}}\left( X \right) \right]^2\\
\operatorname{\mathbb{E}}\left( X^2 \right) &= \mu^2 + \sigma^2 \\
\operatorname{Cov}\left( X, X \right) &= \operatorname{Var}\left( X \right) \\
\operatorname{Cov}\left( X,Y \right) &= \operatorname{\mathbb{E}}\left( XY \right) - \operatorname{\mathbb{E}}\left( X \right)\operatorname{\mathbb{E}}\left( Y \right) \\
\operatorname{Cov}\left( X, a \right) &= 0 \\
\operatorname{Cov}\left( X, Y+Z \right) &= \operatorname{Cov}\left( X, Y \right) + \operatorname{Cov}\left( X, Z \right) \\
\operatorname{Cov}\left( aX, bY \right) &= ab \operatorname{Cov}\left( X, Y \right)
\end{align}$$

```{margin}
Be careful about the notations $\sigma_X ^2$ and $\sigma_{X,X}$

$$
\sigma_X^2 = \operatorname{Var}\left( X \right) = \operatorname{Cov}\left( X, X \right) = \sigma_{X,X}
```



If $X$ and $Y$ are independent,

$$\begin{align}
\operatorname{\mathbb{E}}\left( XY \right) &= \operatorname{\mathbb{E}}\left( X \right)\operatorname{\mathbb{E}}\left( Y \right) \\
\operatorname{Cov}\left( X, Y \right) &= 0 \\
\operatorname{Var}\left( aX + bY \right) &= a^2\operatorname{Var}\left( X \right) + b^2\operatorname{Var}\left( Y \right) \\
\end{align}$$

### Linear Combinations


For $n$ random variables $X_1, X_2, \ldots, X_n$, consider a linear combination $\sum_i^n a_i X_i$. Though we have no information about the dependence between $X_i$'s, the expectation of the sum equals to the sum of the expectations

$$
\operatorname{\mathbb{E}}\left( \sum_i^n a_i X_i \right)
 = \sum_i^n \operatorname{\mathbb{E}}\left(a_i X_i \right)
 = \sum_i^n a_i\operatorname{\mathbb{E}}\left( X_i \right)
$$



In this sense, expectation is a linear operator.


In particular, for independently and identically distributed $X_1, X_2, \ldots, X_n$ with common mean $\operatorname{\mathbb{E}}\left( X_i \right)=\mu$, the expectation of the average value is

$$\begin{align}
\operatorname{\mathbb{E}}\left( \bar X \right) &= \operatorname{\mathbb{E}}\left( \frac{1}{n}\sum_{i} X_i \right)\\
&= \frac{1}{n}\sum_i \operatorname{\mathbb{E}}\left( X_i \right)\\
&= \mu \\
\end{align}$$


In general, the variance of a sum of a linear combination is

$$
\begin{aligned}
\operatorname{Var}\left(\sum_{i=1}^{n} a_{i} X_{i}\right)
&= \operatorname{Cov}\left( \sum_i a_i X_i, \sum_i a_i X_i \right)\\
&=\sum_{i, j=1}^{n}   \operatorname{Cov}\left(a_{i}X_{i}, a_{j}X_{j}\right) \\
&=\sum_{i=1}^{n}  \operatorname{Var}\left(a_{i}X_{i}\right)+\sum_{i \neq j}   \operatorname{Cov}\left(a_{i}X_{i}, a_{j}X_{j}\right) \\
&=\sum_{i=1}^{n} a_{i}^{2} \operatorname{Var}\left(X_{i}\right)+2 \sum_{1 \leq i<j \leq n} a_{i} a_{j} \operatorname{Cov}\left(X_{i}, X_{j}\right)
\end{aligned}
$$


:::{admonition,tip} Tip
One can imagine that there is a $n \times n$ covariance table with the $i,j$-th entry being $\operatorname{Cov}\left( a_i X_i, a_j X_j \right)$, and the required variance is the sum of all the entries, which consists of
- the sum of the diagonal entries as $\sum_i\operatorname{Var}\left(a_{i}X_{i}\right)$
- the sum of the off-diagonal entries as $\sum_{i\ne j}\operatorname{Cov}\left(a_{i}X_{i}, a_{j}X_{j}\right)$
:::


In particular, the variance of the average value of the IID sum is

$$\begin{align}
\operatorname{Var}\left( \bar X \right) &= \operatorname{Var}\left( \frac{1}{n}\sum_{i} X_i \right)\\
&= \frac{1}{n^2}\sum_i \operatorname{Var}\left( X_i \right)\\
&= \frac{1}{n} \sigma^2 \\
\end{align}$$



### Expectation of Nonnegative Random Variables

For nonnegative random variables, the expectation can be computed from the complementary cumulative distribution function $1 - F(x) = \operatorname{\mathbb{P}}\left( X > x \right)$.

- discrete case

$$
\operatorname{\mathbb{E}}\left( X \right)=\sum_{n=0}^{\infty} \operatorname{\mathbb{P}}\left( X>n \right)
$$

- continuous case

$$
\operatorname{\mathbb{E}}\left( X \right) = \int_{0}^{\infty} \operatorname{\mathbb{P}}(X \geq x) \mathrm{~d} x
$$

:::{admonition,dropdown,seealso} *Proof by changing the order of summation/integral*

- discrete case

  $$\begin{align}
  \sum_{n=0}^{\infty} \operatorname{\mathbb{P}}\left( X>n \right)
  &= \sum_{n=0}^{\infty} \sum_{k=n+1}^{\infty} \operatorname{\mathbb{P}}\left( X=k \right) \\
  &= \sum_{k=1}^{\infty} \sum_{n=1}^k \operatorname{\mathbb{P}}\left( X=k \right) \\
  &= \sum_{k=1}^{\infty} k \operatorname{\mathbb{P}}\left( X=k \right) \\
  &= \sum_{k=0}^{\infty} k \operatorname{\mathbb{P}}\left( X=k \right) \\
  &= \operatorname{\mathbb{E}}\left( X \right)
  \end{align}$$

- continuous case

  $$
  \begin{aligned}
  \int_{0}^{\infty} \operatorname{\mathbb{P}}(X \ge x) \mathrm{~d} x
  &=\int_{0}^{\infty} \int_{x}^{\infty} f_{X}(y) \mathrm{~d} y \mathrm{~d} x \\
  &=\int_{0}^{\infty} \int_{0}^{y} f_{X}(y) \mathrm{~d} x \mathrm{~d} y \\
  &=\int_{0}^{\infty} f_{X}(y) \int_{0}^{y} 1 \mathrm{~d} x \mathrm{~d} y \\
  &=\int_{0}^{\infty} y f_{X}(y) \mathrm{~d} y \\
  &=\operatorname{\mathbb{E}}\left( X \right)
  \end{aligned}
  $$

  $\square$

:::


### Law of Total Expectation

Aka law of iterated expectations, tower rule, smoothing theorem.

Given the conditional expectation $\operatorname{\mathbb{E}}\left( X \mid Y \right)$, the law of total expectation states that we can obtained the unconditional expectation $\operatorname{\mathbb{E}}\left( X \right)$ by

$$
\operatorname{\mathbb{E}}\left( X \right)=\operatorname{\mathbb{E}}\left[ \operatorname{\mathbb{E}}(X \mid Y) \right]
$$


:::{admonition,note} Note
The inside expectation is taken w.r.t. $X$ and the outside expectation is taken w.r.t. $Y$, since the conditional expectation $\operatorname{\mathbb{E}}\left(X \mid Y \right)$ is a function $g(Y)$ that depends on the random variables $Y$. To emphasize this we can write

$$
\operatorname{\mathbb{E}}_X\left( X \right)=\operatorname{\mathbb{E}}_Y\left[ \operatorname{\mathbb{E}}_X(X \mid Y) \right]
$$
:::

In general, we can partition the sample space into finite or countably infinite sets $A_i$, then

$$
\operatorname{\mathbb{E}}(X)=\sum_{i} \operatorname{\mathbb{E}}\left(X \mid A_{i}\right) \operatorname{\mathbb{P}}\left(A_{i}\right)
$$

For instance, we can compute the expectation as a weighted sum of the expectation of the positive part and the expectation of the negative part on respective probabilities.

$$
\operatorname{\mathbb{E}}(X)=\operatorname{\mathbb{E}}\left(X \mid X>0\right) \operatorname{\mathbb{P}}\left(X>0\right) + \operatorname{\mathbb{E}}\left(X \mid X<0\right) \operatorname{\mathbb{P}}\left(X<0\right)
$$

:::{admonition,dropdown,seealso} *Proof*

By definition

$$
\begin{aligned}
\operatorname{\mathbb{E}}\left( \operatorname{\mathbb{E}}\left( X \mid Y \right) \right) &=\operatorname{\mathbb{E}}\left[\sum_{x} x \cdot \operatorname{\mathbb{P}}(X=x \mid Y)\right] \\
&=\sum_{y}\left[\sum_{x} x \cdot \operatorname{\mathbb{P}}(X=x \mid Y=y)\right] \cdot \operatorname{\mathbb{P}}(Y=y) \\
&=\sum_{y} \sum_{x} x \cdot \operatorname{\mathbb{P}}(X=x, Y=y) \\
&=\sum_{x} x \sum_{y} \operatorname{\mathbb{P}}(X=x, Y=y) \\
&=\sum_{x} x \cdot \operatorname{\mathbb{P}}(X=x) \\
&=\operatorname{\mathbb{E}}(X)
\end{aligned}
$$

$\square$

:::


### Law of Total Variance

Aka law of iterated variances, variance decomposition formula, Eve's law.

$$
\operatorname{Var}(X)=\operatorname{\mathbb{E}}[\operatorname{Var}(X \mid Y)]+\operatorname{Var}(\operatorname{\mathbb{E}}[X \mid Y])
$$


:::{admonition,note} Note
Here both $\operatorname{Var}\left( X \mid Y \right)$ and $\operatorname{\mathbb{E}}\left( X \mid Y \right)$ are random. The outside expectation and variance are taken w.r.t. the conditioned variable, $Y$.
:::


The first and the second term can be interpreted as the unexplained and the explained components of the variance of $X$ by knowing $Y$. Imagine that there is a deterministic relation $X=f(Y)$, then $\operatorname{Var}\left( X \mid Y \right) = 0$ so that the first term is 0, and the second term becomes $\operatorname{Var}\left(  f(Y) \right) = \operatorname{Var}\left( X \right)$.

:::{admonition,dropdown,seealso} *Proof*

Note that the relation $\operatorname{Var}\left( X \right) = \operatorname{\mathbb{E}}\left( X^2 \right) - \left[ \operatorname{\mathbb{E}}\left( X \right) \right]^2$ holds in a similar fashion when conditioning on $Y$

$$
\operatorname{Var}\left( X \mid Y \right) = \operatorname{\mathbb{E}}\left( X^2 \mid Y \right) - \left[ \operatorname{\mathbb{E}}\left( X \mid Y\right) \right]^2
$$

By the law of total expectation, we can compute $\operatorname{\mathbb{E}}\left( X^2 \right)$ by

$$
\begin{align}
\operatorname{\mathbb{E}}\left( X^2 \right)
&= \operatorname{\mathbb{E}}\left[ \operatorname{\mathbb{E}}\left( X^2 \mid Y \right) \right] \\
&=  \operatorname{\mathbb{E}}\left\{ \operatorname{Var}\left( X \mid Y \right) + \left[ \operatorname{\mathbb{E}}\left( X \mid Y\right) \right]^2  \right\}
\end{align}
$$

and compute $\operatorname{\mathbb{E}}\left( X \right)$ by $\operatorname{\mathbb{E}}\left[ \operatorname{\mathbb{E}}\left( X\mid Y \right) \right]$. Hence,

$$\begin{align}
\operatorname{Var}\left( X \right) &= \operatorname{\mathbb{E}}\left( X^2 \right) - \left[ \operatorname{\mathbb{E}}\left( X \right) \right]^2\\
&=  \operatorname{\mathbb{E}}\left\{ \operatorname{Var}\left( X \mid Y \right) + \left[ \operatorname{\mathbb{E}}\left( X \mid Y\right) \right]^2  \right\} - \left\{ \operatorname{\mathbb{E}}\left[ \operatorname{\mathbb{E}}\left( X\mid Y \right) \right] \right\}^2\\
&= \operatorname{\mathbb{E}}[\operatorname{Var}(X \mid Y)]+\operatorname{Var}(\operatorname{\mathbb{E}}[X \mid Y])
\end{align}$$

$\square$

:::



:::{admonition,warning} Warning
From above we see that the identity that holds for expectation

$$
\operatorname{\mathbb{E}}(X)=\sum_{i} \operatorname{\mathbb{E}}\left(X \mid A_{i}\right) \operatorname{\mathbb{P}}\left(A_{i}\right)
$$

does **not** hold for variance

$$
\operatorname{Var}(X) \ne \sum_{i} \operatorname{Var}\left(X \mid A_{i}\right) \operatorname{\mathbb{P}}\left(A_{i}\right)
$$

unless $\operatorname{Var}(\operatorname{\mathbb{E}}[X \mid A]) = 0$, which implies that $\operatorname{\mathbb{E}}\left( X \mid A \right) = \text{constant}$, i.e., $X$ and the partitioning $A_i$ are independent.
:::

## Inequalities

There are two important inequalities that connect probability, expectation and variance.

### Markov's Inequality

Markov's inequality upper bounds right-tail probability $\operatorname{\mathbb{P}}\left( X\ge \lambda \right)$ by $\frac{\mu}{\lambda}$. For a **nonnegative** random variable $X$ and $\lambda>0$,

$$
\operatorname{\mathbb{P}}(X \geq \lambda) \leq \frac{\mu}{\lambda}
$$

A more useful form is to substitute $\lambda \leftarrow \lambda \mu$, then

$$
\operatorname{\mathbb{P}}(X \geq \lambda \mu) \leq \frac{1}{\lambda}
$$

That is, the probability of exceeding expectation by more than a factor  of $\lambda$ is at most $\frac{1}{\lambda}$.

:::{admonition,dropdown,seealso} *Proof*

- By the law of total expectation, and since $\operatorname{\mathbb{E}}(X \mid X<\lambda)\ge 0$ and $\operatorname{\mathbb{E}}(X \mid X \geq \lambda)\ge \lambda$, we have

    $$\begin{aligned}
    \operatorname{\mathbb{E}}(X) & =  \operatorname{\mathbb{E}}(X \mid X<\lambda) \cdot \operatorname{\mathbb{P}}(X<\lambda) +  \operatorname{\mathbb{E}}(X \mid X \geq \lambda) \cdot \operatorname{\mathbb{P}}(X \geq \lambda)\\
    & \ge 0 \cdot \operatorname{\mathbb{P}}(X<\lambda) +\operatorname{\mathbb{E}}(X \mid X \geq \lambda) \cdot \operatorname{\mathbb{P}}(X \geq \lambda) \\
    & \geq \lambda \cdot \operatorname{\mathbb{P}}(X \geq \lambda)
    \end{aligned}$$

    $\square$

- By the definition of expectation,

    $$\begin{aligned}
    \operatorname{\mathbb{E}}(X) &= \int_{0}^{\lambda} x f(x) \mathrm{~d} x+\int_{\lambda}^{\infty} x f(x) \mathrm{~d} x \\
    & \geq \int_{\lambda}^{\infty} x f(x) \mathrm{~d} x \\
    & \geq \int_{\lambda}^{\infty} \lambda f(x) \mathrm{~d} x \\
    & =\lambda \int_{\lambda}^{\infty} f(x) \mathrm{~d} x \\
    &=\lambda \operatorname{\mathbb{P}}(X \geq \lambda)
    \end{aligned}$$

    $\square$

:::

### Chebyshev's Inequality

The probability of deviating from $\mu$ by more than $\lambda \sigma$ is at most $\frac{1}{\lambda^2}$. For any $\lambda>0$,

$$
\operatorname{\mathbb{P}}(|X-\mu| \geq \lambda \sigma) \leq \frac{1}{\lambda^{2}}
$$

For instance, taking $\lambda=\sqrt{2}$ gives

$$
\operatorname{\mathbb{P}}\left( \mu - \sqrt{2}\sigma \le X \le \mu - \sqrt{2}\sigma \right) > \frac{1}{2}
$$

Substituting $\lambda \leftarrow \frac{\lambda}{\sigma}$ gives

$$
\operatorname{\mathbb{P}}(|X-\mu| \geq \lambda ) \leq \frac{\sigma ^2}{\lambda^{2}}
$$

In general,

$$
\operatorname{\mathbb{P}}(|\mathrm{X}-\mu|>\lambda) \leq \frac{\mathbb{E}\left[(X-\mu)^{p}\right]}{\lambda^{p}}
$$

:::{admonition,dropdown,seealso} *Proof by the law of total expectation*

$$\begin{align}
\sigma^{2} &=\operatorname{\mathbb{E}}\left[(X-\mu)^{2}\right] \\
&=  \operatorname{\mathbb{E}}\left[(X-\mu)^{2} \mid \lambda \sigma \leq| X-\mu \mid\right] \cdot \operatorname{\mathbb{P}}\left( \lambda \sigma \leq|X-\mu| \right)\\
& \, +  \operatorname{\mathbb{E}}\left[(X-\mu)^{2} \mid \lambda \sigma>| X-\mu \mid\right] \cdot \operatorname{\mathbb{P}}\left( \lambda \sigma>|X-\mu| \right) \\
& \geq(\lambda \sigma)^{2} \operatorname{\mathbb{P}}\left( \lambda \sigma \leq|X-\mu| \right)+0 \cdot \operatorname{\mathbb{P}}\left( \lambda \sigma>|X-\mu| \right) \\
&=\lambda^{2} \sigma^{2} \operatorname{\mathbb{P}}[\lambda \sigma \leq|X-\mu|]
\end{align}$$

$\square$

:::


### Chernoff Bound

Suppose $X_{1}, \cdots, X_{n}$ are independent r.v. with $X_{i} \in[0,1]$. Let $S=\sum_{i} X_{i}$ and $\mu=\mathbb{E}[X]$. Then for $\lambda \in (0,1)$,

$$
\operatorname{\mathbb{P}}(S>(1+\lambda) \mu)<e^{-\frac{\lambda^{2} \mu}{3}} \qquad \text{(upper tail)}
$$

and,

$$
\operatorname{Pr}(S<(1-\lambda) \mu)<e^{-\frac{\lambda^{2} \mu}{2}} \qquad \text{(lower tail)}
$$



### Cauchy-Schewarz Inequality in Probability

For two random variables $X, Y$ we have

$$
\left[ \operatorname{Cov}\left( X,Y \right) \right]^2 \le \operatorname{Var}\left( X \right) \operatorname{Var}\left( Y \right)
$$


The equality holds iff there is a deterministic linear relation between $X$ and $Y$, $Y = aX + b$.

:::{admonition,dropdown,seealso} *Proof*

Recall the Cauchy-Schewarz inequality for vectors $\boldsymbol{u}, \boldsymbol{v}$ of an innver product space,

$$
|\langle\mathbf{u}, \mathbf{v}\rangle|^{2} \leq\langle\mathbf{u}, \mathbf{u}\rangle \cdot\langle\mathbf{v}, \mathbf{v}\rangle
$$

Define an inner product on the set of random variables using the expectation of their product

$$
\langle X, Y\rangle:=\operatorname{\mathbb{E}}(X Y)
$$

Then the Cauchy-Schewrz inequality becomes

$$
|\operatorname{\mathbb{E}}(X Y)|^{2} \leq \operatorname{\mathbb{E}}\left(X^{2}\right) \operatorname{\mathbb{E}}\left(Y^{2}\right)
$$

Substituting $X$ by $X-\mu_X$ and $Y$ by $Y-\mu_Y$ gives

$$
\begin{aligned}
|\operatorname{Cov}(X, Y)|^{2} &=\left\vert \operatorname{\mathbb{E}}\left[ (X-\mu_X)(Y-\mu_Y) \right] \right\vert^{2} \\
&\le \operatorname{\mathbb{E}}\left[ (X-\mu_X)^{2} \right] \operatorname{\mathbb{E}}\left[ (Y-\mu_Y)^2\right]\\
&=\operatorname{Var}(X) \operatorname{Var}(Y)
\end{aligned}
$$

$\square$

:::


## Exercise

### Coin Flips

- Count trials: What is the expected number of coin flips to get two heads in a row?

  ::::{admonition,dropdown,seealso} *Solution 1: Law of Total Expectation*

  Denote the required number of flips by $X$. We can partition the sample space into **three** parts:
  - $A_T$: the first flip is a tail
  - $A_{HT}$: the first two flips are head, tail
  - $A_{HH}$: the first two flips are head, head

  It's easy to see

  $$
  \operatorname{\mathbb{P}}\left( A_T \right) = \frac{1}{2}, \operatorname{\mathbb{P}}\left( A_{HT} \right) = \operatorname{\mathbb{P}}\left( A_{HH} \right) = \frac{1}{4}
  $$

  But what are $\operatorname{\mathbb{E}}\left( X \mid A_T \right), \operatorname{\mathbb{E}}\left( X \mid A_{HT} \right), \operatorname{\mathbb{E}}\left( X \mid A_{HH} \right)$?

  - If the first flip is T, then we start over, and waste 1 flip
  - If the first two flips are HT, then we start over, and waste 2 flips
  - If the first two flips are HH, then done! We use 2 flips

  As a result, we have

  - $\operatorname{\mathbb{E}}\left( X \mid A_T \right) = \operatorname{\mathbb{E}}\left( X \right) + 1$
  - $\operatorname{\mathbb{E}}\left( X \mid A_{HT} \right) = \operatorname{\mathbb{E}}\left( X \right)+ 2$
  - $\operatorname{\mathbb{E}}\left( X \mid A_{HH} \right) = 2$

  Then by the law of total expectation

  $$
  \begin{align}
  \operatorname{\mathbb{E}}\left( X \right)
  &= \operatorname{\mathbb{E}}\left( X \mid A_T \right) \operatorname{\mathbb{P}}\left( A_T \right)
  +  \operatorname{\mathbb{E}}\left( X \mid A_{HT} \right) \operatorname{\mathbb{P}}\left( A_{HT} \right)
  +  \operatorname{\mathbb{E}}\left( X \mid A_{HH} \right) \operatorname{\mathbb{P}}\left( A_{HH} \right) \\
  &= \left[ \operatorname{\mathbb{E}}\left( X \right)
   + 1 \right]\cdot \frac{1}{2} + \left[ \operatorname{\mathbb{E}}\left( X \right) + 2\right] \cdot \frac{1}{4}
   + 2 \cdot \frac{1}{4} \\
  \end{align}
  $$

  Solving the equation gives $\operatorname{\mathbb{E}}\left( X \right) = 6$


  :::{admonition,note} Note
  One may also partition the sample space to two parts ${A_H}$ and $A_T$, but to compute $\operatorname{\mathbb{E}}\left( X \mid A_H \right)$, it requires to partition $A_H$ into $A_{HT}$ and $A_{HH}$, and then use the law of total expectation again, which is complicated and easy to make mistakes. So it would be better to partition $A$ to three parts at the beginning.
  :::

  In general, what is the expected number of coin flips to get $n$ heads in a row? In fact, we just need to continue to partition $A_{HH}$ into $A_{HHT}$ and $A_{HHH}$, and so on. By the law of total expectation the equation becomes

  $$
  \operatorname{\mathbb{E}}\left( X_n \right)
  = \left[ \operatorname{\mathbb{E}}\left( X_n \right) + 1 \right]\cdot \frac{1}{2}
  + \left[ \operatorname{\mathbb{E}}\left( X_n \right) + 2\right] \cdot \frac{1}{4}
   + \ldots
   + \left[ \operatorname{\mathbb{E}}\left( X_n \right) + n\right] \cdot \frac{1}{2^n}
   + n \cdot \frac{1}{2^n} \\
  $$

  The solution is

  $$
  \operatorname{\mathbb{E}}\left( X_n \right) = 2 \left( 2^n-1 \right)
  $$
  ::::


  :::{admonition,dropdown,seealso} *Solution 2: Recurrence Relation*

  One can also derive the solution from a recurrence relation between $\operatorname{\mathbb{E}}\left( X_n \right)$ and $\operatorname{\mathbb{E}}\left( X_{n-1} \right)$.

  Let $Y_{n} = X_n - X_{n-1}$ be the number of additional flips required to get $n$ heads in a row, given that we already got $n-1$ heads in a row. Then by the law of total expectation,

  $$
  \begin{align}
  \operatorname{\mathbb{E}}\left( Y_{n} \right)
  &= \operatorname{\mathbb{E}}\left( Y_n \mid \text{the $n$-th flip is H} \right) \operatorname{\mathbb{P}}\left( \text{the $n$-th flip is H}  \right) \\
    &\ + \operatorname{\mathbb{E}}\left( Y_n \mid \text{the $n$-th flip is T} \right) \operatorname{\mathbb{P}}\left( \text{the $n$-th flip is T}  \right) \\
  &= 1 \cdot \frac{1}{2} + \left[ 1 + \operatorname{\mathbb{E}}\left( X_n \right) \right] \cdot \frac{1}{2}
   \end{align}
  $$

  Hence, we have the recurrence relation

  $$
  \operatorname{\mathbb{E}}\left( X_n \right) = 2 \operatorname{\mathbb{E}}\left( X_{n-1} \right) + 2
  $$

  Let $f(n) = \operatorname{\mathbb{E}}\left( X_n\right) + 2$ then we have $f(n) = 2f(n-1)$. Since $f(1) = \operatorname{\mathbb{E}}\left( X_1 \right)+2 = 4$, we have $f(n) = 2^{n+1}$. Therefore,  

  $$\operatorname{\mathbb{E}}\left( X_n \right) = 2^{n+1}-2$$

  :::


- **Count rows**: what is the expected number of times to see $k$ heads in a row, i.e., HH...HH, in $n$ flips of a coin?

  :::{admonition,dropdown,seealso} *Solution*

  In $n$ flips of a coin, there are $n-k+1$ places where the string HH...HH can start to appear, each with a (non-independent) probability $\frac{1}{2^k} $ of happening. Let $X$ be the number of times to see the string HH...HH, and $X_i$ be the indicator variable that is $1$ if the string starts to appear at the $i$-th flip, then

  $$
  X = \sum_{i=1}^{n-k+1} X_i
  $$

  and hence

  $$\begin{align}
  \operatorname{\mathbb{E}}\left( X \right) &= \operatorname{\mathbb{E}}\left( \sum_{i=1}^{n-k+1} X_i \right)\\
  &= \sum_{i=1}^{n-k+1} \operatorname{\mathbb{E}}\left( X_i \right)\\
  &= \frac{n-k+1}{2^k} \\
  \end{align}$$

  The first second last line holds even if $X_i$'s are not independent.

  :::


- **Count runs**: a coin with a probability $p$ to get a head is flipped $n$ times. A "run" is a maximal sequence of consecutive flips that are all the same. For instance, HTHHHTTH has five runs and $n=8$. What is the expected number of runs?

  :::{admonition,dropdown,seealso} *Solution*

  Let $X_i$ be the indicator for the event that a run starts at the $i-th$ toss. Let $X = \sum_i X_i$ be the total number of runs. It is easy to see $\operatorname{\mathbb{E}}\left( X_1 \right) = 1$. For $i>1$,


  $$
  \begin{aligned}
  \operatorname{\mathbb{E}}\left(X_{i}\right)=& \operatorname{\mathbb{P}}\left(X_{i}=1\right) \\
  =& \operatorname{\mathbb{P}}\left(i \text { -th toss is } \mathrm{H} \mid(i-1) \text { -th toss is } \mathrm{T}\right) \times \operatorname{\mathbb{P}}\left((i-1) \text { -th toss is } \mathrm{T}\right) \\
  &+\operatorname{\mathbb{P}}\left(i \text { -th toss is } \mathrm{T} \mid(i-1)\text {-th} \text { toss is } \mathrm{H}\right) \times \operatorname{\mathbb{P}}\left((i-1)\text {-th } \text { toss is } \mathrm{H}\right) \\
  =& p(1-p)+(1-p) p \\
  =& 2 p(1-p)
  \end{aligned}
  $$

  Therefore,

  $$
  \begin{aligned}
  \operatorname{\mathbb{E}}(X) &=\operatorname{\mathbb{E}}\left(X_{1}+X_{2}+\cdots+X_{n}\right) \\
  &=\operatorname{\mathbb{E}}\left(X_{1}\right)+\operatorname{\mathbb{E}}\left(X_{2}\right)+\cdots+\operatorname{\mathbb{E}}\left(X_{n}\right) \\
  &=\operatorname{\mathbb{E}}\left(X_{1}\right)+\left[\operatorname{\mathbb{E}}\left(X_{2}\right)+\cdots+\operatorname{\mathbb{E}}\left(X_{n}\right)\right] \\
  &=1+(n-1) \times 2 p(1-p) \\
  &=1+2(n-1) p(1-p)
  \end{aligned}
  $$
  :::


### Incremental Update of Mean and Variance

*Suppose you have $n$ observations $x_1, x_2, \ldots, x_n$. Now a new value $x_{n+1}$ is observed. Write down recurrence functions for $\bar{x}_n$ and $s^2_n$, and use them to obtain $\bar{x}_{n+1}$ and $s^2_{n+1}$.*

::::{admonition,dropdown,seealso} *Solution*
To update mean,

$$\begin{align}
\bar{x}_{n+1}
&= \frac{\sum_{i=1}^{n+1} x_i}{n+1} \\
&= \frac{x_{n+1} + \sum_{i=1}^n x_i}{n+1} \\
&= \frac{x_{n+1} + n\bar{x}_n}{n+1} \\
&= \bar{x}_n + \frac{1}{n+1}(x_{n+1} - \bar{x}_n) \quad (*)
\end{align}$$

The last line is to avoid computing a large number $n \bar{x}_n$.

The second last line implies that the new sample mean $\bar{x}_{n+1}$ is a weighted average of the current sample mean $\bar{x}_{n+1}$ and the new observed value $x_{n+1}$.

To update variance, we first use the above method to obtain $\bar{x}_{n+1}$, and let $S_n = ns_{n}^2$, then


$$\begin{align}
S_{n+1}
&=  \sum_{i=1}^{n+1}x_i^2  - (n+1) \bar{x}_{n+1}^2  \\
&=  \sum_{i=1}^{n}x_i^2 - n\bar{x}_n^2 + n\bar{x}_n^2 + x_{n+1}^2  - (n+1) \bar{x}_{n+1}^2  \\
&=  S_{n} + n\bar{x}_n^2 + x_{n+1}^2  - (n+1) \bar{x}_{n+1}^2  \\
&=  S_{n}  + x_{n+1}^2  + (n+1)(\bar{x}_n - \bar{x}_{n+1})(\bar{x}_n + \bar{x}_{n+1}) - \bar{x}_{n}^2  \\
&=  S_{n}  + x_{n+1}^2  + (\bar{x}_n - x_{n+1} )(\bar{x}_n + \bar{x}_{n+1}) - \bar{x}_{n}^2 \quad \text{by} \ (*) \\
&=  S_{n}  + x_{n+1}^2  + \bar{x}_n \bar{x}_{n+1} - x_{n+1} (\bar{x}_n + \bar{x}_{n+1}) \\
&=  S_{n}  + (x_{n+1}  - \bar{x}_n)(x_{n+1}  - \bar{x}_{n+1})
\end{align}$$

Finally $s_{n+1}^2 = \frac{1}{n+1} S_{n+1}$.


:::{tip}
The substitution $S_n = ns_n^2$ avoids the computation that involves $\frac{1}{n} $ and $\frac{1}{n+1} $. And the update equation of the mean is also quite useful.
:::

::::

(exp-var-ex)=
### Miscellaneous

1. Two groups of data. In group one, sample standard deviation is $s_1$, in group two it is $s_2$. After merging them, it is $s_3$. Do we always have $s_3 > \max(s_1, s_2)$?

    :::{admonition,dropdown,seealso} *Solution*

    Let $\lambda = \frac{n_1}{n_1 + n_2} \in (0,1)$, then $\bar{x}_3 = \lambda \bar{x}_1 + (1-\lambda)\bar{x}_2$. WLOG assume $\bar{x}_1 - \bar{x}_2 = d \ge 0$.

    $$\begin{aligned}
    s^2 _1 &= \frac{TSS_1}{n_1} \\
    s^2 _2 &= \frac{TSS_2}{n_2} \\
    s^2 _3 &= \frac{TSS_3}{n_3}\\
    s^2 _3 &= \frac{TSS_1 + TTS_2 + n_1 (\bar{x}_1 - \bar{x}_3)^2 + n_2 (\bar{x}_3 - \bar{x}_2)^2 }{n_1 + n_2}\\
    &= \lambda s^2 _1 + (1-\lambda) s^2 _2  + \lambda ((1-\lambda)d )^2 + (1-\lambda) (\lambda d)^2    \\
    &= \underbrace{\lambda s^2 _1 + (1-\lambda) s^2 _2}_{a}  + \underbrace{\lambda(1-\lambda)d^2}_{b}   \\
    \end{aligned}$$

    where

    - $\lambda = \frac{n_1}{n_1 + n_2} \in (0,1)$
    - $d_1 = \bar{x}_1 - \bar{x}_3 = \bar{x}_1 - (\lambda \bar{x}_1 + (1-\lambda)\bar{x}_2) = (1-\lambda)(\bar{x}_1 - \bar{x}_2) = (1-\lambda)c \in [0, c)$
    - $d_2 = \bar{x}_3 - \bar{x}_2 = \lambda \bar{x}_1 + (1-\lambda)\bar{x}_2 - \bar{x}_2 = \lambda(\bar{x}_1 - \bar{x}_2) = \lambda c \in [0, c)$
    - $d_1 + d_2 = c$

    We have

    - $\min(s^2_1, s^2_2) \le a \le \max(s^2_1, s^2_2)$ with equalities iff $s_2^2 = s_2^2$.
    - $0 \le b < d^2$ with equality iff $d=0$.

    Since $a$ and $b$ are independent, we can know for sure that $\min(s^2_1, s^2_2) \le s_3^2$. The other comparison $\max(s^2_1, s^2_2) \text{ vs } s_3^2$ is uncertain, depending on $d$.

    :::

1. Find the distribution of the sum of two independent uniform random variable.

    :::{admonition,dropdown,seealso} *Proof*

    Let $X, Y \overset{\text{iid}}{\sim} \mathcal{U} (0,1)$ and their sum be $Z = X + Y$. Then

    $$\begin{aligned}
    f_{Z}(z)
    &=\int_{-\infty}^{\infty} f_{X}(x) f_{Y}(z-x) \mathrm{~d} x \\
    &=\int_{-\infty}^{\infty} \boldsymbol{1} _{x \in (0,1)} \boldsymbol{1} _{z-x \in (0,1)} \mathrm{~d} x \\
    &=\int_{0}^{1} \boldsymbol{1} _{z-x \in (0,1) } \mathrm{~d} x \\
    &= \left\{\begin{array}{ll}
    \int_{0}^{z} 1 \mathrm{~d} x = z, & \text { if } 0<z<1 \\
    \int_{z-1}^{1} 1 \mathrm{~d} x = 2 - z, & \text { if } 1\le z<2 \\
    \end{array}\right.\\
    \end{aligned}$$

    Hence, $Z$ follows a triangular distribution with lower limit $0$, upper limit $2$, and mode $1$. That is, it's more likely to see $Z$ around $1$, which equals the sum of two expected values. In real life, the sum of two dices is probably around 3.5.

    :::


1. Randomly and independently select two points in $[0, \ell]$, find their expected distance.

    :::{admonition,dropdown,seealso} *Solution 1: integration*

    \ellet $X_1, X_2 \overset{\text{iid}}{\sim} \mathcal{U} [0, \ell]$, their joint density function is

    $$
    f_{X_{1} X_{2}}\left(x_{1}, x_{2}\right)=f_{X_{1}}\left(x_{1}\right) f_{X_{2}}\left(x_{2}\right)=\left\{\begin{array}{ll}
    \frac{1}{\ell^2} & \text { if } \quad x_1, x_2 \in[0, \ell] \\
    0 & \text { otherwise }
    \end{array}\right.
    $$

    Define the distance as

    $$
    g\left(x_{1}, x_{2}\right)=\left|x_{1}-x_{2}\right|=\left\{\begin{array}{lll}
    x_{1}-x_{2} & \text { if } x_{1} \geq x_{2} \\
    x_{2}-x_{1} & \text { otherwise }
    \end{array}\right.
    $$

    Hence the expectation is

    $$
    \begin{aligned}
    \mathbb{E}(g(X_1, X_2)) &=\int_{0}^{\ell} \int_{0}^{\ell} g\left(x_{1}, x_{2}\right) f_{X_{1} X_{2}}\left(x_{1}, x_{2}\right) \mathrm{~d} x_{1} \mathrm{~d} x_{2} \\
    &=\frac{1}{\ell^{2}} \int_{0}^{\ell} \int_{0}^{\ell}\left|x_{1}-x_{2}\right| \mathrm{~d} x_{1} \mathrm{~d} x_{2} \\
    &=\frac{1}{\ell^{2}} \int_{0}^{\ell} \int_{0}^{x_{1}}\left(x_{1}-x_{2}\right) \mathrm{~d} x_{2} \mathrm{~d} x_{1}+\frac{1}{\ell^{2}} \int_{0}^{\ell} \int_{x_{1}}^{\ell}\left(x_{2}-x_{1}\right) \mathrm{~d} x_2 \mathrm{~d} x_1 \\
    &=\frac{\ell^3}{6\ell^2} + \frac{\ell^3}{6\ell^2}  \\
    &=\frac{\ell}{3}
    \end{aligned}
    $$

    :::

    :::{admonition,dropdown,seealso} *Solution 2: random procedure*

    If we randomly select two points, then we cut the interval of length $\ell$ into 3 segments, of length $D_1, D_2, D_3$ respectively. They should be "exchangeable", so $\mathbb{E}\left( D_1 \right) = \mathbb{E}\left( D_2 \right) = \mathbb{E}\left( D_3 \right)$. Since $\mathbb{E}\left( D_1 + D_2 + D_3 \right) = \ell$, we have $\mathbb{E}\left( D_2 \right) = \frac{\ell}{3}$.

    Formally, $(D_1, D_2, D_3) = (\min (X, Y), \max (X, Y)-\min (X, Y), \ell-\max (X, Y))$. One can show that $(D_1, D_2, D_3)$ is an exchangeable sequence, i.e., whose joint probability distribution does not change when the positions in the sequence in which finitely many of them appear are altered.

    :::

   - How about two uniform random points in a compact convex subset in $\mathbb{R} ^n$? For example, interval, disk, square, cube? See this [paper](https://www.cambridge.org/core/journals/bulletin-of-the-australian-mathematical-society/article/average-distance-between-two-points/F182A617B5EC6DB5AD31042A4BDF83AE).
