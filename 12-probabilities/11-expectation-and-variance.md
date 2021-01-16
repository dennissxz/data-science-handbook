# Expectation and Variance

## Definitions

We quickly review the definitions of expectation, variance and covariance.


$$\begin{align}
&&&\text{Population} && \text{Sample} \\
& \text{Mean} & \mu &= \sum_{i=1}^n x_i p(x_i) \text{ or } \int_{\mathcal{X}} x f(x) \mathrm{~d}x &  \bar x &= \frac{1}{n}\sum_i x_i  \\
& \text{Variance} & \sigma^2 &= \operatorname{E}\left[ \left( X-\mu \right)^2 \right]  & s^2 &= \frac{1}{n}\sum_i(x_i - \bar x)^2\\
& \text{Standard deviation}  & \sigma &= \sqrt{\operatorname{E}\left[ \left( X-\mu \right)^2 \right]}  & s &= \sqrt{\frac{1}{n}\sum_i(x_i - \bar x)^2} \\
& \text{Covariance}  & \sigma_{X,Y} &= \operatorname{E}\left[ (X-\mu_X)(Y-\mu_Y) \right] & s_{X,Y} &= \frac{1}{n}\sum_i \left[ (x_i - \bar x)(y_i - \bar y) \right]
\end{align}$$


Also recall the definitions of conditional expectation and conditional variance:

$$\begin{align}
 \operatorname{E}(X \mid Y=y)
 &= \sum_{x} x P(X=x \mid Y=y) \\
 &\text{or} \int_{-\infty}^{\infty} x f_{X \mid Y}(x, y) \mathrm{~d} x \\
\operatorname{Var}\left( X \mid Y=y \right)
 &= \operatorname{E}\left[ (X-\mu_{X\mid Y=y})^{2} \mid Y=y \right] \\
\end{align}$$


```{note}
- The notation $X \mid Y=y$ means that $Y=y$ is observed. In this case, the conditional expectation (variance) is a function of the observed value $y$, i.e., $\operatorname{E}(X \mid Y=y) = g(y)$, which itself is a constant.
- The notation $X \mid Y$ means that $Y$ is a random variable and has not been observed yet. In this case, the conditional expectation (variance) is a function of the random variable $Y$, i.e., $\operatorname{E}(X \mid Y) = g(Y)$, which itself is a random variable.
```


## Identities

### Basics

In general, we have

$$\begin{align}
\operatorname{E}\left( aX + bY \right) &= a \operatorname{E}\left( X \right) + b \operatorname{E}\left( Y \right) \\
\operatorname{Var}\left( aX + bY \right) &= a^2\operatorname{Var}\left( X \right) + b^2\operatorname{Var}\left( Y \right) + 2ab\operatorname{Cov}\left( X, Y \right) \\
\operatorname{Var}\left( X \right) &= \operatorname{E}\left( X^2 \right) - \left[ \operatorname{E}\left( X \right) \right]^2\\
\operatorname{E}\left( X^2 \right) &= \mu^2 + \sigma^2 \\
\operatorname{Cov}\left( X, X \right) &= \operatorname{Var}\left( X \right) \\
\operatorname{Cov}\left( X,Y \right) &= \operatorname{E}\left( XY \right) - \operatorname{E}\left( X \right)\operatorname{E}\left( Y \right) \\
\operatorname{Cov}\left( X, a \right) &= 0 \\
\operatorname{Cov}\left( X, Y+Z \right) &= \operatorname{Cov}\left( X, Y \right) + \operatorname{Cov}\left( X, Z \right) \\
\operatorname{Cov}\left( aX, bY \right) &= ab \operatorname{Cov}\left( X, Y \right)
\end{align}$$


```{note}
Be careful about the notations $\sigma_X ^2$ and $\sigma_{X,X}$

$$
\sigma_X^2 = \operatorname{Var}\left( X \right) = \operatorname{Cov}\left( X, X \right) = \sigma_{X,X}
$$
```

If $X$ and $Y$ are independent,

$$\begin{align}
\operatorname{E}\left( XY \right) &= \operatorname{E}\left( X \right)\operatorname{E}\left( Y \right) \\
\operatorname{Cov}\left( X, Y \right) &= 0 \\
\operatorname{Var}\left( aX + bY \right) &= a^2\operatorname{Var}\left( X \right) + b^2\operatorname{Var}\left( Y \right) \\
\end{align}$$

### Linear Combinations


For $n$ random variables $X_1, X_2, \ldots, X_n$, consider a linear combination $\sum_i^n a_i X_i$. Though we have no information about the dependence between $X_i$'s, the expectation of the sum equals to the sum of the expectations

$$
\operatorname{E}\left( \sum_i^n a_i X_i \right)
 = \sum_i^n \operatorname{E}\left(a_i X_i \right)
 = \sum_i^n a_i\operatorname{E}\left( X_i \right)
$$



In this sense, expectation is a linear operator.


In particular, for independently and identically distributed $X_1, X_2, \ldots, X_n$ with common mean $\operatorname{E}\left( X_i \right)=\mu$, the expectation of the average value is

$$\begin{align}
\operatorname{E}\left( \bar X \right) &= \operatorname{E}\left( \frac{1}{n}\sum_{i} X_i \right)\\
&= \frac{1}{n}\sum_i \operatorname{E}\left( X_i \right)\\
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

```{tip}
One can imagine that there is a $n \times n$ covariance table with the $i,j$-th entry being $\operatorname{Cov}\left( a_i X_i, a_j X_j \right)$, and the required variance is the sum of all the entries, which consists of
- the sum of the diagonal entries as $\sum_i\operatorname{Var}\left(a_{i}X_{i}\right)$
- the sum of the off-diagonal entries as $\sum_{i\ne j}\operatorname{Cov}\left(a_{i}X_{i}, a_{j}X_{j}\right)$
```

In particular, the variance of the average value of the IID sum is

$$\begin{align}
\operatorname{Var}\left( \bar X \right) &= \operatorname{Var}\left( \frac{1}{n}\sum_{i} X_i \right)\\
&= \frac{1}{n^2}\sum_i \operatorname{Var}\left( X_i \right)\\
&= \frac{1}{n} \sigma^2 \\
\end{align}$$



### Expectation of Nonnegative Random Variables

For nonnegative random variables, the expectation can be computed from the complementary cumulative distribution function $1 - F(x) = \operatorname{P}\left( X > x \right)$.

- discrete case

$$
\operatorname{E}\left( X \right)=\sum_{n=0}^{\infty} \operatorname{P}\left( X>n \right)
$$

- continuous case

$$
\operatorname{E}\left( X \right) = \int_{0}^{\infty} \operatorname{P}(X \geq x) \mathrm{~d} x
$$

***Proof***

We prove by changing the **order** of summation/integral.

- discrete case

  $$\begin{align}
  \sum_{n=0}^{\infty} \operatorname{P}\left( X>n \right)
  &= \sum_{n=0}^{\infty} \sum_{k=n+1}^{\infty} \operatorname{P}\left( X=k \right) \\
  &= \sum_{k=1}^{\infty} \sum_{n=1}^k \operatorname{P}\left( X=k \right) \\
  &= \sum_{k=1}^{\infty} k \operatorname{P}\left( X=k \right) \\
  &= \sum_{k=0}^{\infty} k \operatorname{P}\left( X=k \right) \\
  &= \operatorname{E}\left( X \right)
  \end{align}$$

- continuous case

  $$
  \begin{aligned}
  \int_{0}^{\infty} \operatorname{P}(X \ge x) \mathrm{~d} x
  &=\int_{0}^{\infty} \int_{x}^{\infty} f_{X}(y) \mathrm{~d} y \mathrm{~d} x \\
  &=\int_{0}^{\infty} \int_{0}^{y} f_{X}(y) \mathrm{~d} x \mathrm{~d} y \\
  &=\int_{0}^{\infty} f_{X}(y) \int_{0}^{y} 1 \mathrm{~d} x \mathrm{~d} y \\
  &=\int_{0}^{\infty} y f_{X}(y) \mathrm{~d} y \\
  &=\operatorname{E}\left( X \right)
  \end{aligned}
  $$

  $\square$

### Law of Total Expectation

Aka law of iterated expectations, tower rule, smoothing theorem.

Given the conditional expectation $\operatorname{E}\left( X \mid Y \right)$, the law of total expectation states that we can obtained the unconditional expectation $\operatorname{E}\left( X \right)$ by

$$
\operatorname{E}\left( X \right)=\operatorname{E}\left[ \operatorname{E}(X \mid Y) \right]
$$

```{note}
The inside expectation is taken w.r.t. $X$ and the outside expectation is taken w.r.t. $Y$, since the conditional expectation $\operatorname{E}\left(X \mid Y \right)$ is a function $g(Y)$ that depends on the random variables $Y$. To emphasize this we can write

$$
\operatorname{E}_X\left( X \right)=\operatorname{E}_Y\left[ \operatorname{E}_X(X \mid Y) \right]
$$
```

In general, we can partition the sample space into finite or countably infinite sets $A_i$, then

$$
\operatorname{E}(X)=\sum_{i} \operatorname{E}\left(X \mid A_{i}\right) \operatorname{P}\left(A_{i}\right)
$$

For instance, we can compute the expectation as a weighted sum of the expectation of the positive part and the expectation of the negative part on respective probabilities.

$$
\operatorname{E}(X)=\operatorname{E}\left(X \mid X>0\right) \operatorname{P}\left(X>0\right) + \operatorname{E}\left(X \mid X<0\right) \operatorname{P}\left(X<0\right)
$$

***Proof***

By definition

$$
\begin{aligned}
\operatorname{E}\left( \operatorname{E}\left( X \mid Y \right) \right) &=\operatorname{E}\left[\sum_{x} x \cdot \operatorname{P}(X=x \mid Y)\right] \\
&=\sum_{y}\left[\sum_{x} x \cdot \operatorname{P}(X=x \mid Y=y)\right] \cdot \operatorname{P}(Y=y) \\
&=\sum_{y} \sum_{x} x \cdot \operatorname{P}(X=x, Y=y) \\
&=\sum_{x} x \sum_{y} \operatorname{P}(X=x, Y=y) \\
&=\sum_{x} x \cdot \operatorname{P}(X=x) \\
&=\operatorname{E}(X)
\end{aligned}
$$

$\square$

### Law of Total Variance

Aka law of iterated variances, variance decomposition formula, Eve's law.

$$
\operatorname{Var}(X)=\operatorname{E}[\operatorname{Var}(X \mid Y)]+\operatorname{Var}(\operatorname{E}[X \mid Y])
$$

```{note}
Here both $\operatorname{Var}\left( X \mid Y \right)$ and $\operatorname{E}\left( X \mid Y \right)$ are random. The outside expectation and variance are taken w.r.t. the conditioned variable, $Y$.
```

The first and the second term can be interpreted as the unexplained and the explained components of the variance of $X$ by knowing $Y$. Imagine that there is a deterministic relation $X=f(Y)$, then $\operatorname{Var}\left( X \mid Y \right) = 0$ so that the first term is 0, and the second term becomes $\operatorname{Var}\left(  f(Y) \right) = \operatorname{Var}\left( X \right)$.

***Proof***

Note that the relation $\operatorname{Var}\left( X \right) = \operatorname{E}\left( X^2 \right) - \left[ \operatorname{E}\left( X \right) \right]^2$ holds in a similar fashion when conditioning on $Y$

$$
\operatorname{Var}\left( X \mid Y \right) = \operatorname{E}\left( X^2 \mid Y \right) - \left[ \operatorname{E}\left( X \mid Y\right) \right]^2
$$

By the law of total expectation, we can compute $\operatorname{E}\left( X^2 \right)$ by

$$
\begin{align}
\operatorname{E}\left( X^2 \right)
&= \operatorname{E}\left[ \operatorname{E}\left( X^2 \mid Y \right) \right] \\
&=  \operatorname{E}\left\{ \operatorname{Var}\left( X \mid Y \right) + \left[ \operatorname{E}\left( X \mid Y\right) \right]^2  \right\}
\end{align}
$$

and compute $\operatorname{E}\left( X \right)$ by $\operatorname{E}\left[ \operatorname{E}\left( X\mid Y \right) \right]$. Hence,

$$\begin{align}
\operatorname{Var}\left( X \right) &= \operatorname{E}\left( X^2 \right) - \left[ \operatorname{E}\left( X \right) \right]^2\\
&=  \operatorname{E}\left\{ \operatorname{Var}\left( X \mid Y \right) + \left[ \operatorname{E}\left( X \mid Y\right) \right]^2  \right\} - \left\{ \operatorname{E}\left[ \operatorname{E}\left( X\mid Y \right) \right] \right\}^2\\
&= \operatorname{E}[\operatorname{Var}(X \mid Y)]+\operatorname{Var}(\operatorname{E}[X \mid Y])
\end{align}$$

$\square$

```{warning}
From above we see that the identity that holds for expectation

$$
\operatorname{E}(X)=\sum_{i} \operatorname{E}\left(X \mid A_{i}\right) \operatorname{P}\left(A_{i}\right)
$$

does **not** hold for variance

$$
\operatorname{Var}(X) \ne \sum_{i} \operatorname{Var}\left(X \mid A_{i}\right) \operatorname{P}\left(A_{i}\right)
$$

unless $\operatorname{Var}(\operatorname{E}[X \mid A]) = 0$, which implies that $\operatorname{E}\left( X \mid A \right) = \text{constant}$, i.e., $X$ and the partitioning $A_i$ are independent.



```


## Inequalities

There are two important inequalities that connect probability, expectation and variance.

### Markov's Inequality

Markov's inequality upper bounds right-tail probability $\operatorname{P}\left( X\ge a \right)$ by $\frac{1}{a} \operatorname{E}\left( X \right)$.

For a nonnegative random variable $X$ and $a>0$,

$$
\operatorname{P}(X \geq a) \leq \frac{\operatorname{E}(X)}{a}
$$

A simple way to memorize this is: $\operatorname{P}\left( \frac{X}{a} \ge 1 \right ) \le \operatorname{E}\left( \frac{X}{a}  \right)$

***Proof***

- By the law of total expectation, and since $\operatorname{E}(X \mid X<a)\ge0$ and $\operatorname{E}(X \mid X \geq a)\ge a$, we have

    $$\begin{align}
    \operatorname{E}(X) & =  \operatorname{E}(X \mid X<a) \cdot \operatorname{P}(X<a) +  \operatorname{E}(X \mid X \geq a) \cdot \operatorname{P}(X \geq a)\\
    & \ge 0 \cdot \operatorname{P}(X<a) +\operatorname{E}(X \mid X \geq a) \cdot \operatorname{P}(X \geq a) \\
    & \geq a \cdot \operatorname{P}(X \geq a)
    \end{align}$$

    $\square$

- By the definition of expectation,

    $$\begin{align}
    \operatorname{E}(X) &= \int_{0}^{a} x f(x) \mathrm{~d} x+\int_{a}^{\infty} x f(x) \mathrm{~d} x \\
    & \geq \int_{a}^{\infty} x f(x) \mathrm{~d} x \\
    & \geq \int_{a}^{\infty} a f(x) \mathrm{~d} x \\
    & =a \int_{a}^{\infty} f(x) \mathrm{~d} x \\
    &=a \operatorname{P}(X \geq a)
    \end{align}$$

    $\square$

### Chebyshev's Inequality

Chebyshev's inequality upper bounds the probability that a random variable is outside the interval $\left( \mu - k \sigma, \mu + k \sigma \right)$ by $\frac{1}{k^2}$.

For any $k>0$,

$$
\operatorname{P}(|X-\mu| \geq k \sigma) \leq \frac{1}{k^{2}}
$$

For instance, taking $k=\sqrt{2}$ gives

$$
\operatorname{P}\left( \mu - \sqrt{2}\sigma \le X \le \mu - \sqrt{2}\sigma \right) > \frac{1}{2}
$$

***Proof***

By the law of total expectation,

$$\begin{align}
\sigma^{2} &=\operatorname{E}\left[(X-\mu)^{2}\right] \\
&=  \operatorname{E}\left[(X-\mu)^{2} \mid k \sigma \leq| X-\mu \mid\right] \cdot \operatorname{P}\left( k \sigma \leq|X-\mu| \right)\\
& \, +  \operatorname{E}\left[(X-\mu)^{2} \mid k \sigma>| X-\mu \mid\right] \cdot \operatorname{P}\left( k \sigma>|X-\mu| \right) \\
& \geq(k \sigma)^{2} \operatorname{P}\left( k \sigma \leq|X-\mu| \right)+0 \cdot \operatorname{P}\left( k \sigma>|X-\mu| \right) \\
&=k^{2} \sigma^{2} \operatorname{P}[k \sigma \leq|X-\mu|]
\end{align}$$

$\square$

### Covariance and Variances

For two random variables $X, Y$ we have

$$
\left[ \operatorname{Cov}\left( X,Y \right) \right]^2 \le \operatorname{Var}\left( X \right) \operatorname{Var}\left( Y \right)
$$


The equality holds iff there is a deterministic linear relation between $X$ and $Y$, $Y = aX + b$.

***Proof***

Recall the Cauchy-Schewarz inequality for vectors $\boldsymbol{u}, \boldsymbol{v}$ of an innver product space,

$$
|\langle\mathbf{u}, \mathbf{v}\rangle|^{2} \leq\langle\mathbf{u}, \mathbf{u}\rangle \cdot\langle\mathbf{v}, \mathbf{v}\rangle
$$

Define an inner product on the set of random variables using the expectation of their product

$$
\langle X, Y\rangle:=\mathrm{E}(X Y)
$$

Then the Cauchy-Schewrz inequality becomes

$$
|\mathrm{E}(X Y)|^{2} \leq \mathrm{E}\left(X^{2}\right) \mathrm{E}\left(Y^{2}\right)
$$

Substituting $X$ by $X-\mu_X$ and $Y$ by $Y-\mu_Y$ gives

$$
\begin{aligned}
|\operatorname{Cov}(X, Y)|^{2} &=\left\vert \mathrm{E}\left[ (X-\mu_X)(Y-\mu_Y) \right] \right\vert^{2} \\
&\le \mathrm{E}\left[ (X-\mu_X)^{2} \right] \mathrm{E}\left[ (Y-\mu_Y)^2\right]\\
&=\operatorname{Var}(X) \operatorname{Var}(Y)
\end{aligned}
$$

$\square$

## Exercise

### Coin Flips - Count Trials

*What is the expected number of coin flips to get two heads in a row?*

````{dropdown} Solution 1: Law of Total Expectation

Denote the required number of flips by $X$. We can partition the sample space into **three** parts:
- $A_T$: the first flip is a tail
- $A_{HT}$: the first two flips are head, tail
- $A_{HH}$: the first two flips are head, head

It's easy to see

$$
\operatorname{P}\left( A_T \right) = \frac{1}{2}, \operatorname{P}\left( A_{HT} \right) = \operatorname{P}\left( A_{HH} \right) = \frac{1}{4}
$$

But what are $\operatorname{E}\left( X \mid A_T \right), \operatorname{E}\left( X \mid A_{HT} \right), \operatorname{E}\left( X \mid A_{HH} \right)$?

- If the first flip is T, then we start over, and waste 1 flip
- If the first two flips are HT, then we start over, and waste 2 flips
- If the first two flips are HH, then done! We use 2 flips

As a result, we have

- $\operatorname{E}\left( X \mid A_T \right) = \operatorname{E}\left( X \right) + 1$
- $\operatorname{E}\left( X \mid A_{HT} \right) = \operatorname{E}\left( X \right)+ 2$
- $\operatorname{E}\left( X \mid A_{HH} \right) = 2$

Then by the law of total expectation

$$
\begin{align}
\operatorname{E}\left( X \right)
&= \operatorname{E}\left( X \mid A_T \right) \operatorname{P}\left( A_T \right)
+  \operatorname{E}\left( X \mid A_{HT} \right) \operatorname{P}\left( A_{HT} \right)
+  \operatorname{E}\left( X \mid A_{HH} \right) \operatorname{P}\left( A_{HH} \right) \\
&= \left[ \operatorname{E}\left( X \right)
 + 1 \right]\cdot \frac{1}{2} + \left[ \operatorname{E}\left( X \right) + 2\right] \cdot \frac{1}{4}
 + 2 \cdot \frac{1}{4} \\
\end{align}
$$

Solving the equation gives $\operatorname{E}\left( X \right) = 6$

```{note}
One may also partition the sample space to two parts ${A_H}$ and $A_T$, but to compute $\operatorname{E}\left( X \mid A_H \right)$, it requires to partition $A_H$ into $A_{HT}$ and $A_{HH}$, and then use the law of total expectation again, which is complicated and easy to make mistakes. So it would be better to partition $A$ to three parts at the beginning.
```

In general, what is the expected number of coin flips to get $n$ heads in a row? In fact, we just need to continue to partition $A_{HH}$ into $A_{HHT}$ and $A_{HHH}$, and so on. By the law of total expectation the equation becomes

$$
\operatorname{E}\left( X_n \right)
= \left[ \operatorname{E}\left( X_n \right) + 1 \right]\cdot \frac{1}{2}
+ \left[ \operatorname{E}\left( X_n \right) + 2\right] \cdot \frac{1}{4}
 + \ldots
 + \left[ \operatorname{E}\left( X_n \right) + n\right] \cdot \frac{1}{2^n}
 + n \cdot \frac{1}{2^n} \\
$$

The solution is

$$
\operatorname{E}\left( X_n \right) = 2 \left( 2^n-1 \right)
$$
````


```{dropdown} Solution 2: Recurrence Relation
One can also derive the solution from a recurrence relation between $\operatorname{E}\left( X_n \right)$ and $\operatorname{E}\left( X_{n-1} \right)$.

Let $Y_{n} = X_n - X_{n-1}$ be the number of additional flips required to get $n$ heads in a row, given that we already got $n-1$ heads in a row. Then by the law of total expectation,

$$
\begin{align}
\operatorname{E}\left( Y_{n} \right)
&= \operatorname{E}\left( Y_n \mid \text{the $n$-th flip is H} \right) \operatorname{P}\left( \text{the $n$-th flip is H}  \right) \\
  &\ + \operatorname{E}\left( Y_n \mid \text{the $n$-th flip is T} \right) \operatorname{P}\left( \text{the $n$-th flip is T}  \right) \\
&= 1 \cdot \frac{1}{2} + \left[ 1 + \operatorname{E}\left( X_n \right) \right] \cdot \frac{1}{2}
 \end{align}
$$

Hence, we have the recurrence relation

$$
\operatorname{E}\left( X_n \right) = 2 \operatorname{E}\left( X_{n-1} \right) + 2
$$

Let $f(n) = \operatorname{E}\left( X_n\right) + 2$ then we have $f(n) = 2f(n-1)$. Since $f(1) = \operatorname{E}\left( X_1 \right)+2 = 4$, we have $f(n) = 2^{n+1}$. Therefore,  

$$\operatorname{E}\left( X_n \right) = 2^{n+1}-2$$
```

### Coin Flips - Count Rows

*What is the expected number of times to see $k$ heads in a row, i.e., HH...HH, in $n$ flips of a coin?*


```{dropdown} Solution

In $n$ flips of a coin, there are $n-k+1$ places where the string HH...HH can start to appear, each with a (non-independent) probability $\frac{1}{2^k} $ of happening. Let $X$ be the number of times to see the string HH...HH, and $X_i$ be the indicator variable that is $1$ if the string starts to appear at the $i$-th flip, then

$$
X = \sum_{i=1}^{n-k+1} X_i
$$

and hence

$$\begin{align}
\operatorname{E}\left( X \right) &= \operatorname{E}\left( \sum_{i=1}^{n-k+1} X_i \right)\\
&= \sum_{i=1}^{n-k+1} \operatorname{E}\left( X_i \right)\\
&= \frac{n-k+1}{2^k} \\
\end{align}$$

The first second last line holds even if $X_i$'s are not independent.

```


### Coin Flips - Count Runs



````{dropdown} Solution
A coin with a probability $p$ to get a head is flipped $n$ times. A "run" is a maximal sequence of consecutive flips that are all the same. For instance, HTHHHTTH has five runs and $n=8$. What is the expected number of runs?

Let $X_i$ be the indicator for the event that a run starts at the $i-th$ toss. Let $X = \sum_i X_i$ be the total number of runs. It is easy to see $\operatorname{E}\left( X_1 \right) = 1$. For $i>1$,


$$
\begin{aligned}
\mathrm{E}\left(X_{i}\right)=& \operatorname{P}\left(X_{i}=1\right) \\
=& \operatorname{P}\left(i \text { -th toss is } \mathrm{H} \mid(i-1) \text { -th toss is } \mathrm{T}\right) \times \operatorname{P}\left((i-1) \text { -th toss is } \mathrm{T}\right) \\
&+\operatorname{P}\left(i \text { -th toss is } \mathrm{T} \mid(i-1)\text {-th} \text { toss is } \mathrm{H}\right) \times \operatorname{P}\left((i-1)\text {-th } \text { toss is } \mathrm{H}\right) \\
=& p(1-p)+(1-p) p \\
=& 2 p(1-p)
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\mathrm{E}(X) &=\mathrm{E}\left(X_{1}+X_{2}+\cdots+X_{n}\right) \\
&=\mathrm{E}\left(X_{1}\right)+\mathrm{E}\left(X_{2}\right)+\cdots+\mathrm{E}\left(X_{n}\right) \\
&=\mathrm{E}\left(X_{1}\right)+\left[\mathrm{E}\left(X_{2}\right)+\cdots+\mathrm{E}\left(X_{n}\right)\right] \\
&=1+(n-1) \times 2 p(1-p) \\
&=1+2(n-1) p(1-p)
\end{aligned}
$$
````


### Incremental Update of Mean and Variance

*Suppose you have $n$ observations $x_1, x_2, \ldots, x_n$. Now a new value $x_{n+1}$ is observed. Write recurrence functions to update the sample mean $\bar{x}_n$ and variance $s^2_n$.*

````{dropdown} Solution
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

To update variance, we first find $\bar{x}_n$, then let $S_n = ns_{n}^2$,


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


```{tip}
The substitution $S_n = ns_n^2$ avoids the computation that involves $\frac{1}{n} $ and $\frac{1}{n+1} $. And the update equation of the mean is also quite useful.
```
````
