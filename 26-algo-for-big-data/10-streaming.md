# Streaming Model

## Background

Problem
- Given a sequence of input data, $y_1, \ldots, y_n$, where $y \in [m]$, let $\boldsymbol{x} \in [n]^m$ be a container vector of sample frequency, where $x_i$ counts the number of $y$ that has value $i$, for $i=[m]$. We are interested in
  - $\left\| \boldsymbol{x}  \right\| _0$: how many distinct $y_i$ are there? Let it be $\ell_0$, note that $\ell_0\le m$.
  - $\left\| \boldsymbol{x}  \right\| _1$: how many $y_i$ are there, i.e. how large is $n$?
  - $\left\| \boldsymbol{x}  \right\| _2$: what is the Euclidean norm $\ell_2$ of $\boldsymbol{x}$?
  - $x_i$: what is the frequency of $y$ that has value $i$?

Limitation
- the data set of interest is large, cannot fit into the main memory
- we only have sequential access to the data (one pass or few passes), i.e. the data comes as a **stream**.

Goal
- use little memory, e.g. sub-linear in input parameter or input size
- can also solve the problem of interest approximately, i.e. find estimator $\hat{\ell}_p$ for $\ell_p = \left\| x \right\| _p$ that is
  - unbiased $\mathbb{E}[\hat{\ell}_p] = \ell_p$
  - good concentration
      - $(\epsilon, \delta)$ approximation (guarantee): given $0 < \epsilon, \delta< 1$, want the probability of deviation from true value $\ell_p$ by $\epsilon \ell_p$ to be no larger than $\delta$. This event can be viewed as a failure output, and its probability is called failure probability.

        $$\mathbb{P}[ \vert \hat{\ell}_p - \ell_p \vert> \epsilon \ell_p ] < \delta$$

        Often, given required $\epsilon$, we want to find good $\hat{\ell}_p$ that gives lower $\delta$.

      - constant approximation

Overall algorithm design intuition:
- maintain in memory some quantity $Q$ (can be a scalar, vector, matrix)
- upon receiving new $y_i$, update $Q$ **probabilistically**, rather than deterministically:
  - update $Q$ with some probability
    - depending on $Q$ (Morris for $\ell_1$)
    - depending on $y_i$ (FM for $\ell_0$)
  - update $Q \mathrel{+}= q$ with some random quantity $q$ (AMS, CountMin, etc.)
- output some transformation $f(Q)$ of $Q$, such that
  - unbiasedness: $\mathbb{E} [f(Q)] = \ell_p$.
  - good concentration probability $\mathbb{P}[ \vert f(Q) - \ell_p \vert> \epsilon \ell_p ] < \delta$.
- improvement
  - repeat $s$ instantiations to obtain $Q_1, \ldots, Q_s$, report their average $Q_+$ (FM+, Morris+, AMS)
  - repeat $t$ instantiations of $Q_+$, report their median (FM++, Morris++, AMS++).


Metrics for reviewing algorithms
- memory usage
- number of passes
- approximation factor
- (sometimes) query/update time

Types of streams
- Insertion-only: only $\texttt{Insert(i)}: x_i \mathrel{+}= 1$ is allowed.
- Insertion and deletion (dynamic): also allow $\texttt{Delete(i)}: x_i \mathrel{-}= 1$, e.g. numbers, edges of graphs, etc. Assume that $\#\operatorname{deletions} (i) \le \#\operatorname{insertions} (i)$ such that $x_i \ge 0$.
- Turnstile: more general case for vectors and matrices, with operation $\texttt{Add}(i,\Delta): x_i \mathrel{+}= \Delta$

References
- [TTIC 41000](https://www.mit.edu/~mahabadi/courses/Algorithms_for_Massive_Data_SP21/): Algorithms for Massive Data, Spring 2021
- [Harvard CS 226/MIT 6.889](https://www.sketchingbigdata.org/fall17/): Sketching Algorithms for Big Data, Fall 2017

Note: section [Estimate $\ell_1$](streaming-estimate-l1) might be a better appetizer than [Estimate $\ell_0$](streaming-estimate-l0).

(streaming-estimate-l0)=
## Estimate $\ell_0$

How many distinct $y_i$ are there? Aka distinct element counting problem. For instance, $m = 10$, input stream is $(3,6,9,3,4,5,4)$, output $5$.

Reference: https://www.sketchingbigdata.org/fall17/lec/lec2.pdf

### Trivial Solution

- One trivial solution is to keep a bitvector counter $\boldsymbol{c}$ of size $m$. For each number $i \in [m]$, if $c_i = 0$, then increase $x_i$ from 0 to 1.
- Another trivial solution is to just store the stream, which takes $\min(m, n) \log m$ bits.

These methods are not efficient if $m, n$ are large.

### FM Algorithm

<!-- Let $k$ be the true number of distinct element. We find an estimate $\hat{k}$ such that

$$
\mathbb{P}\left( \left\vert \hat{k} - k  \right\vert > \epsilon k\right) < \delta
$$

Consider a decision problem: given a number $d$, return YES if $k \ge d(1 + \epsilon)$ and NO otherwise. If we can solve this decision problem, then we can run $d$ for $(1, 1 + \epsilon, \ldots, (1 + \epsilon)^i, \ldots, n)$ to find where $k$ is.

Algorithms to solve the decision problem:
- sample each of $m$ coordinate w.p. $\frac{1}{d} \ge 1$, obtain $S = \left\{ i_1, \ldots \right\}$
- if a number $i \in S$ is shown in the stream, increase $x_i$
- if $x_i = 0$ for all $i \in S$, return NO, else YES.

Analysis:


$$\begin{aligned}
\mathbb{P}\left( \text{NO}  \right)
&= \mathbb{P}\left( \text{all $k$ elements in the stream are not sampled}  \right)\\
&= \left( 1 - \frac{1}{d}  \right)^k \\
&\approx \exp \left( - \frac{k}{d}  \right)\\
\end{aligned}$$

For small enough $\epsilon$,
- if $k > (1 + \epsilon) d$ then $e^{-(1+\epsilon)}<\frac{1}{\mathrm{e}}-\frac{\epsilon}{3}$
- if $k < (1 - \epsilon) d$ then $e^{-(1-\epsilon)}>\frac{1}{\mathrm{e}}+\frac{\epsilon}{3}$

 -->

We introduce the FM algorithm.

---
**Algorithms**: Flajolet-Martin (FM)

---
- Pick a random hash function $h: [m] \rightarrow [0,1]$.
- Maintain in memory the smallest hash we've seen so far: $X = \min_{i \in \text{stream} } h(i)$
- Upon $\texttt{query()}$, output $\frac{1}{X} - 1$

---


:::{admonition,tip} Intuition

- Random hash result can be viewed as uniform random variable.
- We are partitioning the interval $[0, 1]$ into bins of size $\frac{1}{\ell_0+1}$
- Taking advantage of the property of the minimum of $\ell_0$ uniform r.v.

:::


Correctness
- Unbiasedness $\mathbb{E} [X] = \frac{1}{t+1}$, and hope that $\frac{1}{X} - 1$ is close to $t$.

  :::{admonition,dropdown,seealso} *Proof*

  We can view each $h(i)$ as a uniform random variable.


  $$
  \begin{aligned}
  \mathbb{E}[X] &=\int_{0}^{\infty} \mathbb{P}(X>\lambda) d \lambda \quad \text{(property of positive r.v.)} \\
  &=\int_{0}^{\infty} \mathbb{P}(\forall i \in \operatorname{stream}, h(i)>\lambda) \mathrm{~d} \lambda \\
  &=\int_{0}^{\infty} \prod_{\text{distinct } i \in \text { stream }} \mathbb{P}(h(i)>\lambda) \mathrm{~d} \lambda \\
  &=\int_{0}^{1}(1-\lambda)^{\ell_0} \mathrm{~d} \lambda \\
  &=\frac{1}{\ell_0+1}
  \end{aligned}
  $$

  :::

  The variance is small $\operatorname{Var}\left( X \right) = \frac{\ell}{(\ell_0+1)^{2}(\ell_0+2)}$.

  :::{admonition,dropdown,seealso} *Proof*


  $$
  \begin{aligned}
  \mathbb{E}\left[X^{2}\right] &=\int_{0}^{\infty} \mathbb{P}\left(X^{2}>\lambda\right)\mathrm{~d}\lambda \\
  &=\int_{0}^{\infty} \mathbb{P}(X>\sqrt{\lambda})\mathrm{~d}\lambda \\
  &=\int_{0}^{1}(1-\sqrt{\lambda})^{\ell}\mathrm{~d}\lambda \\
  &=2 \int_{0}^{1} u^{\ell}(1-u)\mathrm{~d}u \quad u=q - \sqrt{\lambda} \\
  &= \frac{2}{(\ell_0+1)(\ell_0+2)}
  \end{aligned}
  $$


  $$
  \operatorname{Var}[X]=\frac{2}{(\ell_0+1)(\ell_0+2)}-\frac{1}{(\ell_0+1)^{2}}=\frac{\ell}{(\ell_0+1)^{2}(\ell_0+2)}
  $$

  :::

### FM+

Similarly to the Morris+, we can upgrade our basic algorithm into FM+ by running it $s = \frac{1}{\epsilon^2\eta}$ times in parallel to obtain $X_1, X_2, \ldots, X_s$. The output is

$$
\frac{1}{\frac{1}{s} \sum_{i=1}^{s} X_{i}}-1
$$

By Chebyshev's inequality

$$
\begin{aligned}
\mathbb{P}\left(\left|\frac{1}{s} \sum_{i=1}^{s} X_{i}-\frac{1}{\ell_0+1}\right|>\frac{\varepsilon}{\ell_0+1}\right) &<\frac{\operatorname{Var}\left[\frac{1}{s} \sum_{i} X_{i}\right]}{\frac{\varepsilon^{2}}{(\ell_0+1)^{2}}} \\
&<\frac{1}{\varepsilon^{2} s}=\eta
\end{aligned}
$$

Total space is $\mathcal{O} (\frac{1}{\epsilon^2 \eta})$.


### FM++

Run $t = \Theta(\log \frac{1}{\eta} )$ independent copies of FM+, each with $\eta = \frac{1}{3}$. Output the median of $t$ FM+ estimates.

Total space is $\mathcal{O} (\log \frac{1}{\delta} \cdot \frac{1}{\epsilon^2}  )$.


(streaming-estimate-l1)=
## Estimate $\ell_1$

How many $y_i$ are there, i.e. How large is $n$? Aka counting problem. A trivial solution is to maintain a counter, which requires space $\mathcal{O}(\log n)$ bits. But when $n$ is large, can we do better? We introduce Morris counter in insertion-only streams. Morris algorithm was developed in 1970s, when memory was expensive.

### Morris

The Morris algorithms is

---
**Algorithms**: Morris Counter

---
- Initialize $X = 0$.
- Upon receiving a new number $y$, increase $X \mathrel{+}= 1$ with probability $\frac{1}{2^X}$
- Return $\hat{n} = 2^X - 1$

---


:::{admonition,tip}

- $X$ is attempting to store a value that is roughly $\log n$. Hence, the amount of space required to store $X$ is $O(\log \log n)$
- To achieve, this, update the counter $X$ probabilistically
- Since $X \ll \ell_0$, output some large number $\gg X$, e.g. $2^X$

:::

Correctness
- Unbiasedness: Let $X_n$ denote $X$ after $n$ updates. Then $\mathbb{E}\left[ 2 ^{X_n} \right]  = n + 1$. Hence $\mathbb{E} [\hat{n}_1] = 2^{\mathbb{E} [X_n]} - 1$ is an unbiased estimator of $\ell_1 = n$.

  :::{admonition,dropdown,seealso} *Proof by induction*

  Base: $n=0, X_0=0$, the identity $\mathbb{E}\left[ 2^{X_n} \right]  = n + 1$ holds.

  Step: Suppose $\mathbb{E}\left[ 2^{X_n} \right]  = n + 1$ holds, then

  $$
  \begin{array}{l}
  \mathbb{E}\left[2^{X_{n+1}}\right]
  &=\sum_{j=0}^{\infty} \operatorname{Pr}\left(X_{n}=j\right) \cdot \mathbb{E}\left[2^{X_{n+1}} \mid X_{n}=j\right] \\
  &=\sum_{j=0}^{\infty} \operatorname{Pr}\left(X_{n}=j\right) \cdot\left(2^{j} \cdot \mathbb{P}[ \text{not increase } X_n \mid X_n = j ] +2^{j+1} \cdot \mathbb{P}[ \text{increase } X_n \mid X_n = j ]\right) \\
  &=\sum_{j=0}^{\infty} \operatorname{Pr}\left(X_{n}=j\right) \cdot\left(2^{j} \cdot\left(1-\frac{1}{2^{j}}\right)+2^{j+1} \cdot \frac{1}{2^{j}}\right) \\
  &=\sum_{j=0}^{\infty} \operatorname{Pr}\left(X_{n}=j\right) \cdot 2^{j}+\sum_{j=0}^{\infty} \operatorname{Pr}\left(X_{n}=j\right) \\
  &=\mathbb{E}\left[2^{X_{n}}\right]+1 \\
  &=(n+1) + 1
  \end{array}
  $$

  :::

- Good concentration $\mathbb{P}[ \left\vert \hat{n} - n \right\vert \ge \epsilon n] \le \frac{1}{2\epsilon^2}$

  :::{admonition,dropdown,seealso} *Proof*

  Recall Chebyshev's inequality

  $$
  \operatorname{\mathbb{P}}(|X-\mu| \geq \epsilon \mu ) \leq \frac{\sigma ^2}{(\epsilon \mu)^{2}}
  $$

  It can be shown that $\mathbb{E}\left[2^{2 X_{n}}\right]=\frac{3}{2} n^{2}+\frac{3}{2} n+1$, hence,

  $$\operatorname{Var}\left( \hat{n} \right) = \operatorname{Var}\left( 2^{X_n} \right) = \mathbb{E}\left[2^{2 X_{n}}\right] -  \mathbb{E}\left[2^{X_{n}}\right]^2 =\frac{n^2}{2} - \frac{n}{2}$$

  Substitute this into Chebyshev's inequality gives

  $$
  \operatorname{\mathbb{P}}(|\hat{n}-n| \geq \epsilon n ) \leq \frac{1}{(\epsilon n)^{2}}\left( \frac{n^2}{2} - \frac{n}{2} \right) \le  \frac{n^2}{2(\epsilon n)^{2}} = \frac{1}{2\epsilon ^2}
  $$

  :::

However, the upper bound $\delta = \frac{1}{2\epsilon^2}$ is not very meaningful. We only have $\delta < 1$ when $\epsilon > 1$, for which we can simply return 0. Is there any way to find a smaller upper bound? Currently $\operatorname{Var}\left( \hat{n} \right) = \Theta(n^2)$, can we find another estimate that does better?

### Morris+

Run in parallel: $s$ independent instantiations of Morris estimator $\hat{n}$, then return the average, denoted $\hat{n}_+$. Hence

$$\operatorname{Var}\left( \hat{n}_+ \right) = \operatorname{Var}\left( \frac{1}{s} \sum_{k=1}^s \hat{n}_k \right) = \frac{1}{s} \operatorname{Var}\left( \hat{n} \right)$$

Note that $\mathbb{E} [\hat{n}_+]$ is still unbiased. The upper bound now becomes $\frac{1}{s}\cdot \frac{1}{2\epsilon^2}$. How to choose $s$? To ensure failure probability is less than $\delta$, we can set $s \ge \frac{1}{2\epsilon ^2 \delta}$. Equivalently, we say that we can set $s = \Theta \left( \frac{1}{\epsilon^2 \delta}  \right)$, for suitable constant.

In total there are $s$ number of $X$, each $X$ takes $\mathcal{O} (\log \log n)$. The total space for an Morris++ estimator $\hat{n}$ with $(\epsilon, \delta)$ guarantee is then $\Theta\left( \frac{1}{\epsilon^{2} \delta} \cdot \log \log n\right)$.

We can further improve the result by using Morris++.

### Morris++

Run in parallel: $t$ independent instantiations of Morris+ estimator $\hat{n}_{+}$ with failure probability at most $\frac{1}{3}$, by setting $s = \Theta \left( \frac{1}{\epsilon^2\frac{1}{3} }  \right) =  \Theta \left( \frac{1}{\epsilon^2} \right)$. Then return their median, denoted as $\hat{n}_{+ +}$.

What's the failure probability of the median $\hat{n}_{+ +}$? If it fails, then at least half of $\hat{n}_{+}$ fails, i.e. at most half succeed. Formally, define

$$
Y_{i}=\left\{\begin{array}{ll}
1, & \text { if the } i \text {-th Morris+ instantiation succeeds. } \\
0, & \text { otherwise }
\end{array}\right.
$$

Let $S = \sum_{i=1}^t Y_i$. Then, the failure probability is

$$\begin{aligned}
\mathbb{P}\left( \hat{n}_{+ +} \text{ fails}  \right)
&= \mathbb{P}\left( S \le \frac{t}{2}   \right) \\
&\le \mathbb{P}\left( S \le \frac{3}{4} \mu_S   \right) \quad \because \mu_S \ge \frac{2t}{3} \\
&\le e^{-(1/4)^2 \mu_S/2} \quad \text{by Chernoff Bound} \\
&\le e^{- t/48} \\
\end{aligned}$$

To get the failure probability at most $\delta$, we can set $t = \Theta(\log \frac{1}{\delta})$.

In total there are $t \times s$ number of $X$. The total space of Morris++ with success probability $\delta$ is then $\Theta\left(\log \frac{1}{\delta} \cdot \frac{1}{\epsilon^{2}} \cdot \log \log n\right)$. In particular, for constant $\epsilon, \delta$ (say each 1/100), the total space complexity is $\mathcal{O} (\log \log n)$ with constant probability. This is exponentially better than the $\mathcal{O} (\log n)$ space achieved by storing a counter.


## Estimate $\ell_2$

AMS algorithm.

maintain a single number $z = \sum_{i=1}^n s_i x_i$.

2-wise independent, 4-wise independent??

$k$-wise independent requires $k\log m$ bits.

Consider turnstile model: a stream of $n$ input $(i, \Delta)$ to update $x_i \mathrel{+}=\Delta$. We want to approximate $\ell_2 = \left\| \boldsymbol{x}  \right\| _2$. In statistics, this is related to the second order moment of $X$, which describe how disperse the distribution is.

We introduce Alon-Matias-Szegedy’96 (AMS) Algorithm to estimate $\ell_2 ^2 = \left\| \boldsymbol{x}  \right\| _2 ^2$.
For another method Johnson-Lindenstrauss (JL) Lemma, see [here](https://www.sketchingbigdata.org/fall17/lec/lec3.pdf). For $\ell_p$-norm approximation, see here [https://www.sketchingbigdata.org/fall17/lec/lec4.pdf].

### AMS

Let $s_i \in \left\{ -1, 1 \right\}$ with equal probability.

---
**Algorithms**: (part of) Alon-Matias-Szegedy’96 (AMS)

---
- Generate $s_1, \ldots, s_m$ for each coordinates. Initialize $Z=0$.
- Upon $\texttt{add}(i, \Delta)$, update $Z \mathrel{+}= s_i \cdot \Delta$
- Upon $\texttt{query()}$, output $Z^2$

---


:::{admonition,tip} Intuition

- $Z = \sum_{i=1}^m s_i x_i = \boldsymbol{s} ^{\top} \boldsymbol{x}$ where $s_i = \pm 1$ with equal probability.

:::


Correctness

- $\mathbb{E} [Z^2] =\left\| \boldsymbol{x}  \right\| _2 ^2$

  :::{admonition,dropdown,seealso} *Proof*

  $$
  \mathbb{E}\left(Z^{2}\right)=\mathbb{E}\left(\boldsymbol{x}^{T} \boldsymbol{s} \boldsymbol{s}^{T} \boldsymbol{x}\right)=\boldsymbol{x}^{T} \mathbb{E}\left(\boldsymbol{s} \boldsymbol{s}^{T}\right) \boldsymbol{x}=\boldsymbol{x}^{T} \boldsymbol{x}=\|\boldsymbol{x}\|_{2}^{2}
  $$

  where we use $s_i \perp s_j, \mathbb{E} [s^2_i]$. In fact, 2-wise independence is enough.

  :::

- Good concentration $\operatorname{Pr}\left[\left|Z^{2}-\ell_2^2\right| \geq \sqrt{2}c \ell_2^2\right] \le \frac{1}{c^{2}}$

  :::{admonition,dropdown,seealso} *Proof*

  We use $\operatorname{Var}\left(Z^{2}\right)=\mathbb{E}\left[Z^{4}\right]-\mathbb{E}\left[Z^{2}\right]^{2}$

  To compute $\mathbb{E}\left[Z^{4}\right]$, note that for distinct $i,j,k,l$,

  $$\begin{aligned}
  x_{i} x_{j}^{2} x_{k} \mathbb{E}\left[s_{i} s_{j}^{2} s_{k}\right] &=0 \\
  x_{i} x_{j} x_{k} x_{l} \mathbb{E}\left[s_{i} s_{j} s_{k} s_{l}\right]&=0 \\
  \end{aligned}$$

  which can be shown using iterative conditioning. Thus,

  $$\begin{aligned}
  \mathbb{E}\left[Z^{4}\right] &=\mathbb{E}\left[\left(\sum_{i} s_{i} x_{i}\right)^{4}\right] \\
  & =\mathbb{E}\left[\sum_{i}\left(s_{i} x_{i}\right)^{4}\right]+3 \cdot \mathbb{E}\left[\sum_{i \ne j}\left(s_{i} s_{j} x_{i} x_{j}\right)^{2}\right]+0 \\
  &= \mathbb{E}\left[\sum_{i} x_{i}^{4}\right]+\mathbb{E}\left[\sum_{i \neq j}\left(x_{i} x_{j}\right)^{2}\right] \\
  &= \sum_{i} x_{i}^{4} + 3 \sum_{i \neq j}\left(x_{i} x_{j}\right)^{2} \\
  &\le 3 \left( \sum_{i} x_{i}^{4} + \sum_{i \neq j}\left(x_{i} x_{j}\right)^{2}  \right) \\
  &= 3 \left\| \boldsymbol{x}  \right\| _2 ^4
  \end{aligned}$$

  In sum, we have

  $$
  \operatorname{Var}\left(Z^{2}\right)
  \le 3 \left\| \boldsymbol{x}  \right\|_2 ^4 - \left\| \boldsymbol{x}  \right\|_2 ^4 \le 2 \left\| \boldsymbol{x}  \right\| _2 ^4
  $$

  By Chebyshev,

  $$
  \operatorname{Pr}\left[\left|Z^{2}-\|x\|_{2}^{2}\right| \geq \sqrt{2} \epsilon\|x\|_{2}^{2}\right] \leq 1 / \epsilon^{2}
  $$

  In fact, 4-wise independence is enough since the maximum degree of the polynomial in $s_i$ is 4.

  :::

This bound is too loose to be informative. The overall AMS algorithm keep multiple estimators $Z_1, \ldots, Z_k$. And output the average of squared values, $Z ^\prime = Z_1 ^2, \ldots, Z_k^2$. It is easy to see that

$$
\mathbb{E}\left[Z^{\prime}\right]=\mathbb{E}\left[\frac{\sum_{i} z_{i}^{2}}{k}\right]=\mathbb{E}\left[Z_{1}\right]=\|x\|_{2}^{2}
$$

and

$$
\operatorname{Var}\left(Z^{\prime}\right)=\operatorname{Var}\left(\frac{\sum_{i} Z_{i}^{2}}{k}\right)=\frac{\sum \operatorname{Var}\left(Z_{i}^{2}\right)}{k^{2}}=\frac{\operatorname{Var}\left(Z_{1}^{2}\right)}{k} \le \frac{2\|x\|_{2}^{4}}{k}
$$

Hence,

$$\operatorname{Pr}\left[\left|Z^{\prime}-\ell_{2}^{2}\right| \geq c \sqrt{2/k}\ell_{2}^{2}\right] \leq 1 / c ^{2}$$

If we set $c=\Theta(1)$ and $k=\Theta\left(1 / \varepsilon^{2}\right)$, we get a $(1 \pm \epsilon)$ approximation with constant probability.

Note that we need 4-wise independence of $s_1, \ldots, s_m$ to bound $\mathbb{E} \left[ \left( \sum_{i=1}^m s_i x_i \right)^2 \right]$, which requires $\mathcal{O} (\log m)$ random bits. ($p\log m$ random bits for $p$-wise independence). Besides, there are $\mathcal{O} (\frac{1}{\epsilon^2})$ number of $Z$'s, with maximum value $nm$, so total we need $\mathcal{O} (\frac{1}{\epsilon^2} \log(mn))$ bits.

We can formulate our algorithm in vector form. Let $\boldsymbol{S}$ be an $k \times n$ matrix where $s_{ij} = \pm 1$ with equal probability, then basically we are just maintaining the vector $\boldsymbol{Z} = \boldsymbol{S} \boldsymbol{x}$ throughout the algorithm and answering the query by outputting $\left\| \boldsymbol{Z}  \right\|_2 ^2 / k$. Here $\boldsymbol{Z}$ is a **linear sketch** $C(\boldsymbol{x})$ of $\boldsymbol{x}$.


### AMS++

```{margin}
Since AMS already use averaging, we denote AMS++ for taking median, as in FM++ and Morris ++.
```

To get a $(\epsilon, \delta)$ approximation, we run $t = \mathcal{O} (\log \frac{1}{\delta})$ independent instantiations of AMS and take the median, similar to FM++ and Morris++.

There are $\mathcal{O} (\log \frac{1}{\delta} \cdot \frac{1}{\epsilon^2} )$ number of $Z$'s, and $\mathcal{O} (\log m)$ random bits for sign $s$'s.


## Estimate $x_i$

(Review $k$-wise independent hash function [here](https://www.sketchingbigdata.org/fall17/lec/lec2.pdf))

Now we want to estimate the frequency $x_i$ of $i$.

Two algorithms:
- **CountMin**: keep track of all coordinates with additive error, i.e., for each coordinate report $\hat{x}_i$ that is within $x_{i} \pm \frac{\|x\|_{1}}{k}$
- **CountSketch**: keep track of all coordinates with additive error, i.e., for each coordinate report $\hat{x}_i$ that is within $x_{i} \pm \frac{\|x\|_{2}}{\sqrt{k}}$


See [here](https://www.mit.edu/~mahabadi/courses/Algorithms_for_Massive_Data_SP21/Lec2.pdf).


.


.


.


.


.


.


.


.
