# Application to Density Fitting

## Discriminators and Generators

### Discriminator

Consider two $p$-dimensional random variables $X \sim P_X, Y \sim P_Y$, we want to find a discriminator function $D$ to discriminate the two distributions, by the objective

$$
\max_{D}\ \mathbb{E} _{X \sim P_X}[D(X)] -  \mathbb{E} _{Y \sim P_Y}[D(Y)]
$$

where $D$ satisfies certain constraints, e.g. the above cost is bounded.

### Generators

Consider a random variable $X \sim P_X$, where $P_X$ is the 'push-forward' measure of base measure $P_0$ on $\mathcal{Z}$, via transformation $T: \mathcal{Z} \rightarrow \mathcal{X}$.

$$
P_X(\left\{ x \in A \right\}):= P_0 (T ^{-1} (A))
$$

In this case, $T$ is a generator. Usually spaces $\mathcal{Z}$ and $\mathcal{X}$ are high-dimensional. Benefits of representing $P$ as a pushforward measure is that we can calculate expectation of $X$ efficiently.

We can represent $D$ and $T$ as neural networks $D_\phi$, $T_\psi$ with some parameters $\phi, \psi$.

### Normalizing Flow

If $T$ is invertible, then

$$
P_X(x) = P_0 (T ^{-1} (x)) \left\vert \operatorname{det}   J _{T ^{-1}} (x) \right\vert
$$

where $J _{T ^{-1}}$ is the Jacobbian of $T ^{-1}$.

To sample from $\mathcal{X}$, we can sample from $\mathcal{Z}$. To comput the expectation of a function $f(X)$, we can use the approximation

$$
\mathbb{E} _{X \in P_X}[f(X)] \approx \sum_{Z_i \in P_0} f(T(Z_i))
$$



## Density fitting

Given a sample distribution $P_X$, we want to fit it by $P_T$, by minimizing the distance between them. The distance measure can be [KL diverggence](kl-divergence), or [1-Wasserstein distance](wasserstein-distance).

Note that we [can](https://www.probabilitycourse.com/chapter4/4_3_2_delta_function.php) write $P_X(x)$ as (generalize) density using delta function and the observations $x_1, x_2, \ldots, x_n$.

$$
P_X(x) = \sum_{i=1}^n \delta_{x_i}(x)
$$

where $\delta_{x_i}(x) = \delta(x - x_i)$. An useful property of delta function is that $\int \delta_{x_i}(x) g(x)\mathrm{~d} x = g(x_i)$.

### By KL Divergence

The objective is

$$
\min_{T}\ \operatorname{KL}(P_X, P_T)
$$

where

$$\begin{aligned}
\operatorname{KL}(P, P_T)
&= \mathbb{E} _{X \in P}[\log P(X)] - \mathbb{E} _{X \in P}[\log P_T(X)]\\
\end{aligned}$$

The first term is a constant, hence the problem is to minimize the second term, which is

$$\begin{aligned}
- \mathbb{E} _{X \in P}[\log P_T(X)]
&= - \int P_X(x) \log P_T(x)\mathrm{~d} x\\
&= - \sum_{i=1}^n \int \delta_{x_i}(x) \log P_T(x)\mathrm{~d} x\\
&= - \sum_{i = 1}^m \log \left[ P_T(x_i) \right] \\
&= - \sum_{i = 1}^m \log \left[ P_0 (T ^{-1} (x_i)) \left\vert \operatorname{det}  J _{T ^{-1}} (x) \right\vert \right] \\
\end{aligned}$$

The last equality is from normalizing flow. We can then represetn the function $T ^{-1}$ by a NN and solve this optimization problem by SGD.

### By 1-Wasserstein Distance

To approximate $P$ by $P_T$, the objective is

$$
\min_{T}\ W_1 (P, P_T)
$$

where

$$
W_1(P, P_T)
= \inf_{\pi \in \mathcal{\Pi} } \int \left\| x - y \right\| \pi(x, y) \mathrm{~d}x \mathrm{~d} y\\
$$

It can be [shown](wasserstein-dual) that the dual form is

$$\begin{aligned}
W_1 (P, P_T)= \sup_{D \text{ is 1-Lipschitz} } \int D (x) P(x) \mathrm{~d} x -  \int D (y) P_T(y) \mathrm{~d} y  \\
\end{aligned}$$

We can then parameterize $D \leftarrow D_\phi$, $T \leftarrow T_\psi$ by NN. Substituting the generalized density function $P(x) = \sum_{i=1} \delta _{x_i}(x)$ and $P_{T_\psi}(x) = \sum_{i=1} \delta _{T_\psi(z_i)}(x)$, the problem becomes

$$
\min_{\psi}\ \max_{\phi}\ \sum_i D_\phi (x_i) - \sum_i D_\phi (T_\psi(z_i))  \\
$$

The parameters $\phi$ and $\psi$ can be updated alternatively.

:::{admonition,note} vs MCMC

If we parameterize the function $P_X(x)$ by $P_\theta (x)$, e.g. $P_\theta(x) = \sigma(\boldsymbol{\theta} ^{\top} x + \theta_0)$, to sample from $P_\theta (x)$, we need MCMC, but it is hard to know when it converges.

Using a generator to approximate $P_X$ is a popular alternative recently. We can efficiently sample from $\mathcal{Z}$, and use normalizing flow to compute $\mathbb{E} _{X \sim P_X}[X]$.

:::

## Scientific Applications

### Learning

Suppose there is a physical process model by PDE with some parameter $a$. Given $a$, there is a solution $u_a$. We want to learn a forward mapping $F: a \rightarrow u_a$, or a backward mapping $B: u_a \rightarrow a$.

Applying NN, we can parameterize $F \leftarrow F_\phi$, $B \leftarrow B_\psi$.

Current research problem:
- how to specify the NN architecture?
- generalizable? scalable?

### Solving

Suppose there is some high-dimensional PDE, e.g. Fokker-Planck, many body shrodinger, etc. We can parameterize some functions therein as NN.
