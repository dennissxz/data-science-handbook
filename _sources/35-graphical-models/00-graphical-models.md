# Graphical Models

Graphical models us graphical language for representing and reasoning about random variables and their distributions. The edge represents dependence structure.

For instance, hair length and height are not independent, since there is a confounding variable of gender. Given gender, they are independent. We can draw a graph to represent this.

$$
\text{gender } G
$$

$$
\swarrow \qquad \searrow
$$

$$
\text{hair length } L  \boldsymbol \quad \overset{\quad c \quad }{\longrightarrow} \quad \text{height } H \qquad
$$


$$
\begin{array}{ll}
p(h \mid l) \neq p(h) & H \not \perp L \\
p(h \mid l, g)=p(h \mid g) & H \perp L \mid G
\end{array}
$$

## Bayesian Networks

Bayesian networks is an **acyclic** directed graphical models.

- Each node is a variable $X_i$, and its parents are denoted by $\pi(X_i)$
- Local probability function $P(X_i \mid \pi(X_i))$

### Factorization

In a Bayesian network, the joint probability can be factorized as

$$
P\left(X_{1}, \ldots, X_{n}\right)=\prod_{i=1}^{n} P\left(X_{i} \mid \pi\left(X_{i}\right)\right)
$$

:::{figure} gm-joint-prob

<img src="../imgs/gm-joint-prob.png" width = "70%" alt=""/>

Joint probability as factorization
:::

How many numbers needed to encode the probabilities?

Suppose $a,b,c,d$ are discrete and each have 10 values
- representing $p(a)$ requires 10 numbers (actually 9)
- representing $p(b \vert a)$ requires 100 numbers

So without factorization, to represent $p(a,b,c,d)$ we need $1e4$ numbers. But with factorization, we only need $1e1 + 1e2 + 1e3 + 1e4 = 1210 \ll 1e4$. That's the advantage of factorization.
