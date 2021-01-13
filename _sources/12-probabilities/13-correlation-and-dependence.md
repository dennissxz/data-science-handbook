# Correlation and Dependence

## Definitions

### Correlation

### Dependence

## Comparison

Does correlation imply dependence? Or vice versa?

## Simpson's Paradox

We have talked about the dependence between two random variables, which says that the information of a random variable $X$ depends on the information of another variable $Y$. In fact, here "the information of $X$" can be generalized to any probablistic objects, such as
- the distribution of a random vector $\boldsymbol{X}=\left[ X_1, \ldots, X_2 \right]$
- the correlation of two random variables $\mathrm{Corr}\left( X_1, X_2 \right)$
- some statistical conclusions.

Simpson's Paradox refers to this case. It is a phenomenon in probability and statistics, in which a trend appears in several different groups of data but disappears or reverses when these groups are combined.

The cause of Simpson's paradox is the existence of a confounding variable, which has dependence with the observed variables.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/f/fb/Simpsons_paradox_-_animation.gif" width="50%" align="center"/>
</p>



Mathematically, if the trends are correlations, it can be formulated as

$$\mathrm{Corr}(X,Y\,\vert\, Z) > 0 \not \Rightarrow \mathrm{Corr}(X,Y) > 0$$

where $Z$ is a confounding variable.

If the trends are proportions, we can illustrate them with vectors. In the below case, for a vector $v$, suppose the angle between it and the $x$-axis is $\theta$. The horizontal projection $\vert \overrightarrow{v}\cos\theta\vert$ is the number of applicants and the vertical projection $\vert \overrightarrow{v}\sin\theta\vert$ is the number of accepted candidates. A steeper vector then represents a larger acceptance rate.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/c/cd/Simpsons_paradox.jpg" width="50%" />
</p>


The Simpson's paradox says that, even if $\overrightarrow{L_{1}}$ has a smaller slope than $\overrightarrow{B_{1}}$ and $\overrightarrow{L_{2}}$ has a smaller slope than $\overrightarrow{B_{2}}$, the overall slop $\overrightarrow{L_{1}} + \overrightarrow{L_{2}}$ can be greater than $\overrightarrow{B_{1}}+\overrightarrow{B_{2}}$. For this to occur, one of the orange vectos must have a greater slope than one of the blue vectors (e.g. $\overrightarrow{L_{2}}$ vs. $\overrightarrow{B_{1}}$), and these will generally be **longer** than the alternatively subscripted vectors (i.e. imbalanced data) – thereby dominating the overall comparison.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Simpson_paradox_vectors.svg/1200px-Simpson_paradox_vectors.svg.png" width="50%">
</p>


## Exercise

### Raining

Tags: Jane Street, Quant, 20Q4

*Suppose the probabilities of raining on Saturday and Sunday are $p$ and $q$ respectively. What is the probability of raining on weekend? What is the probability that it rains on either Saturday or Sunday?*


Note that the question does not specify the dependence of raining on Saturday and Sunday. To be rigorous, we introduce two indicator variables


$$\begin{align}
X &= \mathbb{I}[\text{raining on Saturday}] \\
Y &= \mathbb{I}[\text{raining on Sunday}] \\
\end{align}$$


Hence

$$\begin{align}
\mathrm{P}(X=1) &= p\\
\mathrm{P}(Y=1) &= q\\
\end{align}$$


Suppose $\mathrm{P}(X=1, Y=1)=a$, then the contingency table is

|$X \ \backslash \ Y$| $0$ | $1$| total |
|:-: | :-:| :-:| :-: |
|$0$ | $1-p-q+a$|   $q-a$    | $1-p$ |
|$1$   |  $p-a$ | $a$  | $p$|
|total  | $1-q$  | $q$  | $1$  |

Note that there are constraints on $a$:

$$\begin{align}
1-p-q+a &\ge 0 \\
p-a & \ge 0 \\
q-a & \ge 0
\end{align}$$

So we can obtain the valid range for $a$:

$$\max(p+q-1,0) \le a \le \min(p,q)$$

The required answer is  

$$\begin{align}
p_1&= \mathrm{P}(\text{raining on weekend})  \\
& = 1 - \mathrm{P}(X=0, Y=0)  \\
 & = p+q-a \\
 & \in [\max(p,q), \min(p+q,1)] \\
p_2 &= \mathrm{P}(\text{raining on either Saturday or Sunday})  \\
& = \mathrm{P}(X=1, Y=0) + \mathrm{P}(X=0, Y=1)  \\
 & = p+q-2a \\
 & \in [\vert p-q\vert, \min(p+q, 2-p-q)]
\end{align}$$


### Expected Value of Maximum of Two Uniform Random Variables

*Suppose $X$ and $Y$ are two uniformly distributed random variables over the interval $[0,1]$. What is the expected value $\mathrm{E}[\max(X,Y)]$?*


Let $Z=\max(X,Y)$. Since there is no dependence specified, we start from the special cases.

- If $X$ and $Y$ are independent, then

    $$\begin{align}
    \mathrm{P}(Z\le z) &= \mathrm{P}(\max (X, Y) \le z) \\
    &=\mathrm{P}(X \leqslant z) \mathrm{P}(Y \leqslant z) \\
    &= z^2 \\
    \mathrm{E}(Z) &= \int_{0}^{1}\mathrm{P}(Z\ge z)\mathrm{d}z \\
     &= \int_{0}^{1}\left(1-z^{2}\right) \mathrm{d}z \\
    &=\frac{2}{3}
    \end{align}$$

    Another way without finding the cumulative distribution function $\mathrm{P}\left( Z\le z \right)$:

    $$
    \begin{aligned}
    \mathrm{E}(\max (x, y)) &=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max (x, y) p(x, y) \,\mathrm{d}x \,\mathrm{d}y \\
    &=\int_{0}^{1} \int_{0}^{1} \max (x, y) \,\mathrm{d}x \,\mathrm{d}y \\
    &=\int_{0}^{1} \int_{0}^{x} x \,\mathrm{d}y \,\mathrm{d}x+\int_{0}^{1} \int_{0}^{y} y \,\mathrm{d}x \,\mathrm{d}y \\
    &=\int_{0}^{1} x^{2} \,\mathrm{d}x+\int_{0}^{1} y^{2} \,\mathrm{d}y \\
    &=\left[\frac{x^{3}}{3}\right]_{0}^{1}+\left[\frac{y^{3}}{3}\right]_{0}^{1} \\
    &=\frac{1}{3}+\frac{1}{3} \\
    &=\frac{2}{3}
    \end{aligned}
    $$

    In particular, for $n$ independent uniform random variables,

    $$\begin{align}
    \mathrm{E}(Z) &= \int_{0}^{1}\mathrm{P}(Z\ge z)\mathrm{d}z\\
    &= \int_{0}^{1}\left(1-z^{n}\right) \mathrm{d}z\\
    &= \frac{n}{n+1}\\
    \end{align}$$


- If $X$ and $Y$ has the relation $X=Y$, then

    $$\mathrm{E}(Z)=\mathrm{E}(X)=\frac{1}{2}$$

  In this case

  $$\begin{align}
  \mathrm{Corr}\left( X,Y \right) &= \frac{\mathrm{Cov}\left( X,Y \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( Y \right)}} \\
  &= \frac{\mathrm{Cov}\left( X,X \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( X \right)}} \\
  &= \frac{\mathrm{Var}\left( X \right)}{\mathrm{Var}\left( X \right)}\\
  &= 1
  \end{align}$$


- If $X$ and $Y$ has the relation $X+Y=1$, then by the law of total expectation

    $$\begin{align}
    \mathrm{E}(Z) &=\mathrm{E}[\mathrm{E}(Z \,\vert\, X)]\\
    &=\mathrm{\mathrm{P}\left( X\le \frac{1}{2} \right)} \cdot \mathrm{E}\left( 1-X\,\vert\, X\le \frac{1}{2}  \right) + \mathrm{P}\left( X> \frac{1}{2} \right) \cdot \mathrm{E}\left( X\,\vert\, X > \frac{1}{2}\right)\\
     &= \frac{1}{2} \times \frac{3}{4}  + \frac{1}{2} \times \frac{3}{4}  \\
    &=\frac{3}{4}
    \end{align}$$

  In this case

  $$\begin{align}
  \mathrm{Corr}\left( X,Y \right) &= \frac{\mathrm{Cov}\left( X,Y \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( Y \right)}} \\
  &= \frac{\mathrm{Cov}\left( X,1-X \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( 1-X \right)}} \\
  &= \frac{\mathrm{-Var}\left( X \right)}{\mathrm{Var}\left( X \right)}\\
  &= - 1
  \end{align}$$


It seems that the range is $[\frac{1}{2}, \frac{3}{4}]$.
