# Causal Estimation

(Review the [randomized controlled trials](rct) section)

If we saw some association, e.g.
- positive association in level of ice cream production and number of drowning cases  
- positive association in percentage of population having lung cancer, and percentage of population that smoke in a city.

Can we determine if there exists causal relation among two variables? Or there is something we've missed?
- in the first case, there is a common cause: weather.
- in the second case, there may be a common cause: a gene that cause both lung cancer and addiction to smoking.

Causal estimation from observational data is not possible without some assumptions.

## Model

We consider a common cause $X$ to a treatment variable $T$ and an outcome variable $Y$.

$$\begin{aligned}
& X \\
\swarrow &\quad \searrow \\
A \quad &\longrightarrow Y \\
\end{aligned}$$

We consider a simple case: $A$ is binary, $X$ is discrete, and $(A_i, X_i, Y_i) \overset{\text{iid}}{\sim} P$ where $P$ is some unknown distribution. We want to estimate the average treatment effect, defined as

```{margin}
$\operatorname{do}(A=a)$ is some 'do' calculus to be introduced later.
```

$$
ATE = \mathbb{E}\left( Y \mid \operatorname{do}(A=1) \right) - \mathbb{E}\left( Y \mid \operatorname{do}(A=0) \right)
$$

Since $X$ is discrete, consider stratifying $X$ to find $\mathbb{E}\left( Y \mid A X=x \right)$. Then we can write

$$
\tau := \mathbb{E}_X \left[ \mathbb{E}\left( Y \mid A=1, X=x \right) \right] - \mathbb{E}_X \left[ \mathbb{E}\left( Y \mid A=0, X=x \right) \right]
$$


:::{admonition,note} Note

$\mathbb{E}_X \left[ \mathbb{E}\left( Y \mid A=1, X=x \right) \right] \ne \mathbb{E}\left( Y \mid A=1\right)$

Precisely, on LHS, $X\sim P$. On RHS, it equals $\mathbb{E}_{X \vert A=1} \left[ \mathbb{E}\left( Y \mid A=1, X=x \right) \right]$, i.e. $X$ is draw from $P(\cdot, A=1)$.

:::

$\tau$ can be calculated if we know the distribution $P$.

Theorem
: If there is no unobserved confounders other than $X$, and $P(A=1 \mid x) \in (0,1)$, then $\tau = ATE$.

How to estimate $\tau$? Use randomized control trials, which makes independence between treatment $A$ and common cause $X$. The causal graph change to

$$\begin{aligned}
& X \\
&\quad \searrow \\
A \quad &\longrightarrow Y \\
\end{aligned}$$


Define $Q(a, x) = \mathbb{E}\left( Y \mid a, x \right)$, then we can write

$$
\tau = \mathbb{E}_X \left[ Q(1, x) - Q(0, x) \right]
$$

## Backdoor Adjustment

Steps:

- fit a model $\hat{Q}$ for $Q$, i.e. some regression model that fits value $Y$ by two covariates $A$ and $X$, where $X$ is observed.
- compute $\hat{\tau} = \frac{1}{n} \sum_{i=1}^n \left[ \hat{Q}(1, x_i) - \hat{Q}(0, x_i) \right]$

How do we fit $\hat{Q}$? Recall that we can define mean as $\mu = \arg\min_a \mathbb{E}\left [( X - a \right)^2]$. We can consider mean squared error to select $Q$,

$$
Q^* = \arg\min_Q \mathbb{E} [( Y - Q(A, X) ^2]
$$

The true solution is $Q^*(a, x) = \mathbb{E}\left( Y \mid a, x \right)$.

For instance, if we assume the underlying model is a linear model, we use

$$
Q(A, x) = \beta_0 + \beta_A A + \beta_x x
$$

Then its easy to see $\tau = \beta_A$ and $\hat{\tau} = \hat{\beta}_A$.

## Propensity Score

Define propensity score $g(x) = \mathbb{P} (A=1 \mid x)$, we have the identity

$$
\tau= \mathbb{E} \left[Y \frac{A}{g(X)}- Y \frac{1-A}{1-g(X)}\right]
$$

Hence we can fit $\hat{g}$, then an plug-in estimator is

$$
\hat{\tau}_g := \frac{1}{n} \sum_i \left( y_i \frac{A_i}{\hat{g}(x_i)} - y_i \frac{1-A_i}{1-\hat{g}(x_i)}  \right)
$$

How to estimate $\hat{g}$? We can use any proper scoring rule (loss function), e.g. cross Entropy.

## Question

Consider the causal graph

$$\begin{aligned}
X_A \quad &X_{A \land Y} \quad X_Y\\
\downarrow  \swarrow &\qquad \searrow \quad \downarrow\\
A\quad &\longrightarrow \qquad Y \\
\end{aligned}$$

where $X_A$ are the factors affect $A$ only, $X_Y$ are the factors affect $Y$ only, and $X_{A \land Y}$ are the factors affect both $A$ and $Y$.

- To estimate $Q$, we use $X_Y$ and $X_{A \land Y}$.
- To estimate $g$, we use only $X_{A \land Y}$. Why don't include $X_A$: If we include $X_A$, then we are more certain about $g(x)=\mathbb{P}\left( A=1 \mid x \right)$, i.e. the propensity score approaches 0 or 1. Hence as a denominator, $\tau$ becomes unstable.


AIPTW:


$$
\tau_{AIPW}:= \frac{1}{n} \sum_{i=1}^{n}\left[\hat{Q}\left(1, x_{i}\right)-\hat{Q}\left(0, x_{i}\right) + a_i \frac{y_i -\hat{Q}\left(1, x_{i}\right) }{\hat{g}(x_i)} - (1-a_i) \frac{y_i-\hat{Q}\left(0, x_{i}\right)}{1-\hat{g}(x_i)}\right]
$$

How is this better? Consider influence curve of $\tau$:

$$
\phi(x_i, a_i, y_i; Q, g, \tau) = Q(1, x_i) - Q(0, x_i) + a_i \frac{y_i - Q(1, x_i)}{g(x_i)} - (1- a_i) \frac{y_i - Q(0, x_i)}{1- g(x_i)}  - \tau
$$

Notice:

$$
\hat{\tau}_{AIPW} - \tau = \frac{1}{n} \sum_i \phi(x_i ; \hat{Q}, \hat{g}, \tau) .
$$

Consider a best case: $\hat{Q} = Q, \hat{g}=g$, then by CLT

$$
\sqrt{n}(\hat{\tau}_{AIPW*} - \tau) \rightarrow \mathcal{N} \left( 0, \mathbb{E}\left( \sum \phi^2 \right) \right)
$$

By some algebra,


$$\begin{aligned}
\sqrt{n}(\hat{\tau}_{AIPW*} - \tau) &= \frac{1}{\sqrt{n}} \sum_{i=1}^n \phi(x_i; Q,g, \tau) \\
&+  \frac{1}{\sqrt{n}} \sum_{i=1}^n \left[ \phi(x_i; \hat{Q}, \hat{g}, \tau) - \phi(x_i; Q,g, \tau)- \mathbb{E}\left(  \right) \right] \\
&= \\
&= \\
&= \\
\end{aligned}$$


- First term is ideal case error.
- Second term is data reuse penalty: error due to reuse of data. This converges to 0 quickly, so not a big problem.
- 3rd term is model misspecification error.

ref: https://si.biostat.washington.edu/sites/default/files/modules/chapter1.pdf


## Causality Graph

Assumptions
1. there exists an DAG describing causal relations
1. Markov property holds: In the following triple, $Y$ and $Z$ are conditional independent given $X$

  $$\begin{aligned}
  & X \\
  \swarrow &\quad \searrow \\
  Y \quad &\qquad Z \\
  \end{aligned}$$

1. Faithfulness: the conditional dependency in variables $\Leftrightarrow$ the conditional dependency in DAG.

How to formalize causal effect?

:::{admonition,note} vs Bayesian graphical models

Bayesian graphical models encode factorization of conditional distributions.


:::

### Graph Surgery

Given a graph $G$, we want to estimate $\mathbb{P} (Y \vert A)$, but $A$ may have other parents. A graph surgery is saying that, we remove edges between $A$ and its parents $\operatorname{pa} (A)$, obtain a new graph $G ^\prime$ estimate $\mathbb{P} _{G ^\prime }(Y \vert A)$, denoted as $\mathbb{P} (Y \vert \operatorname{do}(A) )$.

How to compute it?
- If $A$ has no parents, then done.
- Else
  - conduct RCT, if possible, or
  - estimate by $\mathbb{E} [Y \vert \operatorname{do}(A) ] := \mathbb{E} _{\mathbb{P} (Y \vert \operatorname{do}(A))  } [Y \vert A]$, or $\mathbb{E} [Y \vert \operatorname{do}(A=a) ] - \mathbb{E} [Y \vert \operatorname{do}(A=a ^\prime) ]$

What if there is more than 1 causal model that is consistent with the observation data? If so, the causal model is not identifiable.

Causal effect is always identified if we see all variables in the graph.

### Backdoor Adjustment

- Backdoor path: an undirected path between $A$ and $Y$ with an arrow into $A$.
- Backdoor criteria for $S \in X \setminus \left\{ A, Y \right\}$
  - $S$ blocks all backdoor paths between $Y$ and $A$
  - $S$ contains no descendants of $A$

$$\begin{aligned}
V \rightarrow  S \\
\downarrow \qquad \downarrow \\
A \rightarrow Y \\
\end{aligned}$$

- Theorem: if $S$ satisfies the above criteria, then $\mathbb{P} (Y \mid \operatorname{do}(A) )  = \mathbb{E} _S [P (Y \mid A, S)]]$.

### Front door

FD Criteria
: 

.


.


.


.


.


.


.
