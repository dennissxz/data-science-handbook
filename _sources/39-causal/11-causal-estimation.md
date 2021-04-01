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

Since $X$ is discrete, consider stratifying $X$ to find $\mathbb{E}\left( Y \mid A=1, X=x \right)$. Then we can write

$$
\tau := \mathbb{E}_X \left[ \mathbb{E}\left( Y \mid A=1, x \right) \right] - \mathbb{E}_X \left[ \mathbb{E}\left( Y \mid A=0, x \right) \right]
$$

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

Backdoor adjustment:
- fit a model $\hat{Q}$ for $Q$, i.e. some regression model that fits value $Y$ by two covariates $A$ and $X$
- compute $\hat{\tau} = \frac{1}{n} \sum_{i=1}^n \left[ \hat{Q}(1, x_i) - \hat{Q}(0, x_i) \right]$

How do we fit $\hat{Q}$?

.


.


.


.


.


.


.


.
