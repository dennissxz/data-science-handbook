# Missing Values

## Types

An entry $x_{ij}$ can be missing due to various reasons.

### Completely at Random

Missing completely at random (MCAR) means for each variable $j$, every entry is equally likely to be missing

$$
\operatorname{P} (x_{ij} \text{ is missing} ) = p_j
$$

Then we have a smaller sample. This will increase the standard errors of estimators (lower precision), but it does not cause bias.

### At Random

Missing at random (MAR) means that the probability of missing can also depend on some attributes of the subject, say other values $x_{i, -j}$

$$
\operatorname{P} (x_{ij} \text{ is missing} ) = f(x_{i, -j})
$$



### Not at Random

Missing not at random (MNAR) means that the probability of missing can depends on some unobservable variables $Z_{ij}$

$$
\operatorname{P} (x_{ij} \text{ is missing} ) = f(z_{ij})
$$

### Depends on Response

The probability of missing depends on the value of $y$.

In this case, in the missing data, the relation estimated from the observed data may not hold.

## Imputation

Imputation means how we fill the missing entries.

### Drop

Simple drop observation $i$ if any entry $x_{ij}$ is missing. This is acceptable when in MAR, MCAR and when the missing is infrequent.  

### By Mean

We can impute $x_{ij}$ by the column mean $\bar{x}_{\cdot j}$. But if $x_{ij}$ is deterministic on other variables, then after imputation this dependent relation does not hold.

### By Regression

Suppose $X_j$ is deterministic on other explanatory variables $X_{-j}$, we can estimate this relation by regression $x_j$ over all other explanatory variables to maintain this dependent relation in the imputed data.



:::{admonition,note} In linear regression

Clearly, this method increases $\operatorname{VIF}_j$.

In general, after imputation

- $\left\vert \hat{\beta}_j \right\vert$ decreases
- $\hat{\sigma}^2$ increases

:::
