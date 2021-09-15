# Data Issues


Reference
- preprocessing [link](https://mp.weixin.qq.com/s/SCmCY3joCmn6FJcKYwtlwg?utm_medium=email&_hsmi=120495215&_hsenc=p2ANqtz-8IbhySDq8KZwn2MNO0fqXLg0vL7SYTAIiAOHDsWSV5An-vlSjm7VtKDbhA7A-9nsS-IfCqqcJ3Wvs5DNLzgl4I8XCt5A&utm_content=120495215&utm_source=hs_email)


(missing-values)=
## Missing Values

### Types

An entry $x_{ij}$ can be missing due to various reasons.

#### Completely at Random

Missing completely at random (MCAR) means for each variable $j$, every entry is equally likely to be missing

$$
\operatorname{P} (x_{ij} \text{ is missing} ) = p_j
$$

Then we have a smaller sample. This will increase the standard errors of estimators (lower precision), but it does not cause bias.

#### At Random

Missing at random (MAR) means that the probability of missing can also depend on some attributes of the subject, say other values $x_{i, -j}$

$$
\operatorname{P} (x_{ij} \text{ is missing} ) = f(x_{i, -j})
$$



#### Not at Random

Missing not at random (MNAR) means that the probability of missing can depends on some unobservable variables $Z_{ij}$

$$
\operatorname{P} (x_{ij} \text{ is missing} ) = f(z_{ij})
$$

#### Depends on Response

The probability of missing depends on the value of $y$.

In this case, in the missing data, the relation estimated from the observed data may not hold.

### Imputation

Imputation means how we fill the missing entries.

#### Drop

Simple drop observation $i$ if any entry $x_{ij}$ is missing. This is acceptable when in MAR, MCAR and when the missing is infrequent.

#### By Mean or Mode

We can impute $x_{ij}$ by the column mean $\bar{x}_{\cdot j}$ or column mode. But if $x_{ij}$ is deterministic on other variables, then after imputation this dependent relation does not hold.

#### By Regression

Suppose $X_j$ is deterministic on other explanatory variables $X_{-j}$, we can estimate this relation by regression $x_j$ over all other explanatory variables to maintain this dependent relation in the imputed data.


:::{admonition,note} Side-effect

Clearly, this method increases [multicollinearity](lm-multicollinearity) among the variables, measured by $\operatorname{VIF}_j$. In general, in linear regression, after imputation

- $\left\vert \hat{\beta}_j \right\vert$ decreases
- $\hat{\sigma}^2$ increases

:::

### By EM Algorithm

We treat the missing entires as latent variables, and use EM algorithm to impute the values.
1. Make initial guess of missing values
2. Iterate
    - Find maximum likelihood estimates of the parameters $\theta$ of the assumed joint distribution of the variables, using all data $(x_{miss}, x_{obs})$
    - Update missing values $x_{miss}$ by conditional expectation $\mathbb{E} [x_{miss} \mid x_{obs}, \theta]$


## Imbalanced Data

- up/down sampling to make them balance in the data set
  - Synthetic Minority Oversampling Technique (SMOTE)
- up/down weighting in the loss function

## Normality

Some model or methods assume normality of data. If some variables are not from normal distribution, we can try to [transform](transform-normality) them to normal.

## Standardization

When will we use standardization?
- For algorithms using Euclidean distance, e.g. k-means
- For dimension reduction method involves variance, e.g. PCA
- For gradient descent, to reduce noise in the trajectory
