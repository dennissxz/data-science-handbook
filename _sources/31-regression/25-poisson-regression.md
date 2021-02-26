# Poisson Regression

Aka log-linear models.

Poisson regression is used to model count response.

## Data

Counts data are common. Such as contingency tables. It can be modeled by the Poisson distribution.

One can also use data transformation and then fit OLS. Suppose the underlying distribution is Poisson, the by the Delta method, we have $g(Y)-g(\mu) \approx(Y-\mu) g^{\prime}(\mu)$, hence $\operatorname{Var}[g(Y)] \approx\left[g^{\prime}(\mu)\right]^{2} \operatorname{Var}(Y)$. If $Y\sim \operatorname{Poi}$ , then $\sqrt{Y}$ has

$$
\operatorname{Var}(\sqrt{y}) \approx\left(\frac{1}{2 \sqrt{\mu}}\right)^{2} \mu=\frac{1}{4}
$$

The approximation holds for larger $\mu$, for which $\sqrt{Y}$ is more closely linear in a neighborhood of $\mu$.

Then, since the variance is stablized, we and fit OLS. The model is

$$E(\sqrt{y_{i}})=\boldsymbol{x}_{i}^{\top} \boldsymbol{\beta}$$

However, the fit may hold more poorly for $E(\sqrt{y_{i}})$ than other transformations. GLM is more appealing, since it models the transformation of the mean, rather than the mean of the transformation.


## Poisson Log-linear Models

### Link function

Recall the Poisson distribution

$$
f(y)=\exp(y\log\mu-\mu)\frac{1}{y!}
$$

The canonical link is $\log\mu$.

$$
\log\mu_{i}=\boldsymbol{x}_i ^\top \boldsymbol{\beta}
$$

Or equivalently

$$
\mu_{i}=\Pi\left(e^{\beta_{k}}\right)^{x_{ik}}
$$

which assuming that each $x_{ik}$ has a multiplicative effect on $y_{i}$.

### Interpretation

$\mu_i$ is multiplied by $e^{\beta_k}$ if there is one unit increase in $X_k$ .

## Estimation

### Score Equations

For the log-linear link, the score equations are

$$
\sum_{i}(y_{i}-\mu_{i})x_{ij}=0
$$

when the model has an intercept, this implies

$$
\sum_{i}y_{i}=\sum_{i}\mu_{i}
$$


### Covariance of MLE

For the log-linear link, the covariance matrix is

$$
\text{Var}(\hat{\boldsymbol{\beta}})=(\boldsymbol{X} ^{\top} \boldsymbol{W} \boldsymbol{X} )^{-1}
$$

where

$$
w_{i}=v_{i}=\text{Var}(Y_{i})=\mu_{i}
$$


## Hypothesis Testing

### Deviance

$$
D(\boldsymbol{y},\hat{\boldsymbol{\mu}})=2\sum_{i=1}^{n}\left[y_{i}\log\left(\frac{y_{i}}{\hat{\mu}_{i}}\right)-y_{i}+\hat{\mu}\right]
$$

when the model has an intercept, since $\sum_{i}y_{i}=\sum_{i}\mu_{i}$, the deviance becomes

$$
D(\boldsymbol{y},\hat{\boldsymbol{\mu}})=2\sum_{i=1}^{n}\left[y_{i}\log\left(\frac{y_{i}}{\hat{\mu}_{i}}\right)\right]
$$


### Pearson Statistic

$$
X^{2}=\sum_{i=1}^{n}\frac{(y_{i}-\hat{\mu}_{i})^{2}}{\hat{\mu}_{i}}
$$


## One-way Layout

Poisson regression are closely related to one-way layout.

### Problem Settings

Recall a simple two-sample $t$-test, the underlying model is

$$
y_{ij}=\mu_{i}+\epsilon_{ij}
$$


- $y_{ij}$ is the $j$-th observation of the $i$-th group, $i=1,2$, and $j=1,2,\ldots,{n_{i}}$
- $\mu_{i}$ is the group mean
- $\epsilon_{ij}$ is the random error which is assumed to have normal distribution $N(0,\sigma^{2})$.

Essentially, the data generating process is

$$
y_{ij}\sim N({\mu_{i}}, {\sigma^{2}})
$$

And we are interested in comparing the two means $\mu_{1}=\mu_{2}$.

Now, instead, we suppose $y_{ij}$ follows Poisson distribution.

### Model with Poisson

There are $c$ groups, each with $n_{i}$ observations. The total number of observations are $n=\sum_{i=1}^{c}n_{i}$. The data generating process is

$$
y_{ij} \overset{\text{iid}}{\sim} \operatorname{Poi} (\mu_{i}),\quad i=1,2,\ldots, c,\quad j=1,2,\ldots,{n_{i}}
$$

Suppose the means $\mu_{i}$'s can be modeled by

$$
\log(\mu_{i})=\beta_{0}+\beta_{i},\ \beta_{0}=0\text{ for identifiability}
$$

Then we can apply GLM to solve for $\boldsymbol{\beta}$, like we apply OLS to
classification. The model is

$$
\log\boldsymbol{\mu}= \boldsymbol{X} \boldsymbol{\beta}
$$

where

$$
\boldsymbol{\mu}=\left(\begin{array}{c}
\mu_{1}\boldsymbol{1}_{n_{1}}\\
\mu_{2}\boldsymbol{1}_{n_{2}}\\
\vdots\\
\mu_{c}\boldsymbol{1}_{n_{c}}
\end{array}\right),\quad \boldsymbol{X} \boldsymbol{\beta}=\left(\begin{array}{cccc}
\boldsymbol{1}_{n_{1}} & \boldsymbol{0}_{n_{1}} & \cdots & \boldsymbol{0}_{n_{1}}\\
\boldsymbol{0}_{n_{2}} & \boldsymbol{1}_{n_{2}} & \dots & \boldsymbol{0}_{n_{2}}\\
\vdots & \vdots & \ddots & \vdots\\
\boldsymbol{0}_{n_{c}} & \boldsymbol{0}_{n_{c}} & \cdots & \boldsymbol{1}_{n_{c}}
\end{array}\right)\left(\begin{array}{c}
\beta_{1}\\
\beta_{2}\\
\vdots\\
\beta_{c}
\end{array}\right)
$$


### Score Equations

Since $x_{ij}=1$, the likelihood equation for $\beta_{i}$ is

$$
\sum_{j=1}^{n_{l}}\left(y_{ij}-\mu_{i}\right)=0
$$

Thus the MLE is

$$
\hat{\mu}_{i}=\bar{y}_{i}=\frac{\sum_{i}y_{ij}}{n_{i}}
$$

and

$$
\hat{\beta}_{i}=\log\bar{y}_{i}
$$


### Covariance of MLE

Since $\hat{w}_{i}=\hat{v}_{i}=\text{Var}(y_{i})=\hat{\mu}_{i}=\bar{y}_{i}$,
then

$$
\widehat{\text{Var}}(\hat{\boldsymbol{\beta}})=\left(\boldsymbol{X} ^{\top}\hat{\boldsymbol{W} } \boldsymbol{X} \right)^{-1}
$$

with diagonal

$$
\widehat{\text{Var}}(\hat{\beta}_{i})=\frac{1}{n_{i}\bar{y}_{i}}
$$

Since $\frac{\mu_{h}}{\mu_{i}}=\exp(\hat{\beta}_{h}-\hat{\beta}_{i})$, the 95% confidence interval for $\frac{\mu_{h}}{\mu_{i}}$ is

$$
\exp\left[\left(\hat{\beta}_{h}-\hat{\beta}_{i}\right)\pm1.96\sqrt{\left(n_{h}\bar{y}_{h}\right)^{-1}+\left(n_{i}\bar{y}_{i}\right)^{-1}}\right]
$$


### Hypothesis Testing

To test

$$
H_{0}:\mu_{1}=\dots=\mu_{c}
$$

we can compare the deviances of the null model with the one-way layout. The likelihood-ratio test statistic is

$$
2\sum_{i=1}^{c}n_{i}\bar{y}_{i}\log\left(\bar{y}_{i}/\bar{y}\right)
$$

Under $H_{0}$, it has distribution converging to $\chi^2_{c-1}$.

If the data have greater than Poisson variability, the large-sample $\text{Var}(\hat{\beta}_{i})>\frac{1}{n_{i}\mu_{i}}$, and it's better to use a model that permits greater dispersion such as the [negative binomial model](negative-binomial).

### Deviance and Pearson Statistic

From the formula, we can decompose the summation by $c$ groups,

$$
G^{2}=2\sum_{i=1}^{c}\sum_{j=1}^{n_{i}}y_{ij}\log\left(\frac{y_{ij}}{\bar{y}_{i}}\right),\quad X^{2}=\sum_{i=1}^{c}\sum_{j=1}^{n_{i}}\frac{\left(y_{ij}-\bar{y}_{i}\right)^{2}}{\bar{y}_{i}}
$$

when $\bar{y}_{i}$ is large, $G^{2}$ and $X^{2}$ have approximately
$\chi^2_{n-c}$.

## Contingency Tables

Poisson GLM can be used to model the counts in contigency tables.

### Two-Way Contingency Table

Consider an $r\times c$ table for two categorical variables (denote as $A$ and $B$). The Poisson GLM assumes that the count $y_{ij}$ in each cell independently follows a Poisson distributions with mean $\mu_{ij}$.

Consider two scenarios

- If the two categorical variables are **independent**, then

  $$p_{ij}=p_{i+}p_{+j}$$

  where $p_{ij}$ is the proportion of cell $i,j$, $p_{i+}$ is the proportion of row $i$ and $p_{+j}$ is the proportion of column $j$.

  Since $n$ is fixed, we have

  $$\mu_{ij}=np_{ij}=np_{i+}p_{+j}$$

  Taking log gives

  $$
  \log\mu_{ij}=\underbrace{\log n}_{\beta_{0}}+\underbrace{\log p_{i+}}_{\beta_{i}^{A}}+\underbrace{\log p_{+j}}_{\beta_{j}^{B}}
  $$

  For identifiability, two constraints are $\sum_{i}p_{i+}=1$ and $\sum_{j}p_{+j}=1$, or $\beta_{1}^{A}=0$ and $\beta_{1}^{B}=0$. Hence, this model has $\left[1+(r-1)+(c-1)\right]$ free parameters.

  The non-constant part (kernel) of the log-likelihood is

  $$
  \ell(\boldsymbol{\mu} )=\sum_{i=1}^{r}\sum_{j=1}^{c}y_{ij}\log\mu_{ij}-\sum_{i=1}^{r}\sum_{j=1}^{c}\mu_{ij}
  $$

  If we use the canonical link, the score equations are

  $$\begin{aligned}
  \sum_{i,j}\left(y_{ij}-\mu_{ij}\right) &=0\\
  \sum_{j}\left(y_{ij}-\mu_{ij}\right) &=0,\quad i=1,2,\cdots,r\\
  \sum_{i}\left(y_{ij}-\mu_{ij}\right) &=0,\quad j=1,2,\cdots,c
  \end{aligned}$$

  Hence the MLEs are

  $$\begin{aligned}
  \hat{\mu} &=y_{++} \\
  \hat{p}_{i+} &=\frac{y_{i+}}{y_{++}} \\
  \hat{p}_{+j} &=\frac{y_{+j}}{y_{++}} \color{white}{++}
  \end{aligned}$$


- If the two categorical variables has **interaction**, then $p_{ij} \ne p_{i+}p_{+j}$, but we have

  $$
  p_{ij}=p_{i+}p_{+j}\frac{p_{ij}}{p_{i+}p_{+j}}
  $$

  Likewise, the log-linear model is

  $$
  \log\mu_{ij}=\underbrace{\log n}_{\beta_{0}}+\underbrace{\log p_{i+}}_{\beta_{i}^{A}}+\underbrace{\log p_{+j}}_{\beta_{j}^{B}}+\underbrace{\log\frac{p_{ij}}{p_{i+}p_{+j}}}_{\gamma_{ij}^{AB}}
  $$

  Note

  $$
  \sum_{i}\exp\left(\gamma_{ij}^{AB}\right)=\sum_{i}\frac{p_{ij}}{p_{i+}p_{+j}}=\frac{1}{p_{+j}}=\frac{1}{\exp\left(\beta_{j}^{B}\right)},\ j=1,2,\cdots,c
  $$

  $$
  \sum_{j}\exp\left(\gamma_{ij}^{AB}\right)=\sum_{j}\frac{p_{ij}}{p_{i+}p_{+j}}=\frac{1}{p_{i+}}=\frac{1}{\exp\left(\beta_{i}^{A}\right)},\ i=1,2,\cdots,r
  $$

  So there are actually

  $$
  r\times c-(r-1)-(c-1)+1=(r-1)\times(c-1)
  $$

  number of free $\gamma^{AB}$ parameters. The total number of parameters for this model is therefore $r\times c$, which suggests this model is saturated.

  For identifiability, we can add $(r-1)+(c-1)-1$ constraints for the
  last term.

  $$
  \gamma_{1j}^{AB}=\gamma_{i1}^{AB}=0
  $$

  The interactions pertain to odds ratios, for instance, when $r=c=2$,

  $$
  \log\frac{p_{11}/p_{12}}{p_{21}/p_{22}}=\log\frac{\mu_{11}/\mu_{12}}{\mu_{21}/\mu_{22}}=\gamma_{11}^{AB}+\gamma_{22}^{AB}-\gamma_{12}^{AB}-\gamma_{21}^{AB}=\gamma_{22}^{AB}
  $$


### Three-way Contingency Table

Similarly, with different assumptions on independence, we can have
different model parameterization for an $r\times c\times \ell$ table.

- **Mutual Independence**

  If

  $$P(A=i,B=j,C=k)=P(A=i)P(B=j)P(C=k)$$

  then

  $$\log\mu_{ijk}=\beta_{0}+\beta_{i}^{A}+\beta_{j}^{B}+\beta_{k}^{C}$$

- **Joint Independence**

  Joint independence between $A$ and $(B,C)$ says

  $$P(A=i,B=j,C=k)=P(A=i)$$

  then

  $$\log\mu_{ijk}=\beta_{0}+\beta_{i}^{A}+\beta_{j}^{B}+\beta_{k}^{C}+\gamma_{jk}^{BC}$$

- **Conditional Independence**

  Conditional independence of $A$ and $B$ given $C$ says

  $$P(A=i,B=j|C=k)=P(A=i|C=k)P(B=j|C=k)$$

  then

  $$
  \log\mu_{ijk}=\beta_{0}+\beta_{i}^{A}+\beta_{j}^{B}+\beta_{k}^{C}+\gamma_{ik}^{AC}+\gamma_{jk}^{BC}
  $$

- **Homogeneous Association**

  An interpretation of this model is that any two pairs are dependent, but the dependence does not change with the value of the third variable.

  $$
  \log\mu_{ijk}=\beta_{0}+\beta_{i}^{A}+\beta_{j}^{B}+\beta_{k}^{C}+\gamma_{ik}^{AC}+\gamma_{jk}^{BC}+\gamma_{ij}^{AB}
  $$


## Relation to

### Relation to Multinomial Distribution

For independent poisson counts $\left(Y_{1},\cdots,Y_{c}\right)$ with mean $\mu_{i}$, the total counts $n=\sum_{i}Y_{i}$ follows $\operatorname{Poi} (\mu)$ where $\mu=\sum_{i}\mu_{i}$. Conditional on $n$, the joint density of $\left(Y_{1},\cdots,Y_{c}\right)$ is

$$
\frac{P\left(Y_{1}=n_{1},\cdots,Y_{c}=n_{c}\right)}{P\left(\sum_{i}Y_{i}=n\right)}=\left(\frac{n!}{\prod_{i}n_{i}!}\right)\prod_{i=1}^{c}p_{i}^{n_{i}}
$$

which implies

$$
\left(Y_{1},\cdots,Y_{c}\right)\sim \operatorname{Multinomial} \left(\frac{\mu_{1}}{\mu},\cdots,\frac{\mu_{c}}{\mu}\right)
$$


### Different Data Structure with Logistic Models

Both log-linear models and logistic models can be used to model data in a contingency table shape.


- The log-linear models treat **all** categorical variables symmetrically
and regard the cell **counts** as response.
- The logistic models use **some** categorical variables as covariates,
to model **one** remaining categorical variable as response.


### Equivalency to Logistic Models

Suppose in a three-way contingency able, $r=2$, i.e. $A$ is a binary variable with $A\in\left\{ 1,2\right\}$. Then from the formula for Homogeneous Association above,

$$
\begin{aligned}\log\frac{P(A=1|B=j,C=k)}{P(A=2|B=j,C=k)} & =\log\mu_{1jk}-\log\mu_{2jk}\\
\text{use Homogeneous Association } & =\left(\beta_{1}^{A}-\beta_{2}^{A}\right)+\left(\gamma_{1j}^{AB}-\gamma_{2j}^{AB}\right)+\left(\gamma_{1k}^{AC}-\gamma_{2k}^{AC}\right)
\end{aligned}
$$

Equivalently,

$$
\operatorname{logit}[P(A=1|B=j,C=k)]=\lambda+\delta_{j}^{B}+\delta_{k}^{C}
$$

which is a logistic regression model. For $r>2$, it is a baseline-category logit model.

The Poisson log-linear model and logistic model also have the same score equations.

## Over-Dispersed Counts

### Over-dispersion

We say over-dispersion occurs if the actual data variance $\operatorname{Var}\left(  Y_i\right)$ is greater than the variance $v^{*}(Y_{i})$ from model specification.

Detection: Plot $(y_{i}-\hat{\mu}_{i})^{2}$ v.s. $\hat{v}^{*}(\hat{\mu}_{i})$. If the specification $\operatorname{Var} (Y_{i}) = v^{*}(Y_{i})$ is true, then the points should scatter around the line $45^{\circ}$ line. If most points lie above the $45^{\circ}$ line, then over-dispersion exists. Our assumption for the randomness of $y_{i}$ is problematic.

Some alternative models that allow over-dispersion include negative binomial, beta-binomial.

(negative-binomial)=
### Negative-Binomial Model

If $Y_{i}\sim\operatorname{Poi} \left(\lambda_{i}\right)$ and $\lambda \sim \Gamma\left(\mu_{i},k_{i}\right)$, then ${Y_{i} \sim \operatorname{NB} \left(\mu_{i},k_{i}\right)}$. We have

$$
\operatorname{E} (Y_{i})=\mu_{i},\quad \operatorname{Var} (y_{i})=\mu_{i}+\gamma_{i}\mu_{i}^{2}>\mu_{i}
$$

where $\gamma_{i}=1/k_{i}$ is the dispersion parameter.

For NB GLM, We further assume $\gamma_{i}\equiv\gamma$ for all $i$, and the link is $\log(\mu_{i})=\boldsymbol{x}_i ^\top \boldsymbol{\beta}$.


:::{admonition,note} Note
Note that when $\gamma=0$, it is Poisson. So Poisson is a nested model for Negative Binomial. We can perform likelihood-ratio test to compare the dispersion assumption. However, $\gamma=0$ is on the boundary of the parameter space. Thus, the likelihood-ratio statistic does not have an asymptotic null chi-squared distribution. Rather, it is an equal mixture of a single-point distribution at 0 (which occurs when $\hat{\gamma}=0$) and chi-squared with $df=1$. The $p$-value is half that from treating the statistic as chi-squared with $df=1$.
:::


### Beta-Binomial Model

Assume $n_{i}{y_{i}\sim} \operatorname{Bin} {(n_{i},p_{i})}$ and ${p_{i}\sim \operatorname{Beta} \left(\alpha_{1},\alpha_{2}\right)}$. Let $\mu=\frac{\alpha_{1}}{\alpha_{1}+\alpha_{2}}$ and $\theta=\frac{1}{\alpha_{1}+\alpha_{2}}$, then the Beta-binomial distribution has the property

$$
{\operatorname{E} (Y_{i})=\mu_{i},\quad\operatorname{Var}(Y_{i})=\frac{1}{n_i} \left[1+(n_{i}-1)\frac{\theta}{1+\theta}\right]\mu(1-\mu)}>\mu(1-\mu)\quad \text{if }n_{i}>1
$$

The term $\rho=\frac{\theta}{1+\theta}$ is the measure of over-dispersion, It is also the correlation in completely dependent Bernoulli trials, where the variance function has the same form as here.

As $\theta\rightarrow 0$, the Beta distribution converges to a degenerate distribution at $\mu$. Hence, the Beta-binomial distribution converges to the $\operatorname{Bin} (n_{i},\mu_{i})$.

We may use logit link,

$$
{\operatorname{logit}\left(\mu_{i}\right)=\boldsymbol{x}_i ^\top \boldsymbol{\beta} }
$$

Both $\boldsymbol{\beta}$ and $\theta$ are unknown but we can estimate them using MLE.

## Zero-inflated Counts

In practice, the frequency of 0 outcomes is often larger than expected under standard discrete models. A Poisson GLM is inadequate when means can be relatively large but the modal response is 0.

This could occur for the frequency of an activity in which many subjects never participate but many others quite often do. Then a substantial fraction of the population necessarily has a zero outcome, and the remaining fraction follows some distribution that may have small probability of a zero outcome, such as the number of times during some period of having an alcoholic drink, or smoking marijuana, or having sexual intercourse.

### Zero-inflated Poisson (ZIP)


#### Model

We may observe $y=0$ with high frequency than that under assumption, so se can model the counts as a mixture of 0 and Poisson.

$$
y_{i}\sim\left\{ \begin{array}{l}
0 &\text{w.p. }1-\phi_{i}\\
\operatorname{Poi} \left(\lambda_{i}\right)&\text{w.p. }\phi_{i}
\end{array}\right.
$$

with mean $\operatorname{E} \left(Y_{i}\right)=\phi_{i}\lambda_{i}$ and variance
$$
\operatorname{Var}\left(Y_{i}\right)=\phi_{i}\lambda_{i}\left[1+\left(1-\phi_{i}\right)\lambda_{i}\right]> \operatorname{E} \left(Y_{i}\right),\ \text{over-dispersion}
$$

There is a latent variable $Z_{i}\sim \operatorname{Ber}(\phi_{i})$. If $z_{i}=0$
then $y_{i}=0$ and if $z_{i}=1$ then $y_{i}\sim \operatorname{Poi} (\lambda_{i}).$

A common assumption for the links are

$$
\operatorname{logit}\left(\phi_{i}\right)=\boldsymbol{x}_{1i}^{\top}\boldsymbol{\beta}_{1},\quad\log\left(\lambda_{i}\right)=\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}
$$

so we have two linear predictors, one for $\phi_{i}$ and the other
for $\lambda_{i}$.

#### Estimation

The likelihood is

$$
L\left(\boldsymbol{\beta}_{1},\boldsymbol{\beta}_{2}\right)=\prod_{i=1}^{n}\left(1-\phi_{i}\right)^{\mathbb{I}\left(y_{i}=0\right)}\left[\phi_{i}f\left(y_{i};\mu_{i}\right)\right]^{1-\mathbb{I}\left(y_{i}=0\right)}
$$

The log-likelihood is

$$
\begin{aligned}\ell\left(\boldsymbol{\beta}_{1},\boldsymbol{\beta}_{2}\right)= & \sum_{y_{i}=0}\log\left[1+e^{x_{1i}}\boldsymbol{\beta}_{1}\exp\left(-e^{x_{2i}\boldsymbol{\beta}_{2}}\right)\right]-\sum_{i=1}^{n}\log\left(1+e^{x_{1i}}\boldsymbol{\beta}_{1}\right)\\
 & +\sum_{y_{i}>0}\left[\boldsymbol{x}_{1i}^{\top}\boldsymbol{\beta}_{1}+y_{i}\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}-e^{x_{2i}\boldsymbol{\beta}_{2}}-\log\left(y_{i}!\right)\right]
\end{aligned}
$$


#### Disadvantage

- Larger number of parameters compared with ordinary Poisson or negative binomial models.

- Parameters do NOT directly describe the effects of explanatory variables on $E(Y_{i})$

- The correlation between $\boldsymbol{x}_{1i}$ and $\boldsymbol{x}_{2i}$ could cause further problems with interpretation.

- If over-dispersion occurs conditional on $z_{i}=1$, then the mean variance equality does not hold, and we resort to Zero-inflated Negative Binomial.

### Zero-inflated Negative Binomial (ZINB)

Likewise, we model by a mixture of 0 and negative binomial.

$$
Y_{i}\sim\left\{ \begin{array}{l}
0&\text{w.p. }1-\phi_{i}\\
\operatorname{NB}\left(\lambda_{i},k\right)&\text{w.p. }\phi_{i}
\end{array}\right.
$$


### Hurdle Models

Hurdle models handle zeroes separately by a zero-truncated distribution.

Suppose that

- the first part of the process is governed by probabilities $\operatorname{P} (Y_{i}>0)=\pi_{i}$ and $\operatorname{P} (Y_{i}=0)=1-\pi_{i}$

- $Y_{i}\vert Y_{i}>0$ follows a truncated-at-zero probability mass function $f(y_{i};\mu_{i})$, where $y_i > 0$ such as a truncated Poisson.

The complete distribution is

$$
\begin{array}{l}
P\left(y_{i}=0\right)=1-\pi_{i}\\
P\left(y_{i}=j\right)=\pi_{i}\frac{f\left(j;\mu_{i}\right)}{1-f\left(0;\mu_{i}\right)},\quad j=1,2,\ldots
\end{array}
$$

Likewise, we have two linear predictors

$$
\operatorname{logit}\left(\pi_{i}\right)=\boldsymbol{x}_{1i}^{\top}\boldsymbol{\beta}_{1},\quad\log\left(\mu_{i}\right)=\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}
$$

The likelihood is

$$
L\left(\boldsymbol{\beta}_{1},\boldsymbol{\beta}_{2}\right)=\prod_{i=1}^{n}\left(1-\pi_{i}\right)^{\mathbb{I}\left(y_{i}=0\right)}\left[\pi_{i}\frac{f\left(y_{i};\mu_{i}\right)}{1-f\left(0;\mu_{i}\right)}\right]^{1-\mathbb{I}\left(y_{i}=0\right)}
$$

The log-likelihood separates into two terms,

- one is the log-likelihood function for the binary process

  $$
  \begin{aligned}\ell_{1}\left(\boldsymbol{\beta}_{1}\right) & =\sum_{y_{i}=0}\left[\log\left(1-\pi_{i}\right)\right]+\sum_{y_{i}>0}\log\left(\pi_{i}\right)\\
   & =\sum_{y_{i}>0}\boldsymbol{x}_{1i}^{\top}\boldsymbol{\beta}_{1}-\sum_{i=1}^{n}\log\left(1+e^{\boldsymbol{x}_{1i}^{\top}\boldsymbol{\beta}_{1}}\right)
  \end{aligned}
  $$

- and the other is the log-likelihood function for the truncated model

  $$
  \ell_{2}\left(\boldsymbol{\beta}_{2}\right)=\sum_{y_{i}>0}\left\{ \log f\left(y_{i};\exp\left(\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}\right)\right)-\log\left[1-f\left(0;\exp\left(\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}\right)\right)\right]\right\}
  $$

Examples (Hurdle Models with truncated Poisson)
: If we use truncated Poisson, then

  $$
  \ell_{2}\left(\boldsymbol{\beta}_{2}\right)=\sum_{y_{i}>0}\left\{ y_{i}\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}-e^{\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}}-\log\left[1-\exp\left(-e^{\boldsymbol{x}_{2i}^{\top}\boldsymbol{\beta}_{2}}\right)\right]\right\} -\sum_{y_{i}>0}\log\left(y_{i}!\right)
  $$

  Conditional on $Y_{i}>0$, the mean and variance are

  $$
  \operatorname{E} \left(Y_{i}\right)=\frac{\lambda_{i}}{1-e^{-\lambda_{i}}}<\operatorname{Var}\left(Y_{i}\right)=\frac{\lambda_{i}}{1-e^{-\lambda_{i}}}-\frac{\lambda_{i}^{2}e^{-\lambda_{i}}}{\left(1-e^{-\lambda_{i}}\right)^{2}}
  $$

  When over-dispersion occurs, use truncated NB.

#### Compared to Zero-inflated Mean

Zero-inflated models are more natural than the hurdle model when the population is naturally regarded as a mixture, with one set of subjects that necessarily has a 0 response.

However, the hurdle model is also suitable when, at some settings, the data have **fewer** zeros than are expected under standard distributional assumptions.

### Compared to Negative binomial

Zero-inflation is less problematic for negative binomial GLMs, because that distribution can have a mode of 0 regardless of the value of the mean. However, a negative binomial model fits **poorly** when the data are strongly bimodal, with a mode at zero and a separate mode around some considerably higher value.
