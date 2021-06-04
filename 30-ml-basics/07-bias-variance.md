## Bias-variance tradeoff

This section tryies to clarify bias and variance for estimator and for test error. They are different quantities but are related.

Notations:

- $\mathcal{D} = \left\{ (\boldsymbol{x} ,y) \right\}_{1, \ldots, n}$: i.i.d. data points.
- $\boldsymbol{\beta}$: true coefficient


### MSE of an estimator $\hat{\boldsymbol{\beta}}$

Let $\hat{\boldsymbol{\beta}}$ be an estimator of parameter $\boldsymbol{\beta}$, its mean squared error can be decomposed into a variance component and a bias $^2$ component.

$$\begin{aligned}
\operatorname{MSE}(\hat{\boldsymbol{\beta}} )
&= \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}]^2\\
&= \mathbb{E} _{\mathcal{D}} \left[ \left\| \hat{\boldsymbol{\beta}} - \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}}] \right\| \right] ^2 + \left\| \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}} ] - \boldsymbol{\beta}   \right\| ^2\\
&= \underbrace{\operatorname{tr}\left( \operatorname{Cov}\left( \hat{\boldsymbol{\beta}} \right) \right)}_{\text{Variance} } + \underbrace{\left\| \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}} ] - \boldsymbol{\beta}   \right\| ^2}_{\text{Bias}^2 }
\\
\end{aligned}$$

where $\mathbb{E} _{\mathcal{D}}$ means the expectation is taken over randomly drawn data set $\mathcal{D}$.

#### Ridge Regression

$$
\min_{\boldsymbol{\beta}}\ \left\| \boldsymbol{y} - \boldsymbol{X} \boldsymbol{\beta}  \right\|^2 + \lambda \left\| \boldsymbol{\beta}  \right\| _2^2
$$

Estimator:

$$
\hat{\boldsymbol{\beta}}_{\text{Ridge} }=\left(\boldsymbol{X} ^{\top} \boldsymbol{X} +\lambda \boldsymbol{I} \right)^{-1} \boldsymbol{X}^{\top} \boldsymbol{y}
$$

- when $\lambda = 0$, $\hat{\boldsymbol{\beta}}_{\text{Ridge} } = \hat{\boldsymbol{\beta}}_{\text{OLS} }$.
- when $\lambda \rightarrow \infty$, $\hat{\boldsymbol{\beta}}_{\text{Ridge} } \rightarrow 0$, but every entry is non-zero.

|Term| Compared to OLS |As $\lambda$ increases|
|-|-| -|
| bias of $\hat{\boldsymbol{\beta}}_{\text{Ridge} }$ |larger | increases from $0$ to $\left\| \boldsymbol{\beta} \right\|$ $\color{red}{(?)}$ |
| variance of $\hat{\boldsymbol{\beta}}_{\text{Ridge} }$  | smaller  | decreases to 0 $\color{red}{(?)}$ |
| MSE of $\hat{\boldsymbol{\beta}}_{\text{Ridge} }$  | smaller for some $\lambda$  | first decreases then increases |

Reference: https://www.statlect.com/fundamentals-of-statistics/ridge-regression


#### Lasso Regression

$$
\min _{\beta}\|\boldsymbol{y}-\boldsymbol{X} \boldsymbol{\beta}\|^{2}+\lambda\|\boldsymbol{\beta}\|_{1}
$$

- when $\lambda = 0$, $\hat{\boldsymbol{\beta}}_{\text{Lasso} } = \hat{\boldsymbol{\beta}}_{\text{OLS} }$.
- when $\lambda \rightarrow \infty$, $\hat{\boldsymbol{\beta}}_{\text{Lasso} } \rightarrow 0$, many entries are zeros.

|Term| Compared to OLS |As $\lambda$ increases|
|-|-| -|
| bias of $\hat{\boldsymbol{\beta}}_{\text{Lasso} }$  |larger | increases from $0$ to $\left\| \boldsymbol{\beta} \right\|$ $\color{red}{(?)}$|
| variance $\hat{\boldsymbol{\beta}}_{\text{Lasso} }$  | smaller  | decreases to 0 $\color{red}{(?)}$|
| MSE of $\hat{\boldsymbol{\beta}}_{\text{Lasso} }$  | smaller for some $\lambda$  | first decreases then increases |

### Expected Test Error (ETE)

Given an algorithm (e.g. OLS, SVM), we draw a data set $\mathcal{D}$, use the algorithm to train a linear model $h$, then evaluate it on an unseen data point $(\boldsymbol{x} ,y)$. Note that all $\mathcal{D} , \boldsymbol{x} , y$ are random. This expectation is called **expected test error**. It can be decomposed into three components: variance, bias, and noise (aka irreducible error)


$$
\underbrace{\mathbb{E}_{\boldsymbol{x}, y, \mathcal{D}}\left[\left(h_{\mathcal{D}}(\boldsymbol{x})-y\right)^{2}\right]}_{\text {Expected Test Error }}=\underbrace{\mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left(h_{\mathcal{D}}(\boldsymbol{x})-\bar{h}(\boldsymbol{x})\right)^{2}\right]}_{\text {Variance }}+\underbrace{\mathbb{E}_{\boldsymbol{x}, y}\left[(\bar{y}(\boldsymbol{x})-y)^{2}\right]}_{\text {Noise }}+\underbrace{\mathbb{E}_{\boldsymbol{x}}\left[(\bar{h}(\boldsymbol{x})-\bar{y}(\boldsymbol{x}))^{2}\right]}_{\text {Bias }^{2}}
$$

where

- **Expected Label**: $\bar{y}(\mathbf{x})=\mathbb{E}_{y \mid \mathbf{x}}[Y]=\int_{y} y \operatorname{Pr}(y \mid \mathbf{x}) \mathrm{~d} y$
- **Expected Predictor**: $\bar{h}=\mathbb{E}_{\mathcal{D}}\left[h_{\mathcal{D}}\right]=\int_{\mathcal{D}} h_{\mathcal{D}} \operatorname{Pr}(\mathcal{D}) \mathrm{~d} \mathcal{D}$

Reference: https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html

Given an algorithm (e.g. OLS, SVM) and a data set $\mathcal{D}$, we can use cross-validation to compute the 'sample' test error. The cross-validation procedure mimics the random procedures of drawing $\boldsymbol{x} ,y$ and $\mathcal{D}$. For instance, for a $K$-fold cross-validation, this quantity is computed as

$$
\frac{1}{K} \sum_{k=1}^K \frac{1}{n_{k}} \sum_{i=1}^{n_k}(h_k (\boldsymbol{x} _i) - y_i)^2
$$

This quantity is also known as (cross-validation) 'mean squared error' since it has the 'mean' and 'square' forms. If the algorithm depends on some parameter, e.g. Ridge with $\lambda$, then we can compute this quantity for different values of $\lambda$ and compare them, to select an optimal $\lambda$ that minimizes this quantity.


### Expected Test Error given a model

Given a trained model $h$ and a test set $\mathcal{D}_{\text{test} } = \left\{ (\boldsymbol{x} ,y) \right\}_{1, \ldots, n_{\text{test} }}$, a commonly used notion is (test set) 'mean squared error'

$$
\operatorname{MSE}(h)  = \frac{1}{n_{\text{test} }} \sum_{i=1}^{n _{\text{test} }} (h(\boldsymbol{x}_i) - y_i)^2
$$

In expectation

$$
\mathbb{E}_{\boldsymbol{x}, y}\left[\left(h(\boldsymbol{x})-y\right)^{2}\right] =\underbrace{\mathbb{E}_{\boldsymbol{x}, y}\left[(\bar{y}(\boldsymbol{x})-y)^{2}\right]}_{\text {Noise }}+\underbrace{\mathbb{E}_{\boldsymbol{x}}\left[(h(\boldsymbol{x})-\bar{y}(\boldsymbol{x}))^{2}\right]}_{\text {Bias }^{2}}
$$

(Imagine a fitted line and a true line in linear regression)

### Confusion

When people talk about bias-variance tradeoff, sometimes they refer to the bias and variance components in the MSE of the estimator

$$\begin{aligned}
\operatorname{MSE}(\hat{\boldsymbol{\beta}} )
&= \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}]^2\\
&= \mathbb{E} _{\mathcal{D}} \left[ \left\| \hat{\boldsymbol{\beta}} - \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}}] \right\| \right] ^2 + \left\| \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}} ] - \boldsymbol{\beta}   \right\| ^2\\
&= \underbrace{\operatorname{tr}\left( \operatorname{Cov}\left( \hat{\boldsymbol{\beta}} \right) \right)}_{\text{Variance} } + \underbrace{\left\| \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}} ] - \boldsymbol{\beta}   \right\| ^2}_{\text{Bias}^2 }
\\
\end{aligned}$$

but sometimes refer to the bias and variance components of the expected test error (or its realized value: cross-validation MSE).

$$
\underbrace{\mathbb{E}_{\boldsymbol{x}, y, \mathcal{D}}\left[\left(h_{\mathcal{D}}(\boldsymbol{x})-y\right)^{2}\right]}_{\text {Expected Test Error }}=\underbrace{\mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left(h_{\mathcal{D}}(\boldsymbol{x})-\bar{h}(\boldsymbol{x})\right)^{2}\right]}_{\text {Variance }}+\underbrace{\mathbb{E}_{\boldsymbol{x}, y}\left[(\bar{y}(\boldsymbol{x})-y)^{2}\right]}_{\text {Noise }}+\underbrace{\mathbb{E}_{\boldsymbol{x}}\left[(\bar{h}(\boldsymbol{x})-\bar{y}(\boldsymbol{x}))^{2}\right]}_{\text {Bias }^{2}}
$$

In Ridge regression, as $\lambda$ increases, we have two facts

1. $\operatorname{Var}\left( \hat{\boldsymbol{\beta}} _{\text{Ridge} } \right)$ decreases but $\operatorname{Bias}\left( \hat{\boldsymbol{\beta}} _{\text{Ridge} } \right)$ increases, and overall $\operatorname{MSE}\left( \hat{\boldsymbol{\beta}} _{\text{Ridge} } \right)$ first decreases and then increases
2. Expected test error (cross-validation MSE) first decreases and then increases

Many online notes conclude that *'Ridge estimator is biased but it reduces prediction error'*. However, the *'bias'* refer to the bias term in fact 1 while *'prediction error'* refers to the expected test error in fact 2, which is confusing.

**How are the bias and variance terms in $\operatorname{MSE}\left( \hat{\boldsymbol{\beta}} _{\text{Ridge} } \right)$ related to those in Expected test error (ETE)?** When the bias in $\operatorname{MSE}\left( \hat{\boldsymbol{\beta}} _{\text{Ridge} } \right)$ increases, does the bias in ETE also increases? How about variance and overall MSE?

- For the bias $^2$ term in ETE, consider a linear model $\boldsymbol{y} = \boldsymbol{x} ^{\top} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$, then

  $$\begin{aligned}
  \mathbb{E}_{\boldsymbol{x}}\left[(\bar{h}(\boldsymbol{x})-\bar{y}(\boldsymbol{x}))^{2}\right]
  &=\mathbb{E}_{\boldsymbol{x}}\left[(\boldsymbol{x} ^{\top} \bar{\hat{\boldsymbol{\beta}}} -\boldsymbol{x} ^{\top} \boldsymbol{\beta} )^{2}\right]  \\
  &=\mathbb{E}_{\boldsymbol{x}}\left[(\boldsymbol{x} ^{\top} \mathbb{E}_{\mathcal{D}}[\hat{\boldsymbol{\beta}}] -\boldsymbol{x} ^{\top} \boldsymbol{\beta} )^{2}\right]  \\
  &=\mathbb{E}_{\boldsymbol{x}}\left[(\boldsymbol{x} ^{\top}( \mathbb{E}_{\mathcal{D}}[\hat{\boldsymbol{\beta}}] - \boldsymbol{\beta} ))^{2}\right]  \\
  \end{aligned}$$

  Recall that $\operatorname{Bias}(\hat{\boldsymbol{\beta}} ) = \left\| \mathbb{E} _{\mathcal{D} }[\hat{\boldsymbol{\beta}} ]-\boldsymbol{\beta}   \right\|$. Let $\boldsymbol{b} = \mathbb{E} _{\mathcal{D} }[\hat{\boldsymbol{\beta}} ]-\boldsymbol{\beta}$, then

  $$\begin{aligned}
  &= \mathbb{E}_{\boldsymbol{x}}\left[(\boldsymbol{x} ^{\top}\boldsymbol{b})^{2}\right]\\
  &= \mathbb{E}_{\boldsymbol{x}}\left[\boldsymbol{b} ^{\top} \boldsymbol{x} \boldsymbol{x} ^{\top} \boldsymbol{b} ^{2}\right]\\
  &= \boldsymbol{b} ^{\top} \mathbb{E}_{\boldsymbol{x}}\left[\boldsymbol{x} \boldsymbol{x} ^{\top}\right] \boldsymbol{b}\\
  \end{aligned}$$

  In general, as $\operatorname{Bias}^2(\hat{\boldsymbol{\beta}} ) = \left\| \boldsymbol{b}  \right\|^2 = \boldsymbol{b} ^{\top} \boldsymbol{b}$ increases, this quantity increases $\color{red}{(?)}$.

- For the variance term in ETE,

  $$\begin{aligned}
  \mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left(h_{\mathcal{D}}(\boldsymbol{x})-\bar{h}(\boldsymbol{x})\right)^{2}\right]
  &= \mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left(\boldsymbol{x}  ^{\top} \hat{\boldsymbol{\beta}}_{\mathcal{D}}-\boldsymbol{x} ^{\top}\mathbb{E}_{\mathcal{D}}[\hat{\boldsymbol{\beta}}] \right)^{2}\right]\\
  &= \mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left( \boldsymbol{x}  ^{\top}\left( \hat{\boldsymbol{\beta}}_{\mathcal{D}}- \mathbb{E}_{\mathcal{D}}[\hat{\boldsymbol{\beta}}] \right) \right)^{2}\right]\\
  \end{aligned}$$

  Recall $\operatorname{Var}(\hat{\boldsymbol{\beta}} ) = \mathbb{E} _{\mathcal{D}} \left[ \left\| \hat{\boldsymbol{\beta}} - \mathbb{E} _{\mathcal{D}}[\hat{\boldsymbol{\beta}}] \right\| \right] ^2$. Let $\hat{\boldsymbol{b}}_{\mathcal{D}} = \hat{\boldsymbol{\beta}}_{\mathcal{D}} -\mathbb{E} _{\mathcal{D} }[\hat{\boldsymbol{\beta}} ]$, then


  $$\begin{aligned}
  &=\mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[\left( \boldsymbol{x}  ^{\top} \hat{\boldsymbol{b}}_{\mathcal{D}} \right)^{2}\right] \\
  &=\mathbb{E}_{\boldsymbol{x}, \mathcal{D}}\left[ \hat{\boldsymbol{b} }_{\mathcal{D}} ^{\top} \boldsymbol{x} \boldsymbol{x} ^{\top} \hat{\boldsymbol{b} } _{\mathcal{D}}\right] \\
  &=\mathbb{E}_{\mathcal{D}}\left[ \hat{\boldsymbol{b} }_{\mathcal{D}} ^{\top} \mathbb{E}_{\boldsymbol{x}} [\boldsymbol{x} \boldsymbol{x} ^{\top}]  \hat{\boldsymbol{b} } _{\mathcal{D}}\right] \\
  \end{aligned}$$

  In general, as $\operatorname{Var}(\hat{\boldsymbol{\beta}} ) = \mathbb{E}_{\mathcal{D}} \left[ \left\| \hat{\boldsymbol{b}}_{\mathcal{D}}  \right\|^2 \right] = \mathbb{E}_{\mathcal{D}} [\hat{\boldsymbol{b}}_{\mathcal{D}} ^{\top} \hat{\boldsymbol{b}}_{\mathcal{D}}]$ increases, this quantity increases $\color{red}{(?)}$.
