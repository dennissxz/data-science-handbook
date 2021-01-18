# Model Selection

## Sources of Error in Learning

**Inherent uncertainty** (e.g. noise variance) in the data distribution: no way to reduce, other than collect better data

**Approximation error**: model class does not contain best possible model (e.g., linear vs non-linear) to approximate the true model. Improve by considering a bigger model family.

**Estimation error**: model chosen by learning from the class is not the best model in that class (e.g. linear models with varying $\lambda$). Improve by getting more data, simplifying model, regularization.

**Optimization error**: fail to properly minimize the learning objective and get an incorrect $\theta^*$, so the learned model is not the best one for the training data. Improve via better optimization, different objective (e.g. convex.)


## Goodness of fitting

Underfitting
: - a model is not complex enough to capture the relation $\boldsymbol{\phi}: \mathcal{X} \rightarrow \mathcal{Y}$
  - can tell by looking at training error

Overfitting
: - a model perform well (good capcity) in training set but it don't generalize $\boldsymbol{\phi}$  well to test set.
: - we cannot tell on training set, need to look at validation set.

[picture]
caption?


## Regularization

To avoid overfitting, one can avoid complex models, or introduce a penalty term to the loss, called a regularizer, which is a measure of the model complexity. The regularized empirical risk minimization is to minimize

$$
\sum_{i=1}^{N} \ell\left(f\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right), y_{i}\right)+\lambda R(\boldsymbol{\theta})
$$

where $\lambda$ controls the weight of the regularizer: larger $\lambda$ means we are favoring less complex models and smaller $\lambda$ means we favoring more complex models.

Common choices of the regularizers include
- $L_1$ regularizer: $R(\boldsymbol{\theta}) = \left\Vert \boldsymbol{\theta}  \right\Vert _1^1$
  - aka shrinkage since some parameters $\theta_j$ in the optimal solution $\boldsymbol{\theta} ^*$ is 0, i.e., sparsity.
- $L_2$ regularizer: $R(\boldsymbol{\theta}) = \left\Vert \boldsymbol{\theta}  \right\Vert _2^2$
- $L_p$ regularizer: $R(\boldsymbol{\theta}) = \left\Vert \boldsymbol{\theta}  \right\Vert _p^p$


Minimizing the objective function with a $L_p$ regularizer

$$
\min _{\theta}\left\{\sum_{i} \ell\left(f\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right), y_{i}\right)+\lambda\|\boldsymbol{\theta}\|_{p}^{p}\right\}
$$

is equivalent to minimizing a constrained form

$$
\min _{\|\boldsymbol{\theta}\|_{p}^{p} \leq \tau} \sum_{i} \ell\left(f\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right), y_{i}\right)
$$

where large $\lambda$ corresponds to small $\tau$. The resulting optimization is solved by Lagrange multipliers, gradient descent, etc.

In this sense, model families corresponding to decreasing value of $\lambda$ (gradually relaxed constraints) are nested. To find a good choice of $\lambda$, we look for the low point on test error.

[pic]

## Hyperparameter Tuning

The parameter $\lambda$ introduced above is a hyperparameter. Hyperparameters are tuned on validation/tuning data, instead of training data.

Assumption
: training, validation, test data are drawn from the true distribution, so validation is a proxy for test.


- Automatically: Bayesian hyperparameter optimization, gradient descent on held-out set
- Learning to learn: meta-learning



## The Curse of Dimensionality

- parametric models for classification and regression
- curse of dimensionality
- non-parametric: K-nearest neighbors
