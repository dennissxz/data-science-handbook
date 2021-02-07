# Taxonomy

There are various kinds machine learning models. A good taxonomy helps us understand this field. The criteria to create a taxonomy can be
- supervised / unsupervised
- parametric / non-parametric
- discriminative / generative


## Supervised vs Unsupervised

The easiest way to create a taxonomy is to look at the input and the objective of the learning tasks.

### Supervised Learning

In supervised learning, the input data set is $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^n$, where $\boldsymbol{x_i}$ is called features, and attributes or covariates $y_i$ is called labels. The objective is to predict $y_i$ by learning the relation between $\boldsymbol{x}_i$ and $y_i$, or we say it aims to build a mapping from feature space $\mathcal{X}$ to label space $\mathcal{Y}$, given training set $\mathcal{D} = \left\{ (\boldsymbol{x}_i , y_i) \right\}$ with $\boldsymbol{x}_i \in \mathcal{X} , y_i \in \mathcal{Y}$.

When $y_i$ is a real value variable, i.e., $\mathcal{Y} = \boldsymbol{\mathbb{R}}$ the learning tasks are called **regression**. When $y_i$ is categorical, i.e., $\mathcal{Y} = \left\{ 1, \ldots, C \right\}$, the learning tasks are called **classification**.

A model uses a function $f(\boldsymbol{x} ; \boldsymbol{\theta})$ parameterized by $\boldsymbol{\theta}$ to predict $\hat{y}$. In regression, this function returns the predicted value of $y$; in classification, it returns the per-class score $f_{y=c}(\boldsymbol{x}; \boldsymbol{\theta})$ for each class $c$, and make prediction by choosing the class with the highest score.


Learning

: There is a loss function $\ell$ penalizing for predicting $\hat{y}$ when eh ground truth is $y$. Learning is basically minimizing the **empirical loss**


  $$
  \boldsymbol{\theta}^{*}=\underset{\boldsymbol{\theta}}{\operatorname{argmin}} \frac{1}{N}  \sum_{i} \ell\left(f\left(\boldsymbol{x}_i ; \boldsymbol{\theta}\right), y_{i}\right)
  $$


  ```{note}
  - The coefficient $\frac{1}{N}$ is often omitted since it is a constant term and does not affect the optimization process.
  - The loss function $\ell$ is usually related to the specific task, but often is not exactly the same as the task loss (e.g., error rate, F1 score, etc.).
  ```

Assumptions
: - observations $(\boldsymbol{x}_i ,y_i)$ are drawn i.i.d. from a joint probability distribution $p(\boldsymbol{x} ,y)$
: - the joint distributions are the same in both training and test data.

The ultimate goal is to minimize the **expected loss**, aka. **risk**.

$$
\operatorname{E}_{(\boldsymbol{x} , y) \sim p(\boldsymbol{x} ,y)}\left[\ell \left(f \left( \boldsymbol{x}_i ; \boldsymbol{\theta}\right), y_{i} \right)\right]
$$

When the training set $\mathcal{D}$ is a good representatie of the underlying distribution $p(\boldsymbol{x} , y)$, the empirical loss serves as a proxy for the risk.

### Discriminative vs Generative

Models for **classification** can be divided into two types.

- generative: typically fit **per-class density** $p(\boldsymbol{x} \mid y)$ and **class prior** $p(y)$ to estimate the actual density or distribution $p(y, \boldsymbol{x} ; \boldsymbol{\theta} )$ by


$$
p(y \mid \boldsymbol{x} ; \boldsymbol{\theta}) = \frac{p( \boldsymbol{x} ; \boldsymbol{\theta})}{p(\boldsymbol{x})}  \propto p(y, \boldsymbol{x} ; \boldsymbol{\theta} )
$$

- discriminative
  - probabilistic: directly estimate the conditional probability $p(y \mid \boldsymbol{x} ; \boldsymbol{\theta} )$
  - non-probabilistic: directly optimize loss of $f(\boldsymbol{x}  ; \boldsymbol{\theta})$

When to use which? If $\operatorname{dim}(\boldsymbol{x})$ is large but only a few $X_i$'s in $\boldsymbol{x}$ are helpful to discriminate $y$, then modeling $p(\boldsymbol{x} \mid y)$ or $p(\boldsymbol{x})$ may be computationally intractable and unnecessary. It's better to directly estiamte $p(y\mid \boldsymbol{x} l \boldsymbol{\theta})$

On the other hand, if $p(y\mid \boldsymbol{x} ; \boldsymbol{\theta} )$ is hard to estimate while $p(\boldsymbol{x} \mid y)$ is easy to estimate (e.g. Gaussian), then we prefer generative models.


```{note}
Generative classification models can be seen as unsupervised learning (density estimation) in the service of a supervised objective.
```


### Unsupervised Learning

: In unsupervised learning, the input data set is $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^n$, i.e., no labels. The objective is to find interesting patterns in the data, aka knowledge discovery. There could be many tasks depending on what the patterns we are interested in. The two most common tasks are clustering and dimensionality reduction.

Let the objective function to be $J(\mathcal{D}; \boldsymbol{\theta})$. We can also add a regularizer term such that the optimization becomes


$$
\min \left\{ J(\mathcal{D} ; \boldsymbol{\theta} ) + \lambda R(\boldsymbol{\theta} ) \right\}
$$


```{note}
The function $J$ may or may not decompose over observations $\boldsymbol{x}_i $. If yes then we can write $\sum _i J(\boldsymbol{x}_i ; \boldsymbol{\theta} )$.
```

What is $J$? It depends on the pattern we are interested to find.



The four tasks mentioned above are summarized below.

| Task | Input | Objective | Application |
| - | - | - | - |
| Regression  | $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$ where $y_i\in \mathbb{R}$  |  given a new $\boldsymbol{x}_i$, predict  $\hat y_i$ | predict home price  |
| Classification  | $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$, where $y_i$ is categorical  |  given a new $\boldsymbol{x_i}$, predict $\hat{y_i}$ | predict disease |
| Clustering  | $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$, no labels  |   estimate the number of clusters $C$ and determine which cluster each observation $i$ belongs to | cluster users into groups |  
| Dimensionality reduction <br>  (representation learning) |  $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$, no labels |  reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the **essence** of the data | compression, visualization, downstream learning tasks  |

There are other unsupervised learning tasks, such as
- discovering graph structure
- matrix completion
- generation / density estimation

### Semi-supervised Learning

Besides, there are also** semi-supervised learning** tasks, where there are lots of unlabeled data $\boldsymbol{X} _U$ and a little labeled data $\boldsymbol{X}_L$. The objective is to learn representation from unlabeled data and use the learned representation to improve performance of supervised learner on labeled data.

```{note}
- Supervised learning usually doesn't optimize the loss we really care about.
- The boundary between supervised and unsupervised learning is fuzzy. They are somehow "differently supervised".
```

## Parametric vs Non-parametric

Whether there is a fixed number of parameters in the model can be another criteria to create a taxonomy.

Parametric
: Parametric models have a fixed number of parameters
  - pros: faster to use
  - cons: stronger assumptions

Non-parametric
: In non-parametric models, the number of parameters grow with the amount of training data.
  - pros: flexible, no or few assumptions
  - cons: computationally intractable for large datasets



Table 8.1 in the book *Machine Learning: a Probablistic Perspective* summarized the models as below.

| Model                                | Classif/reg | Discr/Gen | Param/Non |
|--------------------------------------|-------------|-----------|-----------|
| Discriminant analysis                | Classif     | Gen       | Param     |
| Naive Bayes classifier               | Classif     | Gen       | Param     |
| Tree-augmented Naive Bayes clssifier | Classif     | Gen       | Param     |
| Linear regression                    | Regr        | Discrim   | Param     |
| Logistic regression                  | Classif      | Discrim   | Param          |
| Sparse linear/ logistic regression | Both      | Discrim   | Param           |
| Multilayer perceptron (MLP) / Neural network | Both      | Discrim   | Param           |
| Conditional random field (CRF) | Classif      | Discrim   | Param           |
| $K$ nearest neighbor classifier | Classif      | Gen  | Non           |
| (Infinite) Mixture Discriminant analysis | Classif      | Gen  | Non           |
| Classification and regression trees (CART) | Both      | Discrim  | Non           |
| Boosted model | Both      | Discrim  | Non           |
| Sparse kernelized lin/logreg (SKLR) | Both      | Discrim  | Non           |
| Support vector machin (SVM) | Both      | Discrim  | Non           |
| Gaussian processes (GP) | Both      | Discrim  | Non           |
| Smoothing splines | Both      | Discrim  | Non           |


## Others

Reinforcement Learning

Multi-agent Learning
