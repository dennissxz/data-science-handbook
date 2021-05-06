# Classification

Probabilistic classification can be carried out in many different approaches, and the criteria of goodness of the classifier can be at various levels. For example,
- Use discriminant functions
- Model per-class score $f_{y=c}(\boldsymbol{x} ; \boldsymbol{\theta} )$
- Model the posterior distribution $\mathbb{P} (c \vert \boldsymbol{x})$

When learning the classifiers, some loss functions are used. Two popular surrogate losses are

- log-loss (aka. cross entropy): assumes that $f_y (\boldsymbol{x} ; \boldsymbol{\theta} ) \propto \log p(y\vert \boldsymbol{x} )$,

  $$
  \ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, y_{i}\right)=-\log p\left(y_{i} \vert \boldsymbol{x}_{i} ; \boldsymbol{\theta}\right)
  $$

- hinge loss (no probabilistic assumption)

  $$
  \ell\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}, y_{i}\right)=\max \left\{0,1+\max _{c \neq y_{i}} f_{c}\left(\boldsymbol{x}_{i} ; \boldsymbol{\theta}\right)-f_{y_{i}}\left(\boldsymbol{x}_{i} ; \boldsymbol{\theta}\right)\right\}
  $$

Empirical metrics
- AUC
- F1 score
- see [confusion-matrix](../13-statistics/33-confusion-matrix.md)
