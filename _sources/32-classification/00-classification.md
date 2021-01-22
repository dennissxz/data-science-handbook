# Classification


Recall that a classification model computes per-class score $f_{y=c}(\boldsymbol{x} ; \boldsymbol{\theta} )$. If perfect separation of training data is impossible, it is intractable to learn a separator minimizing the number of mistakes. Instead, we learn classifiers by minimizing surrogate loss functions. Two popular surrogate losses are

- log loss (aka. cross entropy): assumes that $f_y (bx
    ; \boldsymbol{\theta} ) \propto \log p(y\mid bx
      )$,
  $$
  \ell\left(\boldsymbol{\theta}, \mathbf{x}_{i}, y_{i}\right)=-\log p\left(y_{i} \mid \mathbf{x}_{i} ; \boldsymbol{\theta}\right)
  $$

- hinge loss (no probabilistic assumption)


  $$
  \ell\left(\boldsymbol{\theta}, \mathbf{x}_{i}, y_{i}\right)=\max \left\{0,1+\max _{c \neq y_{i}} f_{c}\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right)-f_{y_{i}}\left(\mathbf{x}_{i} ; \boldsymbol{\theta}\right)\right\}
  $$

- metrics
  - AUC
  - F1 score
  - see [confusion-matrix](../13-statistics/33-confusion-matrix.md)
