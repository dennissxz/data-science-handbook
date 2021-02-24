# Semi-supervised Learning

(Figure credits: Jerry Zhu ICML 2007 tutorial unless otherwise indicated)

## Question

We often have access to a small amount of labeled data plus a large amount of unlabeled data.

Why not just get more labeled data?
- Some labeled data is easy to get more of, e.g. tagged images
- Some is expensive/time consuming/difficult/unethical to get:
  - Medical measurements (e.g. scans, neural recordings)
  - Phonetic transcription of speech: roughly 400x real time, requires
experts
  - Text parsing: takes years to collect a few 1000 sentences’ worth

We introduce semi-supervised learning techniques for classification tasks.

Notions:

- Input $x$, label $y$
- Function to be learned: $f:\mathcal{X} \rightarrow \mathcal{Y}$
- Labeled data of size $\ell$: $(X_\ell, X_\ell) = \left\{ (x_{1:\ell}, y_{1:\ell}) \right\}$
- Unlabeled data $X_u = \left\{ x_{\ell + 1:n} \right\}$, where $\ell <<n$
- Test data $X_{test} = {x_{n+1:}}$


If we have labeled data, how can unlabeled data help?

Unlabeled data tells us something about the distribution of the data, and therefore where “reasonable” class boundaries may be.


## Review

- **Transductive learning**

  Goal is to label the unlabeled training data (only. not necessary learn a labeling function)

- **Inductive learning** (semi-supervised learning)

  Goal is to learn a function that can be applied to new test data.

- **Weak labels/distant supervision**

  We have labels, but they are not exactly for the target task

  - pairs from “Same/different” class, instead of class labels

  - subset of latent variables labeled but not all (e.g. object category
in an image, but not boundaries of that object)

  - “downstream” labels (e.g. click-through counts instead of search
result correctness)


## Self-training

Algorithm

- Train classifier $f$ over labeled data $\mathcal{L}$
- Predict on some unlabeled data $x_u \in X_u$
- Add some $(x_u, f(x_u))$ to labeled data, e.g. with high confidence
- Repeat


## Semi-supervised Generative Models

Recall a generative model with on Gaussian per class can be used for classification.


$$
\begin{aligned}
p(x, y \mid \theta) &=p(y \mid \theta) p(x \mid y, \theta) \\
&=w_{y} \mathcal{N}\left(x ; \mu_{y}, \Sigma_{y}\right)
\end{aligned}
$$

Train by ML (closed-form solution)

Classify by finding the label with maximum probability $p(y \mid x, \theta)=\frac{p(x, y \mid \theta)}{\sum_{y^{\prime}} p\left(x, y^{\prime} \mid \theta\right)}$


## Cluster-then-label

## Graph-based algorithms

Transductive method, no out-of-sample prediction.

## Co-Training























0
