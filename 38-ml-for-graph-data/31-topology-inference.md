# Topology Inference

Can we do inference on graphs (e.g. connectivity) like we can do for usual data matrix? Do we have the concepts and tools such as statistical consistency, efficiency, and robustness? Unfortunately, there is at present no single coherent body of formal results on
inference problems over graphs.

The frameworks developed in these settings naturally take various forms, as dictated by context, with differences driven primarily by the nature of the topology to be inferred and the type of data available.

## Link Prediction

Given
- full knowledge of all the vertices attributes $\mathbf{x}=\left(x_{1}, \ldots, x_{N_{v}}\right)^{\top}$
- status of some of the edges/non-edges $\boldsymbol{Y}^{obs}$

Infer
- the rest of the edges/non-edges $\boldsymbol{Y}^{miss}$, using $\boldsymbol{Y}^{obs}$ and $\boldsymbol{x}$.

Sometimes we need additional modeling for the mechanisms of missingness.
- missing at random: probability of missing $Y_{ij}$ only depends on the values of those other edge variables
- informative missingness: probability of missing $Y_{ij}$ depends on themselves

A basic framework:

$$
\mathbb{P}\left(\mathbf{Y}^{m i s s} \mid \mathbf{Y}^{o b s}=\mathbf{y}^{o b s}, \mathbf{X}=\mathbf{x}\right)
$$

But there are many challenges to predict $Y_{ij}^{miss}$ jointly. Many methods predict individual $Y_{ij}^{miss}$, as introduced below.

### Scoring

Scoring methods are based on the use of score functions. These methods are less formal than the model-based methods, but can be quite effective, and often serve as a useful starting point.

For each potential edge $(i, j) \in V^{(2)}_{miss}$ , a score $s(i, j)$ is computed. A set of predicted edges may then be returned by
- applying a threshold $s^*$ to these scores, or
- ordering them and keeping those pairs with the top $n^*$ values

There are many scores, designed to assess certain structural characteristics of a graph $G^{obs}$.
- $- \operatorname{dist}_{G^{obs}}(i, j)$: inspired by the small-world principal, more close, more likely to form an edge
- $\left\vert N_i^{obs} \cap N_j^{obs} \right\vert$: more common neighbors, more likely to form an edge
- $\frac{\left\vert N_i^{obs} \cap N_j^{obs} \right\vert}{\left\vert N_i^{obs} \cup N_j^{obs} \right\vert}$: a standardized version of the above value, called **Jaccard coefficient**
- $\sum_{k \in N_{i}^{obs} \cap N_{j}^{o b s}} \log \frac{1}{\left|N_{k}^{o b s}\right|}$: variation of the above, weighting more heavily those common neighbors of $i$ and $j$ that are themselves **not** highly connected.

There score functions only assess local structure in $G^{obs}$. For others defined through spectral characteristics of $G^{obs}$, see [SAND 261].

### Classification

Can we approach link prediction as a classification problem?
- (binary) labels: $\boldsymbol{y} ^{obs}$
- features: $\boldsymbol{x}^{obs}$ and $\boldsymbol{y} ^{obs}$
- predict: $\boldsymbol{Y} ^{miss}$.

#### Logistic Regression

A common choice is logistic regression.

$$
\log \left[
\frac{\mathbb{P}_{\beta}\left(Y_{i j}=1 \mid \mathbf{Z}_{i j}=\mathbf{z}\right)}{\mathbb{P}_{\beta}\left(Y_{i j}=0 \mid \mathbf{Z}_{i j}=\mathbf{z}\right)}
\right]=\beta^{\top} \mathbf{z}
$$

- $\boldsymbol{Z} _{ij}$ is a vector of explanatory variables indexed in the unordered pairs $(i, j)$.
  - the explanatory variables can be score functions introduced above, or
  - some transformation of $\boldsymbol{Y} ^{obs}_{(-ij)}$ and $\boldsymbol{X}$: $\mathbf{Z}_{i j}=\left(g_{1}\left(\mathbf{Y}_{(-i j)}^{o b s}, \mathbf{X}\right), \ldots, g_{K}\left(\mathbf{Y}_{(-i j)}^{o b s}, \mathbf{X}\right)\right)^{\top}$
- the coefficient $\boldsymbol{\beta}$ is assumed common to all pairs.

In prediction, we compare the predicted value vs some threshold, e.g. 0.5

$$
\mathbb{P}_{\hat{\beta}}\left(Y_{i j}^{m i s s}=1 \mid \mathbf{Z}_{i j}=\mathbf{z}\right)= \frac{\exp (\hat{\boldsymbol{\beta} } ^{\top} \boldsymbol{z} )}{1 + \exp (\hat{\boldsymbol{\beta} } ^{\top} \boldsymbol{z} )}
$$

Issues
- Need to consider the missing mechanism. If $\boldsymbol{Y} ^{miss}$ is **not** at random, the accuracy of the classification approach is will suffer.
- In a graph $Y_{ij}$ are usually not independent given explanatory variables $\boldsymbol{Z}$, which is assumed in logistic models (no formal work to date exploring the implications on prediction accuracy of ignoring possible dependencies in this manner). Introducing latent variable solve this issue, as discussed below

#### Latent Variables

The use of latent variables is an intuitively appealing way to indirectly model unobserved factors driving the formation of network structure. Let $\boldsymbol{M}$ be an unknown random, symmetric $N_v \times N_v$ matrix of latent variables, defined as

$$
\boldsymbol{M} = \boldsymbol{U} ^{\top} \boldsymbol{\Lambda} \boldsymbol{U} + \boldsymbol{E}
$$

- $\boldsymbol{U}$ is an $N_v \times N_v$ random orthonormal matrix,
- $\boldsymbol{\Lambda}$ is an $N_v \times N_v$ random diagonal matrix,
- $\boldsymbol{E}$ is a symmetric matrix of i.i.d. noise variables

Then each entry of $\boldsymbol{M}$ is

$$
M_{ij} = \boldsymbol{u} _i ^{\top} \boldsymbol{\Lambda} \boldsymbol{u} _j + \epsilon_{ij}
$$

Intuition: The latent variable matrix $\boldsymbol{M}$ is intended to capture effects of network structural characteristics or processes not already described by the observed explanatory variables $\boldsymbol{Z} _{ij}$. We add $M_{ij}$ as an explanatory variable. The model becomes

$$
\log \left[
\frac{\mathbb{P}_{\beta}\left(Y_{i j}=1 \mid \mathbf{Z}_{i j}=\mathbf{z}, M_{ij}=m\right)}{\mathbb{P}_{\beta}\left(Y_{i j}=0 \mid \mathbf{Z}_{i j}=\mathbf{z}, M_{ij}=m\right)}
\right]=\beta^{\top} \mathbf{z} + m
$$

Now $Y_{ij}$ are conditionally independent given $\boldsymbol{Z} _{ij}$ and $\boldsymbol{M} _{ij}$, but conditionally *dependent* given only the $\boldsymbol{Z} _{ij}$.

Distributions for $\boldsymbol{U} , \boldsymbol{\Lambda} , \boldsymbol{E}$.
- $\boldsymbol{U}$: uniform distribution on the space of all $N_v \times N_v$ orthonormal matrices
- $\boldsymbol{\Lambda} , \boldsymbol{E}$: multivariate Gaussian (facilitate MCMC sampling)

Prediction: compare the expected probability of $Y_{ij}=1$ with some threshold, which may be approximated numerically to any desired accuracy by the corresponding sample average of draws from the posterior indicated

$$
\mathbb{E}\left(\frac{\exp \left\{\beta^{T} \mathbf{Z}_{i j}+M_{i j}\right\}
}{1+\exp \left\{\beta^{T} \mathbf{Z}_{i j}+M_{i j}\right\}}
 \mid \mathbf{Y}^{o b s}=\mathbf{y}^{o b s}, \mathbf{Z}_{i j}=\mathbf{z}\right)
$$

Cons: MCMC computation cost, mainly driven by the need to draw $N_v ^2$ unobserved variables $U_{ij}$. Sol: let $\boldsymbol{U}$ have only $K$ non-zero column vectors for $K \ll N_v$, hence low-rank of $\boldsymbol{M}$. In fact $K=2, 3$ work well in practice. [SAND 200 201]

For a case study see [SAND pg.205]

## Association Networks

Non-trivial level of association (e.g. correlation) between certain characteristics of the vertices, but is itself unobserved and must be inferred from measurements reflecting these characteristics.

Given
- no knowledge of edge status anywhere
- relevant measurements at all of the vertices

Infer
- edge status using these measurements

## Tomographic Inference

Measurements are available only at vertices that are somehow at the ‘perimeter’ of the network, and it is necessary to infer the presence or absence of both edges and vertices in the ‘interior.’

Given
- measurements at only a particular subset of vertices

Infer
- topology of the rest
