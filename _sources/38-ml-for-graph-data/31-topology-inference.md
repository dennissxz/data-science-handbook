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

- $\boldsymbol{Z} _{ij}$ is a vector of explanatory variables indexed in the unordered pairs $(i, j)$. In general it is some transformation of $\boldsymbol{Y} ^{obs}_{(-ij)}$ and/or $\boldsymbol{X}$: $\mathbf{Z}_{i j}=\left(g_{1}\left(\mathbf{Y}_{(-i j)}^{o b s}, \mathbf{X}\right), \ldots, g_{K}\left(\mathbf{Y}_{(-i j)}^{o b s}, \mathbf{X}\right)\right)^{\top}$
  - network structure measures using $\boldsymbol{Y} ^{obs}_{-ij}$, e.g. score functions introduced above
  - similarity measures between $X_{ik}$ and $X_{jk}$ for some (univariate) vertex attribute $k$.
    - additive $X_{ik} + X_{jk}$ for continuous values
    - indicator $\mathbb{I} \left\{ X_{ik} = X_{jk} \right\}$ for discrete values
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

Intuition: The latent variable matrix $\boldsymbol{M}$ is intended to capture effects of network structural characteristics or processes not already described by the observed explanatory variables $\boldsymbol{Z} _{ij}$. We add $M_{ij}$ as an explanatory variable (random??). The model becomes

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

For a case study see [SAND pg.205].

## Association Networks

Non-trivial level of association (e.g. correlation) between certain characteristics of the vertices, but is itself unobserved and must be inferred from measurements reflecting these characteristics.

Given
- no knowledge of edge status anywhere
- relevant measurements at all of the vertices $\left\{ \boldsymbol{x}_1, \ldots, \boldsymbol{x}_{N_v} \right\}$

Infer
- edge status $\boldsymbol{Y}$ using these measurements $\boldsymbol{x}$

### Correlation Networks

An intuitive measure of similarity between a vertex pair $(i, j)$ is correlation.

$$
\operatorname{sim}(i, j)  = \rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}
$$

#### Buildup

Suppose for each vertex, we have $n$ independent observations $\left\{ x_{i1}, \ldots, x_{in} \right\}$, e.g. gene expression levels from $n$ experiments. We can then form an $n \times N_v$ matrix $\boldsymbol{X}$, and compute the sample covariance matrix $\hat{\Sigma}=\frac{1}{n-1}(\mathbf{X}-\overline{\mathbf{X}})^{\top}(\mathbf{X}-\overline{\mathbf{X}})$, and hence obtain the entries $\hat{\sigma}$ and compute $\hat{\rho}$.

The corresponding association graph $G$ is the graph with edge set

$$
E=\left\{\{i, j\} \in V^{(2)}: \rho_{i j} \neq 0\right\}
$$

Hence, the the task is to infer the set of non-zero correlations, which can be approached through hypotheses testing


$$
H_{0}: \rho_{i j}=0 \quad \text { versus } \quad H_{1}: \rho_{i j} \neq 0
$$

Problems
- what test statistics?
- whats the null distribution of that test statistic?
- there are $N_v (N_v - 1)/2$ potential edges, which implies multiple testing problem.

#### $p$-value

If $(X_i, X_j)$ follow bivariate Gaussian, then $\hat{\rho}_{ij}$ under $H_0: \rho_{ij}=0$ has a closed-form but the computation of $p$-values is hard. Therefore, some transformed versions of $\hat{\rho}_{ij}$ may be preferable
- $z_{i j}=\frac{\hat{\rho}_{i j} \sqrt{n-2}}{\sqrt{1-\hat{\rho}_{i j}^{2}}} \sim t_{n-1}$, and under $H_0$ is it robust to departures of $X_i$ from Gaussianity.
- $z_{i j}=\tanh ^{-1}\left(\hat{\rho}_{i j}\right)=\frac{1}{2} \log \left[\frac{\left(1+\hat{\rho}_{i j}\right)}{\left(1-\hat{\rho}_{i j}\right)} \right]$ Fisher transformation.
  - for bivariate Gaussian pairs, the distribution of $z_{ij}$ does not have a simple exact form. But under $H_0$ this distribution is well approximated by $\mathcal{N} (0, \frac{1}{n-3} )$ even for moderately large $n$.

Permutation methods can also be used, but is computationally intensive for large $N_v$.

#### Multiple Testing

Recall the false discovery rate is defined to be

$$
\mathrm{FDR}=\mathbb{E}\left(\frac{R_{\text {false }}}{R} \mid R>0\right) \mathbb{P}(R>0)
$$

where $R$ is the number of rejections among our tests and $R_{\text {false }}$ is the number of false rejections.

To guarantee $\mathrm{FDR} \le \gamma$, we use the original method proposed by Benjamini and Hochberg [SAND 33],
- sort the $p$-values from our $N =N_v (N_v−1)/2$ tests, yielding a sequence $p_{(1)}\le p_{(2)} \le \ldots \le p_{(N)}$,
- reject the null hypothesis for all potential edges for which $p_{(k)} \leq(k / N) \gamma$.

Alternatively, we can use Storey [SAND 370] method and declare edges to be present using a particular $q$-value. Then only $qN$ of the edges will be included erroneously.

When dependency of tests [??] exists, the first method still holds, and there are other methods.

### Partial Correlation Networks

If it is felt desirable to construct a graph $G$ where the inferred edges are more reflective of direct influence among vertices, rather than indirect influence through some common neighbor, the notion of partial correlation becomes relevant.

#### Partial Correlation

Definition (Partial correlation)
: The partial correlation of attributes $X_i$ and $X_j$ of vertices $i, j \in V$ w.r.t. the attributes $X_{k_1}, \ldots, X_{k_m}$ of vertices $k_1, \ldots, k_m \in V \setminus \left\{ i, j \right\}$, is the correlation between $X_i$ and $X_j$ left over, after adjusting for those effects common to both. Let $S_m = \left\{ k_1, \ldots, k_m \right\}$, the partial correlation of $X_i$ and $X_j$ adjusting for $\boldsymbol{X} _{S_m} = (X_{k_1}, \ldots, X_{k_m}) ^{\top}$ is defined as

  $$
  \rho_{i j \mid S_{m}}=\frac{\sigma_{i j \mid S_{m}}}{\sqrt{\sigma_{i i\mid S_{m}} \sigma_{j j \mid S_{m}}} }
  $$

To compute it, let $\boldsymbol{W} _1 = (X_i, X_j) ^{\top}$ and $\boldsymbol{W} _2 = \boldsymbol{X} _{S_m}$. We can partition the covariance matrix to

$$
\operatorname{Cov}\left(\begin{array}{l}
\mathbf{W}_{1} \\
\mathbf{W}_{2}
\end{array}\right)=\left[\begin{array}{ll}
\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}
\end{array}\right]
$$

Then the $2 \times 2$ partial covariance matrix is

$$
\boldsymbol{\Sigma}_{11 \mid 2}=\boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}
$$

The values $\sigma_{ii\vert S_m}, \sigma_{jj\vert S_m}$ and $\sigma_{ij\vert S_m} = \sigma_{ji\vert S_m}$ are diagonal and off-diagonal elements of $\boldsymbol{\Sigma}_{11 \mid 2}$

In particular,
- if $m=0$, the partial correlation reduces to the Pearson correlation.
- if $\left(X_{i}, X_{j}, X_{k_{1}}, \ldots, X_{k_{m}}\right)^{\top}$ has a multivariate Gaussian, then $\rho_{ij \vert S_m}=0$ if and only if $X_i$ and $X_j$ are independent conditional on $\boldsymbol{X} _{S_m}$.


:::{admonition,note} Computation Issue of $\hat{\rho}_{i j \mid S_{m}}$

- To compute $\rho_{ij \mid S_m}$ for all $S_m$ is hard. It is more computationally efficient to use recursive expressions between $\rho_{ij \mid S_m}$ and $\rho_{ij \mid S_{m-1}}$, see Anderson [SAND 11].
- If $m <n$ is large w.r.t. $n$, then $\hat{\rho}_{i j \mid S_{m}}$ is a bad poor estimates of $\rho_{i j \mid S_{m}}$.
- $m=2$ is advocated in the context of inference of biochemical networks.
- An algorithmic definition of this value is that it is the result of
  1. performing separate multiple linear regressions of the observations of $X_i$ and $X_j$, respectively, on the observed values of $\boldsymbol{X} _{S_m}$, and then
  1. computing the empirical Pearson correlation between the two resulting sets of residuals.

:::

For more general distributions, however, zero partial correlation will not necessarily imply independence (the converse, of course, is still true).

#### Buildup

Given $m$, there are many ways to define edge set using partial correlations. For instance, there is an edge $e(i,j)$ iff the partial correlation $\rho_{i j \mid S_{m}} \neq 0$ regardless of which $m$ other vertices are conditioned upon.

  $$E=\left\{\{i, j\} \in V^{(2)}: \rho_{i j \mid S_{m}} \neq 0 \ \forall \ S_{m} \in V_{\backslash\{i, j\}}^{(m)}\right\}$$

The testing problem is then

$$
H_{0}: \rho_{i j \mid S_{m}}=0 \quad \text { for some } \quad S_{m} \in V_{\backslash\{i, j\}}^{(m)}
$$

versus

$$
H_{1}: \rho_{i j \mid S_{m}} \neq 0 \quad \text { for all } \quad S_{m} \in V_{\backslash\{i, j\}}^{(m)}
$$

Then we select a test statistic, construct an appropriate null distribution, and adjust for multiple testing, as the correlation networks above.

#### $p$-value

The above test can be considered as a collection of smaller testing sub-problems of the form

$$
H_{0}^{\prime}: \rho_{i j \mid S_{m}}=0 \quad \text { versus } \quad H_{1}^{\prime}: \rho_{i j \mid S_{m}} \neq 0
$$

Under the joint Gaussian assumption, the null empirical distribution $\hat{\rho}_{ij \mid S_m}$ is known but hard to compute the $p$-value. Fisher transformation can also be used here

$$z_{i j \vert S_m}=\tanh ^{-1}\left(\hat{\rho}_{i j\vert S_m}\right)=\frac{1}{2} \log \left[\frac{\left(1+\hat{\rho}_{i j\vert S_m}\right)}{\left(1-\hat{\rho}_{i j\vert S_m}\right)} \right] \rightarrow \mathcal{N} \left( 0, \frac{1}{n-m-3} \right)$$

Then, we can aggregate the $p$-values from sub-problems and define

$$
p_{i j, \max }=\max \left\{p_{i j \mid S_{m}}: S_{m} \in V_{\backslash\{i, j\}}^{(m)}\right\}
$$

to be the $p$-value for the original testing problem.

:::{admonition,warning} Different from Correlation

In practice, we may see
- significant $\rho_{ij} > 0$ but insignificant $\rho_{ij \mid S_m}$, or
- both significant $\rho_{ij} > 0$ and $\rho_{ij \mid S_m} < 0$, i.e. reverse sign after conditioning.

:::

#### Multiple Testing

Given the full collection of $\left\{ p_{ij, \max} \right\}$, over all potential edges $(i, j)$, an FDR procedure may be applied to this collection to choose an appropriate testing threshold, analogous to the manner described above.





### Case Study of Gene

Example: Though experimentally infeasible, can we construct the gene regulatory (activation or repression) networks as a problem of network inference, given measurements sufficiently reflective of gene regulatory activity?

Definition
: - **Genes** are sets of segments of DNA that encode information necessary to the proper functioning of a cell.
  - such information is utilized in the **expression** of genes, whereby biochemical products, in the form of RNA or proteins, are created
  - The **regulation** of a gene refers to the control of its expression.
  - A gene that plays a role in controlling gene expression at transcription stage (DNA is copied to RNA) is called a **transcription factor** (TF), and the genes that are controlled by it, gene **targets**.
  - The problem of inferring regulatory interactions among genes in this context refers to the identification of **TF/target gene pairs**.

Measurements
- The relative levels of RNA expression of genes in a cell, under a given set of conditions, can be measured efficiently on a genome-wide scale using **microarray** technologies.
- In particular, for each gene $i$, the vertex attribute vector $\boldsymbol{x}_i \in \mathbb{R} ^m$ typically consists of RNA relative expression levels measured for that gene over a compendium of $m$ experiments.

Challenge
- a TF can actually be a target of another TF. And so direct correlation between measurements of a TF and a gene target may actually just be a reflection of the regulation of that TF by another TF. Sol: use partial correlation

## Tomographic Inference

Measurements are available only at vertices that are somehow at the ‘perimeter’ of the network, and it is necessary to infer the presence or absence of both edges and vertices in the ‘interior.’

Given
- measurements at only a particular subset of vertices

Infer
- topology of the rest
