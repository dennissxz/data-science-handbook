# Processes on Graphs

In this section we consider the network graph to be known. We focus on models and associated problems of statistical inference and prediction for **processes** defined on network graphs.

A vertex attribute can be associated with the interactions with and the attributes of other vertices
- the behaviors and beliefs of people can be strongly influenced by their social interactions;
- proteins that are more similar to each other, with respect to their DNA sequence
information, often are responsible for the same or related functional roles in a cell;
- computers more easily accessible to a computer infected with a virus may in turn
themselves become more quickly infected;
- the relative concentration of species in an environment (e.g., animal species in a forest or chemical species in a vat) can vary over time as a result of the nature of the relationships among species.

Quantities associated with such phenomena can usefully be thought of as stochastic processes defined on network graphs. Formally, we define a collection of r.v. $\left\{ X_i \right\}$, for $i \in V$, or $\left\{ X_i(t) \right\}$, for $i \in V$ and $t \in \mathbb{T}$, where $\mathbb{T}$ is a (discrete or continuous) range of times. We refer to $\left\{ X_i \right\}$ as static processes and $\left\{ X_i(t) \right\}$ as dynamic processes.

Problems include modeling, inference of parameters, and prediction.

## Markov Random Fields

We introduce the class of Markov random field (MRF) models for vertex attributes on graphs and discuss statistical issues relating to inference and prediction with
these models.

The concept of an MRF can be seen as a generalization of a Markov chain. It has its root in statistical mechanics, e.g. Ising [211] on ferromagnetic fields. It is also used in spatial statistics and in image analysis.

### Specification

Let $G = (V, E)$ be a graph and $\boldsymbol{X} = [X_1, \ldots, X_{N_v}] ^{\top}$ be a collection of random variables defined on $V$. We say that $\boldsymbol{X}$ is a Markov random field if

$$
\mathbb{P} (\boldsymbol{X} = \boldsymbol{x} )  > 0, \quad \text{for all possible outcomes } \boldsymbol{x}
$$

and the Markov condition holds

$$
\mathbb{P}\left(X_{i}=x_{i} \mid \boldsymbol{X}_{(-i)}=\boldsymbol{x}_{(-i)}\right)=\mathbb{P}\left(X_{i}=x_{i} \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right) \qquad (*)
$$

where $\boldsymbol{X} _{(-i)}$ is the vector $[X_1, \ldots, X_{i-1}, X_{i+1}, \ldots, X_{N_v}] ^{\top}$ and $\boldsymbol{X} _{\mathscr{N}_v}$ is the vector of all $X_j$ for neighbors $j \in \mathscr{N}_i$. The Markov condition asserts that $X_i$ is conditionally independent of all other non-neighbor vertices $X_k$, given the values of its neighbors $X_j$, where the neighborhood structure is determined by $G$.


#### R.t. Gibbs random fields

Gibbs random fields define a random vectors $\boldsymbol{X}$ with distributions of the form

$$
\mathbb{P}(\boldsymbol{X}=\boldsymbol{x})=\left(\frac{1}{\kappa}\right) \exp \{U(\boldsymbol{x})\}
$$

where
- $\boldsymbol{\kappa}=\sum_{\boldsymbol{X}} \exp \{U(\boldsymbol{x})\}$ is called the **partition function**.
- $U(\cdot)$ is called the **energy function**,
  - It can be decomposed as a sum over cliques (complete subgraphs) in $G$ in the form $U(\boldsymbol{x})=\sum_{c \in \mathcal{C}} U_{c}(\boldsymbol{x})$, where $\mathcal{C}$ is the set of all cliques of all sizes in $G$. A clique of size 1 consists of just a single vertex $v \in V$.
  - The functions $U_c( \cdot)$ are called **clique potentials**.

Under appropriate conditions, MRFs are equivalent to Gibbs random fields [SAND 36]. It can be shown that the Markov condition (*) holds

$$
\begin{aligned}
\mathbb{P}\left(X_{i}=x_{i} \mid \boldsymbol{X}_{(-i)}=\boldsymbol{x}_{(-i)}\right) &=\frac{\mathbb{P}(\boldsymbol{x})}{\sum_{\boldsymbol{X}^{\prime}: \boldsymbol{x}_{(-i)}^{\prime}=\boldsymbol{x}_{(-i)}} \mathbb{P}\left(\boldsymbol{x}^{\prime}\right)} \\
&=\frac{\exp \left\{\Sigma_{c \in \mathscr{C}_{i}} U_{c}(\boldsymbol{x})\right\}}{\sum_{\boldsymbol{X}^{\prime}: \boldsymbol{x}_{(-i)}^{\prime}=\boldsymbol{x}_{(-i)}}  \exp \left\{\sum_{c \in \mathscr{C}_{i}} U_{c}\left(\boldsymbol{x}^{\prime}\right)\right\}}
\end{aligned}
$$

where
- $\mathcal{C}$ is the set of all cliques involving vertex $i$,
- the denominator summation is over all vectors $\boldsymbol{x} ^\prime$ such that the sub-vector $\boldsymbol{x} _{(-i)} ^\prime$ is fixed to be $\boldsymbol{x} _{(-i)}$.

Cons
: the expression can be extremely complicated. Though good for richness on the one hand, hard for interpretability and computations. Solutions:
  - simplified by assumptions of homogeneity, $U_c$ is assumed not to depend on the particular positions of the clique $c \in \mathcal{C}$.
  - cliques of only a limited size re defined to have non-zero partition functions $U_c$, which reduces the complexity of the decomposition $U(\boldsymbol{x})=\sum_{c \in \mathcal{C}} U_{c}(\boldsymbol{x})$.

#### R.t. Exponential Families

Besag [SAND 36] suggested introducing additional assumptions on MRF
- $U_c \ne 0$ for only cliques $c \in \mathcal{C}$ of size one or two, aka 'pariwise-only dependence'
- the conditional probability $\mathbb{P}\left(X_{i}=x_{i} \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)$ have an exponential family form.

Under these conditions, Besag developed **auto-models** where the energy function takes the form

$$
U(\boldsymbol{x})=\sum_{i \in V} x_{i} H_{i}\left(x_{i}\right)+\sum_{\{i, j\} \in E} \beta_{i j} x_{i} x_{j}
$$

for some functions $H_i(\cdot)$ and coefficients $\left\{ \beta_{ij} \right\}$.


:::{admonition,note} Auto-logistic Model

In particular, suppose $X_i$ are binary random variables. Under appropriate normalization conditions of $H_i$, the energy function is equivalent to

$$
U(\boldsymbol{x})=\sum_{i \in V} \alpha_{i} x_i +\sum_{\{i, j\} \in E} \beta_{i j} x_{i} x_{j}
$$

It is easy to see that

$$
\mathbb{P}\left(X_{i}=1 \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)=\frac{\exp \left(\alpha_{i}+\sum_{j \in \mathscr{N}_{i}} \beta_{i j} x_{j}\right)}{1+\exp \left(\alpha_{i}+\sum_{j \in \mathscr{N}_{i}} \beta_{i j} x_{j}\right)}
$$

Therefore, we obtain a logistic regression form, with covariates being the neighboring of $x_j$'s. This is called the **auto-logistic model**. The Ising model in statistical mechanics is a particular case of this model, with $G$ defined to be a regular lattice.

Assumptions of homogeneity can further simplify this model. For example, specifying that $\alpha_i = \alpha$ and $\beta_{ij} = \beta$, the probability reduces to

$$
\mathbb{P}\left(X_{i}=1 \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)=\frac{\exp \left(\alpha+\sum_{j \in \mathscr{N}_{i}} \beta x_{j}\right)}{1+\exp \left(\alpha+\sum_{j \in \mathscr{N}_{i}} \beta x_{j}\right)}
$$

The log-odds is then

$$
\log \frac{\mathbb{P}\left(X_{i}=1 \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)}{\mathbb{P}\left(X_{i}=0 \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)}=\alpha+\beta_{1} \sum_{j \in \mathscr{N}_{i}} x_{j}
$$

which scales linearly in the number of neighbors $j$ of $i$ with the value $X_j=1$. Similarly, the log-odds can be made to scale linearly as well in the number of neighbors $j$ of $i$ with the value $X_j = 0$, through the parameter choice $\alpha_i = \alpha + \left\vert \mathcal{N} _i \right\vert \beta_2$ and $\beta_{ij} = \beta_1 - \beta_2$.

$$
\log \frac{\mathbb{P}\left(X_{i}=1 \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)}{\mathbb{P}\left(X_{i}=0 \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)}=\alpha+\beta_{1} \sum_{j \in \mathscr{N}_{i}} x_{j}+\beta_{2} \sum_{j \in \mathscr{N}_{i}}\left(1-x_{j}\right)
$$

Other variations follow similarly.

:::

The auto-logistic model has been extended to the case where the $X_i$ take on values $\left\{ 0,1, \ldots, m \right\}$, for arbitrary positive integer $m$, yielding a class of models called **multilevel logistic** or multi-color models. See Strauss [371]. Other auto-models of interest include the **auto-binomial**, the **auto-Poisson**, and the **auto-Gaussian**. In auto-Gaussian, aka **Gaussian Markov random fields**, the PMF is replaced by Gaussian PDF. The conditional expectations and variances take the form

$$\begin{aligned}
\mathbb{E}\left(X_{i} \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)
&=\alpha_{i}+\sum_{j \in \mathscr{N}_{i}} \beta_{i j}\left(x_{j}-\alpha_{j}\right) \\
\mathbb{V}\left(X_{i} \mid \boldsymbol{X}_{\mathscr{N}_{i}}=\boldsymbol{x}_{\mathscr{N}_{i}}\right)
&= \sigma^2
\end{aligned}$$

Under the conditions that $\beta_{ii}=0$ and $\beta_{ij}=\beta_{ji}$, the joint distribution of $\boldsymbol{X}$ is [multivariate Gaussian](multi-gaussian), with mean vector  $\boldsymbol{\mu} = \boldsymbol{\alpha}$ and covariance matrix $\boldsymbol{\Sigma} = \sigma^2 (\boldsymbol{I} - \boldsymbol{B} )^{-1}$ where $\boldsymbol{B} = [\beta_{ij}]$. If we impose homogeneity assumptions such as $\alpha_i = \alpha$ and $\beta_{ij} = \beta$, then $\boldsymbol{\mu} = \alpha \boldsymbol{1}$ and $\boldsymbol{\Sigma} = \sigma^2 (\boldsymbol{I} - \beta \boldsymbol{A} )^{-1}$ where $\boldsymbol{A}$ is the adjacency matrix. See [335] for details.

MRFs can be used alone or to specify just a component (e.g. prior) of a larger complex model.

### Inference

Given the probability distribution of $\boldsymbol{X}$ parameterized by $\boldsymbol{\theta}$,

$$
\mathbb{P}_{\theta}(\boldsymbol{X}=\boldsymbol{x})=\left(\frac{1}{\kappa(\boldsymbol{\theta} )}\right) \exp \{U(\boldsymbol{x} ; \boldsymbol{\theta})\}
$$

How can we infer $\boldsymbol{\theta}$ from observed $\boldsymbol{X}$?

#### Maximum Pseudo-likelihood

If we use MLE, the the joint log-likelihood can be expressed in the simple form

$$
\log \mathbb{P}_{\boldsymbol{\theta}}(\mathbf{X}=\mathbf{x})=U(\mathbf{x} ; \boldsymbol{\theta} )-\log \kappa(\boldsymbol{\theta})
$$

But the summation in the partition function $\boldsymbol{\kappa}(\boldsymbol{\theta})=\sum_{\boldsymbol{X}} \exp \{U(\boldsymbol{x}, \boldsymbol{\theta})\}$ is impossible to compute for all but the smallest of the problems. We resort to maximum pseudo-likelihood, that maximize the (log) joint conditional probabilities

$$
\log \prod_{i=1}^n \mathbb{P}_{\theta}\left(X_{i}=x_{i} \mid \mathbf{X}_{(-i)}=\mathbf{x}_{(-i)}\right) = \sum_{i \in V} \log \mathbb{P}_{\theta}\left(X_{i}=x_{i} \mid \mathbf{X}_{\mathscr{N}_{i}}=\mathbf{x}_{\mathscr{N}_{i}}\right)
$$

which does not involve the partition function $\boldsymbol{\kappa}(\boldsymbol{\theta})$ and is easier to compute.

Cons: since the LHS of the two equations above do not equal, MLPE can produce estimates that differ substantially from the MLE when the **dependencies** inherent in the full joint distribution are too substantial to be ignored.


:::{admonition,note} MLE and MPLE in the Auto-logistic model

Recall that in auto-logistic model under homogeneity assumption we have

$$
U(\boldsymbol{x}; \alpha, \beta)=\sum_{i \in V} \alpha x_{i}+\sum_{\{i, j\} \in E} \beta x_{i} x_{j}
$$

Hence, MLE is defined as

$$
\left.(\hat{\alpha}, \hat{\beta})_{M L E}=\arg \max _{\alpha, \beta}\left[\alpha M_{1}(\mathbf{x})+\beta M_{11}(\mathbf{x})-n \log \kappa(\alpha, \beta)\right)\right]
$$

where $M_1(\boldsymbol{x})$ is the number of vertices with attribute value 1, and $M_{11}(\boldsymbol{x} )$ is the number of pairs of vertices where both attribute values 1. Computation of the last term $\kappa(\alpha, \beta)$ is prohibitive, which requires evaluation of $M_1$ and $M_{11}$ across all $2^{N_v}$ number of binary vectors $\boldsymbol{x}$ of length $N_v$.

In contrast, the MPLE is

$$
(\hat{\alpha}, \hat{\beta})_{M P L E}=\arg \max _{\alpha, \beta}\left\{\alpha M_{1}(\mathbf{x})+\beta M_{11}(\mathbf{x})-\sum_{i=1}^n \log \left[1+\exp \left(\alpha+\beta \sum_{j \in \mathscr{N}_{i}} x_{j}\right)\right]\right\}
$$

where the last term can be computed easily. The overall estimate can be computed using standard software for logistic regression, with the $N_v$ pairs of response $x_i$ and predictor $\sum_{j \in \mathscr{N} _i }x_j$. (??)

Other parameterization and variation of auto-logistic model have analogous formulas. In general, if we only have one observation $\boldsymbol{x}$, some sort of homogeneity assumption will be necessary to, at the very least, make the model identifiable.

:::

#### Coding Methods




#### Mean-field Methods



### Prediction

\mathscr{C} -> \mathcal{C}
\mathscr{N}_i -> \mathscr{N}_i

.


.


.


.


.


.


.


.
