# Random Graph Models

## Background

By a model for a graph we mean a collection

$$
\left\{ \mathbb{P} \theta (G), G \in \mathcal{G}: \theta \in \Theta  \right\}
$$

where
- $\mathcal{G}$ is a collection ('ensemble') of possible graphs
- $\mathbb{P}_\theta$ is a probability distribution on $\mathcal{G}$
- $\theta$ is a vector of parameters ranging over values in $\Theta$.

### Estimate $\eta(G)$

In traditional statistical sampling theory, there are two main approaches to constructing estimates of population parameters $\eta(G)$ from a sample $G^*$: design-based and model-based.
- design-based: inference is based entirely on the random mechanism by which
a subset of elements were selected from the population to create the sample. We have see examples in the previous section.
- model-based approach: on the other hand, a model is given to specify a relationship between the sample and the population. Model-based estimation strategies including least-squares, method-of-moments, maximum-likelihood, etc are then used for constructing estimators for $\eta(G)$.

In more recent decades, the distinction between these two approaches has become more blurred.

### Assess Significance of $\eta(G^{obs})$

Suppose that we have a graph $G^{obs}$ derived from observations of some sort (i.e., **not** necessarily through a formal network sampling mechanism). We often interested in whether $\eta(G^{obs})$ is 'significant', in the sense that unusual or unexpected.

To measure this, we need a reference, like a 'null hypothesis' in hypothesis testing. A RGM can be used to create a reference distribution which, under the accompanying assumption of uniform likelihood of elements in $\mathcal{G}$, takes the form,

$$
\mathbb{P}_{\eta, \mathcal{G}} (t)  = \frac{\# \left\{ G \in \mathcal{G}: \eta(G) \le t \right\}}{\left\vert \mathcal{G} \right\vert}
$$

If $\eta(G^{obs})$ is found to be sufficiently unlikely under this distribution, this is taken as evidence **against** the hypothesis that Gobs is a uniform draw from $G$.

Some issues:
- How to choose $\mathcal{G}$?
- Usually it is not possible to enumerate all elements in $\mathcal{G}$, hence, cannot compute $\mathbb{P}_{\eta, \mathcal{G}} (t)$ exactly $\rightarrow$ approximation.

## Classical Random Graph Models

### Erdos and Renyi

Equal probability on all graphs of a given order and size:

$$\mathcal{G} (N_v, N_e) = \left\{ G = (V, E): \left\vert V \right\vert = N_v, \left\vert E \right\vert = N_e\right\}$$

It is easy to find $\left\vert \mathcal{G} (N_v, N_e) \right\vert = \binom{\binom{N_v}{2}}{N_e}$, hence

$$\mathbb{P} (G) = \binom{\binom{N_v}{2}}{N_e} ^{-1}$$

### Gilbert

A collection $\mathcal{G} (N_v, p)$ is defined to consist of all graphs $G$ of order $N_v$ that may be obtained by assigning an edge **independently** to each pair of distinct vertices with probability $p$.

$$\mathcal{G} (N_v, p) = \left\{ G = (V, E): \left\vert V \right\vert = N_v, \left\vert E \right\vert = N_e\right\}$$

The level of connectivity is related to the relation between $p$ and $N_v$. Let $p = \frac{c}{N}$ for $c > 0$, then

- $c > 1$: w.h.p. $G$ will have a single connected component ('giant component') consisting of $\alpha_c N_v$ vertices, for some constant $\alpha_c > 0$, with the remaining components having only on the order of $\mathcal{O} (\log N_v)$ vertices.
- $c < 1$: w.h.p. all components will have on the order of $\mathcal{O} (\log N_v)$ vertices.

In term of degree distribution, w.h.p.

$$
(1-\varepsilon) \frac{c^{d} e^{-c}}{d !} \leq f_{d}(G) \leq(1+\varepsilon) \frac{c^{d} e^{-c}}{d !}
$$

That is, for large $N_v$, $G$ will have a degree distribution that is like a Poisson distribution with mean $c = p N_v$. This is intuitive since from the perspective of a vertex $i \in V$, it has edge $(i, j)$ w.p. $p$ for $N_v - 1$ number of $j$, hence its expected degree is $p(N_v - 1)$.

Thus, we observe
- **concentrated** degree distribution with expressions decay tails, rather than broad degree distribution observed in many large-scale real-world networks.
- **less clustering**: recall that assortativity is the probability that two neighbors of a randomly chosen vertex are linked is just $p$, which tend to zero as $N_v$ grows.
- **small-world property**: the diameter of the graph very like $\mathcal{O} (\log N_v)$ w.h.p as $N_v \rightarrow \infty$.

## Generalized Random Graph Models

Equal probability on all graphs of a given order and some particular characteristic(s) $\eta^*$:

$$\mathcal{G} (N_v, \eta^*) = \left\{ G = (V, E): \left\vert V \right\vert = N_v, \eta(G) = \eta^*\right\}$$

Erdos Renyi random graph is a particular case of this, with $\eta^* = N_e$. $\eta^*$ can be more general, for instance, degree sequence $\left\{d_{(1)}, \ldots, d_{\left(N_{v}\right)}\right\}$ in ordered form.

.


.


.


.


.


.


.


.
