# Linear Discriminant Analysis



## Fisher's Linear Discriminant Function

### Objective

We have two groups of data in $\mathbb{R} ^p$
- $\left\{ \boldsymbol{x} _{1i} \right\}, i = 1, 2, \ldots, n_1$
- $\left\{ \boldsymbol{x} _{2i} \right\}, i = 1, 2, \ldots, n_2$

We want to project them to a one dimensional line in $\mathbb{R}$, such that the projection by this line yields good separation of the two classes.

Let the projection be $\boldsymbol{a} ^{\top} \boldsymbol{x}$ and let the projected data be
- $\left\{ y _{1i} = \boldsymbol{a} ^{\top} \boldsymbol{x} _{1i} \right\}$
- $\left\{ y_{2i} = \boldsymbol{a} ^{\top} \boldsymbol{x} _{2i} \right\}$

The goodness of separation is measured by th ratio of between-class difference and
within-class variance.

$$
\max _ {\boldsymbol{a} \in \mathbb{R} ^p} \frac{(\bar{y}_1- \bar{y}_2)^2}{s_y^2}
$$

where $s_y^2$ is the pooled variance, if we assume equal covariance structure in the two groups.

$$
s_{y}^{2}=\frac{\sum_{i=1}^{n_{1}}\left(y_{1 i}-\bar{y}_{1}\right)^{2}+\sum_{i=1}^{n_{2}}\left(y_{2 i}-\bar{y}_{2}\right)^{2}}{n_{1}+n_{2}-2}=\boldsymbol{a} ^{\top} \boldsymbol{S}_{x} \boldsymbol{a}=\boldsymbol{a}^{\top} \boldsymbol{S}_{\text {pool }} \boldsymbol{a}
$$

:::{figure} lda-fisher
<img src="../imgs/lda-fisher.png" width = "80%" alt=""/>

Illustration of projection of data to a linear direction [C. Bishop, 2006]
:::

### Assumptions

- equal covariance matrix $\boldsymbol{\Sigma} _1 = \boldsymbol{\Sigma} _2 = \boldsymbol{\Sigma}$
- full rank $\operatorname{rank}\left( \boldsymbol{\Sigma}  \right) = p$
- without normality assumption that the population are from multivariate normal.

### Learning

Note that the objective function

$$
\frac{(\bar{y}_1- \bar{y}_2)^2}{s_y^2} = \frac{[\boldsymbol{a} ^{\top} (\bar{\boldsymbol{x} }_1-\bar{\boldsymbol{x} }_2)]^2}{\boldsymbol{a}^{\top} \boldsymbol{S}_{\text {pool }} \boldsymbol{a}}
$$

has the form of generalized [Rayleigh quotient](rayleigh-quotient). Hence the solution is given by

$$
\boldsymbol{a} ^*= \boldsymbol{S} _{\text{pool} } ^{-1} (\bar{\boldsymbol{x} }_1-\bar{\boldsymbol{x} }_2)
$$

with maximum

$$
(\bar{\boldsymbol{x} }_1-\bar{\boldsymbol{x} }_2) \boldsymbol{S} _{\text{pool} } ^{-1} (\bar{\boldsymbol{x} }_1-\bar{\boldsymbol{x} }_2) =: D^2
$$

The maximum $D^2$ can be viewed as the square of the Mahalanobis distance between the population means of the original data.

### Prediction

For a new data point $\boldsymbol{x} _0$, we compute $y_0 = \boldsymbol{a} ^{* \top } \boldsymbol{x} _0$, and use the midpoint of the transformed means $m = \frac{1}{2}(\bar{y}_1 + \bar{y}_2)$ as the partition point, assign it to closer class.

Suppose $\bar{y}_1 \ge \bar{y}_2$

- If $y_0 > m$, then assign $\boldsymbol{x} _0$ to class 1
- otherwise, class 2.

In other words, we assign $\boldsymbol{x} _0$ to class $j^* = \min_{j=1, 2} \left\vert y_0 - \bar{y}_j \right\vert$.

### R.t. Two-sample Means

The quantity $D^2$ is [used](multi-two-sample) in Hotelling's $T^2$ to test if the two means are equal. The test can be used here to check if the separation of the two population is significant enough to apply classification. Under the assumption of normal distribution of equal variance for the two populations, the test statistic is


$$
\frac{n_{1}+n_{2}-p-1}{\left(n_{1}+n_{2}-2\right) p} \cdot \frac{n_{1} n_{2}}{n_{1}+n_{2}} D^{2} \sim F_{p, n_{1}+n_{2}-p-1} \quad \text { under } \quad H_{0}: \boldsymbol{\mu}_{1}=\boldsymbol{\mu}_{2}
$$

There is not much point to conduct classification if the difference of the class means is not significant in the first place. On the other hand, significant difference of the class means is not sufficient to guarantee a good classification.




.


.


.


.


.


.


.


.
