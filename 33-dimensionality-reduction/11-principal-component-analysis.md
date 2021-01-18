# Principal Component Analysis

Proposed by Pearson in 1901 and further deveoped by Hotelling in 1993.


## Objective

Given $X_1, X_2, \ldots, X_p$, we want to extract the most useful information of $p$ measurements such that

1. **explore underlying dimension** behind the $p$ original measurements to explain the variation of $p$ original measurements, which may have interesting or revealing interpretations, such as size, shape and contrasts in natural science;

2. **estimate latent variables** (i.e. variables that cannot be measured or observed.) which can explain the variation of the $p$ original measurements, especially in
social behavioral sciences.

3. **simplify the dimension** of the observed data set. Lower dimension can be chosen from the data set such that the variations of measurements can be captured with an acceptable level. For example, $m \ll p$ latent variables are chosen to capture 90% of variation of $p$ original measurements. Indeed, this can be regarded as the data reduction or dimension reduction.


Consider a $p$-dimensional random vector $\boldsymbol{x} = \left( X_1, X_2, \ldots, X_p \right)^\top$ with mean vector $\boldsymbol{\mu} = \left( \mu_1, \ldots, \mu_p \right)^\top$ and covariance matrix $\boldsymbol{\Sigma}$. PCA aimes to obtain the variables $Y_1, Y_2, \ldots, Y_m$ which are the **linear combinations** of $X_1, X_2, \ldots, X_p$ and $m \le p$, such that

- The sum of the new individual variances

  $$
  \operatorname{Var}\left( Y_1 \right) + \operatorname{Var}\left( Y_2 \right) + \ldots + \operatorname{Var}\left( Y_m \right)
  $$

  is **close** to the sum of the original individual variances

  $$
  \operatorname{Var}\left( X_1 \right) + \operatorname{Var}\left( X_2 \right) + \ldots + \operatorname{Var}\left( X_m \right)
  $$

- The linear combinations $Y_i$ and $Y_j$ are **uncorrelated** for $i\ne j$. This imply that each variable in $\boldsymbol{y} = \left( Y_1, Y_2, \ldots, Y_m \right)^\top$ can be analyzed by using **univariate** techniques.


Another formulation: Find a linear mapping $\boldsymbol{W}$ (assume $\boldsymbol{X}$  is centered)

- Minimize reconstruction residuals

  $$\begin{align}
  \boldsymbol{W}^*  = \underset{\boldsymbol{\boldsymbol{W} } }{\operatorname{argmin}} \, & \sum_i^n \left\Vert \boldsymbol{x}_i - \hat{\boldsymbol{x} }_i \right\Vert ^2    \\
   \text{s.t.}  & \boldsymbol{W} ^\top \boldsymbol{W} = \boldsymbol{I}  
  \end{align}$$

- Maximize the variance of projected data $\boldsymbol{W} ^\top \boldsymbol{X}$

$$\begin{align}
\boldsymbol{W}^*  = \underset{\boldsymbol{\boldsymbol{W} } }{\operatorname{argmax}} \, & \operatorname{tr}\left( \boldsymbol{W} ^\top \boldsymbol{X} \boldsymbol{X} ^\top \boldsymbol{W} \right)   \\
 \text{s.t.}  & \ \boldsymbol{W} ^\top \boldsymbol{W} = \boldsymbol{I}  
\end{align}$$

## Learning

### Sequential Maximization

The first variable in $\boldsymbol{y}$, i.e. $Y_1 = \boldsymbol{\alpha} \boldsymbol{x}$ is obtained to maximize its variance, i.e.,

$$
\lambda_{1} \equiv \operatorname{Var}\left(Y_{1}\right)=\max _{\left\Vert \boldsymbol{\alpha}  \right\Vert _2^2 = 1 } \boldsymbol{\alpha}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}
$$

Suppose the maximum is achieved at $\boldsymbol{\alpha} = \boldsymbol{\alpha} _1$ and we call $Y_1$ given below the first population principal component


$$
Y_1 = \boldsymbol{\alpha} _1^T \boldsymbol{x}
$$

Successively for $i=2, \ldots, m$ the variance of $Y_i$ can be obtained by the following maximization


$$
\begin{aligned}
&&&\lambda_{i} \equiv \operatorname{Var}\left(Y_{i}\right)=\max _{\alpha} \boldsymbol{\alpha}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}\\
& &\mathrm{s.t.}  \quad &\boldsymbol{\alpha}^{\top} \boldsymbol{\alpha}=1 \\
& & & \ \boldsymbol{\alpha}^{\top} \boldsymbol{x} \text { being uncorrelated with } Y_{1}, \ldots, Y_{i-1}  
\end{aligned}
$$

The maximum is achieved at $\boldsymbol{\alpha} = \boldsymbol{\alpha} _i$ and the $i$-th population principal component is

$$
Y_i = \boldsymbol{\alpha} _i^\top \boldsymbol{x}
$$


```{dropdown}


We consider the maximization problem:

$$\begin{align}
\max _{\boldsymbol{\alpha}} \quad & \boldsymbol{\alpha}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}  \\
\text {s.t.} \quad & \boldsymbol{\alpha}^{\top} \boldsymbol{\alpha}=1
\end{align}$$

$$
\quad
$$

The Lagrangean is

$$
\begin{equation}
L(\boldsymbol{\alpha}, \theta)=\boldsymbol{\alpha}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}-\lambda\left(\boldsymbol{\alpha}^{\top} \boldsymbol{\alpha}-1\right)
\end{equation}
$$

The first order conditions are

$$
\begin{aligned}
\frac{\partial L}{\partial \boldsymbol{\alpha}}
&= 2 \boldsymbol{\Sigma} \boldsymbol{\alpha}-2 \lambda \boldsymbol{\alpha} \\
&=\mathbf{0} \\
\Rightarrow \quad \quad \boldsymbol{\Sigma} \boldsymbol{\alpha} &=\lambda \boldsymbol{\alpha}  \quad \quad \quad \quad (1)
\end{aligned}
$$

and

$$
\begin{aligned}
\frac{\partial L}{\partial \lambda}
&= 1-\boldsymbol{\alpha}^{\top} \boldsymbol{\alpha} \\
&= 0 \\
\Rightarrow \quad \quad  \boldsymbol{\alpha}^{\top} \boldsymbol{\alpha}
&=1 \quad \quad \quad \quad (2)
\end{aligned}
$$

Premultiply $(1)$ by $\boldsymbol{\alpha} ^\top$ we have
$$
\boldsymbol{\alpha}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}=\lambda \boldsymbol{\alpha}^{\top} \boldsymbol{\alpha}
$$

Hence,

$$
\lambda = \boldsymbol{\alpha} ^\top \boldsymbol{\Sigma} \boldsymbol{\alpha}
$$

Note that $(1)$ also gives

$$
(\boldsymbol{\Sigma}-\lambda \boldsymbol{I}) \boldsymbol{\alpha} =\mathbf{0}
$$

which implies $\lambda$ is the eigenvalue of $\boldsymbol{\Sigma}$.


Therefore, the maximized variance $\boldsymbol{\alpha} ^\top \boldsymbol{\Sigma} \boldsymbol{\alpha}$ equals to the largest eigenvalue of $\boldsymbol{\Sigma}$.

```

### Eigenvalue Decomposition

Rather than obtaining the principal components sequentially, the principal components and their variances can be obtained simultaneously by solving for the eigenvectors and eigenvalues of $\boldsymbol{\Sigma}$. Using the Spectral Decomposition Theorem,



$$
\boldsymbol{\Sigma} = \sum_i^p \lambda_i \boldsymbol{\alpha} _i \boldsymbol{\alpha} _i ^\top = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^\top
$$

where
- $\lambda_1 > \lambda_2 > \dots> \lambda_p \ge 0$ are ordered eigenvalues of $\boldsymbol{\Sigma}$: $\boldsymbol{\Lambda} =  \operatorname{diag}(\lambda_1, \lambda_2, \ldots, \lambda_p)$
- $\boldsymbol{\alpha} _1, \ldots, \boldsymbol{\alpha} _p$ are their corresponding normalized eigenvectors forming the column vectors of the orthogonal matrix $\boldsymbol{U} = \left( \boldsymbol{\alpha} _1\  \boldsymbol{\alpha} _2 \ \ldots \  \boldsymbol{\alpha} _p \right)$, where $\boldsymbol{U}  ^\top \boldsymbol{U}   = \boldsymbol{I}$ or $\boldsymbol{\alpha} _i ^\top \boldsymbol{\alpha} _j = 1$ if $i=j$ and 0 otherwise.



The $k$-th population principal component is defined as

$$
Y_{k}=\boldsymbol{\alpha}_{k}^{\top} \boldsymbol{x}=\alpha_{1 k} X_{1}+\alpha_{2 k} X_{2}+\cdots+\alpha_{p k} X_{p}, \quad k=1, \ldots, p
$$

The principal component transform is then

$$
\boldsymbol{y} = \boldsymbol{U} ^\top \boldsymbol{x}
$$

## Special Cases

### Variables are Uncorrelated

If $\boldsymbol{X_i}$ are uncorrelated, then $\boldsymbol{\Sigma}$ is a diagonal matrix, i.e., $\boldsymbol{\Sigma} = \operatorname{diag}\left( \sigma_{11}, \sigma_{22}, \sigma_{pp} \right)$. Without loss of generality, assume $\sigma_{11} > \sigma_{22} > \sigma_{pp}$, then from its spectral decomposition $\boldsymbol{\Sigma} = \boldsymbol{U} ^\top \boldsymbol{\Lambda} \boldsymbol{U}$, we have
- $\boldsymbol{U} = \boldsymbol{I}$
- $\boldsymbol{\Lambda} = \operatorname{diag}\left( \sigma_{ii} \right)$, or $\lambda_i = \sigma_{ii}$.

Hence, the principal component is

$$
Y_i = X_i
$$

Clearly, it is **not** necessary to perform PCA in this case.

### Variables are Perfectly Correlated

In this case, the covariance matrix is not of full rank, i.e., $\left\vert \boldsymbol{\Sigma}  \right\vert = 0$. Then, some eiganvalues equal zero. In other words,

$$
\lambda_1 > \lambda_2 > \ldots, > \lambda_m > \lambda_{m+1} = \ldots = \lambda_p = 0
$$

Only $m$ eigenvectors $\boldsymbol{\alpha} _i$ can be obtained with $\left\Vert \boldsymbol{\alpha}_i  \right\Vert _2 ^2 =  1$ .

### Few Variables Have Extremely Large Variances

If a few variables have extremely large variances in comparison with other variables, they will dominate the first few principal components and give the foregone conclusion that a few principal components is sufficient in summarizing information. That conclusion may even be spurious, as the measurement scales, which affect the variances, are quite arbitrary in a lot of applications.

For example, $X_1$ is measured in meters while $X_2$ and $X_3$ are measured in kilometers. The first PC should have particularly large variance ($\lambda_1$ is particularly large relative to $\lambda_2$ and $\lambda_3$). This property suggests that if $\boldsymbol{x}$  are on different, or non-commensurable, measurement units, we should standardize them,

$$
Z_i = \frac{X_i - \mu_i}{\sigma_i}
$$

before performing PCA.

## Properties

1. All principal components are uncorrelated, i.e., $\operatorname{Cov}\left( Y_i, Y_j \right) = 0$ for $i \ne j$


    ```{dropdown} Proof

    $$
    \begin{aligned}
    \operatorname{Cov}\left(Y_{i}, Y_{j}\right) &=\operatorname{Cov}\left(\boldsymbol{\alpha}_{i}^{\top} \boldsymbol{x}, \boldsymbol{\alpha}_{j}^{\top} \boldsymbol{x}\right) \\
    &=\mathrm{E}\left(\boldsymbol{\alpha}_{i}^{\top}(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\alpha}_{j}\right) \\
    &=\boldsymbol{\alpha}_{i}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}_{j} \\
    &=\boldsymbol{\alpha}_{i}^{\top} \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^{\top} \boldsymbol{\alpha}_{j} \\
    &=\left(\boldsymbol{\alpha}_{i}^{\top}\right)\left(\boldsymbol{\alpha}_{1} \boldsymbol{\alpha}_{2} \cdots \boldsymbol{\alpha}_{p}\right) \boldsymbol{\Lambda}\left(\begin{array}{c}
    \boldsymbol{\alpha}_{1}^{\top} \\
    \boldsymbol{\alpha}_{2}^{\top} \\
    \vdots \\
    \boldsymbol{\alpha}_{p}^{\top}
    \end{array}\right) \boldsymbol{\alpha}_{j} \\
    &=\boldsymbol{e}_{i}^{\top} \boldsymbol{\Lambda} \boldsymbol{e}_{j} \\
    &=0
    \end{aligned}
    $$

    ```


1. The variance of the $i$-th principal component is $\lambda_i$, i.e. $\operatorname{Var}\left( Y_i \right) = \lambda_i$.


    ```{dropdown} Proof
    $$\begin{align}
    \operatorname{Var}\left( Y_i \right)
    &= \operatorname{Var}\left( \boldsymbol{\alpha} _i ^\top \boldsymbol{x}  \right) \\
    &= \boldsymbol{\alpha} _i ^\top \boldsymbol{\Sigma} \boldsymbol{\alpha} _i \\
    &= \boldsymbol{e}_i ^\top \boldsymbol{\Lambda} \boldsymbol{e}_i  \\
    &= \lambda_i
    \end{align}$$
    ```

1. The first principal component $Y_1 = \boldsymbol{\alpha} _1 ^\top \boldsymbol{x}$ has the largest variance among all linear combinations of $X_i$'s. The $i=2, \ldots, p$, the $i$-th principal component has the largest variance among all linear combinations of $X_i$'s, which are uncorrelated with the first $(i-1)$ principal components.


1. The principal component preserve the total variance

    $$
    \sum_{i=1}^{p} \operatorname{Var}\left(Y_{i}\right)=\sum_{i=1}^{p} \operatorname{Var}\left(X_{i}\right)
    $$

    or

    $$
    \sum_{i=1}^{p} \lambda_{i}=\sum_{i=1}^{p} \sigma_{i i}
    $$


    ```{dropdown} Proof
    $$
    \begin{aligned}
    \sum_{i=1}^{p} \sigma_{i i} &=\operatorname{tr}(\boldsymbol{\Sigma}) \\
    &=\operatorname{tr}\left(\sum_{i=1}^{p} \lambda_{i} \boldsymbol{\alpha}_{i} \boldsymbol{\alpha}_{i}^{\top}\right) \\
    &=\sum_{i=1}^{p} \lambda_{i} \operatorname{tr}\left(\boldsymbol{\alpha}_{i} \boldsymbol{\alpha}_{i}^{\top}\right) \\
    &=\sum_{i=1}^{p} \lambda_{i} \operatorname{tr}\left(\boldsymbol{\alpha}_{i}^{\top} \boldsymbol{\alpha}_{i}\right) \\
    &=\sum_{i=1}^{p} \lambda_{i}
    \end{aligned}
    $$
    ```


1. The correlation between a principal component $Y_j$ and an original variable $X_i$ is given by

    $$
    \operatorname{Corr}\left( X_i, Y_j \right) = \frac{\sqrt{\lambda_j}a_{ij}}{\sqrt{\sigma_{ii}}}
    $$

    where $\alpha_{ij}$ denotes the $i$-th element of $\boldsymbol{\alpha} _j$.


    ```{dropdown} Proof

    $$
    \begin{aligned}
    \operatorname{Cov}\left(X_{i}, Y_{j}\right) &=\operatorname{Cov}\left(X_{i}, \boldsymbol{\alpha}_{j}^{\top} \boldsymbol{x}\right) \\
    &=\operatorname{Cov}\left(\boldsymbol{e}_{i}^{\top} \boldsymbol{x}, \boldsymbol{\alpha}_{j}^{\top} \boldsymbol{x}\right) \\
    &=\boldsymbol{e}_{i}^{\top} \boldsymbol{\Sigma} \boldsymbol{\alpha}_{j} \\
    &=\boldsymbol{e}_{i}^{\top} \sum_{k=1}^{p} \lambda_{k} \boldsymbol{\alpha}_{k} \boldsymbol{\alpha}_{k}^{\top} \boldsymbol{\alpha}_{j} \\
    &=\lambda_{j} \boldsymbol{e}_{i}^{\top} \boldsymbol{\alpha}_{j} \boldsymbol{\alpha}_{j}^{\top} \boldsymbol{\alpha}_{j} \\
    &=\lambda_{j} \boldsymbol{e}_{i}^{\top} \boldsymbol{\alpha}_{j} \\
    &=\lambda_{j} \alpha_{i j}
    \end{aligned}
    $$

    and then

    $$\begin{align}
    \operatorname{Corr}\left(X_{i}, Y_{j}\right)
    &=\frac{\operatorname{Cov}\left(X_{i}, Y_{j}\right)}{\sqrt{\operatorname{Var}\left(X_{i}\right) \operatorname{Var}\left(Y_{j}\right)}} \\
    &=\frac{\lambda_{j} \alpha_{i j}}{\sqrt{\sigma_{i i} \lambda_{j}}} \\
    &=\frac{\sqrt{\lambda_{j}} \alpha_{i j}}{\sqrt{\sigma_{i i}}}
    \end{align}$$
    ```

1. If the correlation matrix $\boldsymbol{\rho} = \boldsymbol{D}^{-1}\boldsymbol{\Sigma} \boldsymbol{D}^{-1}$ instead of the covariance matrix $\boldsymbol{\Sigma}$ is used, i.e. variables $X_1, X_2, \ldots, X_p$ are standardized, then


   $$
   \sum_i^p \lambda_i = \sum_i^p \sigma_{ii} = p
   $$


## Tuning

There are several ways to choose the number of principal components to retain.

1. **Cumulative proportion cutoff**:

    Include the components such that the cumulative proportion of the total variance explained is just more than a threshold value, say 80%, i.e., if

    $$
    \begin{equation}
    \frac{\sum_{i=1}^{m} \ell_{i}}{\sum_{i=1}^{p} \ell_{i}} >0.8
    \end{equation}
    $$

    This method keeps $m$ principal components.

1. **Proportion cutoff**

    Select the components whose eigenvalues are greater than a threshold value, say average of eigenvalues; for correlation matrix input, this average is $p^{-1} \sum_{i=1}^{p} \ell_{i}=p^{-1} p=1$ if we use the correlation matrix $\boldsymbol{\rho}$.

1. **Scree plot**

    Construct the so-called scree plot of the eigenvalue $\ell_i$ on the vertical axis versus $i$ on horizontal axis with equal intervals for $i = 1, 2, \ldots, p$, and join the points into a decreasing polygon. Try to find a “clean-cut” where the polygon “levels off” so that the first few eigenvalues seem to be far apart from the others.

    [picture]

1. **Hypothesis testing**

    Perform formal significance tests to determine the larger an unequal eigenvalues and retain the principal components to these eigenvalues.

1. **reconstruction loss**
    We can look at the expansion
    $$
    \hat{\boldsymbol{x} }=\mu_{\boldsymbol{x}} +\sum_{j=1}^{k}\left(\phi_{j}^{T} \boldsymbol{x} \right) \phi_{j}
    $$
    and examine the residual $\left\Vert \boldsymbol{x} - \hat{\boldsymbol{x} } \right\Vert _ $
    [image, pg22]

    note: expected residual corresponds to variance in the remaining subspace.

1. **Downstream task performance**


## Interpretation

### Geometric Meaning: Direction of Variation

For the distribution of $\boldsymbol{x}$, thelcenter location is determined by $\boldsymbol{\mu} _ \boldsymbol{x}$ and the variation is captured by each principal direction $\boldsymbol{\alpha} _i$



For the multinormal distribution, the family of **contours** of $\boldsymbol{x}$ (on each of which the pdf is a constant) is a family of ellipsoids in the original coordinate system $\boldsymbol{x}$ satisfying the following equation for a
constant $c$,

$$
\begin{equation}
(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})=c^{2}
\end{equation}
$$

where $c$ serves as an index of the family. This family of ellipsoids have orthogonal principal axes

$$
\pm c\lambda_i^{1/2}\boldsymbol{\alpha}_i, i= 1, 2, \ldots, p
$$

with length
- $2c\lambda_i^{1/2}$
- directional cosines as coefficients given in $\boldsymbol{\alpha} _i$ for the $i$-th axis.

[digit image, page 20]


### Proportion Explained

The proportion of total variance explained by $Y_i$, which is

$$\frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$

is considered as a measure of **importance** of $Y_i$ in a more parsimonious description of the system.

### Score of an Observation in Sample Data

For a data set of $n$ observations, we decompose the sample covariance matrix $S$ as

$$
\begin{equation}
\boldsymbol{S} =\sum_{i=1}^{p} \ell_{i} \boldsymbol{a} _{i} \boldsymbol{a} _{i}^{\top}=\boldsymbol{A} \boldsymbol{L}  \boldsymbol{A} ^\top
\end{equation}
$$

where $\lambda_i$ are eigencalues of $\boldsymbol{S}$ and $\boldsymbol{a} _i$'s are their corresponding normalized eigenvectors.

The $i$-th **sample** principal component is defined as

$$
\begin{equation}
Y_{i}=\boldsymbol{a}_{i}^{\top} \boldsymbol{x}=a_{1 i} X_{1}+a_{2 i} X_{2}+\cdots+\alpha_{p i} X_{p}
\end{equation}
$$

where $\begin{equation}
\boldsymbol{a}_{i}^{\top}=\left(\begin{array}{llll}
a_{1 i} & a_{2 i} & \cdots & a_{p i}
\end{array}\right)
\end{equation}$.

The data layout is

$$
\begin{equation}
\begin{array}{cccccccc}
&& \text{Data} \ \ \boldsymbol{X}  &&&&\text{PC} \ \ \boldsymbol{Y} &\\
\hline X_{1} & X_{2} & \cdots & X_{p} & \quad \quad Y_{1} & Y_{2} & \cdots & Y_{p} \\
x_{11} & x_{12} & \cdots & x_{1 p} & \quad \quad y_{11} & y_{12} & \cdots & y_{1 p} \\
x_{21} & x_{22} & \cdots & x_{2 p} & \quad \quad y_{21} & y_{22} & \cdots & y_{2 p} \\
& & \vdots & & & & \vdots & \\
x_{n 1} & x_{n 2} & \cdots & x_{n p} & \quad \quad y_{n 1} & y_{n 2} & \cdots & y_{n p}
\end{array}
\end{equation}
$$

where the corresponding row vectors on the data matrices are related as

$$\boldsymbol{y} _i ^\top = \boldsymbol{x} _i ^\top \boldsymbol{A} , i= 1, 2, \ldots, n$$

where $\boldsymbol{y} _i$ can be interpreted as a vector of principal component scores for the $i$-th observation.


```{note}
Properties of the population principal components are all valid in the sample context, by replaceing

$$\boldsymbol{\mu}, \boldsymbol{\Sigma} , \boldsymbol{\rho}, \lambda_i, \boldsymbol{\alpha} _i$$

by

$$\bar{\boldsymbol{x}}, \boldsymbol{S} , \boldsymbol{R} , \ell_i, \boldsymbol{a} _i$$
```



## Cons

**Sensitive to variable transformation**

The results of PCA are not invariant under a linear transformation and, even worse, there is no easy correspondence between the two sets of results $\boldsymbol{y}$ and $\boldsymbol{y} ^\prime$, before and after the linear transformation. For example, the PCA using $\\boldsymbol{\Sigma}$ is not the same as the PCA using $\boldsymbol{\rho}$ and we cannot use the PCA from $\boldsymbol{\rho} $ to get the PCA results from the original variables.

If the two sets of results are consistent to each other, the PCA based on $\boldsymbol{\Sigma}$  may be preferred in some situation. If they are very different, or even contradictory, subject-matter knowledge and/or wisdom are needed to make a choice.

The PCA based on covariance matrix is preferred when the original measurements units are very important, like in many applications in
natural sciences. However, when the units of measurement are of artificial nature, like scores in some questions as frequently used in social sciences, the PCA based on correlation matrix is preferred.


## Relation to

### SVD

Recall the SVD of the data matrix

$$
X = \boldsymbol{U} \boldsymbol{S} \boldsymbol{V} ^\top
$$

Suppose $X$ is centered, then


$$
\boldsymbol{\Sigma} = \boldsymbol{X} ^\top \boldsymbol{X} = \boldsymbol{V} \boldsymbol{S} ^\top \boldsymbol{S} \boldsymbol{V} ^\top
$$

So the right singular vectors $\boldsymbol{V}$ are the eigenvectors of $\boldsymbol{X} ^\top \boldsymbol{X}$, the eigenvalues of $\boldsymbol{X} ^\top \boldsymbol{X}$ are proportional to the squared singular values of $\sigma_i$

So we can compute the PCS via an SVD.

### Compression

Instead of storing the $n \times p$ data matrix $\boldsymbol{X}$, not we need to store the $p \times 1$ mean vector $\boldsymbol{\mu} _ \boldsymbol{x}$ and the $m\times p$ projection matrix $\boldsymbol{W}$, and the $n \times m$ projected data matrix $\boldsymbol{Y}$.

To transmit $N$ examples, we need $p+pm+nm$ numbers instead of $np$.

### Gaussians

PCA essentially models variance in the data. What distribution is characterized by variance? Gaussian.
can be described by Gaussians

Probabilistic PCA

### Classification

For a classification task, we can perform PCA on the features before fitting the data to a classifier. The classifier might be more accurate since PCA reduces noise.


But note that the direction of largest variance need not to be the most disriminative direction.

[image pg29]

If we knew the labels, we could use a supervised dimensionality reduction, e.g. linear discriminant analysis.

## Extension

Probabilistic PCA  is a method of fitting a constrained Gaussian, where some variances are equal.

$$
\begin{equation}
\boldsymbol{\Sigma}=\boldsymbol{U}\left[\begin{array}{ccccccc}
\lambda_{1} & \ldots & 0 & \ldots & \ldots & \ldots \\
& \ddots & 0 & \ldots & \ldots & \ldots \\
0 & \ldots & \lambda_{k} & \ldots & \ldots & \ldots \\
0 & \ldots & 0 & \sigma^{2} & 0 & \ldots \\
& & & & \ddots & \\
0 & \ldots & \ldots & \ldots & 0 & \sigma^{2}
\end{array}\right] \boldsymbol{U}^{T}
\end{equation}
$$

Estimate for the noise variance $\sigma^2$


$$
\begin{equation}
\sigma^{2}=\frac{1}{d-k} \sum_{j=k+1}^{d} \lambda_{j}
\end{equation}
$$
