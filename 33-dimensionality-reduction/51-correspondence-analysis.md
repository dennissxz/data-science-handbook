# Correspondence Analysis

Correspondence Analysis may have been evolved from methods used in multidimensional scaling. However Correspondence Analysis deals with a different type of data: count data in a two-way contingency table.

Specifically, CA is a method for graphically displaying both the rows and columns of a **two-way contingency table**. The CA graphical procedure aims to present associations between the row variable and the column variable.

$$
\begin{array}{c|cccccc}
& \text { col-var level }1  & \text { col-level }2  & \cdots & \text { col-level }j  & \cdots & \text { col-level }J  \\
\hline \text { row-var level }1 & n_{11} & n_{12} & \cdots & n_{1 j} & \cdots & n_{1 J} \\
\text { row-var level }2  & n_{21} & n_{22} & \cdots & n_{2 k} & \cdots & n_{2 J} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\text { row-var level }i  & n_{i 1} & n_{i 2} & \cdots & n_{i j} & \cdots & n_{i j} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\text { row-var level }I  & n_{I 1} & n_{I 2} & \cdots & n_{I j} & \cdots & n_{I J}
\end{array}
$$

- Two categorical variables are of $I$ and $J$ categories respectively,
- The variables are represented as row and column variables.
- $n_{ij}$ is the count of item belong to category $i$ of the row variable and
category $j$ of the column variable.
