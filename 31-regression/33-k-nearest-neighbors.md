# K Nearest Neighbors




- task: classification
- parametric: no
- prediction: $p(y=c\vert \boldsymbol{x}, \mathcal{D}, K) = \frac{1}{K} \sum_{j \in N_K (\boldsymbol{x}, \mathcal{D})} \mathbb{I}\left( y_j =c \right)$
- description
  - it simply look at the $K$ points in the training set that are nearest to the test input $\boldsymbol{x}_i$
  - it is an example of **memory-based learning** or **instance-based learning**
  - it requires a distance measure $d(\boldsymbol{x}_i, \boldsymbol{x}_j)$, such as Euclidean distance, to identity $N_K \left( \boldsymbol{x} \mathcal{D} \right)$, the $K$ nearest points to $\boldsymbol{x}$
- cons:
  - poor performance in high dimensional settings
