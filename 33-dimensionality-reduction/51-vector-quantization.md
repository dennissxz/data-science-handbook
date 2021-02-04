# Vector Quantization

Quantization = discretization = compression

VQ is a clustering-based discrete representation learning method. Feature vectors are often a vector of real values per data sample
Can either model the continuous space directly, or represent as discrete symbols
For some applications, compact representation may be important Sometimes, the goal is compactness (compression, coding) Quantization incurs some loss of information, however
Example VQ codebooks for 2 dimensional vectors:



# K-means clustering


K-means clustering is a iterative algorithm for clustering.

---
**K-means clustering**

---

- Initialize

- While

---

## Objective

minimize total distortion

local minimum, may not be global, affected by final result.

[img30]


## Pros Cons

Pros

- Work well for clusters with spherical shape and similar size

Cons

- affected by random initialization.
  - sol: try several initializations, keep the result with lowers D

- Work bad for clusters with non-ideal attributes

- Choice of distance measure interacts with cluster shape


# Agglomerative Clustering


## Learning
- Iteratively find the closest pair of clusters and merge them into single cluster until some stop condition.

- Until

Clustering methods differ by distance measure.

table

### Ward'd methods




##
