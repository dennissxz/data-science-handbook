# Trainability

## Problems of Gradients

Dead ReLU: neurons with negative value (before ReLU) are not undated. Recovered if the updated outputs from the previous layer make its value positive.

### Vanishing Gradients

### Exploding Gradients

## Initialization

Exercise: what happen if we initialize all parameter to be 0? No update.

Variance of the value of a node (before activation) grows with number of in-flow outputs.

We want to keep variance of all neurons roughly the same upon unitialization.

### Xavier Initialization

### Kaiming Initialization


## Normalization

## Adding an Affine Transformation

## Residual Connections
