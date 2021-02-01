# Autoencoders

An autoencoder has two parts

- One neural network f(x) (encoder) produces a representation of the input

- A second neural network g(x) (decoder) tries to reproduce the input from the encoded representation

Typical loss for continuous input: $\|g(f(\mathbf{x}))-\mathbf{x}\|^{2}$

Q: we can just learn an identical mapping that produces zero loss.

Only makes sense if we constrain $f(\cdot)$ somehow

- Limit dimensionality (number of units at the output) of f(x)
- Add a penalty to the loss, e.g. to induce sparsity in the learned
representation
- Denoising autoencoders: Add noise to the input, but try to reproduce
the *clean* input. The autoencoder learn to extract essential information of the input.

[img23]

## Extension

The decoder part can be viewed as representing a distribution $p(\boldsymbol{x} \mid \boldsymbol{z})$ over the encoder input $\boldsymbol{x}$ given the output $\boldsymbol{z} = f(\boldsymbol{x})$. c.f. probabilistic PCA.

If $p(\boldsymbol{x} \mid \boldsymbol{z})$ is, say, a spherical Gaussian with mean g(z), then optimizing the likelihood over a training set is equivalent to the usual squared loss.

Other distributions produce different losses.
