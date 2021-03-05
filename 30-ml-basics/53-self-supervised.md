# Self-supervised Learning

```{margin}
The term "self-supervised" is used fairly loosely, and the boundary between self-supervised and other types of unsupervised learning is often unclear
```

In supervised learning, we have $\left(\mathbf{x}_{i}, y_{i}\right)$ and an objective $\min \sum_{i} L\left(f\left(\mathbf{x}_{i}, \theta\right), y_{i}\right)$.

:::{figure} selfsup-sup
<img src="../imgs/selfsup-sup.png" width = "50%" alt=""/>

Data $(\boldsymbol{x} _i, y_i)$ in supervised learning [Shakhnarovich 2021]

:::

In unsupervised learning, we only have $\boldsymbol{x} _i$, the objective is specified as a function of $(\boldsymbol{x} _i, f(\boldsymbol{x} _i))$, e.g. clustering, dimensionality reduction.

In self-supervised learning, data comes in form of multiple channels $(\boldsymbol{x}_i ,\boldsymbol{z}_i )$. Our goal is to predict $\boldsymbol{z}$ from $\boldsymbol{x}$. This is similar to supervised learning, but with no manual labels $y_i$, and $\boldsymbol{z}$ is inherent in data. The $\boldsymbol{z}$ here are some “proxy” or “pretext”. In order to predict it well, the model need to have a good understanding/representation of $\boldsymbol{x}$.

:::{figure} selfsup-selfsup
<img src="../imgs/selfsup-selfsup.png" width = "50%" alt=""/>

Data $(\boldsymbol{x} _i, \boldsymbol{z} _i)$ in self-supervised learning [Shakhnarovich 2021]

:::


## Proxy Task

Proxy tasks include


- Colorization

  Color gray scale images

  :::{figure} selfsup-colorization
  <img src="../imgs/selfsup-colorization.png" width = "50%" alt=""/>

  Colorization [Shakhnarovich 2021]
  :::

- Inpainting (mask reconstruction)

  Fill a masked part in an image

  :::{figure} selfsup-inpainting
  <img src="../imgs/selfsup-inpainting.png" width = "50%" alt=""/>

  Inpainting [Shakhnarovich 2021]
  :::

- Given two patches of an image, determine their relative position.

  :::{figure} selfsup-location
  <img src="../imgs/selfsup-location.png" width = "60%" alt=""/>

  Relative position [Shakhnarovich 2021]
  :::

- Solving jigsaw puzzles

  Learn to identify more probable permutations of image pieces

- Learn to predict soundtrack (more precisely, a cluster to which the soundtrack should be assigned) from a single video frame [Owens et al., 2016]

- Predicting video frame from previous frames

## Contrastive Learning

“Simple framework for Contrastive Learning of visual Representations” (Chen et al., 2020)

Consider pairs of views from the same image. Goal is to maximize similarity for matching pairs and dissimilarity for non-matching pair

:::{figure} selfsup-contrastive-views
<img src="../imgs/selfsup-contrastive-views.png" width = "70%" alt=""/>

Views (transformations)
:::

Computation graph:

:::{figure} selfsup-contrastive-graph
<img src="../imgs/selfsup-contrastive-graph.png" width = "30%" alt=""/>

Computational graph of contrastive learning
:::

## Contrastive Predictive Coding

For objects of spatial or temporal order.

- Vision: predict some part of the image from another part

  - Encode image patches (using CNN)
  - Predict, from the context embedding of the patches above some level, a patch below that level
  - Actual embedding of the patch $\boldsymbol{z} _i$, predicted $\hat{\boldsymbol{z}}_i$
  - Loss for patch $i$

    $$
    -\log \frac{\widehat{\mathbf{z}}_{i} \cdot \mathbf{z}_{i}}{\widehat{\mathbf{z}}_{i} \cdot \mathbf{z}_{i}+\sum_{n} \widehat{\mathbf{z}}_{i} \cdot \mathbf{z}_{n}}
    $$

    where $n$ goes over (sampled) other patches, both in this image and in other images

  :::{figure} selfsup-cpc-vision
  <img src="../imgs/selfsup-cpc-vision.png" width = "50%" alt=""/>

  CPC for image
  :::

- Speech: predict next speech segment

  :::{figure} selfsup-cpc-speech
  <img src="../imgs/selfsup-cpc-speech.png" width = "50%" alt=""/>

  CPC for speech
  :::

- Language: masked word reconstruction

  :::{figure} selfsup-bert
  <img src="../imgs/selfsup-bert.png" width = "60%" alt=""/>

  Word reconstruction
  :::

In these reconstruction tasks, some layers in the model can be regarded as the learned representation of the image/text/speech, which can be used for downstream tasks.

## Others

CLIP: Learning Transferable Visual Models From Natural Language Supervision (OpenAI, 2021)
