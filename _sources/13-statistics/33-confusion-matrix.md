# Confusion Matrix

| | condition positive $+$ | condition negative $-$|
|:-:|:-:|:-: |
| predicted positive $\widehat{+}$  |  TP | FP  |
| predicted negative $\widehat{-}$ |  FN | TN  |


Metrics defined by column total (true condition)

- true positive rate, recall, sensitivity, probability of detection, power.

$$\operatorname{\mathbb{P}}\left( \widehat{+} \vert + \right) =  \frac{\mathrm{TP}}{\text{condition positive}} $$

- false negative rate, miss rate

$$\operatorname{\mathbb{P}}\left( \widehat{-} \vert + \right)  = \frac{\mathrm{FN}}{\text{condition positive}} $$

- false positive rate, fall-out, probability of false alarm, type I error

$$\operatorname{\mathbb{P}}\left( \widehat{+} \vert - \right) = \frac{\mathrm{FP}}{\text{condition negative}} $$

- true negative rate, specificity, selectivity

$$ \operatorname{\mathbb{P}}\left( \widehat{-} \vert - \right) = \frac{\mathrm{TN}}{\text{condition negative}} $$

Metrics defined by row total (predicted condition)

- false discovery rate

$$\operatorname{\mathbb{P}}\left(- \vert \widehat{+}  \right) = \frac{\mathrm{FP}}{\text{predicted positive}} $$

- precision

$$\operatorname{\mathbb{P}}\left(+ \vert \widehat{+}  \right) = \frac{\mathrm{TP}}{\text{predicted positive}} $$

Metrics defined by overall table

- accuracy

  Probability that the predicted result is correct

  $$\operatorname{\mathbb{P}}\left( \widehat{+} \cap +\right) + \operatorname{\mathbb{P}}\left( \widehat{-} \cap -\right) = \frac{\mathrm{TP}+\mathrm{TN}}{\text{total population}} $$

- prevalence

  Proportion of true condition in a population

  $$ \operatorname{\mathbb{P}}\left( + \right) = \frac{\mathrm{TP}+\mathrm{FN}}{\text{total population}} $$


- $F_1$ score

  The $F_1$ score is the harmonic mean of precision and recall. Hence, tt is often used to balance precision and recall.

  $$F_1 = \frac{1}{\text{precision}^{-1} + \text{recall}^{-1}} = \frac{\mathrm{TP}}{\mathrm{TP} + \frac{1}{2}\left( \mathrm{FP} + \mathrm{FN} \right)  } $$

- Macro $F_1$ score

  Macro $F_1$ score extends $F_1$ score for multiple binary labels, or multiple classes. It is computed by first computing the F1-score per class/label and then averaging them. Aka Macro F1-averaging.

  $$
  \text{Macro } F_1 = \frac{1}{K} \sum_{k=1}^K  F_{1, k}
  $$

  where $K$ is the number of classes/labels.

ROC curve: receiver operating characteristics curves,
- y-axis: true positive rate, aka TPR, recall, sensitivity
- x-axis: false positive rate, aka FPR, 1-specificity
- varying a parameter controlling the discrimination between positives and negatives
- classifiers with curves pushing more into the upper left-hand corner are generally considered more desirable. 45 degree line is random guessing.
- AUC: area under the curve, close to 1 is good. 0.5 is random guessing.

:::{figure} roc
<img src="../imgs/roc.png" width = "70%" alt=""/>

ROC curves [[Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)]
:::
