# Confusion Matrix

| | condition positive $+$ | condition negative $-$|
|:-:|:-:|:-: |
| predicted positive $\widehat{+}$  |  TP | FP  |
| predicted negative $\widehat{-}$ |  FN | TN  |


Metrics defined by column total (true condition)

- true positive rate, recall, sensitivity, probability of detection, power.

$$\mathrm{P}\left( \widehat{+} \vert + \right) =  \frac{TP}{\text{condition positive}} $$

- false negative rate, miss rate

$$\mathrm{P}\left( \widehat{-} \vert + \right)  = \frac{FN}{\text{condition positive}} $$

- false positive rate, fall-out, probability of false alarm, type I error

$$\mathrm{P}\left( \widehat{+} \vert - \right) = \frac{FP}{\text{condition negative}} $$

- true negative rate, specificity, selectivity

$$ \mathrm{P}\left( \widehat{-} \vert - \right) = \frac{TN}{\text{condition negative}} $$

Metrics defined by row total (predicted condition)

- false discovery rate

$$\mathrm{P}\left(- \vert \widehat{+}  \right) = \frac{FP}{\text{predicted positive}} $$

- precision

$$\mathrm{P}\left(+ \vert \widehat{+}  \right) = \frac{TP}{\text{predicted positive}} $$

Metrics defined by over all table

- accuracy

  Probability that the predicted result is correct

  $$\mathrm{P}\left( \widehat{+} \cap +\right) + \mathrm{P}\left( \widehat{-} \cap -\right) = \frac{TP+TN}{\text{total population}} $$

- prevalence

  Proportion of true condition in a population
  $$ \mathrm{P}\left( \widehat{+} \right) = \frac{TP+FP}{\text{total population}} $$



- $F_1$ score

  The $F_1$ score is the harmonic mean of precision and recall.

$$F_1 = \frac{1}{\text{precision}^{-1} + \text{recall}^{-1}} = \frac{TP}{TP + \frac{1}{2}\left( FP + FN \right)  } $$
