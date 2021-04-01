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

  The $F_1$ score is the harmonic mean of precision and recall.

$$F_1 = \frac{1}{\text{precision}^{-1} + \text{recall}^{-1}} = \frac{\mathrm{TP}}{\mathrm{TP} + \frac{1}{2}\left( \mathrm{FP} + \mathrm{FN} \right)  } $$
