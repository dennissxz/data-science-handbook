# Linear Models - Advanced Topics

We introduced how to use linear models for advanced analysis, e.g. causal effect.


## Natural Experiments

In natural experiments, we use observed data, but it is reasonable to suspect treatment uncorrelated with error without an actual experiment, i.e. it is "as if" there had been randomization.


- A **cross-section** of data is a sample drawn from a population at a single point in time
- **Repeated cross-sections** are samples drawn from the same **population** at several points in time
- If we have information on the same **individuals** at more than one point in time, we have **panel data**

We often use the “difference-in-differences” estimator to study repeated cross-sections or panel data when we have a natural experiment.

### Difference-in-difference

In DiD, we uses a control group to net out any trends over time.

| | Treatment | Control
| - | - |- |
|Before   |  $A$ | $B$  |
|After   |  $C$ | $D$  |
|Difference | $C-A$  | $D-B$  |

```{margin}
Other statistic of interest or more robust e.g. median, percentiles, log, can also be used.
```

Let $\bar{y}_A, \bar{y}_B, \bar{y}_C, \bar{y}_D$ be the corresponding means of each group. DiD is computed as

$$
DiD = (\bar{y}_C - \bar{y}_A) - (\bar{y}_D- \bar{y}_B)
$$

Then we can conduct a two-sample $t$-test. We can also use regression with two categorial variables (After, Treat) and one interaction term.

$$
Y=\beta_{0}+\beta_{1} \text {Treated} * \text {After}+\beta_{2} \text {Treated}+\beta_{3} \text {After}+\varepsilon
$$

where

- $\hat{\beta}_0 = \bar{y}_B$
- $\hat{\beta}_2 = \bar{y}_A - \bar{y}_B$
- $\hat{\beta}_3 = \bar{y}_D - \bar{y}_B$
- $\hat{\beta}_1 = (\bar{y}_C - \bar{y}_A) - (\bar{y}_D - \bar{y}_B) = DiD$

and we can test $\hat{\beta}_1 = 0$.

To control other variables, we add them into the model (note about over controlling)

$$
Y=\beta_{0}+\beta_{1} \text { Treated } * \text { After }+\beta_{2} \text { Treated }+\beta_{3} \text { After }+\sum_k \gamma_k X_k+\varepsilon
$$


### Choice of Control Group

It is often hard to choose control group. One criteria is the common trends assumption.

Assumption (Common trends):
: In a before/after study, whatever factors change with time must affect the treatment and control group the **same**.
  - Use both theory and data to assess common trends
    - theory: logical reasoning
    - data: plot $y$ for the two groups along time before the treatment, see if they have similar ups and downs.
  - If some unobserved factor changes more for treated group than for control, then we have bias.

A control group must enable you to approximate the counterfactual for the treated group – what would have happened to them if they had not received treatment?

For instance, in the project *Taxes on the Rich* (Feldstein (1995), Goolsbee (2000)). The tax policy impose a decrease in marginal tax rates for high income earners in 1986. Goolsbee examines increase in marginal rates in 1993 for high income earners. Lower (but still high) earners are control.


### Internal Validity

Internal validity refers to whether one can validly draw the inference that within the context of the study, to conclude that the differences in the dependent variables were caused by the differences in the relevant explanatory variables.

Some issues are

1. Omitted Variables: events, other than the experimental treatment, occurring between pre-intervention and post- intervention observations that provide alternative explanations for the results.

2. Trends in Outcomes: processes within the units of observation producing changes as a function of the passage of time per se, such as inflation, aging, and wage growth.

3. Misspecified Variances: omission of group error terms. Bertrand (2004) et al.

4. Mismeasurement: changes in definitions or survey methods that may produce changes in the measured variables. NHIS, CPS.

5. Political Economy: endogeneity of policy changes due to governmental responses to variables associated with past or expected future outcomes. Campbell and drunk driving.

6. Simultaneity: endogeneity of explanatory variables due to their joint determination with outcomes. Think price and quantity.

7. Selection: assignment of observations to treatment groups in a manner that leads to correlation between assignment and outcomes in the absence of treatment. Selection can take many forms. Trainees often do well relative to their recent past...

8. Attrition: the differential loss of respondents from treatment and comparison groups.

9. Omitted Interactions: differential trends in treatment and control groups or omitted variables that change in different ways for treatment and control groups. For example, a time trend in a treatment group that is not present in a comparison group. The exclusion of such interactions is a common identifying assumption in the designs of natural experiments. This is the common trends assumption.

## Regression Discontinuity

[[Wiki](https://en.wikipedia.org/wiki/Regression_discontinuity_design)]

We want to analyze an policy effect to different group of people. For instance,

- effect of extended unemployment insurance benefits on willingness to work (measured by actual unemployment period), where the benefits are different for different age group, characterized by age cutoffs.
- effect of medicaid on health (measured by mortality or hospitalization rate), where the medicaid are different for different age group, characterized by birth date cutoffs
- in loan application, a rule of thumb is that applicants with credit score greater than 620 have low delinquency probability and hence more likely to get accepted.

There are other cutoffs, like earnings.

In sum, there is a sharp policy at cutoff point $a^*$, while other characteristics that influence outcome ($y$) are very similar around $a^*$. It should be as if we **randomized** and those just above $a^*$ are the treatment group and just below a* are the control group. To analyze the policy effect at this cutoff, the regression discontinuity equation is


$$
Y_{i a}=\beta_{0}+\beta_{1} D_{a \geq a *}+f(a)+\varepsilon_{i a}
$$

where

- $i$ is individual, $a$ is so called **running variable** (e.g. age, score, time)
- $y_{ia}$ is an outcome variable of interest
- $D_{a \ge a^*}$ is an indicator for an individual being above the cutoff $a^*$
- $f(a)$ is a function of $a$, often linear or quadratic, and often with different slopes above and below $a^*$
- The observations used to run this regression are those around $a^*$, e.g. there is a bandwidth/window width.
- We can run this equation at each cutoff points in the policy.

We are interested in $\beta_1$.

Note

- RD requires a large sample
- wider window (larger sample) increase precision as well as bias, precision v. bias tradeoff
- In some cases the cutoff is "hard" (unemployment benefit), and some times it is soft (credit score). If it is soft we call this fuzzy RD

## Instrumental Variables

Suppose we are in


## Panel Data







## Categorical Data

TBD...

dummy variables $X_{ij}$

when $c = 2$,

interpretation
- $\hat{\beta}_1$: difference in means between the group with $X=1$ and $X=0$.
- $\hat{\beta}_0$: mean of the group with $X=0$.

TBD

https://www.1point3acres.com/bbs/thread-703302-1-1.html


.


.


.


.


.


.


.


.
