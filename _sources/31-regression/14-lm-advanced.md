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

Recall the common trends assumption. Can we generalize it? What if there are more than 2 periods and more than 2 groups? We introduce first difference and fixed effect, for more than s.

Panel data, aka. longitudinal data, are data constructed from repeated samples of the same entities/individuals $i$ at different points $t$ in time.

$$
Y_{i t}=\beta_{0}+\beta_{1} X_{1 i t}+\ldots+\beta_{k} X_{k i t}+u_{i t}
$$

where entities $i=1, \ldots, N$ and time $t=1, \ldots T$, if balanced. There can also be unbalanced panel data such that the total number of observations is less than $NT$.

For instance,

- Graduation rate at each school in Chicago over last 10 years
- Poverty rate for each state over several years
- Earnings of workers over time before and after disability


:::{admonition,note} Note

Panel data is different from repeated cross-section data that have multiple observations per sample in multiple time periods.

Whether an analysis uses repeated cross-section or panel data sometimes depends on how raw data are manipulated. Consider a random sample of 100 people from each state, taken every year.
- Different people each year – so it is repeated cross-section if $i$ indexes people
- If the individuals from a state are aggregated into a state average, then since we have the same states year after year it is panel — now $i$ indexes states

:::

Panel data enable us to remove bias from certain types of omitted variables.

- If omitted variables for entity $i$ do not change over time,
- Or if omitted variables for time period $t$ do not differ across entity,

Then panel data gives unbiased estimates.

### First Difference

Suppose we have a panel data set at two time points $t_1$ and $t_2$. Suppose the true model is

$$Y_{it} = \beta_0 + \beta_1 X_{1it} + \beta_2 X_{2it} + u_{it}$$

where $\mathbb{E}(u_{it} \vert X_{1it}, X_{2it})=0$. But we only observe $X_{1}$ and omit $X_2$. Then running a regression model $Y_i \sim X_{1i}$ at each time point leads to a biased estimate of $\beta_1$. However, the difference is

$$\Delta Y_{i} = \beta_1 \Delta X_{1i} + \beta_2 \Delta X_{2i} + \Delta u_{i}$$

If $\Delta X_{2i}=0$ for each $i$, i.e., for $X_{2it}$ does not change over time $t$ for entity $i$, then we can run regression $\Delta Y_i \sim \Delta X_{1i}$ (include intercept) and obtain an unbiased estimate of $\beta_1$. The intercept estimate $\tilde{\beta}_0$ can be interpreted as the change of $\beta_0$ over time.

addimg *3

Rationales of FD
- FD regress the change in $Y$ against the change in $X$
- If omitted variables are constant over time (time invariant), then they will be unrelated to changes in $Y$ and $X$ for a given $i$.
- Differencing gets rid of time invariant unobservables, as well as time invariant observables.

For $T > 2$ case.

We can run compute $\Delta Y_{it} = Y_{i(t+1)} - Y_{it}$ and $\Delta X_{1it}$, for each $t=1,\ldots, T-1$, and use all these $(T-1)n$ number of $(\Delta Y, \Delta X)$ pairs to run a regression to obtain $\hat{\beta}_1$.


Pros
- Solve bias caused by time invariant variables.

Cons
- Cannot solve bias caused by time varying variables, if they are correlated with $\Delta X$. It's like we omit $\Delta X_{2i} \ne 0$ in $\Delta Y_{i} = \beta_1 \Delta X_{1i} + \beta_2 \Delta X_{2i} + \Delta u_{i}$
- Cannot estimate coefficient for time invariant variables ($\beta_2$ in the above case)
- Lower variation in independent variable sine $\sigma_{\Delta X}^{2} \ll \sigma_{X}^{2}$. Higher standard error of estimate
- May exaggerate measurement error since signal is reduced but noise variance is larger.


### Fixed Effect

#### Entity Fixed Effect

In FD we drop time invariant variable $\beta_2 X_{2it} = \beta_2 X_{2i}$ to estimate $\beta_1$, but in fixed effect model we estimate these $\beta_2 X_{2i}$. Suppose again the true model is

$$Y_{it} = \beta_0 + \beta_1 X_{1it} + \beta_2 X_{2it} + u_{it}$$

If $X_{2it}$ is time invariant, then we can write

$$Y_{it} = (\underbrace{\beta_0 + \beta_2 X_{2i}}_{\alpha_i}) + \beta_1 X_{1it} + u_{it}$$

which suggests that each entity $i$ has a different intercept $\alpha_i$. Hence, the new equation can be estimated by letting each entity $i$ have a unique intercept. This is called fixed effects regression.

$$Y_{i t}=\beta_{1} X_{i t}+\sum_{i=1}^{N} \alpha_{i} d_{i}+u_{i t}$$

where $d_i$ is a dummy variable indicating if an observation is in group $i$. The total number of observations in this regression is $NT$, and number of parameters is $N+1$.

Note that
- If $T=2$ then FD=FE.
- If there is autocorrelation of errors within an entity, use clustered standard error.
- de-mean interpretation
- precision of $a_i$ depends on the number of observations in entity $i$.


#### Time Fixed Effect

Some omitted variables very over time but are the same across entities within a time period.

$$Y_{i t}=\beta_{1} X_{i t}+\sum_{t=1}^{T} \alpha^{t} d^{t}+u_{i t}$$

Examples:
- Federal policy may affect all states the same
- Macroeconomic shocks may affect many workers the same
- Technological change may affect all firms the same
- Quarterly seasonal effect
- Fall/summer effect for agricultural data

The total number of observations in this regression is $NT$, and number of parameters is $T+1$.

#### Both

We can include both entity and time fixed effect. This will eliminate both time invariant unobservables within each entity, and entity invariant unobservables within each time period.

$$Y_{i t}=\beta_{1} X_{i t}+\sum_{i=1}^{N} \alpha_{i} d_{i}+\sum_{t=1}^{T} \alpha^{t} d^{t}+u_{i t}$$

The total number of observations in this regression is $NT$, and number of parameters is $N + T+1$.

Example:
- Drinking culture fixed within states ($a_i$)
- Federal policy changes affect all states the same ($a_t$)

Pros
- Can eliminate bias due to time invariant unobservable factors or entity invariant unobservable factors

Cons
- Time varying unobservables that are unique to each state (not a common shock) can still cause bias.

## Time Series

Time series data are data that are temporally ordered.



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
