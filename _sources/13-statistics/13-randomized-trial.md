# Causality and Randomized Trial


## Rubin Model of Causation

### Definitions

First we define some basic concepts.

Unit
: an object $i$ at a given point int time

Treatment and control
: A treatment $t$ is an intervention, whose effects we want to estimate as compared to a control $c$, which is the lack of an intervention.

State $S$
: The state of the world $S$ for unit $i$ is its treatment status, $S \in {t, c}$

Potential outcomes
: **Before** treatment is assigned, we can imagine the measure of interest $Y$ for each unit, under both treatment and control, labeled $Y_i^t$ and $Y_i^c$.

Assignment mechanism
: is the algorithm that determines whether $i$ receives $t$ or $c$.

Causal effect (treatment effect)
: is the difference between potential outcomes for a given unit $i$

$$T_i = Y_i^t - Y_i^c$$

Fundamental problem of causal inference
: Once the state $S \in {t, c}$ is determined, we observe one outcome $Y_i^t$ or $Y_i^c$. It is impossible to observe the value $Y_i^t$ and $Y_i^c$ for the same unit and, therefore, it is impossible to observe the effect of treatment $t$ versus control $c$ on object $i$.

Identification problem
: Because of the fundamental problem of causal inference, we cannot get an unbiased estimate (identify) causal parameters unless the assignment mechanism meets certain conditions.


### Statistical Solution

We give up on calculating $T_i = Y_i^t - Y_i^c$, and settle for $d$, the average treatment effect

$$
d = \operatorname{E}\left( Y^t - Y^c \right) = \operatorname{E}\left( Y^t\right) - \operatorname{E}\left(Y^c \right) = \operatorname{E}\left( T \right)
$$

Note that we cannot observe $Y^t$ and $Y^c$ at the same time. What we observe is actually $Y^t \mid S=t$ and $Y^c \mid S=c$. Consider the conditional expectations

$$
\operatorname{E}\left( Y^t \mid S=t  \right) \text{ and } \operatorname{E}\left( Y^c \mid S=c  \right)
$$

If $S$ is **independent** with $Y^t$ and $Y^c$, then

$$\begin{align}
\operatorname{E}\left( Y^t \mid S=t  \right)  &= \operatorname{E}\left( Y^t \right) \\
\operatorname{E}\left( Y^c \mid S=c  \right) &= \operatorname{E}\left( Y^c \right)
\end{align}$$

and hence

$$
d = \operatorname{E}\left( Y^t \mid S=t  \right)  - \operatorname{E}\left( Y^c \mid S=c  \right)
$$

Note that we can get unbiased estimators of the conditional expectations

$$\begin{align}
\hat{\operatorname{E}}\left( Y^t \mid S=t  \right)
 &= \frac{1}{n_t} \sum_{S = t} Y^t \\
\hat{\operatorname{E}}\left( Y^c \mid S=c  \right)  
&= \frac{1}{n_c} \sum_{S = c} Y^c
\end{align}$$

Therefore, if $S$ is independent with $Y^t$ and $Y^c$, then we have an unbiased estimator for $d$:

$$\begin{align}
\hat{d}
 &= \frac{1}{n_t} \sum_{i: S = t} Y^t  - \frac{1}{n_c} \sum_{i: S = c} Y^c \\
\operatorname{E}\left( \hat{d} \right)&= d
\end{align}$$

Random assignment of $S$ makes independence (at the sample level) plausible, but not certain

### Example of Assignment Mechanism: Rubin’s Perfect Doctor

Let's see an example where assignment $S$ and potential outcome $Y$ is not independent.

Suppose that patients with a particular form of cancer can have surgery ($t$) or not ($c$). We want to know the effect on years remaining. Imagine that there is a perfect doctor who can see both potential outcomes (years remaining):

| Person | $Y^t$ (Surgery) | $Y^c$ (Not) | $T$ |
|--------|----------|-----|-----|
| A      | 8        | 12  | -4  |
| B      | 1        | 8   | -7  |
| C      | 12       | 10  | 2   |
| D      | 2        | 4   | -2  |
| E      | 10       | 8   | 2   |
| F      | 11       | 8   | 3   |

From the table we have

$$\begin{align}
\operatorname{E}\left( Y^t \right) &= 44/6\\
\operatorname{E}\left( Y^c \right)&= 50/6\\
d&= \operatorname{E}\left( T \right)\\
&= -1
\end{align}$$

So on average, the treatment is **bad**.

Now suppose the omnipotent doctor assigns the treatment to those who will **benefit**. As researchers we would see


| Person | $Y_i^t$ (Surgery) | $Y_i^c$ (Not) | $T_i$ |
|--------|----------|-----|-----|
| A      |    ?     | 12  | $c$  |
| B      |     ?    | 8   | $c$   |
| C      | 12       | ?  | $t$   |
| D      |        ? | 4   | $c$   |
| E      | 10       |  ?  | $t$    |
| F      | 11       |   ? | $t$   |

and calculate

$$\begin{align}
\operatorname{E}\left( Y^t \mid S = t\right) &= 33/3\\
\operatorname{E}\left( Y^c \mid S = c\right)&= 24/3\\
d & =\operatorname{E}\left( T \right)\\
&= 3
\end{align}$$

and conclude that the treatment is **good**.

The cause of this contradiction is that the assignment $S$ is dependent on $Y$.

## Randomized Control Trial

### Randomized Control Trial Makes Independence Plausible

If treatment determined randomly, then sample independence is plausible (not guaranteed). Randomized control trial uses such a random assignment mechanism, e.g., flip a fair coin for each patient to determine the assignment. In this way, The two groups should have approximately the same distribution of unobservable characteristics, such as the potential outcome and other confounding variables.

We will often refer to exogenous treatment, versus endogenous. Exogenous meant to denote “outside control of optimizing agent”, makes independence plausible.

### Check for Randomization

By chance, RCT could result in a poor randomization. For instance, there are $2^6 = 64$ ways to assign treatment $t$ to 6 patients in the Perfect Doctor example. Thus, there is $\frac{1}{64}$ probability that RCT would give same (bad) result as Perfect Doctor.

To check for bad randomization, one can perform balance tests.
- Test for independence of $S$ and exogenous characteristics (not outcomes $Y$).
- If randomization is good, treated and control should be similar on observable attributes.


```{note}
Often as a first step we want to know the (non-causal) relationships between variables and then we try and figure out what is behind the patterns, i.e. if they relationships are possibly causal.
```

### Potential Problems and Limitations

#### Attrition

Participants may leave a study (i.e., attrit). So we don't observe $Y$. For instance, job training, drug trial, diet program, etc.
- If attrition is uncorrelated with $Y$, we still have independence between $S$ and $Y$ in remaining sample.
- If attrition is correlated with $Y$, then the independence is broken and there is selection bias. We will observe non-random set of treated outcomes.

#### Compliance

Sometimes people refuse to follow the treatment, but you still observe their outcomes.
- Perhaps treated people refuse to take treatment
- Perhaps control people seek treatment outside program
If you observe outcomes $Y$, you can still get a valid estimate of something called the **intent to treat** parameter.

Consider a compliance matrix of four groups of participants.

||Assigned $t$| Assigned $c$|
|-|-|-|
|Received $t$   | A  |  B |
|Received $c$   | C  |  D |

Here
- $A$ and $D$ are compliers
- $B$ and $C$ are non-compliers

A researcher might compare $(A,B)$ to $(C,D)$ to estimate the average treatment effect. But if compliance is a function of $Y$, then independence is lost.

Intent to treat analysis compares $(A,C)$ to $(B,D)$


#### Cost

Randomized controlled trial tend to be ery expensive compared with observational studies. What if you already have many observational datasets?

#### External validity

An estimate is internally valid if it is a good (unbiased) estimate for the population and setting of the trial.

An estimate is externally valid if it is a good (unbiased) estimate for populations and settings outside those of the trial.

All studies (RCT or not) have issues of external validity.
