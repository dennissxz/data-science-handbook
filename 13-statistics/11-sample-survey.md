# Sample Survey

*Explain commonly used sampling methods.*



## Simple random sample
- Every member and set of members has an equal chance of being included in the sample.  

- Technology, random number generators, or some other sort of chance process is needed to get a simple random sample.

- Example: A teachers puts students' names in a hat and chooses without looking to get a sample of students.

- Pros: Random samples are usually fairly representative since they don't favor certain members.

- Cons: Some groups may have no members being selected.


## Stratified random sample

- The population is first split into **groups**. The overall sample consists of some members from **every** group. The members from each group are chosen randomly. That is, each group has **at least one** member being selected.

- Example: A student council surveys 100 students. It is known that in the population the ratio of the freshmen, sophomores, juniors and seniors is 4:3:2:1. Then they can get random samples of 40 freshmen, 30 sophomores, 20 juniors, and 10 seniors.

- Pros: A stratified sample guarantees that members from each group will be represented in the sample, so this sampling method is good when we want some members from every group.

- Cons:
    - Requires a stratification variable, which can be difficult.

    - May also be expensive to implement since every group is visited.

    - Requires small within-group difference and large across-group difference.


## Cluster random sample

- The population is first split into groups. The overall sample consists of **every** member from some of the groups. The groups are selected at random.

- Example: An airline company wants to survey its customers one day, so they randomly select 5 flights that day and survey every passenger on those flights.

- Pros: A cluster sample gets every member from some of the groups, so it's good when each group reflects the population as a whole.

- Cons: Larger error than simple random sampling since the selected members are clustered.

## Systematic random sample

- Members of the population are put in some order. A starting point is selected at random, and every $k$-th element from the population is selected, where $k=\frac{\text{population size}}{\text{sample size}}$.

- Example: A principal takes an alphabetized list of student names and picks a random starting point. Every $20^{\text{th}}$, start superscript, start text, t, h, end text, end superscript student is selected to take a survey.

- Pros: Easy to implement. Smaller error than simple random sampling since the members are more evenly distributed.

- Cons: Require the order to be irrelevant to the experiment. If periodicity is present and the period is a multiple or factor of the interval used, the sample is especially likely to be **unrepresentative** of the overall population, making the scheme less accurate than simple random sampling. For instance, $\circ \bullet \circ \circ \circ \bullet \circ \circ \circ \bullet \circ \circ$ and $\circ \bullet \bullet  \bullet  \bullet \circ \circ \circ \circ \circ$.
