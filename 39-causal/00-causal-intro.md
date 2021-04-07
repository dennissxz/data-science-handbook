# For Causal Inference

In the intersection of machine learning and causal inference, there are two branches:

- Machine learning for causality

  Question: among users taking a certain drug, many of them develop a cancer. Did the drug cause the cancer?

  Experiments to answer this question are not feasible: not ethical, take long time, and costly.

  Can we answer this question from observational data set using machine learning methods?

- Causality for machine learning

  Issues in machine learning:
  - move the model out of certain domains, accuracy drops
  - dependence of irrelevant features: change some feature value (word in sentence), prediction (sentiment) changes.

  Can use causal inference to solve these issues?

Reference:
- Advanced Data Analysis from an Elementary Point of View Part III[link](https://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/)
