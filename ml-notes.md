Anomaly detection
=================

Model p(x)
p(xtest) < epsilon  # flag anomaly
p(xtest) >= epsilon # ok

Applications
============

* Fraud detection
* Manufacturing
* Datacenter Monitoring

Gaussian Distribution
=====================

sigma = standard deviation
sigma^2 = variance

If x is distrubuted Gaussian with mean mu and variance sigma^2 :
x ~ N(mu,sigma^2)
~ -> "distributed as"

Formula

Parameter estimation

Algorithm
=========

1. Choose features xi that you think might be indicative of anomalous examples
2. Fit parameters mu1, ..., mun, sigma1^2, ..., sigman^2
3. Compute "gaussian probability" p(x) for each new x. If p(x)<epsilon, anomaly detected

Development of an anomaly detection system
==========================================

1. Fit model
2. Predict label
  Algorithm evaluation: y=1 (anomaly), y=0 (normal)
3. Evaluation metrics:
  TP, FP, FN, TN
  Precision/recall
  F1-score

use cross-validation set to choose epsilon

Anomaly detection vs Supervised learning
========================================

AD small number of positive examples (y=1)
SL large number of positive examples (y=1)
AD large number of negative examples (y=0)
AD different type of anomalies
SL enough positive examples from training set to sense what are like
AD future anomalies do not look like the existing

Applications
============

* AD:
  * Fraud detection
  * Manufacturing
  * Datacenter monitoring
* SL:
  * spam classification
  * weather prediction
  * cancer classification

Multivariate Gaussian distribution
========================

Don't model p(x1), p(x2), ... seperately but all in one go.
p(x;mu,sigma)f

Original model vs multivariance Gaussian
=========================

* OM -> manually create features to capture anomalies, where features take unusual combination of values
* MG -> automatically captures correlations between features
* OM -> computationally cheaper, scales better to large n
* MG -> computationally expensive
* OM -> ok if m is small
* MG -> must have m>n or else Sum is non-invertible, m>=10n

Recommender Systems
===================

Collaborative Filtering algorithm

Feature learning

Low rank matrix factorization

Mean normalization


Online Learning
===============

y=1 (use service)
y=0 (not use service)

p(y=1|x;theta)

Repeat forever{
	Get (x,y) corresponding to user
	Update Theta using (x,y)
	Theta:=Theta-a(h(x)-y)*x (j=0,..,n)
}

Map-reduce and data parallelism
===============================

Split the training set calculation on multiple computers or cores. The sum the results and divide by m.

Artificial data synthesis
=========================

Use only when we have a low bias classifier.

* Create new data from scratch (ex use computer fonts with random background for the photo OCR case)
* Use examples that we currently have and create additional data to amplify the training set (ex artificial distrortions of characters, background noise in audio)
* Collect/label data by yourself
* Crowd sourcing (ex. amazon mechanical turk)
