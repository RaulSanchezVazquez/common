# common
[BETA] Machine Learning Utilities.

Whenever it is possible we follow sklearn-like most common methods and practices, this means:

- Models have the well known `fit()`, `predict()` and `predict_proba()` methods.
- Data Transformers have the `transform()` and `inverse_transform()` methods.


What it provides:

- Pytorch AutoEncoder out-of-the-box.
- Date utilities (Enable to use dates for Models that relie on Structured data).
- Evaluation utilities (classification and regression metric reports)
- Factorization Machines (Own implementation [in construction])
- Random Search:
  - XGB
  - LGBM
  - Factorization Machines
- Utilities for Random Search
- Mean Confidence Intervals
- Spanish NLP string cleaning
- Parallel Map on lists
- Plot utilities
- Weight of Evidence Scaler
