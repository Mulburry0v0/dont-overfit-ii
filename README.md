# dont-overfit-ii

  This is my EE551 individual project. It is a playground prediction competition on Kaggle.
## Proposal:

  This project is a competition that challenges mere mortals to model a 20000*300 matrix of contiuous variables using 250   training samples. The result should be a prediction of the binary target associated with each row, without overfitting to the minimal set of training examples provided.
## Files:
  * train.csv - the training set. 250 rows.
  * test.csv - the test set. 19750 rows.
  * sample_submission.csv - a sample submission file in the correct format
## Packages may use:
  numpy, pandas, matplotlib.pyplot, seaborn, lightgbm, sklearn etc.
## General Steps:
1. EDA on the features:
  * Read Dataset;
  * Display Dataset.
2. Try different models:
  * linear regression;
  * decision tree etc.
3. Feature selection:
  * Plot Features Importance;
  * Low Importance Features;
  * Removing Features.
4. Model training.
