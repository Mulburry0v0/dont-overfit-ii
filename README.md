# dont-overfit-ii

  This is my EE551 individual project. It is a playground prediction competition on Kaggle. | Name: Lijin Zhou
## Proposal:

  This project is a competition that challenges mere mortals to model a 20000*300 matrix of contiuous variables using 250   training samples. The result should be a prediction of the binary `target` associated with each row, without overfitting to the minimal set of training examples provided.
## Files:
  * `train.csv` - the training set. 250 rows.
  * `test.csv` - the test set. 19750 rows.
  * `sample_submission.csv` - a sample submission file in the correct format
  * `dont-over-fit.py` - project code
  * `dont-over-fit.iqynb` - project code in jupyter notebook
  * `dont-over-fit.pdf` - project report
  * `submission0.csv`, `submission1.csv`, `submission2.csv`, `submission3.csv` - predict results of different methods, output from project code
## Packages need to install:
  ipython, numpy, pandas, matplotlib.pyplot, seaborn, eli5, sklearn etc.
## General Steps:
### 1. EDA on the features:
  * Read Dataset;
  * Display Dataset.
### 2. Try different models:
  * logistic regression;
  * decision tree;
  * lasso regression.
### 3. Feature selection:
  * Plot Features Importance;
  * Low Importance Features;
  * Removing Features.
### 4. Model improvement:
  * feature selection;
  * parameter optimistic;
  * add statistics;
  * add distance.
### 5. Current Result:
  * The best score I have is around 0.85.
### 6. Keep learning and improving:
  * Will try more models, including RandomForest, Adaboost, Bagging, GradientBoosting;
  * Essembling predictions of different methods;
  * Read more articles to find methods to avoid overfitting, since the methods I know could not improve the result sharply.
