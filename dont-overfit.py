#!/usr/bin/env python
# coding: utf-8

# # Don't over fit II competition

# This is my EE551 individual project. It is a playground prediction competition on Kaggle.

# ## Exploratory Data Analysis(EDA)
# ### Datacollection

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


# In[6]:


train.shape


# In[7]:


train.head()


# In[8]:


sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.head()


# In[9]:


test.head()


# In[10]:


train.tail()


# In[11]:


train.columns


# In[12]:


print(train.info())


# ### Visualization

# In[13]:


train['target'].value_counts().plot.bar();


# In[14]:


import seaborn as sns
f,ax = plt.subplots(1,2,figsize=(12,6))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct ='%1.1f%%',ax=ax[0],shadow = True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('target', data = train, ax = ax[1])
ax[1].set_title('target')
plt.show()


# In[15]:


plt.figure(figsize = (26,24))
for i, col in enumerate(list(train.columns)[2:30]):
    plt.subplot(7, 4, i+1)
    plt.hist(train[col])
    plt.title(col)


# Values in columns are more or less similar.

# In[16]:


plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
train[train.columns[2:]].mean().plot('hist')
plt.title('Distribution of stds of all columns')
plt.subplot(1,2,2)
train[train.columns[2:]].std().plot('hist')
plt.title('Distribution of means of all columns')
plt.show()


# Columns have mean of 0 +/- 0.15 and std of 1 +/- 0.1.

# In[17]:


corr = train.corr()['target'].sort_values(ascending = False)


# In[18]:


corr.head(10)


# In[19]:


corr.tail(10)


# ## Logistic regression

# In[20]:


from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
X_train = train.drop(['id', 'target'], axis = 1)
y_train = train['target']
X_test = test.drop(['id'], axis = 1)


# Find the best parameters for function 'LogisticRegression'.

# In[21]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(penalty = 'l1', random_state = 42)
params = {'solver': ['liblinear', 'saga'],
         'C': [0.001, 0.1, 1, 10, 50],
         'tol': [0.00001,  0.0001, 0.001, 0.005],
         'class_weight': ['balanced', None]}
log_gs = GridSearchCV(log, params, cv = StratifiedKFold(n_splits = 5), verbose = 1, n_jobs = -1, scoring = 'roc_auc')

log_gs.fit(X_train, y_train)

log_best = log_gs.best_estimator_

print(log_best)
print(log_gs.best_score_)


# Define a function to plot learning curve.

# In[22]:


def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv = cv, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha = 0.1,color = 'r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha = 0.1,color = 'g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r',label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g',label = "Cross_validation score")
    plt.legend(loc = 'best')
    return plt


# Plot the learning curve of log_best.

# In[23]:


learningCurve = plot_learning_curve(log_best, "LR learning curves",X_train, y_train, cv = StratifiedKFold(n_splits = 5))


# Define a function to draw roc curve.

# In[24]:


from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from scipy import interp
def plot_roc(clf, X = X_train, y = y_train, n = 6):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 0
    classifier = clf
    cv = StratifiedKFold(n_splits = n)
    for train, test in cv.split(X,y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle = '--', lw =2, color = 'r', label = 'Chance', alpha =.8)
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color = 'b', label = r'Mean ROC(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw =2, alpha = .8)
    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr,1)
    tprs_lower = np.maximum(mean_tpr - std_tpr,0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color ='grey', alpha = 0.2, label = r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc = 'lower right')
    plt.show()
                 
                 


# Plot the roc curve of log_best.

# In[25]:


roc = plot_roc(log_best)


# cv_score is far away from training score. It is overfitting. C is responsible for level of regularization and the smaller it is, the bigger the level of regularization it is. 
# First try C = 0.1

# In[26]:


log_p0 = LogisticRegression(class_weight = 'balanced', penalty = 'l1', C = 0.1, solver = 'saga', random_state = 42)
learningCurve0 = plot_learning_curve(log_p0, "LR learning curves", X_train, y_train, cv = StratifiedKFold(n_splits = 5))


# In[27]:


roc0 = plot_roc(log_p0)


# Try C = 0.05.

# In[28]:


log_p1 = LogisticRegression(class_weight = 'balanced', penalty = 'l1', C = 0.05, solver = 'saga', random_state = 42)
learningCurve1 = plot_learning_curve(log_p1, "LR learning curves", X_train, y_train, cv = StratifiedKFold(n_splits = 5))


# In[29]:


roc1 = plot_roc(log_p1)


# Try C = 0.15.

# In[30]:


log_p2 = LogisticRegression(class_weight = 'balanced', penalty = 'l1', C = 0.15, solver = 'saga', random_state = 42)
learningCurve2 = plot_learning_curve(log_p2, "LR learning curves", X_train, y_train, cv = StratifiedKFold(n_splits = 5))


# In[31]:


roc2 = plot_roc(log_p2)


# It seems like that when C = 0.1, the model performs best.

# Output the first submission file.

# In[32]:


log_p0.fit(X_train, y_train)
log_pred0 = log_p0.predict_proba(X_test)[:,1]
submission0 = pd.DataFrame({'id':test['id'],
                          'target':log_pred0})
submission0.to_csv('submissions/submission0.csv', index = False)


# ## Feature Selection

# Use eli5 to do the feature selection.

# In[33]:


import eli5
eli5.show_weights(log_p0,top = 50)


# In[34]:


(log_p0.coef_ != 0).sum()


# In[35]:


top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(log_p0).feature if 'BIAS' not in i]
X_train_new = train[top_features]


# In[36]:


learningCurve3 = plot_learning_curve(log_p0, "LR learning curves", X_train_new, y_train, cv = StratifiedKFold(n_splits = 5))


# In[37]:


roc3 = plot_roc(log_p0,X_train_new)


# In[38]:


log_p0.fit(X_train_new, y_train)
X_test_new = test[top_features]
log_pred3 = log_p0.predict_proba(X_test_new)[:,1]
submission1 = pd.DataFrame({'id':test['id'],
                          'target':log_pred3})
submission1.to_csv('submissions/submission1.csv', index = False)


# In[39]:


X_test_new.head()


# ## Add new statistics.

# In[40]:


train['mean'] = train.mean(1)
train['std'] = train.std(1)
test['mean'] = test.mean(1)
test['std'] = test.std(1)
X_train_add = train[top_features + ['mean']]
X_test_add = test[top_features + ['mean']]


# In[41]:


learningCurve4 = plot_learning_curve(log_p0, "LR learning curves", X_train_add, y_train, cv = StratifiedKFold(n_splits = 5))


# In[42]:


roc4 = plot_roc(log_p0,X_train_add)


# In[43]:


log_p0.fit(X_train_add, y_train)
log_pred4 = log_p0.predict_proba(X_test_add)[:,1]
submission2 = pd.DataFrame({'id':test['id'],
                          'target':log_pred4})
submission2.to_csv('submissions/submission2.csv', index = False)


# # Decison Tree

# In[44]:


from sklearn.tree import DecisionTreeClassifier
X_train = train.drop(['id', 'target'], axis = 1)
y_train = train['target']
X_test = test.drop(['id'], axis = 1)
tree = DecisionTreeClassifier()
params = {'criterion':['gini','entropy'],
          'max_depth':[1,3,5,7,10],
         'class_weight': ['balanced', None]}
trees = GridSearchCV(tree, params, cv = StratifiedKFold(n_splits = 5), verbose = 1, n_jobs = -1, scoring = 'roc_auc')

trees.fit(X_train, y_train)

tree_best = trees.best_estimator_

print(tree_best)
print(trees.best_score_)


# In[45]:


learningCurve5 = plot_learning_curve(tree_best, "DT learning curves", X_train_add, y_train, cv = StratifiedKFold(n_splits = 5))


# In[46]:


roc5 = plot_roc(tree_best,X_train_add)


# Decision tree is not suitable for this dataset.

# # Lasso Regression

# In[47]:


from sklearn.linear_model import Lasso
X_train = train.drop(['id', 'target'], axis = 1)
y_train = train['target']
X_test = test.drop(['id'], axis = 1)
las = Lasso(alpha=0.031, tol=0.01, random_state=42, selection='random')

params = {
            'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
            'tol'   : [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]
        }
las_ss = GridSearchCV(las, params, cv = StratifiedKFold(n_splits = 5), verbose = 1, n_jobs = -1, scoring = 'roc_auc')

las_ss.fit(X_train, y_train)

las_best = las_ss.best_estimator_

print(las_ss)
print(las_ss.best_score_)


# In[48]:


learningCurve6 = plot_learning_curve(las_best, "Lasso learning curves", X_train, y_train, cv = StratifiedKFold(n_splits = 5))


# In[49]:


def plot_roc0(clf, X = X_train, y = y_train, n = 6):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    classifier = clf
    cv = StratifiedKFold(n_splits = n)
    for train, test in cv.split(X,y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle = '--', lw =2, color = 'r', label = 'Chance', alpha =.8)
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color = 'b', label = r'Mean ROC(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw =2, alpha = .8)
    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr,1)
    tprs_lower = np.maximum(mean_tpr - std_tpr,0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color ='grey', alpha = 0.2, label = r'$\pm$ 1 std. dev.')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc = 'lower right')
    plt.show()


# In[50]:


roc6 = plot_roc0(las_best,X_train)


# In[51]:


las_best.fit(X_train_add, y_train)
las_best_pred = las_best.predict(X_test_add)
submission3 = pd.DataFrame({'id':test['id'],
                          'target':las_best_pred})
submission3.to_csv('submissions/submission3.csv', index = False)


# In[ ]:




