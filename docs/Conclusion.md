---
layout: page
title: "Results & Conclusion"
description: "Implications and future work"
header-img: "img/home-bg.jpg"
---

# Results

## Comparison of models for disease (MCI and AD) vs. non-disease

To compare the performance of reproducibility and classification for the above models, we printed an accuracy table and drew a ROC curve. 
The accuracy table showed except for the baseline logistic regression model, the other models all generated excellent accuracy score for both training and test dataset. Scores for the training set are always higher than the test set.

```python
## comparasion of accuracy of different model
classifiers=[ovr_model, rf1_best_model,   best_tree_model, AdaBoost_model, qda_model]
names = ['baseline logistic','random forest', 'decision tree', 'boosting','qda']
accuracy_table=pd.DataFrame()
j=0
for i in classifiers:
    accuracy_table.loc[names[j],r'$ accuracy(train)$']=accuracy_score(y_train1, i.predict(X_train1))
    accuracy_table.loc[names[j],r'$ accuracy(test)$']=accuracy_score(y_test1, i.predict(X_test1))
    j=j+1

accuracy_table.loc['optimal logistic',r'$ accuracy(train)$']=accuracy_score(y_train1, ovr_model_sparse.predict(X_train_restrict))
accuracy_table.loc['optimal logistic',r'$ accuracy(test)$']=accuracy_score(y_test1, ovr_model_sparse.predict(X_test_restrict))
accuracy_table
```

<div align="center"><font size="2"><b>Table 1. Accuracy table (Disease vs. Non-disease)</b></font></div>
<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/t1.png" alt="t1" width="350"/></div>

However, accuracy score alone just told us the predictive capability of models, thus ROC curve and AUC calculations were further explored to provide the information of classification. While the baseline logistic model gave the lowest classification performance, the other models all have pretty good AUC score. We will choose random forest and boosting as our final models as they have excellent performance in both reproducibility and classification.

```python
### ROC for decision tree
fpr_dt11, tpr_dt11, thres_dt11 = roc_curve(y_test1, best_tree.predict_proba(X_test1)[:,1])
fpr_dt12, tpr_dt12, thres_dt12 = roc_curve(y_test1, np.zeros((len(y_test1), 1)))
auc_dt11 = roc_auc_score(y_test1, best_tree.predict_proba(X_test1)[:,1])
auc_dt12 = roc_auc_score(y_test1, np.zeros((len(y_test1), 1)))
### ROC for bset random forest
fpr_qda11, tpr_qda11, thres_qda11 = roc_curve(y_test1, qda_model.predict_proba(X_test1)[:,1])
auc_qda11 = roc_auc_score(y_test1, qda_model.predict_proba(X_test1)[:,1])
###ROC for boosting
fpr_boo11, tpr_boo11, thres_boo11 = roc_curve(y_test1, AdaBoost_model.predict_proba(X_test1)[:,1])
auc_boo11 = roc_auc_score(y_test1, AdaBoost_model.predict_proba(X_test1)[:,1])
###ROC for random forest
fpr11_best, tpr11_best, thres11_best = roc_curve(y_test1, rf1_best.predict_proba(X_test1)[:,1])
auc11_best = roc_auc_score(y_test1, rf1_best.predict_proba(X_test1)[:,1])
###ROC for logistic regression
fpr_ovr11, tpr_ovr11, thres_ovr11 = roc_curve(y_test1, ovr_model_sparse.predict_proba(X_test_restrict)[:,1])
auc_ovr11 = roc_auc_score(y_test1, ovr_model_sparse.predict_proba(X_test_restrict)[:,1])

fig, ax = plt.subplots(1,1)
ax.plot(fpr_dt11, tpr_dt11, '-', alpha=0.8, label='Best decision tree' % (auc_dt11 ))
ax.plot(fpr_qda11, tpr_qda11, '-', alpha=0.8, label='QDA' % (auc_qda11 ))
ax.plot(fpr_boo11, tpr_boo11, '-', alpha=0.8, label='Boosting' % (auc_boo11 ))
ax.plot(fpr11_best, tpr11_best, '-', alpha=0.8, label='Best random forest' % (auc11_best))
ax.plot(fpr_ovr11, tpr_ovr11, '-', alpha=0.8, label='Optimized egression' % (auc_ovr11))
ax.plot(fpr_dt12, tpr_dt12, '-', alpha=0.8, label=' 0 classifier' % (auc_dt12))
plt.title("ROC curves model for disease and non-disease")
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.legend()
plt.show()
```
```python
## comparison of accuracy of different model
classifiers=[ovr_model, rf1_best_model, best_tree_model, AdaBoost_model, qda_model]
names = ['baseline logistic','random forest', 'decision tree', 'boosting','qda']
AUC_table=pd.DataFrame()
j=0
for i in classifiers:
    AUC_table.loc[names[j],r'$ AUC $']=roc_auc_score(y_test1, i.predict_proba(X_test1)[:,1])
    j=j+1

AUC_table.loc['optimized logistic',r'$ AUC $']=accuracy_score(y_test1, ovr_model_sparse.predict(X_test_restrict))
AUC_table
```

<img src="https://yueli1201.github.io/Alzheimer/figures/15.jpeg" alt="15" width="750"/><br/>
<div align="center"><font size="2"><b>Fig 15. ROC for different models for disease vs. non-disease</b></font></div>

## Comparison of models for AD vs. MCI

Likewise, we printed an accuracy table and a ROC curve to compare the performance of reproducibility and classification for the above models. Similar to the previous comparison, baseline logistic gave the lowest accuracy score, while random forest, boosting still have the highest performance in terms of reproducibility.

```python
## comparasion of accuracy of different model
classifiers=[ovr_model2, rf2_best_model, best_tree_model2, AdaBoost_model2, qda_model2]
names = ['baseline logistic','random forest', 'decision tree', 'boosting','qda']
accuracy_table=pd.DataFrame()
j=0
for i in classifiers:
    accuracy_table.loc[names[j],r'$ accuracy(train)$']=accuracy_score(y_train2, i.predict(X_train2))
    accuracy_table.loc[names[j],r'$ accuracy(test)$']=accuracy_score(y_test2, i.predict(X_test2))
    j=j+1

accuracy_table.loc['optimized logistic',r'$ accuracy(train)$']=accuracy_score(y_train2, ovr_model_sparse2.predict(X_train_restrict2))
accuracy_table.loc['optimized logistic',r'$ accuracy(test)$']=accuracy_score(y_test2, ovr_model_sparse2.predict(X_test_restrict2))
accuracy_table
```

<div align="center"><font size="2"><b>Table 2. Accuracy table (AD vs. MCI)</b></font></div>
<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/t2.png" alt="t2" width="350"/></div>

In terms of classification performance, we drew ROC curved to visualize the comparisons and calculated corresponding AUC for each model. As we could see, random forest and boosting performed the best for both accuracy score and AUC.

```python
## ROC for decision tree
fpr_dt21, tpr_dt21, thres_dt21 = roc_curve(y_test2, best_tree2.predict_proba(X_test2)[:,1])
fpr_dt22, tpr_dt22, thres_dt22 = roc_curve(y_test2, np.zeros((len(y_test2), 1)))
auc_dt21 = roc_auc_score(y_test2, best_tree2.predict_proba(X_test2)[:,1])
auc_dt22 = roc_auc_score(y_test2, np.zeros((len(y_test2), 1)))
### ROC for bset random forest
fpr_qda21, tpr_qda21, thres_qda21 = roc_curve(y_test2, qda_model2.predict_proba(X_test2)[:,1])
auc_qda21 = roc_auc_score(y_test2, qda_model2.predict_proba(X_test2)[:,1])
###ROC for boosting
fpr_boo21, tpr_boo21, thres_boo21 = roc_curve(y_test2, AdaBoost_model2.predict_proba(X_test2)[:,1])
auc_boo21 = roc_auc_score(y_test2, AdaBoost_model2.predict_proba(X_test2)[:,1])
###ROC for random forest
fpr21_best, tpr21_best, thres21_best = roc_curve(y_test2, rf2_best.predict_proba(X_test2)[:,1])
auc21_best = roc_auc_score(y_test2, rf2_best.predict_proba(X_test2)[:,1])
###ROC for logistic regression
fpr_ovr21, tpr_ovr21, thres_ovr21 = roc_curve(y_test2, ovr_model_sparse2.predict_proba(X_test_restrict2)[:,1])
auc_ovr21 = roc_auc_score(y_test2, ovr_model_sparse2.predict_proba(X_test_restrict2)[:,1])

fig, ax = plt.subplots(1,1)
ax.plot(fpr_dt21, tpr_dt21, '-', alpha=0.8, label='Best decision tree' % (auc_dt21 ))
ax.plot(fpr_qda21, tpr_qda21, '-', alpha=0.8, label='QDA' % (auc_qda21 ))
ax.plot(fpr_boo21, tpr_boo21, '-', alpha=0.8, label='Boosting' % (auc_boo21 ))
ax.plot(fpr21_best, tpr21_best, '-', alpha=0.8, label='Best random forest' % (auc21_best))
ax.plot(fpr_ovr21, tpr_ovr21, '-', alpha=0.8, label='Regression' % (auc_ovr21))
ax.plot(fpr_dt22, tpr_dt22, '-', alpha=0.8, label=' 0 classifier' % (auc_dt22))
plt.title("ROC curves for AD and MCI")
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.legend()
plt.show()
```
```python
## comparasion of accuracy of different model
classifiers2=[ovr_model2, rf2_best_model, best_tree_model2, AdaBoost_model2, qda_model2]
names = ['baseline regression','random forest', 'decision tree', 'boosting','qda']
AUC_table2=pd.DataFrame()
j=0
for i in classifiers2:
    AUC_table2.loc[names[j],r'$ AUC $']=roc_auc_score(y_test2, i.predict_proba(X_test2)[:,1])
    j=j+1

AUC_table2.loc['optimized logistic',r'$ AUC $']=accuracy_score(y_test2, ovr_model_sparse2.predict(X_test_restrict2))
AUC_table2
```

<img src="https://yueli1201.github.io/Alzheimer/figures/16.jpeg" alt="16" width="750"/><br/>
<div align="center"><font size="2"><b>Fig 16. ROC for different models for AD vs. MCI</b></font></div>

# Conclusion
1. The classification models for disease vs. non-disease performed better than the classification models for AD vs. MCI. This is due to the fact that it is hard to differentiate AD and MCI. An AD patient may be diagnosed as MCI by another doctor. 
2. For disease vs. non-disease classification model, boosting yield the highest accuracy rate as well the highest area under the curve (AUC) score. For AD vs. MCI classification model, random forest had a high accuracy rate and the highest AUC score. 
3. To make it a cost-efficient classification method, we wanted to use as less predictors as possible. From the optimized logistic regression model, we can see a single predictor, CDRSB, is good enough to discriminate patients and non-patients. As for the discrimination of AD and MCI, CDRSB, MMSE are needed. Based on our models, we recommended healthcare providers to consider CDRSB as the first choice to help them diagnose AD.

# Future Work
1. More robust cross-validation.
We split our dataset into train and test data, and cross validated our models within the ADNI dataset. If we have access to other database, we can use external data to test the performance of our model.
2. Use longitudinal data to create risk prediction model. Currently, we only used the baseline data for the classification of disease vs. non-disease and AD vs. MCI. In the future, we can use survival analysis to build up risk prediction model to help people predict their risks of developing AD in advance.