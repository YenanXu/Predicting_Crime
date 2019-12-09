---
layout: page
title: "Models"
description: "Classification models for disease vs. non-disease and AD vs. MCI"
header-img: "img/home-bg.jpg"
---

# Contents

<a href="#1.">1. Models to distinguish Disease (MCI and AD) vs. Non-diseases participants</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.1">1.1 Baseline (Simple logistic regression model)</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.2">1.2 Optimized logistic regression model</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.3">1.3 Random forest</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.4">1.4 Decision tree</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.5">1.5 Boosting</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.6">1.6 Quadratic discriminant analysis (QDA)</a><br/>
<a href="#2.">2. Models to distinguish MCI vs. AD participants</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.1">2.1 Baseline (Simple logistic regression model)</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.2">2.2 Optimized logistic regression model</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.3">2.3 Random forest</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.4">2.4 Decision tree</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.5">2.5 Boosting</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.6">2.6 Quadratic discriminant analysis (QDA)</a><br/>
<br>
<br>

<a name="1."> </a>

## 1. Disease (MCI and AD) vs. Non-diseases

On the basis of a criterion for the quality of life and intervention decision, distinguishing disease from non-disease is of great importance. Thus, we provided 6 ways to conduct the classification including logistic regression, random forest, decision tree, boosting, QDA, and random forest models.
<a name="1.1"> </a>

### 1.1 Baseline (Simple logistic regression model)
We fitted the logistic regression model with split train and test dataset. With all covariates, we got an accuracy score of 0.733 for the training set and 0.723 for the test set.

```python
#LogisticRegression with all covariates
ovr=LogisticRegressionCV(multi_class = 'ovr', cv=5)
ovr_model=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train1,y_train1)
print('The accuracy in training dataset is '+"{}".format(ovr_model.score(X_train1, y_train1)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model.score(X_test1, y_test1)))
```
<a name="1.2"> </a>

### 1.2 Optimized logistic regression model

We conducted forward selection to select the most valuable predictors. As the graph has shown, the graph described the accuracy of the models that we get through forward selection. At each point, the model’s accuracy was drawn from variables from the beginning until that point in the test set. The graph described the accuracy of the models that we get through forward selection. From the output, we chose only CDRSB as our predictor in the model as it will generate an accuracy score of 0.978 for training set and 0.963 for test set.

```python
# forward selection for logistic model
# input 'model' is a pre-fit model like LogisticRegressionCV(multi_class = 'ovr', cv = 5)
def forward_selection(model, X_train, X_test, y_train, y_test):
    var_selected = []
    accuracy = []
    for i in range(len(list(X_train))):
        X_train_rest = X_train.drop(var_selected, axis = 1)
        accuracy_temp = []
        for j in range(len(list(X_train_rest))):
            var_now = var_selected.copy()
            var_now.append(list(X_train_rest)[j])
            model_fitted = model.fit(X_train[var_now], y_train)
            y_test_pred = model_fitted.predict(X_test[var_now])
            accuracy_temp.append(accuracy_score(y_test, y_test_pred))
        var_selected.append(list(X_train_rest)[accuracy_temp.index(max(accuracy_temp))])
        accuracy.append(max(accuracy_temp))
    return(var_selected, accuracy)
```
```python
model = LogisticRegressionCV(multi_class = 'ovr', cv = 5)
var_selected1, accuracy1 = forward_selection(model, X_train1, X_test1, y_train1, y_test1)
plt.figure(figsize=(15,5))
plt.plot(var_selected, accuracy)
plt.xticks(rotation=90)
plt.ylabel('accuracy on testing dataset')
plt.title('Accuracy for forward selection on prediction of controls and diseases')
plt.show()
```

<img src="https://yueli1201.github.io/Alzheimer/figures/5.jpeg" alt="6" width="750"/>
<div align="center"><font size="2"><b>Fig 5. forward selection for logistic regression</b></font></div>

```python
## best logistic
X_train_restrict=np.array(X_train1['CDRSB']).reshape(-1,1)
X_test_restrict=np.array(X_test1['CDRSB']).reshape(-1,1)
ovr_model_sparse=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train_restrict, y_train1)
print('The accuracy in training dataset is '+"{}".format(ovr_model_sparse.score(X_train_restrict, y_train1)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model_sparse.score(X_test_restrict, y_test1)))
```
<a name="1.3"> </a>

### 1.3 Random forest

Without tuning the parameters, random forest generates an accuracy score of 0.958 and AUC of 0.972, which indicated good prediction and classification capability. After tuning the parameters, we chose depth = 50 and number of estimaters = 100 and got an accuracy score of 0.960 and AUC of 0.984.

```python
###Random forest baseline model
rf1 = RandomForestClassifier(random_state=100)
display(rf1.fit(X_train1, y_train1))
```
```python
##Tuning parameters for random forest
depths = [1,5,10,50]
num = [10, 20, 50,100]
tune = pd.DataFrame(np.zeros((len(depths)*len(num), 3)))

i = 0
kf = KFold(n_splits=5)

for d in depths:
    for n in num:
        print("depth:{}, n:{}".format(d,n))
        tune_cv = []
        for train, val in kf.split(X_train1):
            train_X, train_y, val_X, val_y = X_train1.iloc[train,:], pd.DataFrame(y_train1).iloc[train,:], X_train1.iloc[val,:], pd.DataFrame(y_train1).iloc[val,:]
            rf_model1 = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=100)
            rf_model1.fit(train_X, train_y.values.ravel())
            tune_cv.append(roc_auc_score(val_y, rf_model1.predict_proba(val_X)[:,1]))

        tune.iloc[i,:] = [d, n, np.mean(tune_cv)]
        i+=1
        
tune.columns = ['max_depth','max_num','AUC']
tune.sort_values(by = 'AUC', ascending=False)
```
```python
# choose the best rf model based on above
rf1_best = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=100)
rf1_best_model=rf1_best.fit(X_train1, y_train1)

y_pred_rf1 = rf1.predict(X_test1)
y_pred_rf1_best = rf1_best_model.predict(X_test1)
accuracy_rf1 = accuracy_score(y_pred_rf1, y_test1)
accuracy_rf1_best = accuracy_score(y_pred_rf1_best, y_test1)
print('Accuracy score for random random forest model is', accuracy_rf1)
print('Accuracy score for best random forest model is', accuracy_rf1_best)
```
```python
### ROC for baseline random forest
fpr11, tpr11, thres11 = roc_curve(y_test1, rf1.predict_proba(X_test1)[:,1])
### ROC for bset random forest
fpr11_best, tpr11_best, thres11_best = roc_curve(y_test1, rf1_best.predict_proba(X_test1)[:,1])
fpr12_best, tpr12_best, thres12_best = roc_curve(y_test1, np.zeros((len(y_test1), 1)))
auc11_best = roc_auc_score(y_test1, rf1_best.predict_proba(X_test1)[:,1])
auc12_best = roc_auc_score(y_test1, np.zeros((len(y_test1), 1)))

fig, ax = plt.subplots(1,1)
ax.plot(fpr11, tpr11, '-', alpha=0.8, label='Baseline random forest' % (auc11))
ax.plot(fpr11_best, tpr11_best, '-', alpha=0.8, label='Best random forest' % (auc11_best))
ax.plot(fpr12_best, tpr12_best, '-', alpha=0.8, label=' 0 classifier' % (auc12_best))
plt.title("ROC curves for random forest model (Disaese vs. Non-disease)")
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.legend()
plt.show()
```

<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/6.png" alt="6" width="450"/><br/></div>
<div align="center"><font size="2"><b>Fig 6. ROC curves for random forest model (Disaese vs. Non-disease)</b></font></div>

On the basis of the selected model, we counted the times that each feature was used at the top node and concluded EcogSPMem, FAQ, and CDRSB, which are a bit different from the output of forward selection generated by regression analysis.

```python
y_train1=pd.DataFrame(y_train1)
y_train1['tmp'] = 1
X_train1['tmp'] = 1
y_train1.columns=['DX','tmp']
train1 = pd.merge(X_train1, y_train1, on=['tmp'])
train1 = train1.drop('tmp', axis=1)
X_train1 = X_train1.drop('tmp',axis=1)
y_train1 = y_train1.drop('tmp',axis=1)

columns=X_train1.columns
#random forest
nrf = [0]*X_train1.shape[1]
for i in range(50):
    for j in range(X_train.shape[1]):
        if rf1_best.estimators_[i].tree_.feature[0] == j:
            nrf[j] = nrf[j]+1

plt.figure(figsize = (20,6))
plt.bar(columns, nrf, width = 0.4, label = 'random forest')
plt.xlabel('$features$',fontsize=16)
plt.ylabel('$counts$',fontsize=16)
plt.title('Times that each feature is used at the top node',fontsize=16)
plt.legend(loc = 'best',fontsize=16)
plt.xticks(rotation=90)
plt.show()
```

<img src="https://yueli1201.github.io/Alzheimer/figures/7.jpeg" alt="7" width="750"/>
<div align="center"><font size="2"><b>Fig 7. Times that each feature is used at the top node</b></font></div>

<a name="1.4"> </a>

### 1.4 Decision tree

According to the graph, the tree achieved the highest cross-validation score at depth=2, therefore, we chose depth=2 for our final decision tree model. Our model yielded 0.978 and 0.963 accuracy rate on train and test dataset respectively.

```python
#decision tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
# find the best depth
depths = list(range(1, 21))
train_scores = []
cvmeans = []
cvstds = []
cv_scores = []
for depth in depths:
    dtree = DecisionTreeClassifier(max_depth=depth)
    # Perform 5-fold cross validation and store results
    train_scores.append(dtree.fit(X_train1, y_train1).score(X_train1, y_train1))
    scores = cross_val_score(estimator=dtree, X=X_train1, y=y_train1, cv=5)
    cvmeans.append(scores.mean())
    cvstds.append(scores.std())

cvmeans = np.array(cvmeans)
cvstds = np.array(cvstds)
# plot means and shade the 2 SD interval
plt.plot(depths, cvmeans, '*-', label="Mean CV")
plt.fill_between(depths, cvmeans - 2*cvstds, cvmeans + 2*cvstds, alpha=0.3)
ylim = plt.ylim()
plt.plot(depths, train_scores, '-+', label="Train")
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Max Depth")
plt.xticks(depths)
plt.show()

##fit best tree
best_tree = DecisionTreeClassifier(max_depth=2)
best_tree_model = best_tree.fit(X_train1, y_train1)
Training_accuracy_best_depth = accuracy_score(y_train1, best_tree.predict(X_train1))
Test_accuracy_best_depth = accuracy_score(y_test1, best_tree.predict(X_test1))
print('The accuracy score with best depth for test set is', Test_accuracy_best_depth)
print('The accuracy score with best depth for train set is', Training_accuracy_best_depth)
```

<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/8.png" alt="8" width="450"/><br/></div>
<div align="center"><font size="2"><b>Fig 8. Accuracy rate vs. the depth of the tree</b></font></div>

<a name="1.5"> </a>

### 1.5 Boosting

Both training set and test set generate high accuracy score with iteration from 0-800. With the depth=2, as iteration increases, the accuracy score increased for training set and reached around perfect score when iteration reached around 150 and fluctuated for the test set around 0.95 to 0.97. The data suggest the boosting model fits the data pretty well with little overfitting.

```python
#boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

AdaBoost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                           n_estimators=800,learning_rate=0.05)
AdaBoost_model=AdaBoost.fit(X_train1,y_train1)

#Plot Iteration based score
train_scores = list(AdaBoost.staged_score(X_train1,y_train1))
test_scores = list(AdaBoost.staged_score(X_test1, y_test1))

plt.plot(train_scores,label='train')
plt.plot(test_scores,label='test')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("Variation of Accuracy with Iterations")
plt.legend()

print(AdaBoost.score(X_train1, y_train1))
print(AdaBoost.score(X_test1, y_test1))
```

<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/9.png" alt="9" width="450"/><br/></div>
<div align="center"><font size="2"><b>Fig 9. Variation of Accuracy with Iterations</b></font></div>

<a name="1.6"> </a>

### 1.6 Quadratic discriminant analysis (QDA)

Using QDA to fit the dataset, we allowed covariances of multivariate Normal(MVN)  distribution within classes to be different, which avoided the disadvantages of using Linear Discriminant Analysis (LDA) when the covariances are not the same in the group. Our model yielded pretty high accuracy scores for both of the training set and test set, which are 0.960, 0.932, respectively.

```python
## qda
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda_model = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train1,y_train1)
#np.mean(cross_val_score(qda, X_train1, y_train1, cv=5))
#np.mean(cross_val_score(qda, X_test1, y_test1, cv=5))

print('Accuracy score using QDA method for training set is', accuracy_score(y_train1, qda_model.predict(X_train1)))
print('Accuracy score using QDA method for test set is', accuracy_score(y_test1, qda_model.predict(X_test1)))
```
<a name="2."> </a>

## 2. MCI vs. AD

Since MCI is a transitional phase from non-cognitive impairment to AD, the accurate diagnosis of AD and MCI is crucial for early intervention of cognitive impairment and for mitigating the progression of AD. Hence, we introduced regression and regression models to explore the best approach to classify AD and MCI.
<a name="2.1"> </a>

### 2.1 Baseline (Simple logistic regression model)

With all covariates, we got an accuracy score of 0.771 for the training set and 0.796 for the test set.

```python
#LogisticRegression
ovr=LogisticRegressionCV(multi_class = 'ovr', cv=5)
ovr_model2=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train2,y_train2)
print('The accuracy in training dataset is '+"{}".format(ovr_model2.score(X_train2, y_train2)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model2.score(X_test2, y_test2)))
```
<a name="2.2"> </a>

### 2.2 Optimized logistic regression model

Forward selection was used again to select predictors. The graph below described the accuracy of the models that we get through forward selection. With CDRSB, the model’s accuracy achieved 0.9. By adding MMSE, EcogSPMem, the accuracy of the model get to 0.95. After that, adding more predictors makes the accuracy rate fluctuating within 0.94 to 0.96. And after we started to introduce predictors, such as FAQ, Entorhinal, MidTemp etc., the accuracy of the model started to go down. The accuracy decreased to 0.796 when we include the predictors. This is because that some of our predictors are correlated with others, which caused the calculation matrix irreversible, and therefore the model had a poor performance. Therefore, we chose CDRSB, MMSE and EcogSPMem as predictors in our optimized logistic regression.

```python
##LogisticRegression Best
X_train_restrict2=X_train2[['CDRSB','MMSE','EcogSPMem']]
X_test_restrict2=X_test2[['CDRSB','MMSE','EcogSPMem']]
ovr_model_sparse2=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train_restrict2, y_train2)
print('The accuracy in training dataset is '+"{}".format(ovr_model_sparse2.score(X_train_restrict2, y_train2)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model_sparse2.score(X_test_restrict2, y_test2)))
```

<img src="https://yueli1201.github.io/Alzheimer/figures/10.jpeg" alt="10" width="750"/>
<div align="center"><font size="2"><b>Fig 10. Accuracy for forward selection on prediction of MCI and AD</b></font></div>

<a name="2.3"> </a>

### 2.3 Random forest

Random forest model gave an accuracy score of 0.906 and AUC of 0.965 if we do not tune the parameters, while the tuned model (depth = 5 and number of estimators = 100) gave an accuracy score of 0.922 and AUC of 0.972, which indicates that tuning parameters could enhance the model’s performance.

```python
###Random forest baseline model
rf2 = RandomForestClassifier(random_state=100)
display(rf2.fit(X_train2, y_train2))
```
```python
##Tuning parameters for random forest
depths = [1,5,10,50]
num = [10, 20, 50,100]
tune = pd.DataFrame(np.zeros((len(depths)*len(num), 3)))

i = 0
kf = KFold(n_splits=5)

for d in depths:
    for n in num:
        print("depth:{}, n:{}".format(d,n))
        tune_cv = []
        for train, val in kf.split(X_train2):
            train_X, train_y, val_X, val_y = X_train2.iloc[train,:], pd.DataFrame(y_train2).iloc[train,:], X_train2.iloc[val,:], pd.DataFrame(y_train2).iloc[val,:]
            rf_model2 = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=100)
            rf_model2.fit(train_X, train_y.values.ravel())
            tune_cv.append(roc_auc_score(val_y, rf_model2.predict_proba(val_X)[:,1]))

        tune.iloc[i,:] = [d, n, np.mean(tune_cv)]
        i+=1
        
tune.columns = ['max_depth','max_num','AUC']
tune.sort_values(by = 'AUC', ascending=False)
```
```python
# choose the best rf model based on above
rf2_best = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=100)
rf2_best_model=rf2_best.fit(X_train2, y_train2)

y_pred_rf2 = rf2.predict(X_test2)
y_pred_rf2_best = rf2_best_model.predict(X_test2)
accuracy_rf2 = accuracy_score(y_pred_rf2, y_test2)
accuracy_rf2_best = accuracy_score(y_pred_rf2_best, y_test2)
print('Accuracy score for random random forest model is', accuracy_rf2)
print('Accuracy score for best random forest model is', accuracy_rf2_best)
```
```python
## ROC for baseline random forest
fpr21, tpr21, thres21 = roc_curve(y_test2, rf2.predict_proba(X_test2)[:,1])
auc21 = roc_auc_score(y_test2, rf2.predict_proba(X_test2)[:,1])
fpr21_best, tpr21_best, thres21_best = roc_curve(y_test2, rf2_best.predict_proba(X_test2)[:,1])
fpr22_best, tpr22_best, thres22_best = roc_curve(y_test2, np.zeros((len(y_test2), 1)))
auc21_best = roc_auc_score(y_test2, rf2_best.predict_proba(X_test2)[:,1])
auc22_best = roc_auc_score(y_test2, np.zeros((len(y_test2), 1)))

fig, ax = plt.subplots(1,1)
ax.plot(fpr21, tpr21, '-', alpha=0.8, label='Baseline random forest' % (auc21))
ax.plot(fpr21_best, tpr21_best, '-', alpha=0.8, label='Best random forest' % (auc21_best))
ax.plot(fpr22_best, tpr22_best, '-', alpha=0.8, label=' 0 classifier' % (auc22_best))
plt.title("ROC curves for random forest model (AD vs. MCI)")
plt.xlabel('1-sepecificity')
plt.ylabel('sensitivity')
plt.legend()
plt.show()
```

<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/11.png" alt="11" width="450"/><br/><br/></div>
<div align="center"><font size="2"><b>Fig 11. ROC curves for random forest model (AD vs. MCI)</b></font></div>

We also learned that FAQ, MMSE, CDRSB contributed the most in classifying AD and MCI.

```python
y_train2=pd.DataFrame(y_train2)
y_train2['tmp'] = 1
X_train2['tmp'] = 1
y_train2.columns=['DX','tmp']
train2 = pd.merge(X_train2, y_train2, on=['tmp'])
train2 = train2.drop('tmp', axis=1)
X_train2 = X_train2.drop('tmp',axis=1)
y_train2 = y_train2.drop('tmp',axis=1)

columns=X_train2.columns
#random forest
nrf = [0]*X_train2.shape[1]
for i in range(50):
    for j in range(X_train.shape[1]):
        if rf2_best.estimators_[i].tree_.feature[0] == j:
            nrf[j] = nrf[j]+1

plt.figure(figsize = (20,6))
plt.bar(columns, nrf, width = 0.4, label = 'random forest')
#plt.xticks(columns)
plt.xlabel('$features$',fontsize=16)
plt.ylabel('$counts$',fontsize=16)
plt.title('Times that each feature is used at the top node',fontsize=16)
plt.legend(loc = 'best',fontsize=16)
plt.xticks(rotation=90)
plt.show()
```

<img src="https://yueli1201.github.io/Alzheimer/figures/12.jpeg" alt="12" width="750"/>
<div align="center"><font size="2"><b>Fig 12. Times that each feature is used at the top node</b></font></div>

<a name="2.4"> </a>

### 2.4 Decision tree

The model achieved the highest cross-validation score at depth=2, therefore, we chose depth=2 for our final decision tree model.

```python
#decision tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
# find the best depth
depths = list(range(1, 21))
train_scores = []
cvmeans = []
cvstds = []
cv_scores = []
for depth in depths:
    dtree = DecisionTreeClassifier(max_depth=depth)
    # Perform 5-fold cross validation and store results
    train_scores.append(dtree.fit(X_train2, y_train2).score(X_train2, y_train2))
    scores = cross_val_score(estimator=dtree, X=X_train2, y=y_train2, cv=5)
    cvmeans.append(scores.mean())
    cvstds.append(scores.std())

cvmeans = np.array(cvmeans)
cvstds = np.array(cvstds)
# plot means and shade the 2 SD interval
plt.plot(depths, cvmeans, '*-', label="Mean CV")
plt.fill_between(depths, cvmeans - 2*cvstds, cvmeans + 2*cvstds, alpha=0.3)
ylim = plt.ylim()
plt.plot(depths, train_scores, '-+', label="Train")
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Max Depth")
plt.xticks(depths);
plt.show()

##fit best tree
best_tree2 = DecisionTreeClassifier(max_depth=2)
best_tree_model2 = best_tree2.fit(X_train2, y_train2)
Training_accuracy_best_depth2 = accuracy_score(y_train2, best_tree2.predict(X_train2))
Test_accuracy_best_depth2 = accuracy_score(y_test2, best_tree2.predict(X_test2))
Test_accuracy_best_depth2
```

<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/13.png" alt="13" width="450"/><br/></div>
<div align="center"><font size="2"><b>Fig 13. Accuracy vs. depth of decision tree</b></font></div>

<a name="2.5"> </a>

### 2.5 Boosting

Training set and test set both had pretty high accuracy score with the iteration from 0 to 800. While it kept increasing with the iteration for training set and reached 1 around iteration = 300, the score basically showed a decreasing trend for test set and reached highest when iteration is around 50. The data suggest that overfitting may happen lightly when iteration is really high, but in general, the model did a terrific job since both of the scores are over 0.9.

```python
#boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

AdaBoost2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                           n_estimators=800,learning_rate=0.05)
AdaBoost_model2=AdaBoost2.fit(X_train2,y_train2)

#Plot Iteration based score
train_scores = list(AdaBoost2.staged_score(X_train2,y_train2))
test_scores = list(AdaBoost2.staged_score(X_test2, y_test2))

plt.plot(train_scores,label='train')
plt.plot(test_scores,label='test')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("Variation of Accuracy with Iterations")
plt.legend()

print(AdaBoost2.score(X_train2, y_train2))
print(AdaBoost2.score(X_test2, y_test2))
```

<div align="center"><img src="https://yueli1201.github.io/Alzheimer/figures/14.png" alt="14" width="450"/><br/></div>
<div align="center"><font size="2"><b>Fig 14. Variation of Accuracy with Iterations</b></font></div>

<a name="2.6"> </a>

### 2.6 Quadratic discriminant analysis (QDA)

Using QDA to analyze our model, we got accuracy scores for the training set and test set, which are 0.934, 0.865, respectively.

```python
## qda
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda_model2 = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train2,y_train2)

print('Accuracy score using QDA method for training set is', accuracy_score(y_train2, qda_model2.predict(X_train2)))
print('Accuracy score using QDA method for test set is', accuracy_score(y_test2, qda_model2.predict(X_test2)))
```
