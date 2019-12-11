---
layout: page
title: "Models"
description: "Classification Models for Crime Types"
header-img: "img/home-bg.jpg"
---

# Contents

<a href="#1."> Models to Predict Crime Types in Boston Area</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.1">1. Baseline (Multiple Logistic Regression model)</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.2">2. Optimized Logistic Regression Model</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.3">3. Neural Network</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.4">4. Decision Tree</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.5">5. Random Forest</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.6">6. Boosting</a><br/>
<br>
<br>

<a name="1."> </a>

## Prediction of Crime Types in Boston Area

For modelling purpose, our data was splitted into train(80%) and test(20%) sets, stratified by crime types. We included here six models, parametric and non-parametric, to address the task of multi-class classification of six different types of criminal misconducts. The models are basic logistic regression, logistic regression optimized by Extra Tree Classifier for feature selections, neural network, decision tree, random forest and boosting.
<a name="1.1"> </a>

## 1. Baseline (Multiple Logistic Regression Model)
The baseline model was fitted using selected variables based on our findings in EDA. The predictors displaying strong correlation with the other predictors (gross area, total assessed building value, lights within 100 meters of the crime scene and median house income) were dropped to prevent multicollinearity. We fitted the model without interaction and polynomial terms. The accuracy scores for the training set and test set are 0.334 and 0.331, respectively.

```python
#Baseline Logistic Regression Model

#Standardize

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_train_scaled[continuous] = scaler.fit_transform(X_train_scaled[continuous])

X_test_scaled = X_test.copy()
X_test_scaled[continuous] = scaler.transform(X_test_scaled[continuous])

dummies = ['HOUR', 'MONTH', 'DAY_OF_WEEK_NUM']

#Turn hour, month and day of week into dummy variables

X_train_dum = pd.get_dummies(X_train_scaled.loc[:,pred_col_log], columns=dummies)
X_test_dum = pd.get_dummies(X_test_scaled.loc[:,pred_col_log], columns=dummies)

model_logcv = LogisticRegressionCV(multi_class='ovr', cv=5).fit(X_train_dum, y_train)

print('The train set accuracy score of the basic logistic model is :', accuracy_score(y_train.values, model_logcv.predict(X_train_dum)))

print('The test set accuracy score of the basic logistic model is :', accuracy_score(y_test.values, model_logcv.predict(X_test_dum)))
```
<a name="1.2"> </a>

### 1.2 Optimized Logistic Regression Model

Besides manual selection of explanatory variables, we conducted feature selection by Extra Tree Classifier, in which feature was assessed by the Gini Importance. The most important features are the ones that lead to the highest averaged decrease in node impurity. After applying the classifier, the remaining variables are `YR_BUILT_m`, `Dist_to_Nearest_Light`, `Lights_within_50m`, `Lights_within_100m`, `HOUR_21`, `HOUR_22`, `MONTH_6`, `MONTH_7`, `MONTH_8`, `MONTH_9`, `DAY_OF_WEEK_NUM_1.0`, `DAY_OF_WEEK_NUM_2.0`, `DAY_OF_WEEK_NUM_3.0`, `DAY_OF_WEEK_NUM_4.0`, `DAY_OF_WEEK_NUM_5.0`, `DAY_OF_WEEK_NUM_6.0`. The optimized logistic regression model using these features generated an accuracy score of 0.313 for training set and 0.313 for test set.

```python
# Variables to be be checked

pred_col_names = ['SHOOTING_DUMMY', 'HOUR', 'MONTH', 'DAY_OF_WEEK_NUM',
                  'AV_BLDG', 'AV_TOTAL', 'LAND_SF', 'GROSS_TAX', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS',
                  'PTYPE_A', 'PTYPE_C', 'PTYPE_EO', 'PTYPE_EP', 'PTYPE_I', 'PTYPE_MU', 'PTYPE_R', 'YR_BUILT_m', 'YR_REMOD_m',
                  'Population Density', 'Young_prop', 'Median household income',
                  'Dist_to_Nearest_Light', 'Lights_within_50m', 'Lights_within_100m']

X_train_log2 = pd.get_dummies(X_train_scaled.loc[:,pred_col_names], columns=dummies,drop_first=True)
X_test_log2 = pd.get_dummies(X_test_scaled.loc[:,pred_col_names], columns=dummies,drop_first=True)
```
```python
#Fit the training data into the classifier

estimator = ExtraTreesClassifier(n_estimators = 10)
featureSelection = SelectFromModel(estimator)
featureSelection.fit(X_train_log2, y_train)

#Select the important features

selectedFeatures = featureSelection.transform(X_train_log2)
Selected_var = X_train_log2.columns[featureSelection.get_support()]

#Fit the logistic model with the selected variables

ovr=LogisticRegressionCV(multi_class = 'ovr', cv=10, max_iter=1000)
ovr_model=LogisticRegressionCV(multi_class = 'ovr', cv=5).fit(X_train_log2.loc[:,list(Selected_var)],y_train)
print('The accuracy in training dataset is '+"{}".format(ovr_model.score(X_train_log2.loc[:,list(Selected_var)], y_train)))
print('The accuracy in testing dataset is '+"{}".format(ovr_model.score(X_test_log2.loc[:,list(Selected_var)], y_test)))
```
<a name="1.3"> </a>

### 1.3 Neural Network

Our neural network is consisted of one input layer, three hidden layers and one output layer. The number of nodes per layer is 256, except for the output layer in which 6 nodes were constructed. ReLu was taken as the activation function, and Softmax was used for the output unit since we have a multi-class outcome variable.  Dropout was used in all layers to prevent overfitting, with the fraction of dropout set to be 0.2. We used Adam as the optimizer, and tuned the learning rate to 0.0008 to ensure the model learns across the epochs. Sparse categorical cross-entropy was applied as loss function, with 1000 as the number of epochs, 0.3 as validation set size and 64 as batch size. The final train set accuracy is 0.500, while the final validation set accuracy is 0.485. The test set accuracy using the predicted classes from the neural network is 0.484.


```python
#Neural Network

nn_model = models.Sequential(
  [layers.Dense(256, activation='relu', input_shape=(X_train_scaled.loc[:,pred_col_names].shape[1],)),
   layers.Dropout(0.2),
   layers.Dense(256, activation='relu'),
   layers.Dropout(0.2),
   layers.Dense(256, activation='relu'),
   layers.Dropout(0.2),
   layers.Dense(256, activation='relu'),

   layers.Dropout(0.2),

  layers.Dense(6, activation='softmax')])


adam = optimizers.Adam(lr=0.0008)
nn_model.compile(loss='sparse_categorical_crossentropy',
               optimizer= adam,
               metrics=['accuracy'])

hist = nn_model.fit(X_train_scaled.loc[:,pred_col_names], y_train, shuffle=True, batch_size=64, epochs=1000, validation_split = 0.2,
                    verbose=1)

#Test set accuracy

accuracy_score(y_test, nn_model.predict_classes(X_test_scaled.loc[:,pred_col_names]))
```
```python
#Plot
val_acc = hist.history["val_accuracy"]
acc = hist.history["accuracy"]

plt.figure(figsize=(15,6))

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.legend()

plt.show()
```
<img src="https://yenanxu.github.io/Predicting_Crime/figures/nn_model.png" alt="4" width="750"/>
<div align="center"><font size="2"><b>Fig 6. Accuracies across the Epochs in Neural Network</b></font></div>

<a name="1.4"> </a>

### 1.4 Decision tree

Individual simple decision trees were constructed using various depths in order to find out an optimized depth. The explanatory variables that are strongly correlated were not dropped from the dataset for modelling as decision trees are immune to multicollinearity. The cross-validation means and standard deviations were then plotted along with train set accuracy for each depth. According to the graph, the tree achieves the highest cross-validation score at depth=46 with no sign of overfitting. We therefore chose depth=46 for our final simple decision tree model. The final train set accuracy is 0.959 and test set accuracy is 0.712.

```python
#Decision Tree

#Function for finding the best depth and fit the tree with it
def Find_BestTree(X_train, y_train, depths):
    cvmeans = []
    cvstds = []
    train_scores = []
    for i in depths:
        tree = DecisionTreeClassifier(max_depth=i)
        model_tree = tree.fit(X_train, y_train)

        score = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=5)
        train_score = accuracy_score(y_train, model_tree.predict(X_train))        
        cvmeans.append(score.mean())
        cvstds.append(score.std())
        train_scores.append(train_score)

    best_depth = depths[cvmeans.index(sorted(cvmeans, reverse=True)[0])]

    # Fit a tree with the best depth
    tree_best = DecisionTreeClassifier(max_depth=best_depth)
    tree_best.fit(X_train, y_train)

    return best_depth, tree_best, cvmeans, cvstds, train_scores

depths = list(range(1, 61))
best_depth, model_BestTree, cvmeans, cvstds, train_scores= Find_BestTree(X_train.loc[:,pred_col_names], y_train, depths)

print('The accuracy score in the train set of the decision tree with the best depth={} is: {}'.format(best_depth, accuracy_score(y_train.values, model_BestTree.predict(X_train.loc[:,pred_col_names]))))

print('The accuracy score in the test set of the decision tree with the best depth={} is: {}'.format(best_depth, accuracy_score(y_test.values, model_BestTree.predict(X_test.loc[:,pred_col_names]))))
```

```python
# Plotting means and standard deviations for different depths
plt.figure(figsize=(10, 6))
plt.fill_between(depths, np.array(cvmeans) + np.array(cvstds), np.array(cvmeans)- np.array(cvstds), alpha=0.5)
plt.ylabel("Cross Validation Accuracy")
plt.xlabel("Maximum Depth")
plt.plot(depths, train_scores, 'b-', marker='o', label = "Train set performance")
plt.plot(depths, cvmeans, 'r-', marker='o', label = "Cross-validation performance")
plt.legend()
plt.show()
```

<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/decision_tree.png" alt="4" width="750"/><br/></div>
<div align="center"><font size="2"><b>Fig 7. Change in Accuracies with Various Depths of Simple Decision Tree</b></font></div>

<a name="1.5"> </a>

### 1.5 Random Forest

For random forest model, we set the depth to be the best depth found in simple decision trees. By varying the number of trees between 400 to 480, we found that the number of trees giving rise to the highest test set accuracy to be 420. The train set accuracy and test set accuracy under this optimized setting are 0.959 and 0.732.


```python
#Random Forest

tree_depth = best_depth

n_trees = np.arange(400,500,20)
rf_train_score = []
rf_test_score = []
for n in n_trees:
    rf_model = RandomForestClassifier(n_estimators=n, max_depth=tree_depth, max_features = 'auto')
    rf_model.fit(X_train.loc[:,pred_col_names], y_train)
    train_sc = accuracy_score(y_train.values, rf_model.predict(X_train.loc[:,pred_col_names]))
    test_sc = accuracy_score(y_test.values, rf_model.predict(X_test.loc[:,pred_col_names]))
    rf_train_score.append(train_sc)
    rf_test_score.append(test_sc)
    print("Finished n={}".format(n))

print('With number of trees = {} and maximum depth = {}, the optimized test set accuracy score reaches {}'.format(n_trees[rf_test_score.index(max(rf_test_score))], best_depth, max(rf_test_score)))
```


<a name="1.6"> </a>

### 1.6 Boosting

After tuning the parameters, we set the maximum depth to be 15, learning rate to be 0.1 and number of iterations to be 100 for the boosting model. The staged accuracy scores across the iterations for both train and test sets were plotted. From the graph, we might observe that the accuracy score of the test set keeps rising, while the train set accuracy score maintains high, suggesting that the model is not overfitted. The final train set accuracy was 0.956 while the final test set accuracy was 0.736.

```python
#Boosting

boost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15),
                                 n_estimators=100,
                                 learning_rate=0.1)
boost_model.fit(X_train.loc[:,pred_col_names], y_train)

staged_boosting_training = list(boost_model.staged_score(X_train.loc[:,pred_col_names], y_train))
staged_boosting_test = list(boost_model.staged_score(X_test.loc[:,pred_col_names], y_test))

plt.figure(figsize=(10, 6))

#Plot
plt.ylabel("Accuracy Score")
plt.xlabel("Number of Iterations")
plt.plot(range(1,101),
         staged_boosting_training,
         'b-', label = "Train set performance")
plt.plot(range(1,101),
         staged_boosting_test,
         'r-', label = "Test set performance")
plt.legend()
plt.show()

```
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/boosting.png" alt="4" width="750"/><br/></div>
<div align="center"><font size="2"><b>Fig 8. Variation of Accuracy Score across 100 Iterations in Boosting Model</b></font></div>
