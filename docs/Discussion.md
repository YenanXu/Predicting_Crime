---
layout: page
title: "Discussion"
description: "Real world applications and future direction"
header-img: "img/home-bg.jpg"
---

# Results

## Geographical distribution of crime type and street lights

To understand the geographical night crime pattern in Boston, the incidence of each crime type occured at night time (n=142,478) was plotted on the map. Crime types were color coded (see legend on the right). Crime incidents are most concentrated in Boston Downtown and its sprawling surroundings. All types of crime are homogeneously distributed but Drug Abuse Violation, which occurred most frequently in Downtown and Roxbury. The regions on the map with few crime incidents are parks (e.g. Franklin Park), cemeteries, lakes or ponds.

```python
fig = px.scatter_mapbox(df_crime_output, lat="Lat", lon="Long",
                        color='Crime_Type_Cat', color_discrete_sequence=[px.colors.qualitative.Light24[0],
                                                                         px.colors.qualitative.Light24[1],
                                                                         px.colors.qualitative.Light24[2],
                                                                         px.colors.qualitative.Light24[4],
                                                                         px.colors.qualitative.Light24[5],
                                                                         px.colors.qualitative.Light24[7]],
                        zoom=11, height=500)

fig.update_traces(marker=dict(size=3, opacity=0.8))
fig.update_layout(mapbox_style="open-street-map", dragmode=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show(block=False)
```  

<div align="center"><font size="2"><b>Figure 9. Geographic Distribution of Crime Type</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/Crimetype.png" alt="t1" width="750"/></div>

The relationship between street lighting and night crime incidence occurred in Boston was first explored by examining through eyebulling. Each street light in our dataset (n=74,065) was also plotted on the map. The density of light distribution of a region seems to be associated with the level of prosperity of that region. The street lights are clustered in the center of the city and its density decreases as we move further outside of the urban area. The regions on the map with no street light are parks (e.g. Franklin Park), cemeteries or lakes/ponds.

```python
fig_lights = px.scatter_mapbox(df_lights, lat="Lat", lon="Long",
                        color_discrete_sequence=[px.colors.qualitative.Set1[5]], zoom=11, height=500)

fig_lights.update_traces(marker=dict(size=3, opacity=0.3))
fig_lights.update_layout(mapbox_style="open-street-map", dragmode=False)
fig_lights.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig_lights.show()
```  

<div align="center"><font size="2"><b>Figure 10. Geographic Distribution of Street Lighting</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/Lighting.png" alt="t2" width="750"/></div>

## Comparison of Model performance

Six machine learning algorithms were used to build our prediction model and explore the relationship between predictive features and the outcome, crime type. To have a straightforward comparison of classification accuracy for each model, a summary table was made. Multi-class confusion matrix for each model were also constructed to provide a visualized comparison among models.

<div align="center"><font size="2"><b>Table 3. Summary Table of Model Performance</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/Model_performance.png" alt="t3" width="400"/></div>

Both logistic regression models fail to show robust classification power under this scenario. Since we have a multi-class target variable, we set the ‘multi_class’ option to ‘ovr’ used the one-vs-rest (OvR) scheme. The modified logistic regression used 'softmax' function instead of the 'sigmoid' function. An important issue with softmax function is that it picks up small differences and enhances it out of proportion. It is more likely to bias the classifier towards a particular class, rather than predict six classes as we desired. The baseline model is built based on our EDA results and several varaibles were excluded to prevent colinearity. The optimized logistic regression model used Extra Tree Classfier and selected variables based on Gini importance, and the performance is inferior than our baseline model. According to the confusion matrix, both models predict most observation crime type to be 'Theft', which has the most observation in the train dataset.

```python
#Confusion matrix for logistic models

crime_labels = ['Theft', 'Robbery', 'Assault', 'Vandalism', 'M/V Accident', 'Drug Abuse Violations']
matrix_logcv = confusion_matrix(y_test.values, model_logcv.predict(X_test_dum))
plt.figure(figsize = (12,10))
sns.set(font_scale=1)
sns.heatmap(matrix_logcv,annot=True,cbar=True, xticklabels=crime_labels, yticklabels=crime_labels)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.ylabel('True Value')
plt.xlabel('Predicted Value');

matrix_log2 = confusion_matrix(y_test.values, ovr_model.predict(X_test_log2.loc[:,list(Selected_var)]))
plt.figure(figsize = (12,10))
sns.set(font_scale=1)
sns.heatmap(matrix_log2,annot=True,cbar=True, xticklabels=crime_labels, yticklabels=crime_labels)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
```  

<div align="center"><font size="2"><b>Figure 11. Confusion Matrix of Baseline Logistic Regression Model</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/CM_Log1.png" alt="t4" width="600"/></div>

<div align="center"><font size="2"><b>Figure 12. Confusion Matrix of Optimized Logistic Regression Model</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/CM_Log2.png" alt="t5" width="600"/></div>

Neural network model also failed to provide good accuracy score for both train and test dataset. Multiple Neural Network architectures have been tried to fit the dataset but only one combination of hyperparameters would make the network model to learn and continue to improve train set accuracy. The activation function of initial layers was set to 'relu' since it usually gives the best result. Given that our outcome is multi-classed, last layer was applied with "softmax" activation which outputs an array of six probability scores(summing to 1). Loss function was set to 'sparse_categorical_crossentropy' so we skipped on-hot-coding step and keeps categorical variables as integers. Since our data structure is quite complex with many dimensions, we constructed added multiple layers (width over depth), and set learning rate to a small number so that ensure to learn optimal set of weight, but at a cost of long training process. Due to limited computational power, we only ran 1000 epoches and stlll saw an improved trend of accuracy rate. Therefore, had given better computational power, more epoches could have been ran to improve model performance. From confusion matrix, we could see that the NN model performs the best at predicting Theft, Assualt and M/V accident.

```python
#Confusion matrix for NN model

matrix_NN = confusion_matrix(y_test.values, nn_model.predict_classes(X_test_scaled.loc[:,pred_col_names]))
plt.figure(figsize = (12,10))
sns.set(font_scale=1)
sns.heatmap(matrix_NN,annot=True,cbar=True, xticklabels=crime_labels, yticklabels=crime_labels)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
```  

<div align="center"><font size="2"><b>Figure 13. Confusion matrix of Neural Network Model</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/NN_matrix.png" alt="t6" width="600"/></div>

we also ran several other models- Decision Tree, Random Forests and Boosting. Surprisingly, their results are better than presumably "more advanced" model. Decision tree is often used for classification problems and it could be that these Decision tree based supervised learning models are more suitable models for our dataset. Confusion matrix for these models are present.

```python
#Confusion matrix for NN model
#Decision Tree Confusion Matrix

matrix_deci_tree = confusion_matrix(y_test.values, model_BestTree.predict(X_test.loc[:,pred_col_names]))
plt.figure(figsize = (12,10))
sns.set(font_scale=1)
sns.heatmap(matrix_deci_tree,annot=True,cbar=True, xticklabels=crime_labels, yticklabels=crime_labels)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.ylabel('True Value')
plt.xlabel('Predicted Value')

#Random Forest Confusion Matrix

rf_model = RandomForestClassifier(n_estimators=420, max_depth=tree_depth, max_features = 'auto')
rf_model.fit(X_train.loc[:,pred_col_names], y_train)

matrix_rf = confusion_matrix(y_test.values, rf_model.predict(X_test.loc[:,pred_col_names]))
plt.figure(figsize = (12,10))
sns.set(font_scale=1)
sns.heatmap(matrix_rf,annot=True,cbar=True, xticklabels=crime_labels, yticklabels=crime_labels)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.ylabel('True Value')
plt.xlabel('Predicted Value')

#Boosting Confusion Matrix

matrix_boost = confusion_matrix(y_test.values, boost_model.predict(X_test.loc[:,pred_col_names]))
plt.figure(figsize = (12,10))
sns.set(font_scale=1)
sns.heatmap(matrix_boost,annot=True,cbar=True, xticklabels=crime_labels, yticklabels=crime_labels)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.ylabel('True Value')
plt.xlabel('Predicted Value')
```  

<div align="center"><font size="2"><b>Figure 14. Confusion matrix of Decision Tree, Random Forest and Boosting Models</b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/DT_RF_BT_matrix.png" alt="t7" width="750"/></div>

Boosting decreases bias comparing to the single model. Boosting model learns by fitting the residual of the threes that preceded the current layer and thus tends to improve accuracy with small risk of less coverage. The trend of missclassification was observed- The model is biased towards theft and result in a lot of false positive for theft. Boosting model improves the prediction of theft, but loses prediction power for Robbery and M/V accident.


## Important features
We then looked at relative variable importance in Random Forest and Boosting, two models that give the best prediction accuracy. Clearly, `MONTH`, `DAY_OF_WEEK_NUM` , `Dist_to_Nearest_Light` and `Lights_within _50m` are an important variables for crime type prediction. However the two  models  differ with some other variables. In Random Forest model, 'Lights within 100m' was also picked as an important feature, but it does not contribute at all for the Boosting model. It is probably because `Lights_within _50m` and `Lights_within _100m` are very similar features and Boosting was able to discern it and excluded one variable. Boosting model also uses information from other property value varaibles for prediction, whereas in Random Forest model, these variables are weighted very low. However, looking at EDA analysis, the correlation between lighting and crime type or property value is not obvious, which could explain why our prediction model only reached an accuracy of 0.736.

```python
#Important Features in RF and Boosting models

boost_predictors_imp = pd.DataFrame({'predictor': X_train.loc[:,pred_col_names].columns, 'relative importance':boost_model.feature_importances_}).sort_values(by='relative importance', ascending=False)
rf_predictors_imp = pd.DataFrame({'predictor': X_train.loc[:,pred_col_names].columns, 'relative importance':rf_model.feature_importances_}).sort_values(by='relative importance', ascending=False)
```

```python
#Plot

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(45,20))
fig.subplots_adjust(wspace = 0.4)
y_pos = np.arange(len(boost_predictors_imp ))
ax[0].barh(y_pos, rf_predictors_imp['relative importance'], align='center')
ax[0].xaxis.set_tick_params(labelsize=25)
ax[0].set_yticks(y_pos)
ax[0].set_yticklabels(rf_predictors_imp['predictor'],fontsize=27)
ax[0].invert_yaxis()  
ax[0].set_xlabel('Relative Importance',fontsize=27)
ax[0].set_title('Random Forest',fontsize=30, fontweight="bold")

ax[1].barh(y_pos, boost_predictors_imp['relative importance'], align='center')
ax[1].xaxis.set_tick_params(labelsize=25)
ax[1].set_yticks(y_pos)
ax[1].set_yticklabels(boost_predictors_imp['predictor'],fontsize=27)
ax[1].invert_yaxis()  
ax[1].set_xlabel('Relative Importance',fontsize=27)
ax[1].set_title('Boosting',fontsize=30, fontweight="bold")

plt.show()
```

<div align="center"><font size="2"><b>Figure 15. Important Features </b></font></div>
<div align="center"><img src="https://yenanxu.github.io/Predicting_Crime/figures/feature_importance.png" alt="t8" width="750"/></div>

# Conclusion

We established a crime type prediction model with a prediction accuracy of 0.736, using boosting model. It is surprising that a presumably more 'advanced' model failed to generate high accuracy model. An important takeaway is that there is no general model that outperforms the others in any context, rather it is important to choose suitable model based on data structure.

There are two main real-world implications of our work:
1. Boston Police Department (BPD) has faced the issue of high incidences of crime in the city. Most of them occur at night. Our analysis has show that street lighting distribution has been clearly associated with the rate of crimes in non-urban area. We are aware that BPD had implemented a Safe Street program in 2012. In the future crime control program, BPD could deploy more resources into adding and fixing lighting, especially in low lighting density areas that shown in our map (Figure 10), such as Roxbury and Mission Hill.  
2. Our prediction model aims to forecast types of crime misconduct at specified time and location, which would ultimately help the police better allocate manpower and resources with prepared equipments and mindset.

There are also some limitation of our work. Firstly, Street name variable currently is not incorporated in the model because there are 2000 more variables and some streets appeared in very low frequency. It is plausible to one-hot-coding method but it probabily will add too much noise to the model. Secondly, the target variable is a little imbalanced and it impacts model training. Thirdly, even though there are 50+ features in our dataset, the information is not rich enough to produce excellent prediction of crime type.

# Future Work
1. To make the predictive model more handy to use, it is better to establish a compatible model that predicts crime incidence occurence at certain location and time.
2. Embedding layers could be used to incorporate the categorical variable street name into the model without adding too much nois.
3. To improve model's prediction accuracy of the model, it will be useful to get access to more relevant situational and environmental variables, such as alcohol outlet, sidewalks, drug arrests incidents and graffiti and trash locations.
