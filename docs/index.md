---
layout: page
title: "Alzheimer’s == True"
description: "Classification of Alzheimer’s Disease with Regression Models and Machine Learning Algorithms"
header-img: "img/home-bg.jpg"
---

# Overview

For our final project, we examined public available data from Alzheimer’s Disease Neuroimaging Initiative (ADNI), Our goals were (1) to create data-driven classification models to distinguish non-patients and patients (Alzheimer’s disease patients and mild cognitive impairment patients); (2) to build up classification models to distinguish Alzheimer’s disease patients and mild cognitive patients. Our criteria for model building is to create cost-efficient, high accuracy model with as less information as possible.

# Motivation

Alzheimer’s disease (AD) is the most common type of dementia and is a complicated multifactorial neurodegenerative disorder. AD is an irreversible process that typically begins after 60 and there is no effective treatment for it right now. Previous studies showed there are around 40 million patients suffering from AD globally. Mild cognitive impairment (MCI) is a transitional phase between healthy aging and AD, which indicates cognitive defects but can strongly affect the patient’s quality of life. The accurate diagnosis of AD, especially for its early stage (MCI) is very important for patients to get early treatment, which can mitigate the progression of AD. Distinguishing disease (AD and MCI) and non-disease is also crutial in terms of quality of life and prevention of disease. Previous studies have found that MRI, PET imaging, and biomarkers, etc are sensitive in diagnosing AD and MCI. But it could be expensive to get all these tests and the results are unintuitive for the general population. Some cognitive tests are easy to conduct while they are not sufficient for AD diagnosis. Thus, using proper predictors to classify diagnosis results between disease and non-disease and disease status between people with AD and MCI are of high importance. To conduct the classification in a cost-efficient manner, we proposed aims as below:

### Aim 1: 
Focus on both the classification between disease (AD and MCI combined) and non-disease and between AD and MCI. We propose a classification model using demographic information, neurological exams, screening labs, vital signs, cognitive assessments, biospecimen collections, medication, diagnosis summary, lumbar puncture, genetic, MRI, PET image, and biospecimen data, etc. Logistic regression and machine learning classification algorithms (boosting, random forest, etc.) will be compared to find the optimal model. 

### Aim 2: 
Find the most cost-efficient way with least amount of covariates to classify people’s disease status. Easiness of getting the information about certain covariate will be considered during the selection of covariates, as we want to maximize the cost-effectiveness. 

# Challenges

When processing the ADNI data, there are some challenges that we need to address to successfully create the models we want.
The ADNI database is a combination of four different cohorts. They followed different protocols and the information collected from each cohort is not the same. Therefore, there are some structurally missing data in this database. Can we combine them together? Can we use data from one cohort to impute missing data in the other. According to the previous publication, the data is performing well when combined. So, we assumed the data are missing at random and used fancyimpute package to impute the missing data.

There are misclassification cases in ADNI database. When examining this longitudinal database, we found that some Alzheimer’s disease patients turn into non-patient status. This violated our current understanding of Alzheimer’s disease, which is an incurable disease. After reviewing currently literatures, we found that currently, there is no clear definition of the difference between Alzheimer’s disease and mild cognitive impairment. A patient who is diagnosed as AD may later be diagnosed as MCI later by another doctor. As we decided to only use the baseline information to build up our classification model, the later inconsistent data will not influence our model building. However, the misclassification within the ADNI database is a challenge we can’t get over.

Although a lot of variables are collected in ADNI database. This data is high singular, which causes a very low performance, when we first created a logistic regression model with full set of covariate, the accuracy rate was very low at around 70%. We used forward selection to select predictors to avoid the problem caused by the high singularity in this database.

