---
layout: page
title: "Literature Review"
description: "Literature review and related work"
header-img: "img/home-bg.jpg"
---

## 1. What is Alzheimer’s disease?
Alzheimer's is the most common cause of dementia, a general term for memory loss and other cognitive abilities serious enough to interfere with daily life. Alzheimer's disease accounts for 60 percent to 80 percent of dementia cases. The cause of Alzheimer's disease is poorly understood. About 70% of the risk is believed to be genetic with many genes usually involved. Other risk factors include a history of head injuries, depression or hypertension. The disease process is associated with plaques and tangles in the brain.

[Click to view source.](https://en.wikipedia.org/wiki/Alzheimer%27s_disease)

## 2. How to diagnose Alzheimer’s disease?
The doctor will conduct multiple tests to give a diagnosis on Alzheimer’s disease.
(1) mental status testing. Mental status test will conducted to test patient’s thinking (cognitive) and memory skills.
(2) Neuropsychological tests. A specialist trained in brian conditions and mental health conditions will assess the patient’s memory and thinking abilities.
(3) Interviews with friends and family. Doctors may ask patient’s family members and friends about the patient’s behavior.
(4) Brain-imaging tests. Alzheimer’s dementia results from the progressive loss of brain cells. This degeneration may show up in variety ways of brian scans, such as Magnetic resonance imaging (MRI), Computerized tomography (CT), Positron emission tomography (PET)
Beyond these process, the doctor also need to evaluate the patient’s past medical history as well as evaluate other physical status to check that he/her doesn’t have other health conditions, such as past strokes, Parkinson’s disease, depression or other medical conditions, that could be causing or contributing to the symptoms. 

[Click to view source.](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/in-depth/alzheimers/art-20048075)

## 3. How to differentiate Alzheimer’s disease and mild cognitive impairment?
MCI is distinguished from dementia by the severity of the difficulty with everyday activities and by the presence or absence of dementia-related symptoms. Someone who has dementia will have obvious problems with activities like keeping track of medications or driving. Additionally, individuals with MCI usually do not display symptoms that are seen in dementia, such as impaired judgment or trouble with reasoning.

Sometimes MCI is called “early Alzheimer’s disease,” although MCI does not always progress to Alzheimer’s. There is some disagreement amongst physicians and researchers about when to give a patient a diagnosis of MCI versus an Alzheimer’s disease diagnosis. The symptoms of the two can be very similar, and it is possible that the same person could get an MCI diagnosis from one doctor and an early-stage Alzheimer’s disease diagnosis from a different doctor. There is also some disagreement about when a person who was originally diagnosed with MCI, and has worsening symptoms, should be diagnosed with Alzheimer’s disease instead.

According to the Alzheimer’s Association, individuals who have been diagnosed with MCI, particularly those who have memory issues, are more likely to later develop Alzheimer’s disease or a related dementia. In fact, it has been found that approximately 32% of individuals diagnosed with MCI develop Alzheimer’s disease within 5 years.

[Click to view source.](https://www.dementiacarecentral.com/aboutdementia/othertypes/mci/)

## 4. Should we use the longitudinal data for our classification model?
With in a longitudinal data, the information will be highly correlated with in each individual. And within a longitudinal dataframe, the assumption that random effects are normally distributed in those at risk at each event time is probably unreasonable. If the covariate is predictive of survival, patients whose covariates trajectories have the steepest negative slopes may be at higher risk for mortality, and get censored early. This may cause the random effects having a distributional shift toward a nonnormal distribution as time progresses. Due to these problems, we decided that we are going to only use the data collected at baseline to build up our classification model.

Reference: \\
_Wulfsohn, M.S. and Tsiatis, A.A., 1997. A joint model for survival and longitudinal data measured with error. Biometrics, pp.330-339._
