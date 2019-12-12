---
layout: page
title: "Predicting Crime Types in Boston Area"
description: "Aiming for a safer city"
header-img: "img/home-bg.jpg"
---

# Overview

For this final project, we examined the public available data from Boston Police Department (BPD). We are going to focus on two main goals: 1) Understanding the geographical and temporal crime pattern in Boston, as well as evaluating if the crime categories are related with certain situational or environmental factors 2) Forecasting the types of criminal misconduct in Boston at night, with the ultimate goal to help the Police Department with a model that offers insightful information to better control the crime rate.

# Motivation

Based on FBI crime data, in 2018, Boston ranked 14 out of 50 in the US with regard to crime rate. The chance of being a victim of either violent or property crime in Boston is as high as 1 in 34, which makes Boston more violent than cities such as New York and Seattle. Even though Boston crime statistics demonstrate an overall downward trend in violent and property crimes in the past years, the city’s crime rate is still a lot higher than the national average crime rate.

Considering that the Boston Police Department has limited resources to foot patrol and guard a wide range of areas at night. Under this circumstance, distributing the limited intervention capacity to the most needed location is of great importance. If we are able to use data analytics to forecast the types of crime misconduct of each district at a certain time, the Boston Police will have better visibility on allocate manpower with targeted preparation. For this project we are also interested to see if certain situational and environmental features are linked with crime types. Understanding this context in which violence occurs has broad applicability to public policies on violence mitigation. The project result will hopefully help BPD to design a more effective intervention program to reduce the crime rate at night.

# Challenges
Data processing was a very challenging task in this project. Although the datasets we used were mostly found on one single open data hub *Analyze Boston*, there were subtle variations in how these different data were encoded and recorded. For instance, in the Street Lights dataset, geographic coordinates were given as the measure of location, while in the Property Value dataset, only street names and zip codes were provided. Even though streets and coordinates were all included in the crime incident reports, connecting these three datasets was obstructed due to inconsistent naming and abbreviations of the streets across the different data. Thus, recoding for a unified column of street names was required prior to data merging. This work was almost impossible to be automated and demanded a lot of manual checking to achieve completion.

In addition, the task of our final project is to conduct multi-class classification, while most data we used in the past assignments had binary outcomes as response variable. The complexity of our final dataset (imbalanced outcomes, independent variables from very different sources) further increased difficulties when it came to model building.

Finally, our data contained more than 100,000 observations, and the computing power of local laptops became a limiting factor during the process of data cleaning and modelling. For example, when we were creating the column `Dist_to_Nearest_Light`, the main challenge we were facing was that per-pair calculations of the distances between all the street lights (more than 70000 observations) and all the crime locations (more than 140000 observations) took extremely long time to run (big shout out to our TF Abhi who helped us with his time-saving solution). Similarly, despite the fact that training and test accuracies still exhibited an increasing trend in our neural network model, we had no choice but restrict the number of epochs to 1000 due to limited computing power.
