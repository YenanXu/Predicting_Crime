---
layout: page
title: "EDA"
description: "Exploratory Data Analysis"
header-img: "img/home-bg.jpg"
---

#### _In this part, the location data provided in four datasets will first be associated using geospatial labels such as zipcodes, street names and coordinates. Exploratory data analysis will then be performed to show the geographic distribution of the main focus (crime types, street light etc.). The correlation among covariates and the distributions of covariates for six different crime types will also be examined in order to choose better predictors for the models._

## Data Description

Data of this project was extracted from the following datasets:
1. [Crime Incident Reports Dataset](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system) provided by Boston Police Department(BPD)
- The 2015 Crime incident datasets is 77 Mb with 440,045 observation and 17 columns. This dataset captures important information about crime type, location, time, etc.
2. [Property Assessment Dataset](https://data.boston.gov/dataset/property-assessment) that gives property, or parcel, ownership together with value information in Boston, published by the Department of Innovation and Technology.
- The 2019 (fiscal year) Property Assessment Dataset is 44 Mb with 174,668 observations and 75 columns, including information about address, property type, value, area and other details.
- Variables description is provided on _Analyze Boston_
3. [Street Lights Dataset](https://data.boston.gov/dataset/streetlight-locations) maintained by The Street Lighting Division of Public Works
4. [Census ZIP Code Demographic Profile Dataset](https://www.nber.org/data/census-2010-zip-code-data.html)
- Selected economic statistics from Census 2010 ZIP Code Tabulation Area Demographic Profile Summary File. Variables include ZIP code, Total population, Population density, Population by female, Population by age, median household income which represents district socioeconomic status.



## Data Cleaning

#### Crime Incident Reports Dataset
The `Crime_Type` of the final data was obtained from the BPD Crime incident reports. We have defined six crime types for prediction by merging similar offense code groups: theft (merging 'Commercial Burglary', 'Auto Theft', 'Other Burglary', 'Larceny', 'Burglary – No Property Taken', 'Larceny From Motor Vehicle', 'Residential Burglary'), robbery ('Robbery'), assault (merging 'Aggravated Assault', 'Harassment', 'Criminal Harassment', 'Simple Assault'), vandalism ('Vandalism'), motor/vehicle accident('Motor Vehicle Accident Response') and drug abuse violations('Drug Violation'). Each crime type was labeled by a different number, with theft as 0, robbery as 1, assault as 2, vandalism as 3, motor/vehicle accident as 4 and drug abuse violations as 5. Since we are interested in the relationship between street lighting and crime type occurred in Boston, the crime incidents were only included if they took place when street lights are on. Whether Boston street lights are turned on or not is based on ambient light. We checked the sunset time and set different time boundaries across months of the year to ensure that the crime incidents in our dataset fall in the “dark time” of the day. More specifically, the time intervals were restricted within 5:00pm-6:00am for November to February, 7:30pm-5:00am for September, October, March and April, and 8:30pm-4:00am for May to August. For the purpose of connecting with the Property Assessment Dataset, the first, second and last (suffix) words were obtained from `STREET` and double-checked by eyeball to be consistent with those variables in the _Property Assessment Dataset_. The number of observations was reduced to 148,467 after restriction and cleaning.

#### Property Assessment Dataset
The _Property Assessment Dataset_ was first explored on missing values. Observations with missing geographic information that is required to merge datasets (`ST_NAME` and `ZIPCODE`) were dropped. We only kept quantitative variables with missing values no more than 50% for subsequent analysis. For better matching, the `ZIPCODE` variable was transformed from float type to string and the first and second words of `ST_NAME` were extracted to new variables. The new street name related variables were double-checked by eyeball to be consistent with those variables in the Crime Dataset. The state class code `PTYPE` was categorized into ‘Multiple Use Property’, ‘Residential Property’, ‘Apartment Property’, ‘Commercial Property’, ‘Industrial Property’, ‘Exempt Ownership’ and ‘Exempt Property Type’ according to the provided _PROPERTY OCCUPANCY CODES TABLE_. The built year and last remodeled year were re-calculated by subtracting the median respectively. At last, the dataset was grouped by street names and prepared for merging.

#### Street Lights Dataset
From the streetlight locations provided by _The Street Lighting Division of Public Works_, we extracted the geographic coordinates of the lights across Boston. Since the same geographic information can be obtained from the crime incident reports, the **Haversine equation** can be applied to calculate the distance between every single place of crime occurrence and streetlight location.
- Haversine formula:
<img src="https://yenanxu.github.io/Predicting_Crime/figures/H_formula.jpg" alt="1" width="750"/>

Where $\varphi$ represents latitude, $\lambda$ represents longitude and the subscript marks the location.

Columns of the distance to the nearest streetlight, counts of streetlights within 50 and 100 meters for each crime incident were created in the purpose of evaluating the influence of ambient light upon criminal activities.

#### Census ZIP Code Demographic Profile Dataset
38 ZIP Codes that appeared on _Property Assessment Dataset_ were recorded and were used to extract district socio-economic data from _Census ZIP Code Demographic Profile Dataset_ using the `uszipcode` package. The percentage of younger population (`Young_prop`) is calculated using `Younger Population`/ `Total Population`, where the younger is defined as people who are under 19-years old. Four ZIP Codes did not have information on Census Dataset. Based on our research, ZIP Code boundaries were adjusted in previous years. In 2010, 02112 was contained within 02111 ZIP Code Boundary, 02133 and 02137 were contained within 02136 ZIP Code Boundary. 02201 is a unique - single entity ZIP Code that only include 2 buildings. Therefore, we extrapolated the data of bigger boundary to predict the situation in 02112, 02133, 02137 area. Missing value of 02201 was extrapolated from the nearest district 02109.

#### Merge the datasets
The _Crime Incident Reports Dataset_ and the _Property Assessment Dataset_ were firstly merged on the combination of the first two words from street name and suffix, and then on the combination of the first word and suffix, lastly on the combination of the first two words to increase our capacity of matching. The _Census ZIP Code Demographic Profile Dataset_ then was merged with the dataset by ZIP codes. The most useful information from _Street Lights Dataset_, `Dist_to_Nearest_Light`, `Lights_within_50m` and `Lights_within_100m` were then calculated. The distribution of crime types from the matched and unmatched data was found to be similar, so 5989 unmatched observations were dropped and the final dataset was reduced to 142,478 observations.

```python
# Check the distribution of crime types for matched and unmatched observations
fig, ax = plt.subplots(1,1, figsize=(12,8))

unmatched = df_merge_CP[df_merge_CP['ZIPCODE'].isnull()]['Crime_Type']
matched = df_merge_CP[~df_merge_CP['ZIPCODE'].isnull()]['Crime_Type']

ax.hist(unmatched, bins=np.arange(7)-0.3, density=True, width=0.3, alpha=0.7, label='Unmatched')
ax.hist(matched, bins=np.arange(7), density=True, width=0.3, alpha=0.7, label='Matched')
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(labels=['Theft', 'Robbery', 'Assault', 'Vandalism', 'M/V Accident', 'Drug Abuse Violations'])
ax.set_xlim([-0.5, 5.5])
ax.legend()
ax.set_xlabel('Crime Type')
ax.set_ylabel('Number of Incidents');
```

<img src="https://yenanxu.github.io/Predicting_Crime/figures/EDA_1.png" alt="2" width="750"/>
<div align="center"><font size="2"><b>Fig 1. Distribution of crime types for matched and unmatched observation</b></font></div>

```python
df_final.head()
```
 | INCIDENT_NUMBER | OFFENSE_CODE	
-- | -- | -- 
0 | I192078623 | 3802











## Covariates Selection and Imputation of Missing Data

We have a longitudinal dataset. But for our classification analysis, we just want to keep the baseline information to build classification models. Because, first, we want to find a cost-efficient way to help the classification of AD. Secondly, in the longitudinal data, the information is highly correlated within each individual. ‘Examdate’, ‘update_stamp’, ‘FLDSTRENG’, ‘FSVERSION’ are not useful for the mdoel because they are not the relavent information of patients. So, we excluded them from our analysis. According many previous publications, the patient everyday cognition scale (Ecog) is very uninformative, especially among those dementia people. So all EcogPt variables were excluded from our analysis. Only very few participants in ADNI1 (less than 5% of the total data) have information on Pittsburgh compound B (PIB) test. Therefore, this variable was excluded from our analysis.

Among the rest of the data, Participants in ADNI1 don’t have information on Everyday Cognition Scale (Ecog), Montreal Cognitive Assessment (MOCA), and AV45. Participants from ADNI3 lack information on APOE4, FDG-PET, AV45, Hippocampus volume, whole brain status, Entorhinal, Fusiform, middle temporal gyrus (MidTemp), intracerebral volume (ICV), Ventricles. But as there are only 46 participants in ADNI3, it will not cause a large proportion missing. There are some other randomly missing data. When combine participants recruited based on four different protocols, we assumed that the data were missing at random.  We used IterativeImputer method in fancyimpute (A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion) to impute these missing data.

```python
# Impute missing data
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler

X=df6.drop(['DX'], axis=1)
Y=df6['DX']
columns = X.columns

X_filled_ii = IterativeImputer().fit_transform(X)
X_filled_ii = pd.DataFrame(X_filled_ii)
X_filled_ii.columns = X.columns
```

```python
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X_filled_ii, Y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

So now we are ready to explore the two questions using the upcomming regression models and machine learning algorithms.
