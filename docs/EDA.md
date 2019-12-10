---
layout: page
title: "EDA"
description: "Exploratory data analysis"
header-img: "img/home-bg.jpg"
---

Exploratory data analysis will be performed to show the geographic distribution of the main focus (crime types, street light etc.). The correlation among covariates and the distributions of covariates for six different crime types will also be examined in order to choose better predictors for the models.

## Variables Description
The response variable in our project is the categorical variable `Crime Type`. Each crime type was labeled by a different number, with theft as 0, robbery as 1, assault as 2, vandalism as 3, motor/vehicle accident as 4 and drug abuse violations as 5. All the variables that might potentially be predictors are listed.

<div align="center"><font size="2"><b>Table 1. Potential predictors for models in the final dataset</b></font></div>

| Variable       | Type        | Description                                                                     |
|----------------|-------------|---------------------------------------------------------------------------------|
| SHOOTING_DUMMY | Categorical | 1 indicates shooting event involved and 0 indicates no shooting event involved. |
| HOUR           | Categorical | The time (hour) that crime occurred.                                            |

## Missing values
The counts and distribution of missing values in the potential predictors were checked to decide the right way to handle missing. Table 2. shows that 7 out of 26 variables had missing values, which accounted for only no more than 3%. The distribution of crime types for missing and unmissing data was also found to be similar (Fig 1.), so we decided to drop all the missings in predictors used for models.

```python
# Variables to be be checked
pred_col_names = ['SHOOTING_DUMMY', 'HOUR', 'MONTH', 'DAY_OF_WEEK_NUM',
                  'AV_BLDG', 'AV_TOTAL', 'LAND_SF', 'GROSS_TAX', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS',
                  'PTYPE_A', 'PTYPE_C', 'PTYPE_EO', 'PTYPE_EP', 'PTYPE_I', 'PTYPE_MU', 'PTYPE_R', 'YR_BUILT_m', 'YR_REMOD_m',
                  'Population Density', 'Young_prop', 'Median household income',
                  'Dist_to_Nearest_Light', 'Lights_within_50m', 'Lights_within_100m']

# Missing values table
missing_columns = df_final[pred_col_names].columns[df_final[pred_col_names].isnull().any()]
missing_count = []
for c in missing_columns:
    missing_count.append(df_final[c].isnull().sum())

df_missing = pd.DataFrame(data={'Count': missing_count}, index=missing_columns.values)

df_missing['Proportion'] = df_missing['Count']/df_final.shape[0]

df_missing
```

<div align="center"><font size="2"><b>Table 2. Summary of missing values in the potential predictors</b></font></div>

|             | Count | Proportion |
|-------------|-------|------------|
| AV_BLDG     | 284   | 0.001993   |
| GROSS_TAX   | 1974  | 0.013855   |
| GROSS_AREA  | 595   | 0.004176   |
| LIVING_AREA | 595   | 0.004176   |
| NUM_FLOORS  | 488   | 0.003425   |
| YR_BUILT_m  | 575   | 0.004036   |
| YR_REMOD_m  | 3318  | 0.023288   |

```python
missing = df_final.loc[(df_final['GROSS_AREA'].isnull())|
             (df_final['LIVING_AREA'].isnull())|
             (df_final['NUM_FLOORS'].isnull())|
             (df_final['YR_BUILT_m'].isnull())|
             (df_final['YR_REMOD_m'].isnull())]
unmissing = df_final.iloc[df_final.index.difference(missing.index)]

# Check the distribution of crime types with missing and without missing values
crime_labels = ['Theft', 'Robbery', 'Assault', 'Vandalism', 'M/V Accident', 'Drug Abuse Violations']
fig, ax = plt.subplots(1,1, figsize=(12,8))

ax.hist(unmissing['Crime_Type'], bins=np.arange(7)-0.3, density=True, width=0.3, alpha=0.7, label='Unmissing')
ax.hist(missing['Crime_Type'], bins=np.arange(7), density=True, width=0.3, alpha=0.7, label='Missing')
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(labels=crime_labels)
ax.set_xlim([-0.5, 5.5])
ax.legend(frameon = True, framealpha=0.9)
ax.set_xlabel('Crime Type')
ax.set_ylabel('Density of Incidents');
```

<img src="https://yenanxu.github.io/Predicting_Crime/figures/missing.png" alt="1" width="600"/>
<div align="center"><font size="2"><b>Fig 1. Distribution of crime types for missing and unmissing observations</b></font></div>

## Correlation Analysis

The potential predictors were then explored by looking at the correlation matrix which excluded missing values. `AV_BLDG` and `AV_TOTAL`, `Young_prop` and `Median household income`, `Lights_within_50m` and `Lights_within_100m` were found to have high correlation that might induce issue of colinearity in some models.

```python
# Correlation Analysis of Variables
fig, ax = plt.subplots(figsize=(10,10))
corr = df_final[pred_col_names].corr()
sns.heatmap(corr, vmin=-1, vmax=1, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax);
```

<img src="https://yenanxu.github.io/Predicting_Crime/figures/correlation.png" alt="2" width="600"/>
<div align="center"><font size="2"><b>Fig 2. Correlation matrix excluding missing values</b></font></div>

## Variables Distribution for Different Crime Types



```python
# Correlation between crime type and other variables
# Continuous variables
continuous = ['AV_BLDG', 'AV_TOTAL', 'LAND_SF', 'GROSS_TAX', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS',
              'PTYPE_A', 'PTYPE_C', 'PTYPE_EO', 'PTYPE_EP', 'PTYPE_I', 'PTYPE_MU', 'PTYPE_R', 'YR_BUILT_m', 'YR_REMOD_m',
              'Population Density', 'Young_prop', 'Median household income',
              'Dist_to_Nearest_Light', 'Lights_within_50m', 'Lights_within_100m']
log_trans = ['AV_BLDG', 'AV_TOTAL', 'LAND_SF', 'GROSS_TAX', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS',
              'YR_BUILT_m', 'YR_REMOD_m', 'Dist_to_Nearest_Light']    # need log-transformation to make a good-looking plot

# plot
fig, ax = plt.subplots(6,4,figsize = (25,40))
fig.subplots_adjust(hspace = 0.3)
ax = ax.ravel()

fig.delaxes(ax[22])
fig.delaxes(ax[23])

for i in range(len(continuous)):
    # log-transformation for some variables
    if continuous[i] in log_trans:
        sns.boxplot(x=df_final['Crime_Type'], y=df_final[continuous[i]].apply(np.log), ax=ax[i])
        ax[i].set_ylabel('LOG_'+continuous[i])
    else:
        sns.boxplot(x=df_final['Crime_Type'], y=df_final[continuous[i]], ax=ax[i])
    ax[i].set_xticklabels(labels=crime_labels, rotation=30, horizontalalignment='right')
```

<img src="https://yenanxu.github.io/Predicting_Crime/figures/continuous_box.png" alt="3" width="750"/>
<div align="center"><font size="2"><b>Fig 3. Distribution of continuous variables for different crime types</b></font></div>

```python
# Categorical variables
categorical = ['SHOOTING_DUMMY', 'HOUR', 'MONTH', 'DAY_OF_WEEK_NUM']

# plot
fig, ax = plt.subplots(len(categorical),1,figsize = (20,40))
ax = ax.ravel()
bar_wid = 0.1

for i in range(len(categorical)):
    for j in range(len(set(df_final['Crime_Type']))):
        x_length = len(set(df_final[categorical[i]]))
        count = df_final.loc[df_final['Crime_Type'] == j].groupby(categorical[i])['Crime_Type'].count().values
        ax[i].bar(np.arange(x_length)+bar_wid*j, count, width = bar_wid, alpha=0.7, label = crime_labels[j])
    ax[i].set_xticks([i+len(set(df_final['Crime_Type']))/2*bar_wid for i in range(x_length)])
    ax[i].set_xlabel(categorical[i])
    ax[i].set_ylabel('Counts of each type of crime')
    ax[i].legend(frameon = True, framealpha=0.9)

ax[0].set_xticklabels(labels=['No', 'Yes'])
ax[1].set_xticklabels(labels=set(df_final['HOUR']))
ax[2].set_xticklabels(labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax[3].set_xticklabels(labels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']);
```

<img src="https://yenanxu.github.io/Predicting_Crime/figures/categorical_hist.png" alt="4" width="750"/>
<div align="center"><font size="2"><b>Fig 4. Distribution of categorical variables for different crime types</b></font></div>
