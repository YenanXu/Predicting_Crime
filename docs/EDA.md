---
layout: page
title: "EDA"
description: "Exploratory data analysis"
header-img: "img/home-bg.jpg"
---

#### _The general EDA questions we want to explore are:_
#### _How do distributions of covariates vary among CN, MCI and AD?_
#### _What variables contribute significantly to the classification of AD, MCI, and CN?_

## Data Description

We use data from ADNI dataset. This dataset is a longitudinal dataset, participants were carried forward from previous phases for continued monitoring while new enrollees were added with each phase to further investigate the evolution of AD. Participants of the study are between 55 to 90 years old, who are from 57 sites in the US and Canada. Participants would go through a series of initial tests and take repeated at intervals over subsequent years. Tests include clinical evaluation, neuropsychological tests, genetic testing, lumbar puncture, and MRI and PET scans. In this dataset, clinical data are obtained on for ADNI protocols, ADNI 1, ADNI 2, ADNI 3, ADNI GO, which comprise clinical information about each subject including recruitment, demographics, physical examinations, and cognitive assessment data, etc. Although they are not all the same regarding the date and completeness of measurement, as we are mainly focused on baseline biological and medical data and screening data for demographic, neurological exams, screening labs, vital signs, etc., which are almost complete for the four protocols, we do not differentiate different ADNI protocols when analyzing the data.

## Variables and Observations Cleaning
For the use of exploratory data analysis (EDA), we first deleted observations that are not baseline values and observations that the response value ‘DX’ is missing. Since we just included necessary, not repetitive, baseline data and non-patient reported data, we dropped 58 variables inclduing ‘PTID’, ’SITE’, ‘COLPROT’, ‘ORIGPROT’, ‘EXAMDATE’, and all variables do not end with ‘bl’, ‘RID’, ‘Month’, ‘M’, ‘updated_stamp’, ‘FLDSTRNG’, ‘FSVERSION’, and all Ecog test results by the patients. Moreover, considering the completeness of data, we dropped variables that have missing values more than 80%. 

```python
df = pd.read_csv('ADNIMERGE.csv')
```
```python
#keep only the baseline value for all the variables. (also deleting "VISCODE")
df2 = {}
var = list(df.columns.drop('VISCODE'))
for i in var:
    df2[i] = df[i][df['VISCODE'] == 'bl']
df2 = pd.DataFrame(df2)
print(len(df)-len(df2), 'observation deleted. And variable "VISCODE" is deleted.')

#drop the observation if 'DX' is missing
df3 = {}
var = list(df2.columns.drop('DX'))
for i in var:
    df3[i] = df2[i][df2['DX'].isnull() == False]
df2['DX'].fillna(-1, inplace = True)
df2_DX = list(df2['DX'])
while -1 in df2_DX:
    df2_DX.remove(-1)
df3['DX'] = df2_DX
df3 = pd.DataFrame(df3)
print(len(df2)-len(df3), 'observation deleted.')
print(len(df3), 'observations left.')
```
```python
#drop variables that are selected based on the criteria above
bad = ['PTID','SITE','COLPROT','ORIGPROT', 'EXAMDATE', 'DX_bl', 'EXAMDATE_bl', 
       'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 
       'RAVLT_perc_forgetting_bl', 'FAQ_bl','FLDSTRENG_bl', 'FSVERSION_bl', 'Ventricles_bl', 'Hippocampus_bl', 
       'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 
       'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl', 'EcogSPMem_bl', 'EcogSPLang_bl', 
       'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'FDG_bl', 'PIB_bl', 'AV45_bl', 
       'Years_bl', 'Month_bl', 'RID', 'Month', 'M', 'update_stamp', 'FLDSTRENG', 'FSVERSION','EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', \
       'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal']
df4 = df3.drop(bad, axis = 1)
print(len(list(df3.columns))-len(list(df4.columns)), 'variables are deleted.')
print('Remaining variables are:', list(df4.columns))
```
```python
#Turn all values of categorical variables into categories named in 0, 1, 2, 3...while keeping NaN and turining all 'Unknown's to NaN

cat = [] #list of names of categorical variables
catdict_raw = {} #dictionary with names of categorical variables as keys, and all categories as values, including nan & 'Unknown'
catdict = {} #dictionary with names of categorical variables as keys, and non-nan categories as values (excluding nan & 'Unknown')

#convert 'APOE4' to string, because it is categorical but will be regarded as continuous later.
df5 = {}
df5['APOE4'] = [-1]*len(df4['APOE4'])
for i in range(len(df4['APOE4'])):
    if list(df4['APOE4'])[i] != np.nan:
        df5['APOE4'][i] = str(list(df4['APOE4'])[i])

var = list(df4.columns.drop('APOE4'))
for i in var:
    df5[i] = list(df4[i])
df5 = pd.DataFrame(df5)
df5['APOE4'].replace(-1, np.nan, inplace = True)

for v in df5.columns:
    if df5[v].dtypes == 'O':
        cat.append(v)
        catdict_raw[v] = list(set(df5[v]))
        df5[v].fillna('Unknown', inplace = True) #replace all NaN first into 'Unknown'
        catdict[v] = list(set(df5[v]))
        catdict['APOE4'] = [0,1,2]
        catdict['DX'] = ['CN','MCI','Dementia']
        if 'Unknown' in catdict[v]:
            catdict[v].remove('Unknown') #'Unknown' should not participate in categorizing
        for i in range(len(catdict[v])):
            df5[v].replace(catdict[v][i], i, inplace = True) #Turn all values of categorical variables into categories named in 0, 1, 2, 3...
        df5[v] = pd.to_numeric(df5[v], errors = 'coerce') #convert all 'Unknown's then to NaN
contin = list(df5.columns.drop(cat))
print('Categorical variables are:', cat, '\n')
print('Continuous variables are', contin, '\n')
print('The dictionary for values of categorical variable corresponding to their original names: \n', catdict)
```
```python
#cout the NaNs in each variables, drop variables that has percentage of NaNs over 80%.
perc = []
bad2 = []
for i in list(df5.columns):
    perc_temp = (1-df5[i].isnull().sum()/len(df5))*100
    perc.append(perc_temp)
    if perc_temp < 20:
        bad2.append(i)
plt.figure(figsize=(17,5))
plt.bar(list(df5.columns), perc)
plt.xticks(list(df5.columns), rotation=90)
plt.axhline(20, c = 'r')
plt.ylabel('percent of vaid data (%)')
plt.show()
df6 = df5.drop(bad2, axis = 1)
print(len(list(df5.columns))-len(list(df6.columns)), 'variables are deleted.')
print('Deleted variables are:', bad2)
```

<img src="https://yueli1201.github.io/Alzheimer/figures/1.jpeg" alt="1" width="750"/>
<div align="center"><font size="2"><b>Fig 1. Completeness of dataset after dropping unnecessary variables and observations</b></font></div>

## Correlation Analysis of Variables

Next, we identified continuous and categorical variables and converted the later to numbers for the convenience of further manipulation. To explore the current dataset, we first took a look at the correlations between any pairs of the variables without imputing missing values because the loss of data is small when doing correlation analysis, the probability of two variables either has a NaN is small, so we just delete any pair that either variable has a NaN. And the correlation matrix is drawn as below. 

```python
import seaborn as sns
f, ax = plt.subplots(figsize=(10,10))
corr = df6.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()
```

<img src="https://yueli1201.github.io/Alzheimer/figures/2.jpeg" alt="2" width="750"/>
<div align="center"><font size="2"><b>Fig 2. Correlation matrix including missing values after dropping unnecessary variables and observations**</b></font></div>

## Distribution Exploration for Variables Grouped by Disease Status

Next, we plot the distribution of the variable across non-disease control (CN) participants, MCI participants, and AD participants. Shown in the below plots, we could see that AD, MCI, and CN are not very different across age(AGE), education level (PTEDUCAT), and intracerebral volume (ICV). The levels of Average FDG-PET of angular, temporal, and posterior cingulate
(FDG), Mini-Mental State Examination(MMSE), Hippocampus, WholeBrain, Entorhinal, Fusiform, middle temporal gyrus (MitTemp), etc. are negatively associated with cognitive impairment severity, while the scores of Clinical Dementia Rating-Sum of Boxes (CDRSB), FAQ, Ventricles, etc. are positively associated with cognitive impairment severity.  

```python
continuous = ['AGE', 'PTEDUCAT', 'ICV',
              'FDG','MMSE','RAVLT_immediate','RAVLT_learning','MOCA', 
              'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp',
              'AV45', 'CDRSB', 'ADAS11', 'ADAS13',   'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'FAQ', 
              'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 
              'EcogSPTotal', 'Ventricles'] 

fig, ax = plt.subplots(7,4,figsize = (20,30))
ax = ax.ravel()

for i in range(len(continuous)):
    sns.boxplot(x=df6['DX'], y=df6[continuous[i]],palette = ['#0343df', 'g', '#e50000'], ax=ax[i])
    fig.tight_layout()
    ax[i].set_xticklabels(['CN', 'MCI','AD'])
```

<img src="https://yueli1201.github.io/Alzheimer/figures/3.jpeg" alt="3" width="750"/>
<div align="center"><font size="2"><b>Fig 3. Distribution of continuous variables grouped by disease status</b></font></div>

```python
covariates = ['APOE4', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY'] 
catdict['DX'] = ['CN', 'MCI', 'AD']
color = ['#0343df', 'g', '#e50000']
fig, ax = plt.subplots(len(covariates), 3, figsize=(15,20))
ax = ax.ravel()
a = 0
df_temp = df6.copy()
for i in covariates:
    for dx in list(set(df_temp['DX'])):
        count = []
        for cat in catdict[i]:
            count.append(sum(df_temp[i][df_temp['DX'] == dx] == catdict[i].index(cat)))
        ax[a].bar(catdict[i], count, alpha=1, color = color[int(dx)])
        ax[a].set_xlabel(i)
        ax[a].set_ylabel('count')
        ax[a].set_title(catdict['DX'][int(dx)])
        if i == 'PTRACCAT':
            ax[a].set_xticklabels(catdict[i], rotation=45)
        fig.tight_layout()
        a = a + 1
```

<img src="https://yueli1201.github.io/Alzheimer/figures/4.jpeg" alt="4" width="750"/>
<div align="center"><font size="2"><b>Fig 4. Distribution of categorical variables grouped by disease status</b></font></div>

According to the bar charts, the three health status have different distribution on demographic features. AD has a higher distribution in male, non-Hispanic, especially white, married people, and with APOE4=1. However, we are not sure about the determinants of classification yet since the results may due to the sampling method.

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