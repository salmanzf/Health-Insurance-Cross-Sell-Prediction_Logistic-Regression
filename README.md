# Health Insurance Cross Sell Prediction using Logistic-Regression (with Data Imbalance Example)
A health insurance company want to create a new Vehicle Insurance. Based on the Health Insurance customer, we want to predict how many interested in the new Vehicle Insurance.

# Table of Content
1. [Data Introduction](#Data-Introduction)
2. [Data Exploration (EDA)](#Data-Exploration-EDA)
3. [Data Preparation (Cleaning)](#Data-Preparation-Cleaning)
   1. [Create Dummy Variable](#Create-Dummy-Variable)
   2. [Split Train Test Dataset](#Split-Train-Test-Dataset)
   3. [Feature Scaling](#Feature-Scaling)
4. [SMOTE Algorithm](#SMOTE-Algorithm)
5. [Recursive Feature Elimination Cross Validation (RFECV)](#Recursive-Feature-Elimination-Cross-Validation-RFECV)
6. [Logistic Regression Algorithm](#Logistic-Regression-Algorithm)
7. [Confusion Matrix & Conclusion](#Confusion-Matrix-&-Conclusion)

# Data Introduction
You can view the table structure in the following link
data source: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

In general, we want to predict potential customer in the future based on the 'Response' conducted in the previous survey.
We will use the Logistic Regression algorithm to predict the potential customer.

```python
df_train = pd.read_csv('train.csv')
```

# Data Exploration (EDA)

![Response](https://github.com/salmanzf/Health-Insurance-Cross-Sell-Prediction_Logistic-Regression/blob/streamlit/Image/1_response.png?raw=true)

Reponse 0 (No) is too dominant over 1 (Yes), in fact 87.74% 'No' Response and only 12.26% 'Yes' Response. This will cause the imbalance in the prediction, and to achieve desirable accuracy the algorithm will only predict 0 (No) Response and still achieving 87.74% accuracy. This is not what we want, because we want to predict potential customer and need to predict the 1 (Yes) Response.

> | Response | id |	Age |	Driving_License |	Region_Code |	Previously_Insured |	Annual_Premium |	Policy_Sales_Channel |	Vintage | 
> | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | 0 |	190611.255476 |	38.178227 |	0.997694 |	26.336544 |	0.521742 |	30419.160276 |	114.851040 |	154.380243 |
> | 1	| 190152.264504 |	43.435560 |	0.999122 |	26.762963 |	0.003383 |	31604.092742 |	91.869086 |	154.112246 |
 - The average with 1 Response is higher than 0 Response
 - The customer with no previous vehicle insurance far more interested in the vehicle insurance service

![Gender](https://raw.githubusercontent.com/salmanzf/Health-Insurance-Cross-Sell-Prediction_Logistic-Regression/streamlit/Image/2_gender.png)

  - Male is more interested in the vehicle insurace service, althought not significant

![Vehicle_Age](https://raw.githubusercontent.com/salmanzf/Health-Insurance-Cross-Sell-Prediction_Logistic-Regression/streamlit/Image/3_VehicleAge.png)

  - Customer with older Vehicle Age are more interested in the new vehicle insurance

![Vehicle_Damage](https://raw.githubusercontent.com/salmanzf/Health-Insurance-Cross-Sell-Prediction_Logistic-Regression/streamlit/Image/4_VehicleDamage.png)

  - Customer with Vehicle Damage far more interested in vehicle insurance

# Data Preparation (Cleaning)
Since we found no NULL values in all of the column, we can proceed to create dummy variable on the categorical column using pandas One Hot Encoding.
## Create Dummy Variable
```python
#Categorical Column
cat_col = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
for var in cat_col:
  cat_list = pd.get_dummies(df_train[var], prefix=var)
  df_train1=df_train.join(cat_list)
  df_train=df_train1

data_vars=df_train.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_col]

#Insert the new column to new variable
df_train_final=df_train[to_keep]
df_train_final.columns.values
```
> array(['id', 'Age', 'Driving_License', 'Region_Code',
       'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel',
       'Vintage', 'Response', 'Gender_Female', 'Gender_Male',
       'Vehicle_Age_1-2 Year', 'Vehicle_Age_< 1 Year',
       'Vehicle_Age_> 2 Years', 'Vehicle_Damage_No', 'Vehicle_Damage_Yes'],
      dtype=object)
 
 We can exclude the 'Region_Code' and 'Policy_Sales_Channel' since it contains too much variable and will make the function convoluted.
 
 ## Split Train-Test Dataset
 We will split the data into training and test set with ratio of 75-25.
 ```python
#Penempatan Variabel
x = df_train_final.loc[:, df_train_final.columns.isin(col_x)]
y = df_train_final.loc[:, df_train_final.columns == 'Response']

#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0)

#Menentukan panjang variabel sebelum digunakan algoritma SMOTE
print('Panjang jumlah data sebelum algoritma SMOTE', len(X_train))
print('Panjang jumlah variabel YES sebelum algoritma SMOTE', len(y_train[y_train['Response']==1]))
print('Panjang jumlah variabel NO sebelum algoritma SMOTE', len(y_train[y_train['Response']==0]))
```
- Data length before SMOTE 285831
- 'Response' = 1 before SMOTE 35035
- 'Response' = 0 before SMOTE 250796

## Feature Scaling
For feature scaling, we use standardization from scikit-learn algorithm.
```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

# SMOTE Algorithm
SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. We can see in the results later when we compare it with
the results of the prediction without using SMOTE algorithm.
```python
#SMOTE Algorithm
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)

columns = pd.DataFrame(X_train).columns.isin(col_x)
os_data_x, os_data_y = os.fit_resample(X_train, y_train)
os_data_x = pd.DataFrame(data=os_data_x,columns=columns )
os_data_y = pd.DataFrame(data=os_data_y,columns=['Response'])

#Panjang variabel setelah oversampling SMOTE
print('Panjang jumlah data setelah algoritma SMOTE', len(os_data_x))
print('Panjang jumlah variabel YES setelah algoritma SMOTE', len(os_data_y[os_data_y['Response']==1]))
print('Panjang jumlah variabel NO setelah algoritma SMOTE', len(os_data_y[os_data_y['Response']==0]))
```
- Data length after SMOTE 501592
- 'Response' = 1 after SMOTE 250796
- 'Response' = 0 after SMOTE 250796
  
  
Comparison data length BEFORE and AFTER SMOTE Algorithm:

| Variable | Before SMOTE | After SMOTE |
| --- | --- | --- |
| data length | 285831 | 501592 |
| 'Response' = 1 | 35035 | 250796 |
| 'Response' = 0 | 250796 | 250796 |

We can see the SMOTE Alogithm oversampling the minority (Response = 1) so data length match with the majority (Response = 0).

# Recursive Feature Elimination Cross Validation (RFECV)
RFECV is an automated Feature Selection to determine which independent variable is optimum for the predetermined criteria.
```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
logit = LogisticRegression()
rfe = RFECV(estimator=logit,
            cv=10,
            scoring='accuracy')
rfe.fit(os_data_x, os_data_y.values.ravel())

print(rfe.support_)
print(rfe.ranking_)
print("Optimum number of features: %d" % rfe.n_features_)
```
> [False False  True False False False False False False False False  True]\
[ 4  8  1  9 11 10  7  5  3  6  2  1]\
Optimum number of features: 2

Only 2 independent variabel fits the criteria, which are 'Previously_Insured' and 'Vehicle_Damage_Yes'\
  
```python
#Menerapkan algoritma RFE ke dalam Train dan Test Set
X_train_selected = rfe.transform(os_data_x)
X_test_selected = rfe.transform(X_test)
```
# Logistic Regression Algorithm
```python
#Training Model
logit.fit(X_train_selected, os_data_y)

#Predicting The Model
from sklearn import metrics
y_pred = logit.predict(X_test_selected)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logit.score(X_test_selected, y_test)))
```

# Confusion Matrix & Conclusion
Now lets see the comparison of performance WITH and WITHOUT using SMOTE Algorithm

| Indicator | SMOTE | no-SMOTE |
| --- | --- | --- |
| Optimal Column with RFECV | 2 | 1 |
| Accuracy | 64 % | 88 % |
| Confusion Matrix | ![SMOTE](https://github.com/salmanzf/Health-Insurance-Cross-Sell-Prediction_Logistic-Regression/blob/streamlit/Image/ConfusionMatrix_SMOTE.png) | ![no_SMOTE](https://github.com/salmanzf/Health-Insurance-Cross-Sell-Prediction_Logistic-Regression/blob/streamlit/Image/ConfusionMatrix_no-SMOTE.png) |

Although with SMOTE algorithm we have less Accuracy, we can see the prediction without SMOTE algoithm only predicts '0' (No) with no '1' (Yes) Response and it can achieve 88% accuracy. If we use the algorithm without SMOTE algorithm, we will not have a potential customer in the future since it is not predicting '1' Response. This is why imbalance dataset causes the misconception in prediction algorithm and where SMOTE algorithm is used. With SMOTE algorithm we can predict 11421 out of 11675 (97.82%) '1' Response. This algorithm is far useful since it can predict alot more potential customer in the future, eventhough it also cost a lot more for the marketing team since it has a lot of False Positive where 34334 out of 83603 (41.07%) are wrong '1' Response prediction.
  
At the end, it's up to business decision maker whether the profit from future potential customer is worth the extra marketing budget.
