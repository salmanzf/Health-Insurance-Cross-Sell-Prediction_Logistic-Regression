# Health Insurance Cross Sell Prediction using Logistic-Regression (with Data Imbalance Example)
A health insurance company want to create a new Vehicle Insurance. Based on the Health Insurance customer, we want to predict how many interested in the new Vehicle Insurance.

## Data Introduction
You can view the table structure in the following link
data source: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

In general, we want to predict potential customer in the future based on the 'Response' conducted in the previous survey.
We will use the Logistic Regression algorithm to predict the potential customer.

```python
df_train = pd.read_csv('train.csv')
```

## Data Exploration (EDA)
```python
sns.countplot(x='Response', data=df_train)
plt.show()
```
Reponse 0 (No) is too dominant over 1 (Yes), in fact 87.74% 'No' Response and only 12.26% 'Yes' Response. This will cause the imbalance in the prediction, and to achieve desirable accuracy the algorithm will only predict 0 (No) Response and still achieving 87.74% accuracy. This is not what we want, because we want to predict potential customer and need to predict the 1 (Yes) Response.

```python
df_train.groupby('Response').mean()
```
 - The average with 1 Response is higher than 0 Response
 - The customer with no previous vehicle insurance far more interested in the vehicle insurance service

```python
fig_gender = pd.crosstab(df_train.Gender, df_train.Response)
fig_gender = fig_gender.div(fig_gender.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
for c in fig_gender.containers:
    fig_gender.bar_label(c, label_type='center')
plt.axhline(y=pct_neg, color='red')
plt.title('Stacked Bar Chart of Age vs Response')
plt.xlabel('Gender')
plt.ylabel('Proportion of Response')
```
  - Male is more interested in the vehicle insurace service, althoughn not significant

```python
fig_vehicle_age = pd.crosstab(df_train.Vehicle_Age, df_train.Response)
fig_vehicle_age = fig_vehicle_age.div(fig_vehicle_age.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
for c in fig_vehicle_age.containers:
    fig_vehicle_age.bar_label(c, label_type='center')
plt.axhline(y=pct_neg, color='red')
plt.title('Stacked Bar Chart of Vehicle Age vs Response')
plt.xlabel('Vehicle Age')
plt.ylabel('Proportion of Response')
```
  - Customer with older Vehicle Age are more interested in the new vehicle insurance

```python
fig_vehicle_dmg = pd.crosstab(df_train.Vehicle_Damage, df_train.Response)
fig_vehicle_dmg = fig_vehicle_dmg.div(fig_vehicle_dmg.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
for c in fig_vehicle_dmg.containers:
    fig_vehicle_dmg.bar_label(c, label_type='center')
plt.axhline(y=pct_neg, color='red')
plt.title('Stacked Bar Chart of Vehicle Damage vs Response')
plt.xlabel('Vehicle Damage')
plt.ylabel('Proportion of Response')
```
  - Customer with Vehicle Damage far more interested in vehicle insurance

## Data Preparation
