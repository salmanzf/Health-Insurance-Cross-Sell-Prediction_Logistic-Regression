# Health Insurance Cross Sell Prediction using Logistic-Regression (with Data Imbalance Example)
A health insurance company want to create a new Vehicle Insurance. Based on the Health Insurance customer, we want to predict how many interested in the new Vehicle Insurance.

## Data Introduction
You can view the table structure in the following link
data source: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

In general, we want to predict potential customer in the future based on the 'Response' conducted in the previous survey.
We will use the Logistic Regression algorithm to predict the potential customer.

## Data Exploration (EDA)
'''python
sns.countplot(x='Response', data=df_train)
plt.show()
'''
