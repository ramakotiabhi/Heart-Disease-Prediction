#  Heart Disease Prediction Using Logistic Regression

 ## Step 1: Introduction

In this project, heart data is used, the target of which is in two classes. People who have heart disease and people who don't. The importance of this type of project in the medical world is very high and it should be given more attention. In this kernel, I create a predictive model with the help of different classification algorithms so that people can be predicted to have or not have a disease.
  
## Step 2:  Data Generation

Let's generate some synthetic data for heart disease prediction. We'll create features like age, cholesterol level, blood pressure, and whether the person smokes or not, along with a binary target variable indicating the presence or absence of heart disease.

import numpy as np
import pandas as pd

#Generate synthetic data
np.random.seed(0)
age = np.random.randint(25, 80, 1000)
cholesterol = np.random.randint(120, 300, 1000)
heart_disease = np.random.randint(0, 2, 1000)

#Introduce some missing values
age[np.random.choice(1000, 50, replace=False)] = np.nan
cholesterol[np.random.choice(1000, 30, replace=False)] = np.nan

#Create a DataFrame
data = pd.DataFrame({'Age': age, 'Cholesterol': cholesterol, 'Heart Disease': heart_disease})

#Handling missing values
data.dropna(inplace=True)

#Handling outliers (if any)
#You may use techniques like Winsorization or Z-score

#Handling duplicates (if any)
data.drop_duplicates(inplace=True)

#Displaying cleaned data
print(data.head())

## Step 3: Data Splitting

We'll split the dataset into training and testing sets

from sklearn.model_selection import train_test_split

#Splitting the data into features and target variable
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

#Splitting the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Displaying the shapes of the split datasets
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
 
## Step 4 : Building the Logistic Regression Model

We'll build a logistic regression model using the training data.

from sklearn.linear_model import LogisticRegression

#Creating and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

## Step 5:  Model Evaluation

We'll evaluate the performance of the model on the test set using various metrics.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Making predictions on the test set
y_pred = model.predict(X_test)

#Calculating evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

#Displaying evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

## Step 6: Analyzing Model Coefficients

We'll analyze the coefficients of the logistic regression model to understand the importance of features.

#Extracting coefficients and corresponding feature names
coefficients = model.coef_[0]
feature_names = X.columns

#Creating a DataFrame to display coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(coef_df)
