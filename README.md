# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries .
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5.Determine training and test data set.
6.Apply decision tree Classifier on to the dataframe.
7.Get the values of accuracy and data prediction. 
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GOPIKA K
RegisterNumber:212222040046  
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Initial data set:
![image](https://github.com/user-attachments/assets/bcc220fa-54ed-438c-8d20-95e8821e058e)

## Data info:
![image](https://github.com/user-attachments/assets/fd082f67-66b9-4c8a-9929-67485fb9668c)

## Optimization of null values:
![image](https://github.com/user-attachments/assets/c12796da-83fb-41f1-80b2-82c503479550)

## Assignment of x and y values:
![image](https://github.com/user-attachments/assets/1e3fd4f1-935c-4748-85e0-e90d660e735f)

## Converting string literals to numerical values using label encoder:
![image](https://github.com/user-attachments/assets/62c12c26-09bd-4eac-b55d-dd1498e38b06)

## Accuracy:
![image](https://github.com/user-attachments/assets/92d68082-3077-41c8-9b04-44e17943e27e)

## Prediction:
![image](https://github.com/user-attachments/assets/b3273c75-2332-4f89-bde6-357ba131e20d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
