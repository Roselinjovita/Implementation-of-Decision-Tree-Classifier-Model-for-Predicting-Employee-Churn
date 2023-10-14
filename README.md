# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.ROSELIN MARY JOVITA
RegisterNumber:  212222230122
*/


import pandas as pd
data=pd.read_csv("/content/Employee.csv")

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
accuracy= metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
# data.head()

![Screenshot 2023-10-14 205720](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/fdbb3b50-c6eb-44ce-9594-5f2c7d628cd1)

# data.info()

![Screenshot 2023-10-14 205735](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/73114c42-c06c-4482-8eb4-0fab603a3b5c)

# isnull() and sum()

![Screenshot 2023-10-14 205745](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/769340ec-1d62-4772-8620-1d4b900b7193)

# data value counts()

![Screenshot 2023-10-14 205751](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/4d52e057-ebd2-4c86-bddc-b0a0744dc5d3)

# data head() for salary

![Screenshot 2023-10-14 205809](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/176d00f5-356f-472d-867f-511492b37082)

# x.head()

![Screenshot 2023-10-14 205838](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/5743beab-0398-4f43-98a7-cb7b7b6b8e6e)

# accuracy value

![Screenshot 2023-10-14 210454](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/9f1035e5-cef1-4f83-a9cd-d22773b47dc1)

# data prediction

![Screenshot 2023-10-14 205901](https://github.com/Roselinjovita/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104296/535baeb1-6238-4e53-82a0-a2618d3d5aed)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
