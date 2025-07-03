import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import os

data=pd.read_csv("titanic.csv")
data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
data.head()
x=data.drop("Survived",axis=1)
y=data['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.head()
x_train.isnull().sum()

step1= ColumnTransformer([
    ('Age_Imputer',SimpleImputer(),[2]),
    ('Embarked_Imputer',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')

step2= ColumnTransformer([
    ('OHE_Sex_Embarked',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
],remainder="passthrough")

step3= ColumnTransformer([
    ('Scaling',MinMaxScaler(),slice(0,10))
])

step4= SelectKBest(score_func=chi2,k=8)

step5= DecisionTreeClassifier()

pipeline = Pipeline([
    ('Step1',step1),
    ('Step2',step2),
    ('Step3',step3),
    ('Step4',step4),
    ('Step5',step5)
])

y_pred=pipeline.predict(x_test)
accuracy_score(y_test,y_pred)

os.makedirs("model",exist_ok=True)
pickle.dump(pipeline,open('model/pipeline.pkl','wb'))