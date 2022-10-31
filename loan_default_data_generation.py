
import numpy as np
import pandas as pd


'''Creating Mock Dataset'''


from faker import Faker as mock


mock.seed(58)
mock_func=mock(locale='en_US')


columns=['id','customer_name','state','loan_length','credit_score','customer_income','education',
                             'asset_ownership_over_200k','loan_amount','interest_rate','loan_status']


lst=[]
nrow=10000
for x in range(nrow):
    lst.append([x,mock_func.name(),mock_func.state(),mock_func.random_int(3,12),mock_func.random_int(620,720),
                   mock_func.random_int(80000,150000),mock_func.random_element(['masters','bachelors','high school','doctorate']),
                   mock_func.random_element([0,1]),mock_func.random_int(150000,300000),
                   mock_func.random_int(6,9)+mock_func.random_int(1,9)*0.1,mock_func.random_element([0,1])])


dataset=pd.DataFrame(lst,columns=columns)



dataset.head()


from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder()
encoded=encoder.fit_transform(dataset[['education']])


dataset[encoder.categories_[0]]=encoded.toarray()

dataset.head()


X=dataset.copy()
del X['education']
del X['id']
del X['loan_status']
del X['customer_name']
del X['state']
Y=dataset['loan_status']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression


train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.25,random_state=46)


logit=LogisticRegression(solver='liblinear')
logit.fit(train_x,train_y)


logit_predicted_y=logit.predict(test_x)


logit_confusion_matrix=confusion_matrix(test_y,logit_predicted_y)
logit_accuracy=accuracy_score(test_y,logit_predicted_y)

