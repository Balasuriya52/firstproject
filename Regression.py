import pandas as pd
import numpy as np
def pout(v1,v2,v3,v4,v5):
    out = rg.predict(np.array([float(v1),float(v2),float(v3),float(v4),float(v5)]).reshape(1,-1))
    return int(out[0])

data = pd.read_csv('F:\\MLOPS\\backend\\Student_Performance.csv')
data = data.dropna()
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
data["Extracurricular_Activities"] = l.fit_transform(data["Extracurricular_Activities"])
x = data.iloc[:,[0,1,2,3,4]]
y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 1,shuffle = True)

from sklearn.linear_model import LinearRegression
rg = LinearRegression()
rg.fit(xtrain,ytrain)