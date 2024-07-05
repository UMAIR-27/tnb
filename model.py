import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("electricity_usage.csv")
col = ['Fan','Refrigerator','AirConditioner', 'Fridge', 'Television', 'MonthlyHours', 'ElectricityBill']
data = data[col]

target='ElectricityBill'
X= data.drop(columns = [target])
y=data[target]

model= LinearRegression()
model.fit(X,y)

pickle.dump(model, open("LRmodel.pkl","wb"))