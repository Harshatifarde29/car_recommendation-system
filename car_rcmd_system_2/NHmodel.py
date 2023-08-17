import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


data = pd.read_csv("car1.csv")
print(data)

feature = data[["person", "Fuel Type", "budget"]]
target = data["car"]

#handel cat data

nfeature = pd.get_dummies(feature)
print(nfeature)

#train and test
x_train, x_test, y_train, y_test = train_test_split(nfeature, target,random_state=120)

model = RandomForestClassifier()
model.fit(x_train,y_train)

print("train score is ",model.score(x_train, y_train))
print("test score is ", model.score(x_test, y_test))

cr = classification_report(y_test, model.predict(x_test))
print(cr)

person = float(input("enter no of person "))

budget = float(input("enter your budget ! minimun is 250 "))
fuel = int(input("Type 1 for Diesel ,2 for Electrical and 3 for petrol :"))
if fuel ==  1:
   d =[person, budget,1,0,0]
elif fuel == 2:
   d =[person,budget,0,1,0]
elif fuel == 3:
   d =[person,budget,0,0,1]

res = model.predict([d])
print("car suitable for u is ", res)

'''f = open("CRP.model", "wb")
pickle.dump(model, f)
f.close()'''