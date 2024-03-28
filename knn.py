import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
data = pd.read_csv('column_2C_weka.csv')
data.head(10)

A = data[data["class"] == "Abnormal"]
N = data[data["class"] == "Normal"]



y=data["class"].values
x_data =data.drop(["class"],axis=1) 

data["class"]=[1 if each == "Abnormal" else 0 for each in data["class"]]


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=1)


# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 8) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(8,knn.score(x_test,y_test)))

score_list=[]
for i in range(1,20):
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,20),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
score_list