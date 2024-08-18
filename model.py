import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_df = pd.read_csv(r"C:\Users\recep\OneDrive\Masaüstü\sonrar_data\Copy of sonar data.csv", header= None)

mean_values = sonar_df.groupby(60).mean()

#print(mean_values)

X = sonar_df.drop(columns=60, axis=1)
Y = sonar_df[60]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size= 0.6, stratify= Y, random_state= 1) 

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("accuracy on training data: ", training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("accuracy on testing data: ", testing_data_accuracy)

input_data = (0.0209,0.0278,0.0115,0.0445,0.0427,0.0766,0.1458,0.1430,0.1894,0.1853,0.1748,0.1556,0.1476,0.1378,0.2584,0.3827,0.4784,0.5360,0.6192,0.7912,0.9264,1.0000,0.9080,0.7435,0.5557,0.3172,0.1295,0.0598,0.2722,0.3616,0.3293,0.4855,0.3936,0.1845,0.0342,0.2489,0.3837,0.3514,0.2654,0.1760,0.1599,0.0866,0.0590,0.0813,0.0492,0.0417,0.0495,0.0367,0.0115,0.0118,0.0133,0.0096,0.0014,0.0049,0.0039,0.0029,0.0078,0.0047,0.0021,0.0011)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print("the object is a rock")
else:
    print("the object is a mine")

