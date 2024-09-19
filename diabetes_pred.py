import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib as plt
import seaborn as sns

#Data Collection & Analysis
data = pd.read_csv("diabetes.csv")

data.head()

data.shape

data.describe()

data['Outcome'].value_counts()

data.groupby('Outcome').mean()

#Separating the data and the labels
from sklearn.model_selection import train_test_split

X = data.drop(['Outcome'], axis =1)
y = data['Outcome']

#Data Standardization
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
y =data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify =y, random_state = 2)

#Training the Model
classifier = svm.SVC(kernel='linear')

#training the SVM
classifier.fit(X_train, y_train)

#Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy Score of the training data:' ,training_data_accuracy)

#Accuracy Score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy Score of the test data:', test_data_accuracy)

#Making a predictive System
#Using an example from a dataset 
input_data = (
    int(input("Enter number of Pregnancies (0 to 17): ")),
    float(input("Enter Glucose level (0 to 199): ")),
    float(input("Enter Blood Pressure (0 to 122): ")),
    float(input("Enter Skin Thickness (0 to 99): ")),
    float(input("Enter Insulin level (0 to 846): ")),
    float(input("Enter BMI (0 to 67.1): ")),
    float(input("Enter Diabetes Pedigree Function (0.078 to 2.42): ")),
    int(input("Enter Age (21 to 81): "))
)

#Changing the input data to numy array
input_data_as_numpy_array = np.asarray(input_data)

#Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#Standardise the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] ==0):
    print('The Patient is possibly Not Diabetic')
else:
    print('The Patient is possibly Diabetic')


