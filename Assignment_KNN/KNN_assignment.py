import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics

df = pd.read_csv('/home/harshita/diabetes.csv')

df.head()

df.isnull().sum()
df.isna().sum()

df_mod=df.fillna(0)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = df_mod[features]
y = df_mod.Outcome

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(xTrain,yTrain)
prediction = model.predict(xTest)

confusion_matrix = metrics.confusion_matrix(yTest, prediction)
confusion_matrix

plt.figure()
plt.matshow(confusion_matrix, cmap='Pastel1')

for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, confusion_matrix[x, y])
        
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()

TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

print("Sensitivity: %.4f" % (TP / float(TP + FN)))
print("Specificy  : %.4f" % (TN / float(TN + FP)))

def standardizeDataset(xTrain, xTest):
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
#The function below takes the dataset as the argument leaving out the target column and returns standardized dataset. Original dataset had 70 features that reduced to 14 after running the code below. 
#The preprocessing module further provides a utility class StandardScaler that implements the Transformer API to compute the mean and standard deviation on a training set so as to be able to later reapply the same transformation on the testing set.
    pca = PCA(n_components = 0.55)
    xTrainStd = pca.fit_transform(xTrain)
    xTestStd = pca.transform(xTest)
    return xTrainStd, xTestStd
# you might subtract the mean and divide by the standard deviation, thereby obtaining a “standard normal” random variable with mean 0 and standard deviation 1
#The function below takes the dataset as the argument leaving out the target column and returns standardized dataset.

xTrainStd, xTestStd = standardizeDataset(xTrain, xTest)

print("Columns before PCA")
print(xTrain.shape[1])
print("\nColumns after PCA")
print(xTrainStd.shape[1])

df.duplicated(keep='first') 

df = df.fillna(df.mean())
## normalising all the column to [0-1] for having same effect in calculations
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df.isnull().sum()
df.isna().sum()

print(x_scaled.shape , x_scaled.shape[0]*0.8)

x_train = x_scaled[0:600,0:8]
y_train = x_scaled[0:600,8].astype(int)
x_test = x_scaled[600:750,0:8]
y_test = x_scaled[600:750,8].astype(int)
print(x_train.shape , y_train.shape , x_test.shape, y_test.shape)

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.dot(vector1-vector2 , vector1-vector2))

D = np.zeros((150,600))
for i in range(150):
    for j in range(600):
        D[i,j] = euclidean_distance(x_train[j] , x_test[i])

np.sum(yTrain[D[i].argsort()[:5]])

## generating test prediction
def KNN(k):
#     k = 5
    y_pred = np.zeros((150,) , dtype = int)
    for i in range(150):
        topknear = D[i].argsort()[:k]
        y_of_neighbour = y_train[topknear]
        y_sum_neighbour = int(np.sum(y_of_neighbour))
        if y_sum_neighbour < (k/2):
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    accuracy = accuracy_score(y_pred , y_test)
    print("ACCURACY for k = " + str(k) + " is :" , accuracy*100 , "%")
    return y_pred

KNN(1) , KNN(3) , KNN(5), KNN(7), KNN(9), KNN(11);

model = LogisticRegression()
model.fit(x_train,y_train)
prediction1 = model.predict(x_test)

confusion_matrix1 = metrics.confusion_matrix(y_test, prediction1)
confusion_matrix1

plt.figure()
plt.matshow(confusion_matrix1, cmap='Pastel1')

for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, confusion_matrix1[x, y])
        
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()

# we ablated (i.e. removed) some of the repeated elements and the 
# following results are below:-
# True Positives (TP): (26) we correctly predicted that they do have diabetes
# True Negatives (TN): (90) we correctly predicted that they don't have diabetes
# False Positives (FP): (26) we incorrectly predicted that they do have diabetes
# False Negatives (FN): (8) we incorrectly predicted that they don't have diabetes



