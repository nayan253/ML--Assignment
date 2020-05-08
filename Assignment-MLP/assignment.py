#IMPORTS
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import matplotlib.pyplot as plt
import seaborn as sns
import random

#%matplotlib inline

#PARAMETERS
SEED = 2020
np.random.seed(SEED)
random.seed(SEED)

numInput = 100
numFeatures = 4

#DATA GENERATION
data = np.random.random((numInput,numFeatures+1))
data[:(numInput//2),4] = 0
data[(numInput//2):,4] = 1
np.random.shuffle(data)

df = pd.DataFrame(data, columns= ['Feature'+str(i) for i in range(1,5)]+['label'])
df.to_csv("RandomDataSet.csv", index=False)

"""part i. Read the dataset from file"""
df = pd.read_csv('RandomDataSet.csv')
df.head(n=10) , df.tail(n=10)

data = df.to_numpy()
X = data[:,:-1]
y = data[:,-1]

"""ii. Split the data into train and test. Ensure the data is split in the same way every time the program runs."""
spl = 0.2
N = len(data)
sample = int(spl*N)

#shuffle the whole matrix data and then split
np.random.shuffle(data)
X_train, X_test, y_train, y_test = X[:75,:], X[75:, :], y[:75, ], y[75:,]
"""iii.	Initialise the weights of the perceptron, learning rate and epochs"""

weights = np.random.randn((numFeatures+1))  ## 1 for bias
epoch = 1000
learning_rate = 0.1

"""iv.	Define the activation function"""

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# sigmoid(np.asarray([5,7]))



"""v.	Train the model i.e. Learn the weights of the perceptron on the training data."""

X_train.shape , weights[1:].shape , (X_train@weights[1:] + weights[0]).shape

def predict(X):
  return sigmoid(X@weights[1:] + weights[0])

def train(X_train, y_train):
  for _ in range(epoch):
    prediction = predict(X_train)
    weights[1:] += learning_rate * ( (y_train - prediction) @ X_train )
    weights[0] += learning_rate * np.sum(y_train - prediction)

train(X_train, y_train)



"""vi.	Print the learned weights and the hyperparameters (epoch and learning rate)"""

print("Weights = " , weights)
print("Epochs = " , epoch)
print("Learning Rate = " , learning_rate)



"""vii.	Predict the outputs on train and test data"""

def predictOutput(X):
  pred = predict(X)
  return (pred < 0.5).astype(int)

pred_train = predictOutput(X_train)
pred_test = predictOutput(X_test)



"""viii.	Print the confusion matrix, accuracy, precision, recall on train and test data"""

np.sum((pred_train == y_train) & (y_train == 1))

def printStats(testLabel, predicted):
  TP = np.sum((pred_train == y_train) & (y_train == 1))
  TN = np.sum((pred_train == y_train) & (y_train == 0))
  FP = np.sum((pred_train != y_train) & (y_train == 0))
  FN = np.sum((pred_train != y_train) & (y_train == 1))
  tot = TP + TN + FP + FN
  accuracy = (TP + TN) / tot
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)

  print("Accuracy = " , accuracy)
  print("Precision = " , precision)
  print("Recall = " , recall)

def ShowConfusionMatrix(testLabel, predicted):
    data = {'Actual Label': testLabel, 'Predicted Label': predicted}
    df = pd.DataFrame(data, columns=data.keys())
    confusion_matrix = pd.crosstab(df['Actual Label'], df['Predicted Label'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize = (10,5))
    sns.heatmap(confusion_matrix, annot=True)

print("TRAIN STATS")
printStats(y_train , pred_train)
ShowConfusionMatrix(y_train , pred_train)

print("TEST STATS")
printStats(y_test , pred_test)
ShowConfusionMatrix(y_test , pred_test)

