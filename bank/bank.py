# 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# load dataset
bank = pd.read_csv("bank-additional-full.csv", sep=";", header=0)

# transform categorical data into numerical
for col in bank:
	if bank[col].dtypes == "object":
		lb = LabelEncoder()
		bank[col] = lb.fit_transform(bank[col])

# separate labels into a new dataset
labels = bank['y']
del bank['y']

# split bank dataset into training and test data (60% training / 40% test)
trainRatio = .5

bankTrain = bank[:int(trainRatio*len(bank.index))]
labelTrain = labels[:int(trainRatio*len(bank.index))]

bankTest = bank[int(trainRatio*len(bank.index)):]
labelTest = labels[int(trainRatio*len(bank.index)):]

# Fit some learning algorithms and print results
alg = GaussianNB()
test = alg.fit(bank, labels)
print("Gaussian Naive Bayes accuracy:%3.0f%%" 
	% (test.score(bankTest, labelTest) * 100))


alg = Perceptron(tol=1e-3, random_state=0)
test = alg.fit(bank, labels)
print("Perceptron accuracy:%3.0f%%" 
	% (test.score(bankTest, labelTest) * 100))


alg = tree.DecisionTreeClassifier(random_state=0)
test = alg.fit(bank, labels)
export = tree.export_graphviz(test, out_file='tree.dot')
print("Decision Tree Classifier accuracy: %3.0f%%" 
	% (test.score(bankTest, labelTest) * 100))


alg = MLPClassifier()
test = alg.fit(bank, labels)
print("Multilayer Perceptron accuracy: %3.0f%%" 
	% (test.score(bankTest, labelTest) * 100))


alg = BaggingClassifier(KNeighborsClassifier(),
 max_samples=0.5, max_features=0.5)
test = alg.fit(bank, labels)
print("Bagged K Nearest accuracy: %3.0f%%" 
	% (test.score(bankTest, labelTest) * 100))

