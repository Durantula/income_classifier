import numpy as np
import pandas
from sklearn import svm
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  


# feature scaling implementation offered by SciLearn
scaler = StandardScaler() 

# import classified data
data = pandas.read_csv("./data/classifiedInput.csv", sep=",")
classifiedCategory = pandas.read_csv("./data/classifiedCategory.csv", sep=",")
unclassifiedData = pandas.read_csv("./data/evaluation_classification_data.csv", sep=",")


#prepare numerical data 
X = data._get_numeric_data()
y = classifiedCategory.values.flatten()


# feature sclae the data 
scaler.fit(X)
X = scaler.transform(X) 

# train model on classified data. Switch between MLP and SVM

# clf = svm.SVC(gamma='scale')
clf = MLPClassifier(alpha=0.01)

clf.fit(X,y)

#feature scale evaluation data 
unclassifiedData = scaler.transform(unclassifiedData)
output = clf.predict(unclassifiedData)


# Save data to output.csv
a = np.asarray(output)
np.savetxt("output.csv",a,delimiter=",")

