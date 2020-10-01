#import
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading the dataset into the variable
iris=datasets.load_iris()

# extracting the data ie-feature and target ie-lable
feature=iris.data
lable=iris.target

print(feature[0],lable[0])

#training the classifier
clf=KNeighborsClassifier()
clf.fit(feature,lable)#this line will be able to classsify the new data 

pred=clf.predict([[5.9,2.1,11.1,12]])
print(pred)

