
# coding: utf-8

# # Classification of Wisconsin Diagnostic Breast Cancer Data(UCI repository)

# This is the code for problem P1.<br>
# In this problem, the objective is to learn a decision tree for classification of the given data.For this, the DecisionTreeClassifier model of the Scikit Learn python library is employed.The built decision tree using the training data is visualized using the graphviz library. 

# In[1]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz


# Here, the training dataset and the corresponding labels are loaded from "trainX.csv" and "trainY.csv" respectively from "P1_data/P1_data/" directory.

# In[2]:


train_data=np.genfromtxt('P1_data/P1_data/trainX.csv',delimiter=',')
train_labels=np.genfromtxt('P1_data/P1_data/trainY.csv',delimiter=',')
feature_means=train_data[:,0:9]
feature_std=train_data[:,10:19]
feature_mode=train_data[:,20:29]


# Here, a DecisionTreeClassifer model is defined.Its parameters like decision criterion is set to 'entropy' to use Shannon entropy based on information gain. This parameter can also be set to 'gini' to use gini index as decision criterion. The random state is initialized to an integer(say 10) for randomizing the data for better accuarcy.Then, the classifier model is trained using the training data using its method 'fit' to build the tree. 

# In[3]:


clf = DecisionTreeClassifier(criterion='entropy',random_state=10)
clf = clf.fit(train_data,train_labels)


# Using graphiz library, the learned decision tree is visualized.The corresponding features used for building the tree are marked. 

# In[4]:


feature_names=['Radius(mean)', 'Texture(mean)', 'Perimeter(mean)', 'Area(mean)', 'Smoothness(mean)', 'Compactness(mean)', 'Concavity(mean)', 'Number of concave portions of contour(mean)', 'Symmetry(mean)', 'Fractal dimension(mean)',
               'Radius(std)', 'Texture(std)', 'Perimeter(std)', 'Area(std)', 'Smoothness(std)', 'Compactness(std)', 'Concavity(std)', 'Number of concave portions of contour(std)', 'Symmetry(std)', 'Fractal dimension(std)',
               'Radius(mode)', 'Texture(mode)', 'Perimeter(mode)', 'Area(mode)', 'Smoothness(mode)', 'Compactness(mode)', 'Concavity(mode)', 'Number of concave portions of contour(mode)', 'Symmetry(mode)', 'Fractal dimension(mode)']
target_names=['Malignant','Benign']
dot_data = export_graphviz(clf, out_file=None,feature_names=feature_names,class_names=target_names,filled=True,rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data) 
graph


# Here, the test dataset and the corresponding labels are loaded from "testX.csv" and "testY.csv" respectively from "P1_data/P1_data/" directory.

# In[5]:


test_data=np.genfromtxt('P1_data/P1_data/testX.csv',delimiter=',')
test_labels=np.genfromtxt('P1_data/P1_data/testY.csv',delimiter=',')
test_size=test_data.shape


# The predict method of the classifier is used here to predict the labels of the test data.Comparing the predicted labels with the actual labels the confusion matrix and misclassification rates are calculated.The total number of nodes and leaves present are also computed.

# In[6]:


predicted_labels=clf.predict(test_data)
tp=0
tn=0
fp=0
fn=0
for i in range(test_size[0]):
    if predicted_labels[i]==1:
        if(predicted_labels[i]==test_labels[i]):
            tp+=1
        else:
            fp+=1
    else:
        if(predicted_labels[i]==test_labels[i]):
            tn+=1
        else:
            fn+=1

conf_mat=[[tn, fp],[fn ,tp]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("Misclassification Rate of Benign:",end="")
print(false_pos_rate)
print("Misclassification Rate of Malignant:",end="")
print(false_neg_rate)

print("Total number of nodes:",end="")
print(clf.tree_.node_count)
print("Number of leaf nodes:",end="")
Out=clf.apply(train_data)
print(len(set(Out)))


# Now, the training size is varied from 10% to 90% and the corresponding decision trees are learned from the limited training data.For each case the percentage train and test accuracy of prediction of the learned tree on the test data are plotted. 

# In[7]:


plt.figure(figsize=(10,10))
train_size=train_data.shape
print("Percentage Accuracy Vs Training Size(Train->Blue,Test->Red):")
for i in range(1,11):
    tr_size=int(0.1*i*train_size[0])
    tr_data=train_data[0:tr_size,:]
    tr_labels=train_labels[0:tr_size]
    clf = DecisionTreeClassifier(criterion='entropy',random_state=10)
    clf = clf.fit(tr_data,tr_labels)
    predicted_labels=clf.predict(test_data)
    predicted_train_labels=clf.predict(train_data)
    accuracy_test=accuracy_score(test_labels,predicted_labels)
    accuracy_train=accuracy_score(train_labels,predicted_train_labels)
    plt.plot(tr_size,accuracy_test*100,'ro')
    plt.plot(tr_size,accuracy_train*100,'bo')
plt.show()


# From the data,it is seen that on increasing the train size the train accuracy increases.But the same cannot be said about test accuracy.In general test accuracy should increase with increasing training size.
