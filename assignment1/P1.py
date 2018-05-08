import numpy as np
import math
from numpy.linalg import inv
import pandas as pd
#from scipy.stats import multivariate_normal
#from sklearn.metrics import confusion_matrix

def gaussian(x,mu,sigma):
    n=x.shape
    x_bar=np.subtract(x,mu)
    x_bar_vec=np.array([x_bar])
    sigma_inv=np.linalg.inv(sigma)
    index=np.matmul(x_bar_vec,np.matmul(sigma_inv,x_bar_vec.T))
    num=math.exp(-0.5*index)
    den=(((2*np.pi)**(n[0]))*np.linalg.det(sigma))**0.5
    return num/den


#df=pd.read_csv("P1_data/P1_data/P1_data_test.csv",header=None)
data=np.genfromtxt('P1_data/P1_data/P1_data_train.csv',delimiter=',')
labels=np.genfromtxt('P1_data/P1_data/P1_labels_train.csv',delimiter=',')
size=data.shape
#print(size[0])

count_five=0
mu_five=np.zeros(size[1])
sigma_five=np.zeros((size[1],size[1]))
count_six=0
mu_six=np.zeros(size[1])
sigma_six=np.zeros((size[1],size[1]))

for i in range(size[0]):
    if labels[i]==5:
        mu_five=np.add(mu_five,data[i])
        count_five+=1
    else:
        mu_six=np.add(mu_six,data[i])
        count_six+=1
mu_five=mu_five/count_five
mu_six=mu_six/count_six

for i in range(size[0]):
    if labels[i]==5:
        x=np.subtract(data[i],mu_five)
        x_vec=np.array([x])
        pd=np.matmul(x_vec.T,x_vec)
        sigma_five=np.add(sigma_five,pd)
    else:
        x=np.subtract(data[i],mu_six)
        x_vec=np.array([x])
        pd=np.multiply(x_vec.T,x_vec)
        sigma_six=np.add(sigma_six,pd)
sigma_five=sigma_five/(count_five-1)
sigma_six=sigma_six/(count_six-1)

#print(gaussian(data[1],mu_five,sigma_five))
#print(multivariate_normal.pdf(data[1],mu_five,sigma_five))

prob_C5=count_five/(count_five+count_six)
prob_C6=1-prob_C5

test_data=np.genfromtxt('P1_data/P1_data/P1_data_test.csv',delimiter=',')
test_labels=np.genfromtxt('P1_data/P1_data/P1_labels_test.csv',delimiter=',')
test_size=test_data.shape

### Empirical Case ###
tp=0
tn=0
fp=0
fn=0
predicted_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    if prob_C5*gaussian(test_data[i],mu_five,sigma_five)>prob_C6*gaussian(test_data[i],mu_six,sigma_six):
        predicted_labels[i]=5
        if(predicted_labels[i]==test_labels[i]):
            tp+=1
        else:
            fp+=1
    else:
        predicted_labels[i]=6
        if(predicted_labels[i]==test_labels[i]):
            tn+=1
        else:
            fn+=1

conf_mat=[[tp ,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
#print(confusion_matrix(test_labels, predicted_labels))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)

### Equal Sigma Case ####
tp=0
tn=0
fp=0
fn=0

count=0
common_mu=np.zeros(size[1])
common_sigma=np.zeros((size[1],size[1]))
for i in range(size[0]):
    common_mu=np.add(common_mu,data[i])
    count+=1
common_mu=common_mu/count
for i in range(size[0]):
    X=np.subtract(data[i],common_mu)
    X_vec=np.array([X])
    pd=np.matmul(X_vec.T,X_vec)
    common_sigma=np.add(common_sigma,pd)
common_sigma=common_sigma/(count-1)

#common_sigma=nv.cov()
predicted_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    if prob_C5*gaussian(test_data[i],mu_five,common_sigma)>prob_C6*gaussian(test_data[i],mu_six,common_sigma):
        predicted_labels[i]=5
        if(predicted_labels[i]==test_labels[i]):
            tp+=1
        else:
            fp+=1
    else:
        predicted_labels[i]=6
        if(predicted_labels[i]==test_labels[i]):
            tn+=1
        else:
            fn+=1

conf_mat=[[tp ,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
#print(confusion_matrix(test_labels, predicted_labels))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)

### Diagonal Sigma Case ####
tp=0
tn=0
fp=0
fn=0

for i in range(size[1]):
    for j in range(size[1]):
        if i!=j:
            common_sigma[i][j]=0
predicted_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    if prob_C5*gaussian(test_data[i],mu_five,common_sigma)>prob_C6*gaussian(test_data[i],mu_six,common_sigma):
        predicted_labels[i]=5
        if(predicted_labels[i]==test_labels[i]):
            tp+=1
        else:
            fp+=1
    else:
        predicted_labels[i]=6
        if(predicted_labels[i]==test_labels[i]):
            tn+=1
        else:
            fn+=1

conf_mat=[[tp ,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
#print(confusion_matrix(test_labels, predicted_labels))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
