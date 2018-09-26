import numpy as np
import math
from numpy.linalg import inv
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt

def multivariate_gaussian(x,mu,sigma):
    n=x.shape
    x_bar=np.subtract(x,mu)
    x_bar_vec=np.array([x_bar])
    sigma_inv=np.linalg.inv(sigma)
    index=np.matmul(x_bar_vec,np.matmul(sigma_inv,x_bar_vec.T))
    num=math.exp(-0.5*index)
    den=(((2*np.pi)**(n[0]))*np.linalg.det(sigma))**0.5
    return num/den

def bin_classify(x,p0,M0,S0,p1,M1,S1):
    if p0*multivariate_gaussian(x,M0,S0)>p1*multivariate_gaussian(x,M1,S1):
        return 0
    else:
        return 1

def visualize(D,p1,M1,C1,p2,M2,C2):
    delta = 0.01
    x = np.arange(-15.0, 15.0, delta)
    y = np.arange(-20.0, 20.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, C1[0][0], C1[1][1], M1[0], M1[1],C1[0][1])
    Z2 = mlab.bivariate_normal(X, Y, C2[0][0], C2[1][1], M2[0], M2[1],C2[0][1])
    #Z=p1*Z1-p2*Z2
    plt.figure()
    CS1 = plt.contour(X, Y, Z1)
    CS2 = plt.contour(X, Y, Z2)
    #CS = plt.contour(X, Y, Z)

    X1, Y1 = np.mgrid[-15:15:3000j, -20:20:4000j]
    points = np.c_[X1.ravel(), Y1.ravel()]

    invC = np.linalg.inv(C1)
    v = points - M1
    g1 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C1))+np.log(p1)
    g1.shape = 3000, 4000

    invC = np.linalg.inv(C2)
    v = points - M2
    g2 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C2))+np.log(p2)
    g2.shape = 3000, 4000

    #plt.figure(1)
    #f, ax = plt.subplots(figsize=(15, 15))
    plt.contour(X1, Y1,g1-g2,levels=[0], cmap="Greys_r")
    plt.show()


# def plot_discriminant(D,p1,M1,C1,p2,M2,C2):
#     X, Y = np.mgrid[-5:5:1000j, -20:20:4000j]
#     points = np.c_[X.ravel(), Y.ravel()]
#
#     invC = np.linalg.inv(C1)
#     v = points - M1
#     g1 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C1))+np.log(p1)
#     g1.shape = 1000, 4000
#
#     invC = np.linalg.inv(C2)
#     v = points - M2
#     g2 = -0.5*np.sum(np.dot(v, invC) * v, axis=1) - D*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C2))+np.log(p2)
#     g2.shape = 1000, 4000
#
#     plt.figure(1)
#     f, ax = plt.subplots(figsize=(15, 15))
#     ax.contour(X, Y,g1-g2,levels=[0], cmap="Greys_r")
#     plt.show()
#
# def plot_iso_prob_contours(p1,M1,S1,p2,M2,S2):
#     delta = 0.01
#     x = np.arange(-5.0, 5.0, delta)
#     y = np.arange(-20.0, 20.0, delta)
#     X, Y = np.meshgrid(x, y)
#
#     Z1 = mlab.bivariate_normal(X, Y, S1[0][0], S1[1][1], M1[0], M1[1],S1[0][1])
#     Z2 = mlab.bivariate_normal(X, Y, S2[0][0], S2[1][1], M2[0], M2[1],S2[0][1])
#     #Z=p1*Z1-p2*Z2
#     plt.figure(1)
#     CS1 = plt.contour(X, Y, Z1)
#     CS2 = plt.contour(X, Y, Z2)
#     #CS = plt.contour(X, Y, Z)
#     #plt.clabel(CS, inline=1, fontsize=10)
#     #plt.title('Simplest default with labels')
#     #plt.show()


data=np.genfromtxt("P2_data/P2_data/P2_train.csv",delimiter=',')
test_data=np.genfromtxt("P2_data/P2_data/P2_test.csv",delimiter=',')
#print(data[:,1])
train_size=data.shape
test_size=test_data.shape

input=data[:,0:2]
labels=data[:,2:3]

test_input=test_data[:,0:2]
test_labels=test_data[:,2:3]

#print(input[0,:])
x=data[:,0]
y=data[:,1]

m0=np.zeros(train_size[1]-1)
n0=0
sigma0=np.zeros((train_size[1]-1,train_size[1]-1))

m1=np.zeros(train_size[1]-1)
n1=0
sigma1=np.zeros((train_size[1]-1,train_size[1]-1))

for i in range(train_size[0]):
    if labels[i]==0:
        m0=np.add(m0,input[i,:])
        n0+=1
    else:
        m1=np.add(m1,input[i,:])
        n1+=1
m0=m0/n0
m1=m1/n1
x_0=np.zeros(n0)
y_0=np.zeros(n0)
x_1=np.zeros(n1)
y_1=np.zeros(n1)
p=0
q=0

for i in range(train_size[0]):
    if labels[i]==0:
        x_0[p]=input[i][0]
        y_0[p]=input[i][1]
        p+=1
        X0=np.subtract(input[i],m0)
        X_vec=np.array([X0])
        pd=np.matmul(X_vec.T,X_vec)
        sigma0=np.add(sigma0,pd)
    else:
        x_1[q]=input[i][0]
        y_1[q]=input[i][1]
        q+=1
        X1=np.subtract(input[i],m1)
        X_vec=np.array([X1])
        pd=np.matmul(X_vec.T,X_vec)
        sigma1=np.add(sigma1,pd)
sigma0=sigma0/(n0-1)
sigma1=sigma1/(n1-1)

prob_C0=n0/(n0+n1)
prob_C1=1-prob_C0

#print(sigma0)
#print(sigma1)

####CaseA###
print("\nCase A:")
tp=0
tn=0
fp=0
fn=0

var_x=np.var(input[:,0:1],axis=0)
#var_y=np.var(input[:,1:2],axis=0)
common_sigma=np.zeros((2,2))
common_sigma[0][0]=var_x
common_sigma[1][1]=var_x

pred_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    pred_labels[i]=bin_classify(test_input[i,:],prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)
#print(pred_labels)
for i in range(test_size[0]):
    if pred_labels[i]==0:
        if pred_labels[i]==test_labels[i]:
            tp+=1
        else:
            fp+=1
    else:
        if pred_labels[i]==test_labels[i]:
            tn+=1
        else:
            fn+=1
conf_mat=[[tp,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
visualize(2,prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)


####CaseA(1)###
print("\nCase A(1):")
tp=0
tn=0
fp=0
fn=0

#var_x=np.var(input[:,0:1],axis=0)
var_y=np.var(input[:,1:2],axis=0)
common_sigma=np.zeros((2,2))
common_sigma[0][0]=var_y
common_sigma[1][1]=var_y

pred_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    pred_labels[i]=bin_classify(test_input[i,:],prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)
#print(pred_labels)
for i in range(test_size[0]):
    if pred_labels[i]==0:
        if pred_labels[i]==test_labels[i]:
            tp+=1
        else:
            fp+=1
    else:
        if pred_labels[i]==test_labels[i]:
            tn+=1
        else:
            fn+=1
conf_mat=[[tp,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
visualize(2,prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)

####CaseB###
print("\nCase B:")
tp=0
tn=0
fp=0
fn=0

var_x=np.var(x);
var_y=np.var(y)
common_sigma=np.zeros((2,2))
common_sigma[0][0]=var_x
common_sigma[1][1]=var_y
#print(common_sigma)

pred_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    pred_labels[i]=bin_classify(test_input[i,:],prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)
#print(pred_labels)
for i in range(test_size[0]):
    if pred_labels[i]==0:
        if pred_labels[i]==test_labels[i]:
            tp+=1
        else:
            fp+=1
    else:
        if pred_labels[i]==test_labels[i]:
            tn+=1
        else:
            fn+=1
conf_mat=[[tp,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
visualize(2,prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)

####CaseC###
print("\nCase C:")
tp=0
tn=0
fp=0
fn=0

common_sigma=np.cov(x,y)
#print(common_sigma)

pred_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    pred_labels[i]=bin_classify(test_input[i,:],prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)
#print(pred_labels)
for i in range(test_size[0]):
    if pred_labels[i]==0:
        if pred_labels[i]==test_labels[i]:
            tp+=1
        else:
            fp+=1
    else:
        if pred_labels[i]==test_labels[i]:
            tn+=1
        else:
            fn+=1
conf_mat=[[tp,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
visualize(2,prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)

####CaseD####
print("Case D:")
tp=0
tn=0
fp=0
fn=0

pred_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    pred_labels[i]=bin_classify(test_input[i,:],prob_C0,m0,sigma0,prob_C1,m1,sigma1)
#print(pred_labels)
for i in range(test_size[0]):
    if pred_labels[i]==0:
        if pred_labels[i]==test_labels[i]:
            tp+=1
        else:
            fp+=1
    else:
        if pred_labels[i]==test_labels[i]:
            tn+=1
        else:
            fn+=1
conf_mat=[[tp,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
visualize(2,prob_C0,m0,sigma0,prob_C1,m1,sigma1)

#plot_iso_prob_contours(prob_C0,m0,sigma0,prob_C1,m1,sigma1)
#plot_discriminant(2,prob_C0,m0,sigma0,prob_C1,m1,sigma1)

####CaseE###

print("\nCase E:")
tp=0
tn=0
fp=0
fn=0

var_x=np.var(input[:,0:1],axis=0)
var_y=np.var(input[:,1:2],axis=0)
C0=np.cov(x_0,y_0)
C1=np.cov(x_1,y_1)
common_sigma=np.zeros((2,2))
common_sigma[0][0]=var_x
common_sigma[1][1]=var_y
common_sigma[0][1]=C0[0][1]
common_sigma[1][0]=C1[1][0]

pred_labels=np.zeros(test_size[0])
for i in range(test_size[0]):
    pred_labels[i]=bin_classify(test_input[i,:],prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)
#print(pred_labels)
for i in range(test_size[0]):
    if pred_labels[i]==0:
        if pred_labels[i]==test_labels[i]:
            tp+=1
        else:
            fp+=1
    else:
        if pred_labels[i]==test_labels[i]:
            tn+=1
        else:
            fn+=1
conf_mat=[[tp,fp],[fn,tn]]
print("Confusion Matrix:")
print(np.array(conf_mat))
false_pos_rate=fp/(fp+tn)
false_neg_rate=fn/(tp+fn)
print("False Negative Rate:",end="")
print(false_neg_rate)
print("False Positive Rate:",end="")
print(false_pos_rate)
visualize(2,prob_C0,m0,common_sigma,prob_C1,m1,common_sigma)
