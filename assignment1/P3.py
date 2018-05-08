import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import linear_model
#from sklearn.preprocessing import PolynomialFeatures

def normalize(a):
	a=a-np.mean(a)
	a=a/np.std(a)
	return a

data = pd.read_csv("Wage_dataset.csv")
year = np.array(data["year"])
#year = year-np.mean(year)
#year = year/np.std(year)
age = np.array(data["age"])
#age = age-np.mean(age)
#age = age/np.std(age)
education = np.array(data["education"])
#education = education-np.mean(education)
#education = education/np.std(education)
wage = np.array(data["wage"])

# data=np.genfromtxt('Problem_3/Problem_3/Wage_dataset.csv',delimiter=',')
#
# age = np.array(data[:,1])
# #age = normalize(age)
# year = np.array(data[:,0])
# #year = normalize(year)
# education = np.array(data[:,4:5])
# #education = normalize(education)
# wage = np.array(data[:,10:11])


##################

loss_array_age = np.zeros(shape=(3000,1))
min_loss_age = math.inf
optimal_n = 0
plt.figure()
for n in range(150,200):
	age_matrix = np.zeros(shape=(3000,n))
	for i in range(3000):
		for j in range(n):
			age_matrix[i][j] = age[i]**j
	age_weight = np.matmul(np.linalg.pinv(age_matrix),wage)
	predicted_wage_age = np.matmul(age_matrix,age_weight)
	for k in range(3000):
		loss_array_age[k] = (wage[k]-predicted_wage_age[k])**2
	loss_age = loss_array_age.sum()
	#loss_age=(2*loss_age/n)**0.5
	print(loss_age)
	plt.plot(n,loss_age,'ro')
	# if (n==1):
	# 	min_loss_age = loss_age
	if (loss_age <= min_loss_age):
		min_loss_age = loss_age
		optimal_n = n

plt.show()
print(optimal_n)

plt.figure()
plt.scatter(age,wage)
plt.scatter(age,predicted_wage_age,c='g')
plt.show()

N = optimal_n
age_matrix = np.zeros(shape=(3000,N))
loss_array_age = np.zeros((3000,1))
for i in range(3000):
	for j in range(N):
		age_matrix[i][j] = age[i]**j
age_weight = np.matmul(np.linalg.pinv(age_matrix),wage)
predicted_wage_age = np.matmul(age_matrix,age_weight)

# poly = PolynomialFeatures(degree=N)
# Age=age.reshape(-1,1)
# x_features=poly.fit_transform(Age)
# #print(x_features)
# clf = linear_model.LinearRegression()
# clf.fit(x_features,wage)
# pred=clf.predict(x_features)

x = np.arange(3000)
plt.plot(x,wage,marker='o')
plt.plot(x,predicted_wage_age,marker='x')
plt.show()
# error=np.subtract(predicted_wage_age,wage)
# loss_age=np.dot(error.T,error)

for k in range(3000):
	loss_array_age[k] = (wage[k]-predicted_wage_age[k])**2
loss_age = loss_array_age.sum()
#loss_age=(2*loss_age/N)**0.5
print(loss_age)


# poly = PolynomialFeatures(degree=N)
# Age=age.reshape(-1,1)
# x_features=poly.fit_transform(Age)
# #print(x_features)
# clf = linear_model.LinearRegression()
# clf.fit(x_features,wage)
# pred=clf.predict(x_features)
# plt.plot(x,pred,marker='*')
# plt.show()

#################

loss_array_year = np.zeros(3000)
min_loss_year = math.inf
optimal_m = 0
plt.figure()
for m in range(1,100):
	year_matrix = np.zeros(shape=(3000,m))
	for i in range(3000):
		for j in range(0,m):
			year_matrix[i][j] = year[i]**j
	year_weight = np.matmul(np.linalg.pinv(year_matrix),wage)
	predicted_wage_year = np.matmul(year_matrix,year_weight)
	for k in range(3000):
		loss_array_year[k] = (wage[k]-predicted_wage_year[k])**2
	loss_year = loss_array_year.sum()
	loss_year=(2*loss_year/m)**0.5
	plt.plot(m,loss_year,'ro')
	if (m==1):
		min_loss_year = loss_year
	if (loss_year < min_loss_year):
		min_loss_year = loss_year
		optimal_m = m

plt.show()
print(optimal_m)

plt.figure()
plt.scatter(year,wage)
plt.scatter(year,predicted_wage_year,c='g')
plt.show()

M=optimal_m
year_matrix = np.zeros((3000,M))
loss_array_year = np.zeros(3000)
for i in range(3000):
	for j in range(0,M):
		year_matrix[i][j] = year[i]**j

year_weight = np.matmul(np.linalg.pinv(year_matrix),wage)
predicted_wage_year = np.matmul(year_matrix,year_weight)
x = np.arange(3000)
plt.plot(x,wage,marker='o')
plt.plot(x,predicted_wage_year,marker='s')
plt.show()
for k in range(3000):
	loss_array_year[k] = (wage[k]-predicted_wage_year[k])**2
loss_year = loss_array_year.sum()
loss_year=(2*loss_year/M)**0.5
print(loss_year)


#############################


loss_array_education = np.zeros(shape=(3000,1))
min_loss_education = math.inf
optimal_l = 0
plt.figure()
for l in range(1,100):
	education_matrix = np.zeros(shape=(3000,l))
	for i in range(3000):
		for j in range(0,l):
			education_matrix[i][j] = education[i]**j
	education_weight = np.matmul(np.linalg.pinv(education_matrix),wage)
	predicted_wage_education = np.matmul(education_matrix,education_weight)
	for k in range(3000):
		loss_array_education[k] = (wage[k]-predicted_wage_education[k])**2
	loss_education = loss_array_education.sum()
	loss_education=(2*loss_education/l)**0.5
	plt.plot(l,loss_education,'ro')
	if (l==1):
		min_loss_education = loss_education
	if (loss_education < min_loss_education):
		min_loss_education = loss_education
		optimal_l = l
plt.show()
print(optimal_l)

plt.figure()
plt.scatter(education,wage)
plt.scatter(education,predicted_wage_education,c='g')
plt.show()

L=optimal_l
education_matrix = np.zeros((3000,L))
loss_array_education = np.zeros(3000)
for i in range(3000):
	for j in range(0,L):
		education_matrix[i][j] = education[i]**j

education_weight = np.matmul(np.linalg.pinv(education_matrix),wage)
predicted_wage_education = np.matmul(education_matrix,education_weight)
x = np.arange(3000)
plt.plot(x,wage,marker='o')
plt.plot(x,predicted_wage_education,marker='s')
plt.show()
for k in range(3000):
	loss_array_education[k] = (wage[k]-predicted_wage_education[k])**2

loss_education = loss_array_education.sum()
loss_education=(2*loss_education/L)**0.5
print(loss_education)
