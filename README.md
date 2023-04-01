# Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Yuvadarshini S
RegisterNumber: 212221230126 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('ex1.txt',header=None)


plt.scatter(data[0],data[1],color="green")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit prediction")


def computeCost(x,y,theta):
  """
  Take in a numpy array x, y, theta and generate the cost function of function using theta as parameter 
  in a linear regression model
  """

  m=len(y)
  h=x.dot(theta) #length of the training data
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err) # returning 3
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1);
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta) # call the function


def gradientDescent(x,y,theta,alpha,num_iters):
  """
  Take in numy array x, y and theta and update theta by taking number_iters gradient steps with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """

  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta -= descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  

theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")


plt.plot(J_history,color="orange")
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost Function using Gradient Descent")


plt.scatter(data[0],data[1],color="green")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="magenta")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of the city (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")


def predict(x, theta):
  """
  Takes in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions = np.dot(theta.transpose(),x)

  return predictions[0]
  
  
predict1 = predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))


predict2 = predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

*/
```

## Output:
### Profit Prediction graph
![s1](https://user-images.githubusercontent.com/93482485/229299252-4a473783-ec3b-4bd5-8726-1e733a016dd1.jpg)

### Compute Cost Value
![s2](https://user-images.githubusercontent.com/93482485/229299273-cf0c1428-b530-420c-986b-a8eddc4ebaab.jpg)

### h(x) Value
![s3](https://user-images.githubusercontent.com/93482485/229299282-fce792db-2018-46fa-a9c9-d407dbf87796.jpg)

### Cost function using Gradient Descent Graph
![s4](https://user-images.githubusercontent.com/93482485/229299287-03126ba3-b495-4191-a3bb-1065327da9e0.jpg)

### Profit Prediction Graph
![s5](https://user-images.githubusercontent.com/93482485/229299310-afbd5506-c94b-4a17-bdb6-d301ea6fad30.jpg)

### Profit for the Population 35,000
![s6](https://user-images.githubusercontent.com/93482485/229299337-b28ae3d6-cabe-456c-83bf-c0d9c22dd72d.jpg)

### Profit for the Population 70,000
![s7](https://user-images.githubusercontent.com/93482485/229299349-1dae33b5-6c79-4917-bb0e-8c5cd3304829.jpg)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
