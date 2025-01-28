import numpy as np
import matplotlib.pyplot as plt

# INPUT DATA
x_input = input("Enter your Data for X (Seperated with Comma): ")
y_input = input("Enter your Data for Y(Seperated with Comma):  ")

#TO CHECK IF SEPERATED BY COMMA
X = np.array([float(i) for i in x_input.split(',')])
Y = np.array([float(i) for i in y_input.split(',')])

#CHECKS IF THE DATA ARE SIMILAR IN LENGTH
if len(X) != len(Y):
    print("Error! Data Length Must be Equal!")
    exit()

#SOLVES THE REGRESSION
n = len(X)
mean_x,mean_y =  np.mean(X), np.mean(Y)
m = sum((X - mean_x) * (Y - mean_y )) / sum((X - mean_x) ** 2)
b = mean_y - m * mean_x

#SHOWS THE REGRESSION
regression = m * X +  b

#LABELS
plt.scatter(X,Y, color= 'green',label = 'Data Points')
plt.plot(X, regression,color='red', label = f'Linear Regression (y={m:.2f}x+{b:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid()


plt.show()