from matplotlib import use, cm
use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn import linear_model

from gradientDescent import gradientDescent
from computeCost import computeCost
from warmUpExercise import warmUpExercise
from plotData import plotData
from show import show

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
warmup = warmUpExercise()
print(warmup)
input('Program paused. Press Enter to continue...')

# ======================= Part 2: Plotting =======================
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = np.vstack(zip(np.ones(m), data[:, 0]))
y = data[:, 1]

# Plot Data
# Note: You have to complete the code in plotData.py
print('Plotting Data ...')
plotData(x=data[:, 0], y=data[:, 1])
show()
input('Program paused. Press Enter to continue...')

# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
theta = np.zeros(2)

# compute and display initial cost
J = computeCost(X, y, theta)
print('cost: %0.4f ' % J)

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ')
print('%s %s \n' % (theta[0], theta[1]))
#print(J_history)

# Plot the linear fit
plt.figure()
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()
input('Program paused. Press Enter to continue...')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1 * 10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2 * 10000))

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Create grid coordinates for plotting
B0 = np.linspace(-10, 10, X.shape[0])
B1 = np.linspace(-1, 4, X.shape[0])
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))
# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = computeCost(X,y, theta=[[xx[i,j]], [yy[i,j]]])

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Contour plot
#CS = ax1.contour(B0, B1, Z, np.logspace(-2, 3, 30), cmap=plt.cm.jet)
CS = ax1.contour(B0, B1, Z)
ax1.scatter(theta[0],theta[1], c='r')

# Surface plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)
show()
input('Program paused. Press Enter to continue...')

# =============Use Scikit-learn =============
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
regr.fit(X, y)

print('Theta found by scikit: ')
print('%s %s \n' % (regr.coef_[0], regr.coef_[1]))

predict1 = np.array([1, 3.5]).dot(regr.coef_)
predict2 = np.array([1, 7]).dot(regr.coef_)
print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1 * 10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2 * 10000))

plt.figure()
plotData(x=data[:, 0], y=data[:, 1])
plt.plot(X[:, 1],  X.dot(regr.coef_), '-', color='black', label='Linear regression with scikit')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
show()

input('Program paused. Press Enter to continue...')
