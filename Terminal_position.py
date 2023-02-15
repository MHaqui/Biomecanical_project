import matplotlib as mp
import matplotlib.pyplot as plt
from numpy import matlib
import numpy as np

x_range = [-np.pi/2, np.pi/2]
n = 100
pas = (-x_range[0] + x_range[1])/n
theta2 = 0
theta1_step = []
theta2_step = []

m1 = 1
m2 = 1
l1 = 1
l2 = 1
g = 9.8


alpha = -(m1*l1 + m2*l2)*g
beta = - m2*l2*g
for i in range(n):

    theta2 = pas + theta2
    theta2_step.append(theta2)
    theta1 = np.arctan(beta*np.sin(theta2) /
                       (alpha + beta*np.cos(theta2)))
    theta1_step.append(theta1)
plt.plot(theta1_step, theta2_step)
plt.show()
