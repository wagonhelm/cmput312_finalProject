import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from sensor_distributions import mean_and_var
from math import sqrt

"""Plots all the indiviudal distributions from sensor_distributions's mean_and_var dictionary"""

# All sensors STD
x = np.arange(-1.5,1.5,0.0001)
y = scipy.stats.norm.pdf(x, 0, mean_and_var['odom']['std'])
y2 = scipy.stats.norm.pdf(x, 0, mean_and_var['ultrasonic']['std'])
y3 = scipy.stats.norm.pdf(x, 0, mean_and_var['gps']['std'])
plt.plot(x, y, color='blue', label=r'Odometry $\mathcal{N}(\mu,\,0.02)$')
plt.plot(x, y3, color='red', label=r'Color Sensor $\mathcal{N}(\mu,\,0.38)$')
plt.plot(x, y2, color='orange', label=r'Ultrasonic $\mathcal{N}(\mu,\,110.5)$')
plt.grid()
plt.legend()
plt.xlabel('Distance (cm)')
plt.ylabel('Probability')
plt.title('Gaussian Distributions for All Sensors')
plt.xlim(-1.5, 1.5)
plt.ylim(0, 3)
plt.savefig("images/all_sensors_distributions.png")
plt.show()


# ODOMETRY
x = np.arange(-2,4,0.01)
y = scipy.stats.norm.pdf(x, mean_and_var['odom']['mean'], sqrt(mean_and_var['odom']['var']))
plt.plot(x, y, color='coral')
plt.grid()
plt.xlim(1.25, 2.75)
plt.ylim(0, 3)
plt.xlabel('Distance (cm)')
plt.ylabel('Probability')
plt.title('Normal Distribution of Distance Traveled in One Timestep')
plt.savefig("images/odometry.png")
plt.show()

# GPS
x = np.arange(25,175,0.01)
y1 = scipy.stats.norm.pdf(x, mean_and_var['gps'][1]['mean'], sqrt(mean_and_var['gps'][1]['var']))
y2 = scipy.stats.norm.pdf(x, mean_and_var['gps'][2]['mean'], sqrt(mean_and_var['gps'][2]['var']))
y3 = scipy.stats.norm.pdf(x, mean_and_var['gps'][3]['mean'], sqrt(mean_and_var['gps'][3]['var']))
plt.plot(x, y1, label='1st GPS Reading')
plt.plot(x, y2, label='2nd GPS Reading')
plt.plot(x, y3, label='3rd GPS Reading')
plt.grid()
plt.xlim(25, 175)
plt.ylim(0, 0.7)
plt.xlabel('Distance (cm)')
plt.ylabel('Probability')
plt.legend()
plt.title('Normal Distributions of Mock GPS Readings ')
plt.savefig("images/gps.png")
plt.show()

# SONAR
x = np.arange(60,140,0.01)
y1 = scipy.stats.norm.pdf(x, mean_and_var['ultrasonic']['mean'], sqrt(mean_and_var['ultrasonic']['var']))
plt.plot(x, y1)
plt.grid()
plt.xlim(60, 140)
plt.ylim(0, 0.04)
plt.xlabel('Distance (cm)')
plt.ylabel('Probability')
plt.legend()
plt.title('Normal Distribution of Ultrasonic Readings ')
plt.savefig("images/ultrasonic.png")
plt.show()
