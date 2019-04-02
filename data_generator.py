# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:33:03 2018

@author: Gregor
"""

#import
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import pandas as pd

#Seed for RNG
np.random.seed(25122019)

#training data
n=100
#fps
m=50
#arm length
R=1.5

#sample physical quantities
g = np.array([0,-constants.g]).reshape(2,1)
#random variable for the initial point
b_rand = np.array([2,1]).reshape(2,1) * np.random.random_sample((2, n)) + np.array([7,1]).reshape(2,1)
#random variable for the initial velocity
v_rand = np.array([-5,2]).reshape(2,1) * np.random.random_sample((2, n)) + np.array([-7,2]).reshape(2,1)

#projectile motion
def s(t, v, b):
   return 0.5*g.dot(t**2) + v.dot(t) + b

#time mesh
t_range= np.linspace(0, 1, m+1).reshape(1,m+1)

#Bahnkurven
y=np.zeros([n,2,m+1])
for i in range(n):
    y[i]=s(t_range, v_rand[:,i].reshape(2,1), b_rand[:,i].reshape(2,1))
    
#configuration space of the robot arm    
def r(a):
    return R*np.array([np.cos(a),np.sin(a)]).reshape(2,a.size)

#angle
a_range= np.linspace(0, constants.pi, m).reshape(1,m)
z_range = r(a_range)

#initialization of variable
#variable for catchpoint
z = np.empty((2,n))
z[:] = np.nan
#bool variable for catchability
catch = np.zeros(n)
#ankle of the kinematic pair
alpha = np.empty(n)
alpha[:] = np.nan

def bisec(a,b,R):
    c=b
    while(np.abs(np.linalg.norm(c)-R)>1e-5):
        c=(a+b)/2
        if np.linalg.norm(c)>R:
            a=c
        else:
            b=c
    return c

#calculation of approximate solutions for each sample
for i in range(n):
    for j in range(m):
        if y[i,1,j]<0:
            break
        elif np.linalg.norm(y[i,:,j])<R:
            catch[i] = 1
            break
        elif j==49:
            catch[i] = 2
            
            '''
            if temp[1] > 0:
                catch[i] = 1
                z[:,i] = temp
                #calculation of inverse kinematics
                if temp[0]>R:
                    alpha[i] = np.arccos(1)
                elif temp[0]<-R:
                    alpha[i] = np.arccos(-1)
                else:
                    alpha[i] = np.arccos(temp[0]/R)
            break
            '''
        

            
    

#plots
fig=plt.plot(np.transpose(y[15,0,:]),np.transpose(y[15,1,:]))
fig=plt.plot([z[0]], [z[1]], marker='o', markersize=3, color="red")
fig=plt.plot(z_range[0,:], z_range[1,:], linestyle='--', label='configuration space')
plt.legend(loc=2)
plt.axis('equal')

fig = plt.gcf()
plt.show()
plt.draw()
#fig.savefig('picture1.pdf', dpi=300)


#export data
d = y.reshape((n,2*(m+1)), order='F')
df = pd.DataFrame(data=d)
df['y']=pd.Series(catch)
#df.to_csv('data.csv', index=False)