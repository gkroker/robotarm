import numpy as np
from scipy import constants
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import time

#Seed for RNG
np.random.seed(25122019)

#methods
def forKin(l1,l2,th1,th2):
    x1 = l1*np.cos(th1)
    y1 = l1*np.sin(th1)
    x2 = l1*np.cos(th1)+l2*np.cos(th1+th2)
    y2 = l1*np.sin(th1)+l2*np.sin(th1+th2)
    if np.isscalar(th1)==True:
        return np.matrix([0,0,x1,y1,x2,y2])
    else:
        n = len(th1)
        return np.matrix([np.zeros(n),np.zeros(n),x1,y1,x2,y2])

def invKin(l1,l2,x,y):
    beta = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
    th2 = -np.arctan2(np.sqrt(1-beta**2),beta)
    
    k_1 = l1+l2*np.cos(th2)
    k_2 = l2*np.sin(th2)
    gamma = np.arctan2(k_2,k_1)
    
    th1 = np.arctan2(y,x)-gamma
    return np.matrix([th1,th2])

def p(koeff,t):
    y = np.tensordot(koeff[0,:],t**2,axes=0)+np.tensordot(koeff[1,:],t**1,axes=0)+np.tensordot(koeff[2,:],t**0,axes=0)
    return y.T

def bisec(a,b,R):
    c=b
    while(np.abs(np.linalg.norm(c)-R)>1e-5):
        c=(a+b)/2
        if np.linalg.norm(c)>R:
            a=c
        else:
            b=c
    return c

#constants
l1=1
l2=0.5
R=l1+l2
#fps
m=50
#points
s=30
#timeseries shape
s_shape = (2,m+1)

#import data
data = pd.read_csv("data.csv")

#preparation
sample = data.values
n =sample.shape[0]
t = np.linspace(0, 1, m+1)
dt = 1/m
y = sample[:,-1]
X = sample[:,:-1]
X = X.reshape((-1,) + s_shape, order='F')
X = np.transpose(X,(0,2,1))
X_noise = X + np.random.normal(scale=0.05,size=X.shape)
error = np.zeros((n,))
toc = np.zeros((n,))
s = [0]*n
y_pred = [1]*n
k = [0]*n
theta = np.zeros((n,2))
x_pred = np.zeros(X.shape)


for j in range(n):
    
    #start
    tic = time.time()

    for i in range(m+1):
        if X_noise[j,i,1]<0:
            y_pred[j] = 0
            k[j] = i
        elif np.linalg.norm(X_noise[j,i,:])<2*R:
            s[j]=i
            break

    if y_pred[j] ==0:
        toc[j] = time.time() - tic
        break
    else:
        x_train, x_test = X_noise[j,:s[j],:],X_noise[j,s[j]:,:]
        t_train, t_test = t[:s[j]],t[s[j]:]
        x_train_lin = x_train[:,:] - np.tensordot([0,-constants.g/2],t_train**2,axes=0).T


        # Create linear regression object and Train the model using the training sets
        koeff = np.zeros((3,2))
        koeff[0,:] = [0, -constants.g/2]
        koeff[1:3,:] = np.polyfit(t_train, x_train_lin,1)

        # Make predictions using the testing set
        x_pred[j,:,:] = p(koeff,t)

        for i in range(m+1-s[j]):
            if x_pred[j,s[j]+i,1]<0 and x_pred[j,s[j]+i,0]>0:
                y_pred[j] = 0
                k[j] = s[j]+i
                break
            elif np.linalg.norm(x_pred[j,s[j]+i,:])<R:
                k[j] = s[j]+i
                break
            elif i == m-s[j]:
                y_pred[j] = 2
                k[j] = s[j]+i
                break
            
        if y_pred[j]==2:
            toc[j] = time.time() - tic
        elif y_pred[j]==0:
            toc[j] = time.time() - tic
        else:
            th = invKin(l1,l2,x_pred[j,k[j],0],x_pred[j,k[j],1])
            #pos = forKin(l1,l2,th[:,0],th[:,1])
            #pos = pos.reshape((3,2))

            #Ende
            toc[j] = time.time() - tic
            theta[j,:] = th
            error[j] = np.linalg.norm(x_pred[j,k[j],:]-X[j,k[j],:])
            #print(toc)

#print(np.linalg.norm(x_pred[k,:]-X[0,k,:]))

#print(t[k]-t[s])

io=1
if io==1:
    p = np.zeros((n,m+1,3,2))
    for j in range(n):
        q = np.zeros((k[j]+1,2))
        q[:s[j],:]= [1, -1]
        #
        if y_pred[j]==0 or y_pred[j]==2:
            q[s[j]:,:]= [1, -1]
        else:
            q[s[j]:k[j],0] = np.linspace(1,theta[j,0],k[j]-s[j])
            q[s[j]:k[j],1] = np.linspace(-1,theta[j,1],k[j]-s[j])
            q[k[j]:,:]=q[k[j]-1,:]
        
        #=np.zeros((k[j]+1,3,2))
        for i in range(k[j]+1):
            p[j,i,:,:]=forKin(l1,l2,q[i,0],q[i,1]).reshape((3,2))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 9), ylim=(-0, 3))
    
    ball, = ax.plot([], [], 'o', markersize = 10)
    arm, = ax.plot([], [], 'o-', markersize = 6, linewidth=3, color = '#aaaaaa')
    ball_m, = ax.plot([], [], 'x', markersize = 10, color = 'k')
    ball_pred, = ax.plot([], [], '--')


    def init():
        ball.set_data([], [])
        arm.set_data([], [])
        ball_m.set_data([], [])
        ball_pred.set_data([], [])
        return ball, arm, ball_m, ball_pred


    def animate(it):
        i = it
        j = 0
        while i > -1:
            i = i-k[j]
            j += 1
        j -= 1
        i += k[j]
        thisx = X[j,i,0]
        thisy = X[j,i,1]
        
        thisx2 = p[j,i,:,0]
        thisy2 = p[j,i,:,1]
            
        thisx3 = X_noise[j,i,0]
        thisy3 = X_noise[j,i,1]

        ball.set_data(thisx, thisy)
        arm.set_data(thisx2, thisy2)
        ball_m.set_data(thisx3, thisy3)
        if i >= s[j]:
            ball_pred.set_data(x_pred[j,:,0],x_pred[j,:,1])
        else:
            ball_pred.set_data([], [])
        return ball, arm, ball_m, ball_pred

    ani = animation.FuncAnimation(fig, animate, np.arange(1, np.sum(k)+1),
                              interval=20, blit=False, init_func=init)


        #writer = FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("movie.mp4", fps=50, dpi=300)
    plt.show()