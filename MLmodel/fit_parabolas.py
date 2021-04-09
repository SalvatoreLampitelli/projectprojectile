import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import r2_score
import csv
def parabolona(x, a, b, c):
    return a*x**2+b*x+c
def fitmyp(x,y, N, j ):
        params, params_covariance = optimize.curve_fit(parabolona, x, y)
        Fitty  = parabolona( x, params[0], params[1], params[2] )
        plt.scatter(x,y)
        plt.plot(x, parabolona( x, params[0], params[1], params[2] ) )
        plt.gca().invert_xaxis()	
        plt.gca().invert_yaxis()
        r2 = round(r2_score(y,Fitty),4)
        s = []
        for i in range(len(Fitty)):
            s = ((y[i]-Fitty[i])**2)/Fitty[i]
        somma =   round(np.sqrt( np.sum(s)/50 ),5)

        plt.text(0, 0, r'$Err=$'+str(somma), fontsize=15)
        plt.xlabel("centro delle X espresso tra [-1,1]")
        plt.ylabel("centro delle Y espresso tra [-1,1]")
        plt.savefig("/home/sal/Ball-Tracking/Tesi/MLmodel/imagefit/"+str(j)+"PRENDIMI"+"predict1") 
        plt.clf()
        return  (y[19] - Fitty[19] )**2 , r2 , params[0], params[1], params[2]

