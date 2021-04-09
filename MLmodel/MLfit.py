import randomforest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv 
import Cnn_fit
import math
import fit_parabolas
#df = pd.read_csv(Location, names=['Names','Births'])
dftrainf  = pd.read_csv( 'data_intero_train_f.csv', header=None)
dftrainl  = pd.read_csv( 'data_intero_train_l.csv', header=None)
dftestf   = pd.read_csv( 'data_intero_test_f.csv' , header=None)
dftestl   = pd.read_csv( 'data_intero_test_l.csv' , header=None)

#X = dftestf.to_numpy()
#Y = dftestl.to_numpy()
#predict = randomforest.RandomForest(dftrainf, dftrainl, dftestf, dftestl, X)
#i = 0
#
#print(Y[0])
#plt.scatter(X[0][0:9], X[0][9:18], label = "punti parabola originale")
#plt.scatter(Y[0][0],Y[0][1], label = "prossimo punto originale")
#plt.scatter(predict[0][0],predict[0][1], label="punto predetto dal RandomForest")                                            
#plt.gca().invert_xaxis()	
#plt.gca().invert_yaxis()
#plt.legend(loc = 'lower center')
#plt.savefig("/home/sal/Ball-Tracking/Tesi/MLmodel/imagefit/predict1")
#
#plt.show()
#
#for p in predict:
#    print( "lo scarto dal punto di test", i+1, "è : ", (p - dftestl[0][i])**2 ) 
#    i = i+1
#
#
dftrainf = dftrainf.to_numpy()
dftrainl = dftrainl.to_numpy()
dftestf  = dftestf.to_numpy()
dftestl  = dftestl.to_numpy()
predetto = Cnn_fit.CNN_fit(dftrainf[0:45,19:38], dftrainl[0:45,1:], dftestf[0:5,19:38],dftestl[0:5,1:0], 1)

X = dftestf
Y = dftestl

def the_greatest_show(Xpredetto,Ypredetto,Xoriginale,Yoriginale,dftestf,dftestl,nome,j):
    #print(predetto[j])
    #print(dftestl[j])

    #print( "lo scarto dal punto di test", j+1, "è : ", (predetto[j] - dftestl[j,1])**2 ) 
    
    xtot = np.zeros(shape=(20))
    ytot = np.zeros(shape=(20))

    X = dftestf
    Y = dftestl
    xtot[0:19] = X[j][0:19]
    xtot[19] = Xoriginale
    ytot[0:19] = X[j][19:38]
    ytot[19] = Ypredetto

    fit_parabolas.fitmyp(xtot,ytot,50,j)

    plt.scatter(X[j][0:19], X[j][19:38], label = "punti parabola originale")
    plt.scatter(Xoriginale,Yoriginale, label = "prossimo punto originale")
    plt.scatter(Xpredetto,Ypredetto, label="punto predetto dal CNN")                                            
    plt.gca().invert_xaxis()	
    plt.gca().invert_yaxis()
    plt.legend(loc = 'lower center')
    plt.savefig("/home/sal/Ball-Tracking/Tesi/MLmodel/imagefit/"+nome+str(j)+"predict") 
    plt.show()   

for i in range(len(X)):
    #print(len(X))
    #print(i)
    the_greatest_show( Y[i,0], predetto[i], Y[i,0], Y[i,1], X, Y, "Cnn", i)

j=0
S = []

for p in predetto:
    print( (p - dftestl[j,1] )**2 )
    S.append((p - dftestl[j,1] )**2 ) 
    j=j+1
m = math.sqrt(sum(S)/5)
su = []
for x in S:
    su.append((x-m)**2)
err = math.sqrt(sum(su)/5)
    
print("la somma è :" ,m , "+/-", err)

S.pop(0)
m = math.sqrt(sum(S)/4)
sup = []
for x in S:
    sup.append((x-m)**2)
err = math.sqrt(sum(sup)/4)
print("la somma senza lo 0 è :" ,m , "+/-", err)
