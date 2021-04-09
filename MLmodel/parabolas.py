import numpy as np
import matplotlib.pyplot as plt
import random as random

from sklearn.model_selection import train_test_split


from Cnn_regressor_model import regressCnn
from tensorflow.keras.optimizers import Adam



def parabola(R,x,i): #Array, x, indice
	y = np.array((1,50))
	y = R[i][2]*(x-R[i][0])**2+R[i][1] #uso y = a(x-Vx)^2+Vy
	noise = np.random.normal(y,0.05)
	ynoise = (y + noise)/2
	return (ynoise)
N=0
numrand=np.zeros((1,3))

while N < 500:
	h = random.uniform(0,1)
	k = random.uniform(0,1)
	a = random.uniform(-10,0)
	numrand = np.append(numrand,[ [h,k,a] ] , axis = 0)
	N=N+1
  
x = np.linspace( 0, 1, 50 )
num = 0
features = np.zeros((1,50))
labels   = np.zeros((1,3))
while num < len(numrand):
	y  = parabola(numrand, x, num)
	#print(y.shape)
	for i, j in enumerate(y):
		if j < 0:
			y[i] = 0
	#print(y)
	yn =  np.delete(y, np.argwhere( y <= 0 ))
	xn =  np.delete(x, np.argwhere( y <= 0 ))
	features = np.append(features,[y], axis = 0)
	labels   = np.append(labels,[ [numrand[num][0],numrand[num][1],numrand[num][2] ] ], axis = 0 )
	plt.plot(x,y)
	num = num+1


#splitta i dati
train_features, test_features, train_labels, test_labels = train_test_split(features, 
																			labels, 
																			test_size = 0.25, 
																			random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


f = open("/home/sal/Ball-Tracking/MineTrack/center.txt", "r")
Cx = np.zeros((1,1))
Cy = np.zeros((1,1))
cont1 = 0
cont2 = 0 





while cont1 < 49:
    for x in f:
        cont1 = cont1+1
        x = x.split(',')
        x[0] = x[0].replace('(', '')
        x[0] = x[0].replace(')', '')
        x[1] = x[1].replace('(', '')
        x[1] = x[1].replace(')', '')
        if x[0]!=0 :
            Cx = np.append(Cx,[[int(x[0])/600]] ,axis = 1)
            Cy = np.append(Cy,[[int(x[1])/1064]], axis = 1)
            #print(Cy)
    cont1 = cont1 + 1
    Cx = np.append(Cx,[[0]],axis = 1)	
    Cy = np.append(Cy,[[0]],axis = 1)	


Cx = np.flip(Cx,axis = 1)
Cy = np.flip(Cy,axis = 1)

ForestPred = regressor.predict(Cy)	
f_Vx = ForestPred[0][0]
f_Vy = ForestPred[0][1]
f_a  = ForestPred[0][2]
fx = np.linspace( 0, 1, 50 )
print(f_a,f_Vx,f_Vy)
par = np.array(-f_a*(fx-f_Vx)**2 + f_Vy)
plt.scatter(Cx,Cy)
plt.plot(np.flip(fx), np.flip(par))

#cnn now
#print(train_features.shape)
#train_features = np.reshape(train_features, (37550,50, 751,1))
#y_train = train_labels.reshape(1000,1,3,0)
#print(X_train.shape)

#nrows, ncols = train_features.shape
#train_features = train_features.reshape(nrows, ncols, 1)
epochs = 200
batch_size = 50
model = regressCnn(train_features.shape[1]	, regress=True, features=train_features) #train_features.shape[1]
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
print("[INFO] training model...")
model.fit(train_features, train_labels, 
	validation_data=(test_features, test_labels),
	epochs=epochs, batch_size=batch_size)

cnnPredict = model.predict(Cy)
c_Vx = cnnPredict[0][0]
c_Vy = cnnPredict[0][1]
c_a  = cnnPredict[0][2]
fx = np.linspace( 0, 1, 50 )
print(c_a,c_Vx,c_Vy)
fpar = np.array(-c_a*(fx-c_Vx)**2 + c_Vy)
plt.scatter(Cx,Cy)
plt.plot(np.flip(fx), np.flip(fpar))
print("train=",N,"epochs=",epochs,"batch_size=", batch_size)
plt.show()
