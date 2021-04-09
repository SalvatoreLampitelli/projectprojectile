import csv
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import fit_parabolas
from sklearn.model_selection import train_test_split

N = 1 
size = []                     #lunghezza di x e y 
MetaData = []                 #verrano salvati tutti i dati
Parameters = np.zeros((1,5))  #parametri del fit
def Resize(x,a,b,resolution): #riscala x nell'intervallo [a,b]
	m = np.min(x)
	M = np.max(x)
	nx = ( (b-a)*(x-m) ) / (M-m) + a
	return nx
def Resize2(x,a,b,resolution): #riscala x con media 0 
	m = np.mean(x)
	nx = (x-m)/m
	return nx

def taketwenty(x,size):
	c = int(size/2)
	return x[c-10:c+10]
	
while N <= 50 :
	nomefile = "Centerball" + str(N) + ".csv"                           #apre ogni file ball(N)
	Data = pd.read_csv(nomefile)                                        #leggo dal file
	Data.columns = ['x','y']                                            #cambio le colonne del panda
	nData = Data.to_numpy()                                             #trasforma l'array in un numpy
	asize =  (Data.size)/2
	size.append( asize )
	xcenter = taketwenty( Resize2( np.array(Data['x']), -1, 1,  600 ),asize)   # metto tutto tra [-1,1]
	ycenter = taketwenty( Resize2( np.array(Data['y']), -1, 1, 1080 ),asize)   # e sceglo 20 punti a metà
	MetaData.append( [xcenter , ycenter] )                                     #ha 3 dimensioni [0-49][0-1][0-19]
	                                                                           #(numero dati,tipo x o y, specifico dato x o y)
	#plt.scatter(xcenter,ycenter)                                               #salva ogni plot di ogni parabola
	#plt.gca().invert_xaxis()	
	#plt.gca().invert_yaxis()
	#plt.savefig("/home/sal/Ball-Tracking/Tesi/dataset/parabolas/imageparabolas/myparabolasn"+str(N))
	#plt.clf() 
	d , R2 , a , b , c = fit_parabolas.fitmyp(xcenter,ycenter, N, N)
	Parameters = np.concatenate((Parameters, [ [d , R2 , a , b , c]]) ,axis=0) #poi in csv  
	N=N+1

np.savetxt("/home/sal/Ball-Tracking/Tesi/dataset/parabolas/parameters.csv", Parameters ,delimiter=",", fmt='%.6f')	

print ( "in media ho ", np.mean(np.asarray(size)), " punti, ma ne uso solo 20")
print ( "in totale ho ", len(MetaData), "dati")
print ("i dati del fit quadratico sono salvati in Tesi/dataset/parabolas/parameters.csv")

features = []
labels   = []
n_labels = 1         #numero di punti da predire

N=0

while N < len(MetaData):	                           #qui tolgo l'ultimo dato della x e della y
	j = 0
	k = 0										       
	contenitore = []
	contenitore2 = []
	while j <= MetaData[N][0].size - n_labels:          #con il ciclo while elimino l'ultimo x e
		if j == MetaData[N][0].size - n_labels :			#l'ultimo y e lo metto in labels e features
			contenitore2.append(MetaData[N][0][j])	
			#contenitore2.append(MetaData[N][1][j])
		else:
			contenitore.append(MetaData[N][0][j])
			#contenitore.append(MetaData[N][1][j])
		j=j+1
	while k <= MetaData[N][1].size - n_labels:
		
		if k == MetaData[N][1].size - n_labels:
			contenitore2.append(MetaData[N][1][k])
		else:
			contenitore.append(MetaData[N][1][k])
		k = k + 1	
	features.append(contenitore)                        #features è un array di dim 2 [0-49][0-38]
	labels.append(contenitore2)							#labels è un array di dim2 [0-49][0-1]
	N=N+1

def print_to_csv(features,labels, N, nomefile , percentuale = 0.1): #prende features e label e li divide scrivendoli in csv
	m = int(N*percentuale)
	train_features, test_features, train_labels, test_labels = features[0:N-m], features[N-m:N], labels[0:N-m], labels[N-m:N]
	print('Training Features Shape:', len(train_features))
	print('Training Labels Shape:'  , len(train_labels  ))
	print('Testing Features Shape:' , len(test_features ))
	print('Testing Labels Shape:'   , len(test_labels   ))

	df_train_f = pd.DataFrame(data=train_features)
	df_train_l = pd.DataFrame(data=train_labels  )
	df_test_f  = pd.DataFrame(data=test_features )
	df_test_l  = pd.DataFrame(data=test_labels   )

	df_train_f.to_csv ('/home/sal/Ball-Tracking/Tesi/MLmodel/'+ nomefile + 'train_f.csv',   index=False,header=False)
	df_train_l.to_csv ('/home/sal/Ball-Tracking/Tesi/MLmodel/'+ nomefile + 'train_l.csv',   index=False,header=False)
	df_test_f.to_csv  ('/home/sal/Ball-Tracking/Tesi/MLmodel/'+ nomefile +  'test_f.csv' ,   index=False,header=False)
	df_test_l.to_csv  ('/home/sal/Ball-Tracking/Tesi/MLmodel/'+ nomefile +  'test_l.csv' ,   index=False,header=False)
	print("i dati Train(Test) sono salvati in Tesi/MLmodel/")


print_to_csv(features,labels, 50, "data_intero_", 0.1 )


divdataf = np.zeros(shape=(150,18))
divdatal = np.zeros(shape=(150,2))
contdiv  = np.zeros(shape=(150,20))

#voglio dividere i dati in 3 pezzi
d = 0
iii = 0
while iii < 50:

	if d >= 150 : break
	contdiv[d]    = np.concatenate((  MetaData[iii][0][0:10] , MetaData[iii][1][0:10]  ))
	contdiv[d+1]  = np.concatenate((  MetaData[iii][0][5:15] , MetaData[iii][1][5:15]  ))
	contdiv[d+2]  = np.concatenate((  MetaData[iii][0][10:20], MetaData[iii][1][10:20] ))

	divdataf[d]   = np.concatenate((  MetaData[iii][0][0:9]  , MetaData[iii][1][0:9]   ))
	divdataf[d+1] = np.concatenate((  MetaData[iii][0][5:14] , MetaData[iii][1][5:14]  ))
	divdataf[d+2] = np.concatenate((  MetaData[iii][0][10:19], MetaData[iii][1][10:19] ))

	divdatal[d]    = [ MetaData[iii][0][9]     ,   MetaData[iii][1][9]  ]  
	divdatal[d+1]  = [ MetaData[iii][0][14]    ,   MetaData[iii][1][14] ]  
	divdatal[d+2]  = [ MetaData[iii][0][19]    ,   MetaData[iii][1][19] ]
	d = d+3 
	iii=iii+1 


#voglio dividere i dati in 3 pezzi
data = np.zeros(shape=(50,40)) 

for i in range(50):                                              # faccio un merge delle x e delle y
	data[i] = np.concatenate( (MetaData[i][0], MetaData[i][1]) )

print_to_csv(divdataf,divdatal, 150, "terzi_data_", 0.1 )
