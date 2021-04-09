# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPool1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
def regressCnn(dim, regress=False, features = None):
	# define our MLP network
	input_layer =  Input(shape=(dim, 1))
	model = Sequential()
#	model.add( layers.Conv2D(32, (3, 3), activation='relu') )
#	model.add(layers.MaxPooling2D((2, 2)))
#	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#	model.add(layers.MaxPooling2D((2, 2)))
#	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#   model.add(Conv1D(filters=1, kernel_size=1 ,strides=1,     
#                 input_shape=(371,50,1),kernel_initializer= 'uniform',      
#                 activation= 'relu'))
#   model.add(MaxPool1D(pool_size = 10))
#   model.add(Conv1D(filters = 3, kernel_size=10))
#   model.add(Flatten())			  
	model.add(Dense(500, activation="relu"))
	model.add(Dense(10, activation="relu"))

	#inp =  Input(shape=(50, 1))
	#conv = Conv1D(filters=3, kernel_size=10)(inp)
	#pool = MaxPool1D(pool_size=2)(conv)
	#flat = Flatten()(pool)
	#dense = Dense(3, activation="linear")(flat)
	#model = Model(inp, dense)
	#model.summary()
	#check to see if the regression node should be added
	if regress:
		model.add(Dense(3, activation="linear"))
		#model.compile(loss='mse', optimizer='adam')
	# return our model
	return model