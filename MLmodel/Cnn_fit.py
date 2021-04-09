import numpy as np
import keras 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten #, Reshape
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras import regularizers, initializers
from keras import optimizers
import tensorflow as tf
import pandas as pd
from scipy import stats
from keras.utils import plot_model

import seaborn 
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn import preprocessing


def CNN_fit(x_train,y_train,x_val,y_val,n_class):              #x_features e x_labels n_class:= numero di parametri da predire
    # Keras wants an additional dimension with a 1 at the end
    Lx = len(x_train[0])
    Ly = len(y_train[0])
    print(x_val.shape[0])
    x_train = x_train.reshape(x_train.shape[0], Lx, 1)
    #y_train = y_train.reshape(y_train.shape[0], Ly, 1)
    x_val =  x_val.reshape(x_val.shape[0], Lx, 1)
    #y_val = y_val.reshape(y_val.shape[0], Ly, 1)
    y_train
    input_shape = (Lx, 1)
    reg = regularizers.l1(0.1)
    ini = keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)
    NCONV = 1
    model = Sequential()
    if NCONV==1:
        # -----2-----
        model.add(Conv1D(filters=10, kernel_size=2, 
                         kernel_initializer=ini, 
                         kernel_regularizer=reg,
                         activation='exponential', input_shape=input_shape))
        #model.add(MaxPooling1D(10))
        #model.add(AveragePooling1D(10))
        model.add(Conv1D(filters=5, kernel_size=2, 
                         activation='tanh'))
        model.add(Flatten())
        model.add(Dense(5, activation='relu'))
        model.add(Dropout(0.2))
    # NN with two convolutional layers
    if NCONV==2:
        # -----1-----
        model.add(Conv1D(filters=2, kernel_size=2, 
                         kernel_initializer=ini, 
                         kernel_regularizer=reg, ######## TRY WITHOUT !
                         activation='exponential', input_shape=input_shape))
        #model.add(MaxPooling1D(3))
        #model.add(AveragePooling1D(10))
        model.add(Flatten())
        model.add(Dense(38, activation='exponential'))
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='relu'))
        model.add(Dropout(0.2))
        #model.add(Dense(10, activation='relu'))
        #model.add(Dropout(0.3))
    model.add(Dense(n_class, activation='linear')) # #ho bisogno di linear per non avere probabilità

    print('-----',NCONV,'-----')
    print(model.summary())
    # optimizers.SGD? immagino mi serva solo un opt, quale scelgo?
    #opt = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True) # decay=1e-6,
    #opt = optimizers.RMSprop()
    #opt = optimizers.Adam()
    # optimizers.Nadam?
    opt = optimizers.Nadam()

    # compile the model, Nesterov gradient descent
    # categorical: 3 types of outputs
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=opt,metrics=['accuracy'])

    # Hyper-parameters
    # with small minibatch it does not converge!! 
    BATCH_SIZE = 2
    EPOCHS = 30
    print(x_train.shape)
    print(y_train.shape)
    print('-----NCONV=',NCONV,'-----\nFITTING....')
    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    fit = model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,
                    validation_data=(x_val, y_val),
                    verbose=2, shuffle=True) 

    #,callbacks=callbacks_list)

    # Plot training & validation accuracy values
    plt.figure(figsize=(6, 4))
    plt.plot(fit.history['accuracy'], 'r', label='Accuracy of training data')
    # dashed line!!!
    #plt.plot(fit.history['val_accuracy'], 'b--', label='Accuracy of validation data')
    plt.title(NCONV)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(0)
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(6, 4))
    plt.plot(fit.history['loss'], 'r', label='Loss of training data')
    #plt.plot(fit.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title(NCONV)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


    c=['k','r','y','b','m']
    def plot_w(w):
        # Plot weights of convol. layer
        plt.figure(figsize=(6, 4))
        for i in range(len(w)):
            plt.plot(w[i][0], label=str(i))
        plt.title(NCONV)
        plt.ylabel('weight')
        plt.xlabel('index')
        plt.legend()
        plt.show()
        
    w0 = model.layers[0].get_weights()[0]
    w01 = model.layers[0].get_weights()[1]
    w0T = w0.T
    print('w0T=',w0T)
    print('w01=',w01)
    print(len(w0))
    print(len(w0T))
    plot_w(w0T)
    
    plt.plot(w01, 'r')
    plt.ylabel('bias of layer 0')
    plt.xlabel('filter nr')
    plt.show()
    print(w01)

    LABELS = ["absent","positive","negative"]

    def show_confusion_matrix(validations, predictions):

        matrix = metrics.confusion_matrix(validations, predictions)
        print(matrix)
        plt.figure(figsize=(6, 6))
        seaborn.heatmap(matrix,
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    annot=True,
                    fmt='d',
                    linecolor='white',
                    linewidths=1,
                    cmap='coolwarm')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    y_pred_val = model.predict(x_val)
    # Take the class with the highest probability from the val predictions
    #max_y_pred_val = np.argmax(y_pred_val, axis=1)
    ##print(max_y_pred_val)
    #max_y_val = np.argmax(y_val, axis=1)

    #show_confusion_matrix(y_val[0][0:4], y_pred_val[0][0:4])
    #print()
    plot_model(model, to_file='./model.png', show_shapes=True,show_layer_names=True)
    #print(model.summary().as_latex())
    return y_pred_val
