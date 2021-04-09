import numpy as np
import matplotlib.pyplot as plt
import random as random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from Cnn_regressor_model import regressCnn
from tensorflow.keras.optimizers import Adam


def RandomForest(train_features, train_labels, test_features, test_labels, Cy = None):
	#-----<regressor>------
	regressor = DecisionTreeRegressor(random_state      = 0, 
									  max_leaf_nodes    = 2, 
									  min_samples_leaf  = 1, 
									  min_samples_split = 2,
									  max_depth         = 1,
									  splitter          = "random",
									  max_features      = 10,
									  min_weight_fraction_leaf = 0.2,
									  min_impurity_decrease = 1,
									  ccp_alpha         = 1)
	regressor.fit(train_features, train_labels)
	#-----</regressor>------
	print("Accuracy on training set regressor: {:.3f}".format(regressor.score(train_features, train_labels)))
	print("Accuracy on test set regressor: {:.3f}".format(regressor.score(test_features, test_labels)))
	return regressor.predict(Cy)	

