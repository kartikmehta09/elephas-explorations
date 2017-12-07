#-----#
# construct a CNN for classifying MNIST digits and parallelize the model using elephas
# this works essentially by putting the training data into a Spark RDD
# each worker trains on their share of data and synchronizes with the master node
# see also https://github.com/maxpumperla/elephas/blob/master/examples/mnist_mlp_spark.py
#
# note this code is written in Keras 0.3.3 and will break for more recent versions
# run the code like this:
# ./elephas-test.py --nw [number of spark workers] --ne [number of epochs] --bs [batch size]
#-----#

#---keras imports
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

#---elephas imports
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers
from pyspark import SparkContext, SparkConf

#---other imports
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

#---function for building the model
#---adopted from https://github.com/cartopy/keras-0.3.3/blob/master/examples/mnist_cnn.py
def build_model():
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	return model
#-----#

#---When running this using spark-submit you need to create Spark context 
sc = SparkContext()

#---parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nw', dest='N_workers', type=int, default=1)
parser.add_argument('--ne', dest='nb_epoch', type=int, default=1)
parser.add_argument('--bs', dest='batch_size', type=int, default=32)
args = parser.parse_args()

#---load the mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print "X_train shape", X_train.shape
print "y_train shape", y_train.shape
print "X_test shape", X_test.shape
print "y_test shape", y_test.shape

#---reshape the data into the form that keras expects
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

#---normalize and convert to floats
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#---Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#---build the Keras model
model = build_model()

#---compile model (this step is missing in Max's code)
model.compile(loss='categorical_crossentropy', optimizer='adam')

#---Build RDD from trainin examples 
#---(i.e. in training each worker will train on part of the data)
rdd = to_simple_rdd(sc, X_train, y_train)

#---Initialize SparkModel from Keras model and Spark context
#---there are two optimizers needed:
sgd = SGD(lr=0.1) #<---the master optimizer
adagrad = elephas_optimizers.Adagrad() #<---the elephas opimizer
spark_model = SparkModel(sc,
                         model,
                         optimizer=adagrad,
                         frequency='epoch',
                         mode='asynchronous',
                         num_workers=args.N_workers, master_optimizer=sgd)

#---Train Spark model
spark_model.train(rdd, nb_epoch=args.nb_epoch, batch_size=args.batch_size, 
					verbose=1, validation_split=0.25)

#---Evaluate Spark model by evaluating the underlying Keras master model
pred = spark_model.predict(X_test)
print np.shape(pred)
print np.shape(y_test)
acc = accuracy_score([np.argmax(y) for y in y_test], [ np.argmax(p) for p in pred ] )
print "test accuracy: ", acc