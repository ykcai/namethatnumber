import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers

from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# 10 classes (0-9 numbers)
num_classes = y_test.shape[1]
features = input("Number of features for each layer: ")
learningRate = input("Learning rate of optimizer: ")
layers = 3;
	#Change the numner of hidden layers before you run if things changed
print("\n=====Training model with " + features + "input features, " + layers + " hidden layers, and learningRate = " + learningRate + "=====\n")
def baseline_model():
	# create model
	model = Sequential()
    	#first argument of Conv2D is the # feature maps that are 5x5 and a rectifier ativation function
	model.add(Conv2D(int(features), (5, 5), input_shape=(1, 28, 28), activation='relu'))
		#remove this layer to get 2 layers
	model.add(Conv2D(int(features), (3, 3), activation='relu'))
	model.add(Flatten())
		#remove this layer to get 1 layer
	model.add(Dense(int(features), activation='relu'))
    	#output layer 10 neurons for 10 classes and a softmax activation function
    	#to ouput a probablity-like prediction for each class
	model.add(Dense(num_classes, activation='softmax'))
		# Compile model
		# CHANGE LR (LEARNING RATE)
	customOp = optimizers.Adam(lr=float(learningRate), beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=customOp, metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#Change the numner of hidden layers before you run if things changed
print("\n=====Training model with " + features + "input features, " + layers + " hidden layers, and learningRate = " + learningRate + "=====\n")

# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
