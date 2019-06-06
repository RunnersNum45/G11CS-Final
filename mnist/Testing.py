'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# import matplotlib.pyplot as plt
# # plot 4 images as gray scale
# plt.subplot(221)
# plt.imshow(x_train[0], cmap=plt.get_cmap('grey'))
# plt.subplot(222)
# plt.imshow(x_train[1], cmap=plt.get_cmap('grey'))
# plt.subplot(223)
# plt.imshow(x_train[2], cmap=plt.get_cmap('grey'))
# plt.subplot(224)
# plt.imshow(x_train[3], cmap=plt.get_cmap('grey'))
# # show the plot
# plt.show()

if K.image_data_format() == 'channels_first':
	K.logging.set_verbosity(tf.logging.ERROR)
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
				 activation='relu',
				 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25)) 
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])

model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("Test_Result.h5")