import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(trimgs, trlbls), (tsimgs, tslbls) = mnist.load_data()

trimgs = trimgs.reshape(trimgs.shape[0], img_rows, img_cols, 1)
tsimgs = tsimgs.reshape(tsimgs.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

trimgs = trimgs.astype('float32')
tsimgs = tsimgs.astype('float32')
trimgs /= 255
tsimgs /= 255
print('trimgs shape:', trimgs.shape)
print(trimgs.shape[0], 'train samples')
print(tsimgs.shape[0], 'test samples')

# convert class vectors to binary class matrices
trlbls = keras.utils.to_categorical(trlbls, num_classes)
tslbls = keras.utils.to_categorical(tslbls, num_classes)

model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
# 				 activation='relu',
# 				 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25)) 
model.add(Flatten())
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])

record = model.fit(trimgs, trlbls,
		  batch_size=batch_size,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(tsimgs, tslbls))

history = record.history


# Plot training & validation accuracy values
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


score = model.evaluate(tsimgs, tslbls, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights("Test_Weights.h5")

with open("Test_Save.json", "w") as json_file:
  json_file.write(model.to_json())