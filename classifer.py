#***************************************(/\/\/\/\//\/\/:\/\\/\//\/\/\/)*************************#
#________________________________________Storm_Visualizer_______________________________________#
#************************************Arpan Sarkar***********************************************#
########################################import##################################################
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
###########################################import###############################################

m_width , m_height = 320,320

train_data_loc = 'data/train' #data directory
validation_data_loc = 'data/validate' 
train_samples = 678
validation_samples = 201
epochs = 52 #repeatations per data module
batch_size = 15 #20 for faster processing

if K.image_data_format() == 'channels_first': #auto configure input image charecteristics
	input_shape = (3,m_width,m_height) #(3,320,320)
else:
	input_shape = (m_width,m_height,3)

train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)
#dataset generation operations
test_datagen = ImageDataGenerator(rescale=1. / 255) #test data , only rescale
#training dataset generation
train_generator = train_datagen.flow_from_directory(
	train_data_loc,
	target_size=(m_width,m_height),
	batch_size=batch_size,
	class_mode='binary')
#validation dataset generation
#AS2019
validation_generator = test_datagen.flow_from_directory(
	validation_data_loc,
	target_size=(m_width,m_height),
	batch_size=batch_size,
	class_mode='binary')

storm = Sequential()
storm.add(Conv2D(32,(3,3), input_shape=input_shape))
storm.add(Activation('relu'))
storm.add(MaxPooling2D(pool_size=(2,2)))
#ARPAN
storm.summary()

storm.add(Conv2D(32,(3,3)))
storm.add(Activation('relu'))
storm.add(MaxPooling2D(pool_size=(2,2)))#32 feature bases
#SARKAR
storm.add(Conv2D(64,(3,3)))
storm.add(Activation('relu'))
storm.add(MaxPooling2D(pool_size=(2,2)))#multiple learning for accuracy / 64 feature

storm.add(Conv2D(64,(3,3)))
storm.add(Activation('relu'))
storm.add(MaxPooling2D(pool_size=(2,2)))

storm.add(Flatten()) #1D output
storm.add(Dense(64))
storm.add(Activation('relu'))
storm.add(Dropout(0.5))
storm.add(Dense(1))
storm.add(Activation('sigmoid'))

storm.summary()

storm.compile(	loss = 'binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])
#neural network generator
storm.fit_generator(
	train_generator,
	steps_per_epoch=train_samples,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=validation_samples)

storm.save_weights('primary.h5') #CNN learned dataset

storm_pred = image.load_img('data/validate/oo36861909206.jpg', target_size=(320,320)) #testing 1
storm_pred = image.img_to_array(storm_pred)
storm_pred = np.expand_dims(storm_pred, axis = 0)

final = storm.predict(storm_pred)
print(final)
if final[0][0] >= 0.75:
	pred = "Probable Storm"
else:
	pred = "No detected activity"

print(pred) #prints result

