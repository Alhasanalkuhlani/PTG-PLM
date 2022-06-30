from tensorflow import keras
from keras import layers
import pandas as pd
import os
import logging
from configparser import ConfigParser
logger = logging.getLogger(__name__)
def CNNModel(X_train,Y_train,X_valid,Y_valid,config_file='CNN_config.ini'):
	parser = ConfigParser(os.environ)
	if not os.path.exists('CNN_config.ini'):
		raise IOError("Configuration file '%s' does not exist" % config_file)
	logging.info('Loading config from %s', config_file)
	parser.read('CNN_config.ini')
	config_header = 'CNN'

	logger.info('config header: %s', config_header)
	
	filters = parser.getint(config_header, 'filters')
	kernels = parser.getint(config_header, 'kernels')
	dense_layers1 = parser.getint(config_header, 'dense_layers1')
	dense_layers2 = parser.getint(config_header, 'dense_layers2')
	Learning_Rate = parser.getfloat(config_header, 'Learning_Rate')
	dropout = parser.getfloat(config_header, 'dropout')
	Batch_size = parser.getint(config_header, 'Batch_size')
	Epochs = parser.getint(config_header, 'Epochs')
	earlystop = parser.getboolean(config_header, 'earlystop')
	
	
	train_shape=X_train.shape
	inputs = keras.Input(shape=(None,train_shape[1]), dtype="float32")

	kmodel = keras.models.Sequential()
	kmodel.add(layers.Reshape(input_shape=(train_shape[1], train_shape[2]),
					  target_shape=(train_shape[1], train_shape[2])))
	kmodel.add(layers.Conv1D(filters, kernel_size=kernels, padding='same',batch_input_shape=(None, train_shape[1], train_shape[2])))
	kmodel.add(layers.BatchNormalization(name="conv_1_bn"))
	kmodel.add(layers.ReLU(name="conv_1_relu"))

	kmodel.add(layers.Flatten())
	kmodel.add(layers.Dense(dense_layers1, activation = 'relu'))
	kmodel.add(layers.Dense(dense_layers2, kernel_regularizer=keras.regularizers.l2(0.001),
					activity_regularizer=keras.regularizers.l1(0.001)))
	kmodel.add(layers.Activation('relu'))
	kmodel.add(layers.Dropout(dropout))
	kmodel.add(layers.Dense(1,activation='sigmoid'))
	opt = keras.optimizers.Adam(learning_rate=Learning_Rate)
	kmodel.compile(loss= "binary_crossentropy", metrics=["accuracy"],optimizer=opt)
	callback1=keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0005, patience=8, restore_best_weights=True )
	if earlystop:
		kmodel.fit(X_train, Y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(X_valid, Y_valid),callbacks=callback1)
	else:
		kmodel.fit(X_train, Y_train, batch_size=Batch_size, epochs=Epochs, validation_data=(X_valid, Y_valid))
	return kmodel





	