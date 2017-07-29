import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.merge import Concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import dataProcess
from random import randint
from sklearn.cross_validation import StratifiedKFold

class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ( "conv1 shape:",conv1.shape )
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print ( "pool1 shape:",pool1.shape )

		conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ( "conv2 shape:",conv2.shape )
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print ( "pool2 shape:",pool2.shape )

		conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ( "conv3 shape:",conv3.shape )
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print ( "pool3 shape:",pool3.shape )

		conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		print ( "conv4 shape:",conv4.shape )
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		print ( "pool4 shape:",pool4.shape )

		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		print ( "conv5 shape:",conv5.shape )
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		#merge6 = Concatenate([drop4, up6])
		conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		print ( "conv6 shape:",conv6.shape )

		up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		print ( "conv7 shape:",conv7.shape )

		up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		print ( "conv8 shape:",conv8.shape )

		up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print ( "conv9 shape:",conv9.shape )
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		print ( "conv10 shape:",conv10.shape )

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

		return model


	def generate_batch_data_random(self, x, y, batch_size):
		ylen = len(y)
		loopcount = ylen / batch_size
		while (True):
			i = randint(0,loopcount)
			yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
			
	def train(self):

		print ("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print ("loading data done")
		model = self.get_unet()
		print ("got unet")
		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		
		n_folds = 7
		skf = StratifiedKFold(imgs_mask_train[:,0,0,0], n_folds=n_folds, shuffle=False)
		for i, (train, test) in enumerate(skf):
			print("Running Fold", i+1, "/", n_folds)
			#print(train)
			model.fit(imgs_train[train], imgs_mask_train[train], validation_data=(imgs_train[test], imgs_mask_train[test]), batch_size=1, nb_epoch=5, verbose=1, shuffle=True, callbacks=[model_checkpoint])

		print ('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('imgs_mask_test.npy', imgs_mask_test)
		
	def show_model(self, png_path):
		model = self.get_unet()
		from keras.utils import plot_model
		plot_model(model, to_file=png_path)


if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	#myunet.show_model("model.png")








