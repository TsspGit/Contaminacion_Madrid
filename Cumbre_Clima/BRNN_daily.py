
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__ = ' AUTOR: Miguel Cardenas-Montes'

''' RNN Bidireccional en Keras '''


"""Imports: librerias"""
import os
import math
from math import sqrt
import sys
import numpy
import numpy as np
import random
import scipy.stats
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, GRU, Bidirectional 
from keras.layers.normalization import BatchNormalization

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd  

import tensorflow as tf

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates


## Para cambiar las fuentes
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'large'

import matplotlib as mpl
np.set_printoptions(threshold=np.inf)

import pdb # Para depurar
import copy


##########################################
##############  Codigo  ##################
##########################################

"""Codigo principal"""

def main():

	#values = np.loadtxt("../rawdata/Est16AS_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # Arturo Soria
	values = np.loadtxt("./Est16AS_CodPar08NO2_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # Arturo Soria
	#values = np.loadtxt("../rawdata/Est16AS_CodPar06CO_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # Arturo Soria

	#values = np.loadtxt("../rawdata/Est56FL_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # FL
	#values = np.loadtxt("../rawdata/Est56FL_CodPar08NO2_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # FL
	#values = np.loadtxt("../rawdata/Est56FL_CodPar06CO_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # FL

	#values = np.loadtxt("../rawdata/Est24CC_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # CC
	#values = np.loadtxt("../rawdata/Est24CC_CodPar08NO2_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # CC
	#values = np.loadtxt("../rawdata/Est24CC_CodPar06CO_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # CC

	# Mark data
	#values = np.loadtxt("../rawdata/Markdata/O3data.txt", skiprows=1, unpack=True, dtype=np.float ) # CC
	print( len(values))

################### RNN Bidireccional en Keras ###################

	# Normalizar datos (funcion de activacion tanh)
	scaler = MinMaxScaler(feature_range=(-1, 1)) 
	values = np.array(values).reshape(-1,1)
	dataset = scaler.fit_transform(values)

	sample_size = 100 
	ahead = 7
	dataset = np.asarray(dataset)

	nepochs=1 #50

	assert 0 < sample_size < dataset.shape[0] 

	## Creacion de las muestras a partir del array normalizado ##
	X = np.atleast_3d(np.array([dataset[start:start + sample_size] 
		for start in range(0, dataset.shape[0]-sample_size)]))
	y = dataset[sample_size:]
	qf = np.atleast_3d([dataset[-sample_size:]]) 

	# Separamos en datos de entrenamiento y evaluacion
	test_size = 1000 		

	trainX, testX = X[:-test_size], X[-test_size:]
	trainY, testY = y[:-test_size], y[-test_size:]


	nextSteps = np.empty((ahead+1,sample_size,1))
	nextSteps[0,:,:]= np.atleast_3d(np.array([dataset[start:start + sample_size] 
		for start in range(dataset.shape[0]-sample_size,dataset.shape[0]-sample_size+1)]))

	### Creamos la estructura de la RNN con LSTM's ###

	# Neuronas en la primera y segunda capa ocultas, respectivamente
	neurons = [50,50,12] 

	# Base del modelo
	model = Sequential()
	
	# Incorporamos una capa oculta de celulas LSTM
	model.add(Bidirectional(LSTM(neurons[0], activation='tanh', return_sequences=True), 
				input_shape=(X.shape[1],X.shape[2])))

	#model.add(Dropout(0.2)) 	# Dropout para evitar el sobreajuste

	# Introducimos otra capa oculta de celulas LSTM
	#model.add(Bidirectional(LSTM(neurons[1], activation='tanh', return_sequences=True))) 
	#model.add(Dropout(0.6)) 	# Dropout para evitar el sobreajuste

	#model.add(Bidirectional(LSTM(neurons[2], activation='tanh'))) 
	#model.add(Dropout(0.6)) 	# Dropout para evitar el sobreajuste

	#model.add(BatchNormalization()) # tested> does not improve appreciablely 

	model.add(Flatten())

	# Capa de salida con activacion lineal
	model.add(Dense(100, activation='tanh'))
	model.add(Dense(1, activation='linear'))

	# Compilamos el modelo
	model.compile(loss="mse", optimizer="adam", metrics=['mse']) # previously optimizer=adam , o rmsprop

	# Entrenamos la red con xx epocas

	history=model.fit(trainX, trainY, epochs=nepochs, batch_size=100, validation_data=(testX, testY), verbose=2) 
	
	### Pronostico de los datos de test ###
	pred = model.predict(testX)

	# Denormalizamos la informacion
	pred = scaler.inverse_transform(pred)
	testY = scaler.inverse_transform(testY)

	#print ('\n\nReal', '	', 'Pronosticado')
	
	# squeeze() elimina las dimensiones que figuran a 1
	#for actual, predicted in zip(testY, pred.squeeze()):
		#print (actual, '	', predicted)

	# Calcular ECM y EAM
	testScoreECM = mean_squared_error(testY, pred)
	print('ECM: %.4f' % (testScoreECM))

	testScoreEAM = mean_absolute_error(testY, pred)
	print('EAM: %.4f' % (testScoreEAM))

	#print(model.output_shape)
	print(model.summary())
	#print(model.get_config())
	#print ('next value: ', scaler.inverse_transform(model.predict(qf).squeeze()))

	
	newValues = np.zeros(ahead)
	temp=np.zeros(sample_size)

	for i in range(ahead):
		#print('ahead',i)

		#print('prediccion ', model.predict(nextSteps[None,i,:]), scaler.inverse_transform(model.predict(nextSteps[None,i,:])) )
		temp=nextSteps[i,1:,:]
		#print(temp, len(temp))
		temp = np.append(temp,model.predict(nextSteps[None,i,:]), axis=0)
		newValues[i] = scaler.inverse_transform(model.predict(nextSteps[None,i,:]))
		#print(temp, len(temp))

		#print(nextSteps[i,:,:])
		nextSteps[i+1,:,:]= temp
		#print(nextSteps[i+1,:,:])


	print('newValues: ', newValues)

	startday = pd.datetime(2010, 1, 1)
	startdaypred = pd.datetime(2010, 1, 1) + pd.Timedelta( len(values)-len(pred), unit='d' )
	startdayahead = pd.datetime(2010, 1, 1) + pd.Timedelta( len(values), unit='d' )
	print(startday,startdaypred,startdayahead)

	# list all data in history
	#print(history.history.keys())

	#plt.figure(1)
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,6)) # (18,6)
	plt.figure(1)
	xaxis = ax.get_xaxis()
	ax.xaxis.grid(b=True, which='minor', color='0.90', linewidth=0.6)
	ax.xaxis.set_major_locator(mdates.YearLocator())
	ax.xaxis.set_minor_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

	ax.plot(pd.date_range(startday, periods=len(values), freq='D'), values, "ko", alpha=0.4,markersize=2) # 
	ax.plot(pd.date_range(startdaypred, periods=test_size, freq='D'), pred, linewidth=2.5, alpha=0.99, linestyle='--',color='r' ) 

	#ax.plot( np.arange(len(values)), values, "ko", alpha=0.4,markersize=2 ) #
	#ax.plot( np.arange(len(values),len(values)+len(newValuesReal) ), newValuesReal, "go", alpha=0.4,markersize=2 )
	#ax.plot( np.arange(len(values)-len(testY),len(values)), pred, linewidth=2.5, alpha=0.99, linestyle='--',color='r' ) #
	#ax.plot( np.arange(len(values),len(values)+len(newValues)), newValues, linestyle='--', color='b', linewidth=2.5 )

	#plt.suptitle('Daily Predictions of O3 at AS')
	plt.ylabel(r'$[\mu g/m^3]$')
	#plt.savefig('./pred_NO2_Est24_BRNN_weekly_BRNN50_D100_D1_e50_b1e2_ss100_ts1e3_20100101_20180531.eps')


	# summarize history for accuracy
	#fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
	#plt.figure(2)
	#plt.plot(history.history['acc'])
	#plt.plot(history.history['loss'])
	#plt.title('model accuracy')
	#plt.ylabel('loss')
	#plt.xlabel('epoch')
	#plt.savefig('./loss_O3_Est16_BRNN_weekly_BRNN50_D100_D1_e50_b1e3_ss30_ts1e3_20100101_20180531.eps')

	#plt.legend(['loss', 'val_loss'], loc='upper left')
	# summarize history for loss
	#plt.figure(3)
	#plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	#plt.title('model loss')
	#plt.ylabel('loss')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')

	plt.show()

"""Invoking the main."""
main()

