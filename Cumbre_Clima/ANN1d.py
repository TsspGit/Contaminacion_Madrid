#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__ = ' AUTOR:    Ivan Mendez Jimenez y Miguel Cardenas Montes'

''' ANN unidimensional en Keras para contaminantes Madrid '''

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

import keras.backend as K



from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

##Para cambiar las fuentes
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'large'

import matplotlib as mpl
np.set_printoptions(threshold=np.inf)

import pdb # Para depurar
import copy


def personalloss(y_true, y_pred):
	return K.sum(y_true*K.square((y_true-y_pred)))

##########################################
############## Codigo ####################
##########################################

"""Codigo principal"""
def main():

	#values = np.loadtxt("./Est03PlC_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # Plaza del Carmen

	#values = np.loadtxt("../rawdata/Est24CC_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) #  CC
	#values = np.loadtxt("../rawdata/Est24CC_CodPar08NO2_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) #  CC
	#values = np.loadtxt("../rawdata/Est24CC_CodPar06CO_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) #  CC

	values = np.loadtxt("../rawdata/Est16AS_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # AS
	#values = np.loadtxt("../rawdata/Est16AS_CodPar08NO2_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # AS
	#values = np.loadtxt("../rawdata/Est16AS_CodPar06CO_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # AS

	#values = np.loadtxt("../rawdata/Est56FL_CodPar14O3_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # FL
	#values = np.loadtxt("../rawdata/Est56FL_CodPar06CO_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # FL
	#values = np.loadtxt("../rawdata/Est56FL_CodPar08NO2_valuesonly.txt", skiprows=1, unpack=True, dtype=np.float ) # FL

	#print( len(values))


	################ ANN's en Keras ###################

	# Dataset
	dataset = values

	'''Tama\~no de las muestras con las que entrenar la red. 
	No puede ser mayor que el numero total de medianas 
	disponibles (48), y debe tener al menos 1 elemento.'''

	sample_size = 30 
	ahead = 7
	nepochs=10 #50

	assert 0 < sample_size < len(dataset)

	### Creacion de las muestras a partir del array de entrada ###

	''' Para cada muestra de X, 'y' es el siguiente valor en la serie 
	temporal. Ej. muestra 1: (X --> y) = (dataset[0:11] --> dataset[12])'''	

	dataset = np.asarray(dataset)
	
	X = np.atleast_3d(np.array([dataset[start:start + sample_size] 
				for start in range(0, dataset.shape[0]-sample_size)]))
	y = dataset[sample_size:] 


	# Creamos los conjuntos de datos de entrenamiento y de evaluacion. (192 valores)

	test_size = 1000 
 	
	trainX, testX = X[:-test_size], X[-test_size:]
	trainY, testY = y[:-test_size], y[-test_size:]


	nextSteps = np.empty((ahead+1,sample_size,1))
	nextSteps[0,:,:]= np.atleast_3d(np.array([dataset[start:start + sample_size] 
		for start in range(dataset.shape[0]-sample_size,dataset.shape[0]-sample_size+1)]))

	print('trainX.shape',trainX.shape)

	####### Creamos la estructura de la FFNN ###########
	####(usamos ReLU's como funciones de activacion)###
	
	# 2 capas ocultas con 64 y 32 (y 64) neuronas, respectivamente
	neurons = [64, 32] 

	# Creamos la base del modelo
	model = Sequential() 

	# Ponemos una primera capa oculta con 64 neuronas
	model.add(Dense(neurons[0], activation='relu', input_shape=(sample_size, 1)))
	print(model.layers[-1].output_shape)
	
	# Incorporamos una segunda capa oculta con 32 neuronas
	model.add(Dense(neurons[1], activation='relu'))
	print(model.layers[-1].output_shape)
	
	# Aplanamos los datos para reducir la dimensionalidad en la salida
	model.add(Flatten())

	# A\~nadimos la capa de salida de la red con activacion lineal
	model.add(Dense(1, activation='linear'))
	print(model.layers[-1].output_shape)
	# Compilamos el modelo usando el optimizador Adam
	model.compile(loss=personalloss, optimizer="adam") 

	#keras.utils.layer_utils.print_layer_shapes(model,input_shapes=(trainX.shape))

	# Entrenamos la red
	history=model.fit(trainX, trainY, epochs=nepochs, batch_size=100, validation_data=(testX, testY), verbose=2) 

	# Pronosticos 

	pred = model.predict(testX)

	#print('\n\nReal', '	', 'Pronosticado')
	
	#for actual, predicted in zip(testY, pred.squeeze()):
		#print(actual.squeeze(), '	', predicted)

	
	# Calculamos el ECM y el EAM

	evaluacionECM = mean_squared_error(testY, pred)
	print('ECM: %.4f' % (evaluacionECM))

	evaluacionEAM = mean_absolute_error(testY, pred)
	print('EAM: %.4f' % (evaluacionEAM))

	print(model.summary())

	newValues = np.zeros(ahead)
	temp=np.zeros(sample_size)

	for i in range(ahead):
		#print('ahead',i)

		#print('prediccion ', model.predict(nextSteps[None,i,:]), model.predict(nextSteps[None,i,:]))
		temp=nextSteps[i,1:,:]
		#print(temp, len(temp))
		temp = np.append(temp,model.predict(nextSteps[None,i,:]), axis=0)
		newValues[i] = model.predict(nextSteps[None,i,:])
		#print(temp, len(temp))

		#print(nextSteps[i,:,:])
		nextSteps[i+1,:,:]= temp
		#print(nextSteps[i+1,:,:])


	print('newValues: ', *newValues, sep=', ')

	#plt.figure(1)
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,6)) # (18,6)
	ax.plot( np.arange(len(values)), values, "ko", alpha=0.4,markersize=2 ) #
	#ax.plot( np.arange(len(values),len(values)+len(newValuesReal) ), newValuesReal, "go", alpha=0.4,markersize=2 )
	ax.plot( np.arange(len(values)-len(testY),len(values)), pred, linewidth=2.5, alpha=0.99, linestyle='--',color='r' ) #
	#ax.plot( np.arange(len(values),len(values)+len(newValues)), newValues, linestyle='--', color='b', linewidth=2.5 )

	#plt.suptitle('Daily Predictions of O3 at AS')
	#plt.savefig('./pred_CO_Est16_BRNN_weekly_BRNN50_D100_D1_e50_b1e3_ss30_ts1e3_20100101_20180531.eps')

	# summarize history for loss
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))# 6,6
	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	#ax.set_yscale("log")
	#plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.savefig('./loss_Est16_BRNN_weekly_BRNN50_D100_D3_e50_b1e3_ss100_ts1e3_20100101_20180531.eps')


	plt.show()

"""Invoking the main."""
main()
