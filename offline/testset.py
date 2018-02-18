import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
from pylab import *
from random import shuffle
import keras
from keras import metrics
from keras import backend as K
from keras import regularizers
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model

def plot_confusion_matrix(target_test, pred_y):
	"""
	This function prints and plots the confusion matrix	 
	"""
	#cmap = plt.cm.Blues
	#compute confusion matrix
	cm = confusion_matrix(np.argmax(target_test,axis=1), np.argmax(pred_y, axis=1))
	np.set_printoptions(precision=2)
	print(cm)	



EachFile=[]
GESTURE = []
LIST=[]
LIST_TESTSET=[]
GESTURE_TESTSET=[]
######BUILD THE DATASET
### Obtain the name of each gesture
root='/home/prajval10/keras_test/datasetToday'
dirGestures = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]


#I create a list of list of each file for each gestures
for i in dirGestures:
	root2='/home/prajval10/keras_test/datasetToday/'+i
	onlyfile = [f for f in listdir(root2) if isfile(join(root2, f))]		
	EachFile.append(onlyfile)
#print EachFile

label = []
for i in range(len(EachFile)):
	root3=root+'/'+dirGestures[i]
	for j in range(len(EachFile[i])):
		root4=root3+'/'+EachFile[i][j]
		inputFile = open(root4,'r')		
		File = inputFile.read().split()
		for k in range(len(File)):
			aux = map(int, File[k])
			LIST.append(aux)
		label.append(i)
		#print len(LIST)
		LIST=[]
		GESTURE.append(LIST)

#print dirGestures
#here I did the padding it is necessary for work with the network in python.
x_dataset = pad_sequences(GESTURE, maxlen=105, dtype='int32',padding='pre', truncating='pre', value=0.)

#target value converted in matrix
y_dataset = np_utils.to_categorical(label, 6)
#print label
#print x_dataset[1]
maximum_sample = np.shape(x_dataset)

#print len(x_dataset)

ind_list = [i for i in range(len(GESTURE))]
shuffle(ind_list)
train_new  = x_dataset[ind_list,]
target_new = y_dataset[ind_list,]

# valid_x = train_new[-30:]
# valid_y = target_new[-30:]
test_ind=int(train_new.shape[0]*0.2)
train_ind=train_new.shape[0]-test_ind

x_testset = train_new[-test_ind:]
y_testset = target_new[-test_ind:]


train_x = train_new[:len(train_new)-test_ind]
train_y = target_new[:len(target_new)-test_ind]


model = load_model('my_model_new.h5')
adm = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['binary_accuracy', 'fmeasure', 'precision', 'recall'])
model.compile(loss='categorical_crossentropy',optimizer=adm ,metrics=['accuracy'])
print np.shape(train_new)
predict = model.predict(x_testset)
metrics = model.evaluate(x_testset,y_testset)
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
plot_confusion_matrix(y_testset, predict)