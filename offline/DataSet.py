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

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

EachFile=[]
GESTURE = []
LIST=[]
LIST_TESTSET=[]
GESTURE_TESTSET=[]
######BUILD THE DATASET
### Obtain the name of each gesture
root='/home/prajval10/keras_test/NewDataSet'
dirGestures = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]


#I create a list of list of each file for each gestures
for i in dirGestures:
	root2='/home/prajval10/keras_test/NewDataSet/'+i
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

print GESTURE[0]
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
test_ind=int(train_new.shape[0]*0.1)
train_ind=train_new.shape[0]-test_ind

x_testset = train_new[-test_ind:]
y_testset = target_new[-test_ind:]
print len(x_testset)

train_x = train_new[:len(train_new)-test_ind]
train_y = target_new[:len(target_new)-test_ind]


#########BUILD THE MODEL OF THE RNN
output_neurons = 6
hidden_neurons = 100
model = Sequential()
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

adm = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.add(LSTM(hidden_neurons,input_shape=(None,8), activation='tanh', return_sequences=False,use_bias=True,bias_initializer='random_normal', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(output_neurons, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',optimizer=adm ,metrics=['accuracy'])
history = model.fit(train_x,train_y,batch_size=5,epochs=200,validation_split=0.1, shuffle = True, callbacks=[earlyStopping])

predict = model.predict(x_testset)
plot_confusion_matrix(y_testset, predict)
#print precision(y_testset, predict)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#saving model to file
#model.save('my_model_new.h5') 
