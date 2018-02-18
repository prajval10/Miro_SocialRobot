#!/usr/bin/env python

################################################################

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
from geometry_msgs.msg import Twist
import numpy as np
import miro_msgs
from miro_msgs.msg import platform_config,platform_sensors,platform_state,platform_mics,platform_control,core_state,core_control,core_config,bridge_config,bridge_stream
import os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
from pylab import *

import math
import numpy
import time
import sys
from miro_constants import miro
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from datetime import datetime
import keras
from keras import metrics
from keras import backend as K
from keras import regularizers
import tensorflow as tf

from keras.models import load_model


def fmt(x, f): #function used for conversion from byteArray to String (The values that we get for the head touch sensors are byteArrays)
    s = ""
    x = bytearray(x)
    for i in range(0, len(x)):
        if not i == 0:
            s = s + ", "
        s = s + f.format(x[i])
    return s



class miro_touch_control_ros_client:
	def callback_platform_sensors(self, subscriber_msg):
		# ignore until active
		if not self.active:
			return
		global touch_sensors	
		global new_data
		global dataset
		global send_data		
    		global start_index
    		global end_index
    		global data_flag
    		global pattern
    		global count
    		global model
    		global graph
    			
		
		#print(fmt(subscriber_msg.touch_head, '{0:.0f}')) #Uncomment and see what happens in output, It converts from byteArray to str
		
		new_data = True #set receive flag
		touch_sensors = [0,0,0,0,0,0,0,0]

		touch_sensors[0] = int(fmt(subscriber_msg.touch_head, '{0:.0f}')[0]) # Sensor at 16'oclock on head
		touch_sensors[1] = int(fmt(subscriber_msg.touch_head, '{0:.0f}')[3]) # Sensor at 14'oclock on head
		touch_sensors[2] = int(fmt(subscriber_msg.touch_head, '{0:.0f}')[6]) # Sensor at 10'oclock on head
		touch_sensors[3] = int(fmt(subscriber_msg.touch_head, '{0:.0f}')[9]) # Sensor at 20'oclock on head
		touch_sensors[4] = int(fmt(subscriber_msg.touch_body, '{0:.0f}')[0]) # Sensor at 16'oclock on head
		touch_sensors[5] = int(fmt(subscriber_msg.touch_body, '{0:.0f}')[3]) # Sensor at 14'oclock on head
		touch_sensors[6] = int(fmt(subscriber_msg.touch_body, '{0:.0f}')[6]) # Sensor at 10'oclock on head
		touch_sensors[7] = int(fmt(subscriber_msg.touch_body, '{0:.0f}')[9]) # Sensor at 20'oclock on head
		#print touch_sensors

	def loop(self):
		global new_data
		global touch_sensors
		global dataset
		global data_flag
		global start_index
		global end_index
		global pattern
		global send_data
		gesture_str = ""
		while not rospy.is_shutdown():
			if new_data:
				new_data = False #un-set flag
			
				#print aList
				dataset.append(touch_sensors)
				if len(dataset) == 105:
					data_flag = True
				#print "before %d",len(dataset)
				if data_flag == True:
					pattern = dataset
					x_dataset = np.array(pattern)		    						
    					dataset.pop(0)
					X_train = np.reshape(x_dataset, (1,x_dataset.shape[0], x_dataset.shape[1]))
    					XDataset = pad_sequences(X_train, maxlen=None, dtype='int32',padding='pre', truncating='pre', value=0.)
    					aux =0
					with self.graph.as_default():
						predict = self.model.predict(X_train)
						threshold = 0.5
						for i in range(len(predict[0])):
							if predict[0][i] >= threshold:
								aux = 1

						if aux == 1:
							index = np.argmax(predict) #index of the maximum value
							if index == 0:
								gesture_str = "CarBottomTop"
							elif index == 1:
								gesture_str = "FixedBody"
							elif index == 2:
								gesture_str = "PatHead"
							elif index == 3:
								gesture_str = "FixedHead"
							elif index == 4:
								gesture_str = "PatBody"
							elif index == 5:
								gesture_str = "CarTopBottom"
						else:
							gesture_str = "No_Gesture"

						rospy.loginfo(gesture_str)                                     # It's like rosinfo in C++
						self.pub.publish(gesture_str)

						self.rate.sleep()


	def __init__(self):
		# report
		print("initialising...")
		print(sys.version)
		print(datetime.time(datetime.now()))

		# default data
		self.platform_sensors = None

		# options
		self.robot_name = ""
		
		# handle args
		for arg in sys.argv[1:]:
			f = arg.find('=')
			if f == -1:
				key = arg
				val = ""
			else:
				key = arg[:f]
				val = arg[f+1:]
			if key == "robot":
				self.robot_name = val
			else:
				error("argument not recognised \"" + arg + "\"")

			

		# check we got at least one
		if len(self.robot_name) == 0:
			error("argument \"robot\" must be specified")

		# set inactive
		self.active = False

		# topic root
		topic_root = "/miro/" + self.robot_name
		print "topic_root", topic_root

		# publish
		self.pub = rospy.Publisher('classifyGesture', String, queue_size=10)    # classifyGesture is the name of the topic on which we will publish , string is the parameter

		####### Deleated some functions from here (compared to the orignal file)

		# subscribe
		self.sub_sensors = rospy.Subscriber(topic_root + "/platform/sensors", platform_sensors, self.callback_platform_sensors)
		####### Deleated some functions from here (compared to the orignal file)
		global model
		global graph
		
		self.model = load_model('my_model_new.h5')
		self.model._make_predict_function()
		self.graph = tf.get_default_graph()

		adm = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		self.model.compile(loss='categorical_crossentropy',optimizer=adm ,metrics=['accuracy'])

		# set active
		self.active = True

		self.rate = rospy.Rate(20)

if __name__ == "__main__":
	global count
	global data_flag
	global dataset
	global pattern
	global send_data
	global start_index
    	global end_index
    	global model
    	global graph
    	global touch_sensors
    	global new_data
    	new_data = False
    	touch_sensors = []
	start_index = 0
	end_index = 105
	pattern = []
	send_data = []
	dataset = []
	data_flag = False
	count = 0
	rospy.init_node("miro_touch_control_ros_client_py", anonymous=True)
	main = miro_touch_control_ros_client()
	main.loop()
	rospy.spin()

