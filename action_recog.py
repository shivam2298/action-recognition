#python script for training on gpu
#tensorflow version 1.8.0 and cuDNN 7.1.4 and CUDA version 9.2
import tensorflow as tf
import re
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,SGD
import os
import numpy as np
from PIL import Image ,ImageOps 
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from math import floor,log10
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint
from keras.regularizers import l2




sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
inception_v3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, pooling=None)
model = Sequential()
model.add(inception_v3)
print (model.summary())

print ("----------------------------import and model download done")

def get_image(imgpath):
    desired_size = 299
    im_pth = imgpath
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    read_img = np.array(new_im)
    # mask = read_img!=0
    # read_img[mask] = 255
    return read_img



def build_model(maxlen):
	input_videos = Input((maxlen,2048), dtype='float32')
	X = LSTM(128, return_sequences=False,W_regularizer=l2(0.001), recurrent_dropout=0.5)(input_videos)
	# X = Dropout(0.5)(X)
	# X = LSTM(256, return_sequences=False,recurrent_dropout=0.5)(X)
	# X = Dropout(0.5)(X)
	X = Dense(20)(X)
	X = Activation('softmax')(X)
    
	model = Model(inputs=input_videos, outputs=X)
    
	return model


def getFeatures(X):
    tans = model.predict(X)
    ans = [np.mean(avgpool, axis=(0,1)) for avgpool in tans]
    return ans




def name_process(name):
    if(name[0]=='_'):
        return True
    elif (name[0]=='.'):
        return True
    return False

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#dataset preparation code

"""
directory = 'VideosCrop'

X = []
y = []

print ("computing features............")

for action_folder in os.listdir(directory):
	if name_process(action_folder):
		continue
	action_num = int(re.findall(r'\d+',action_folder)[0])
	if (action_num<1 or action_num>20):
		print ("error")
		break
	path = directory+'/'+action_folder
	video_count_for_curact = 0
	for frame_folder_num, frame_folder in enumerate(os.listdir(path)):
		if(name_process(frame_folder)):
			continue
		one_video = []
		video_count_for_curact = video_count_for_curact+1
		print (action_folder+" : " + frame_folder)
		frame_folder_path = path + '/' + frame_folder
		frame_count = 0
		features = np.zeros((10,2048))
		no_of_frames = len(os.listdir(frame_folder_path))/2
		for i in range(1,no_of_frames+1):        	
			if((i-1)%(no_of_frames/10)):
				continue
			frame_count=frame_count+1
			frame = '0'*(3-int(floor(log10(i))+1))+str(i)+'.png'
			img_path = frame_folder_path +'/' + frame
			reframe = get_image(img_path) 
			one_video.append(reframe)
            
			if(frame_count == 10):
				break

		features[:np.array(one_video).shape[0],:] = getFeatures(np.array(one_video)/255)
		print (action_num , " : " ,np.array(features).shape)
		X.append(features)
		print(np.array(one_video).shape)
    
	y.extend([action_num-1]*video_count_for_curact)
print (np.array(X).shape)


encoded_y = to_categorical(y,20)
print ("encoded y _ ",encoded_y.shape)
print (encoded_y[0])
np.save('X.npy', np.array(X))
np.save('encoded_y.npy',encoded_y)
"""

#----------------------------------------------------------------------
#lstm training code
"""
X = np.load("X.npy")
encoded_y = np.load("encoded_y.npy")

print ("build lstm model and training ..........")
lstm_model = build_model(10)

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train,X_test,Y_train,Y_test = train_test_split(np.array(X),encoded_y,shuffle = True,test_size = 0.10)
filepath = "models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
callbacks = [ 
	 ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto', period=1)
	]
history = lstm_model.fit(X_train, Y_train, callbacks = callbacks, validation_split=0.12, epochs = 150, batch_size = 45, shuffle=True)
score = lstm_model.evaluate(X_test,Y_test,batch_size= 32)
print (score)

np.save('split/X_test.npy', np.array(X_test))
np.save('split/Y_test.npy',Y_test)
np.save('split/X_train.npy', np.array(X_train))
np.save('split/Y_train.npy',Y_train)

#graph of loss and acc

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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  best model evaluation code
"""
from keras.models  import load_model

model = load_model('/home/chavvi/Documents/action recognition/models/best model/weights-improvement-43-0.85.hdf5')
X = np.load('split/X_test.npy')
y = np.load('split/Y_test.npy')
score = model.evaluate(X,y,batch_size= 32)
print (score)
