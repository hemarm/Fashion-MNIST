# -*- coding: utf-8 -*-
"""
@author: Hema M
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import os.path
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16

# from sklearn.metrics import log_loss, confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense

from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

#################################################################################
##                     NON-IMAGE DATA 
#################################################################################


#Check train distribution 
########################################################
## TRAIN DATA

os.chdir('D:/Identify_the_apparel_AV/train_LbELtWX/')
train = pd.read_csv('train.csv')
train.columns, train.shape

#####################################
#VALIDATION DATA 

os.chdir('D:/Identify_the_apparel_AV/test_ScVgIM0/')
val = pd.read_csv('test.csv')
val.columns, val.shape


########################################################
#Create .npy file of csv data (non image data)

train_label = []
train_id = []

train_id = train['id'].tolist()
train_label = train['label'].tolist()


np.save('D:/Assignment/DL/Identify_the_apparel_AV/train_id.npy', train_id)
np.save('D:/Assignment/DL/Identify_the_apparel_AV/train_label.npy', train_label)


#################################################################################
##                             IMAGE DATA 
#################################################################################
#img_height, img_width  = 28, 28
nb_channels = 3
training_data_image_1 = []

#MAKE NPY FILE OF IMAGE DATA
train_image_path = 'D:/Assignment/DL/Identify_the_apparel_AV/train_LbELtWX/train'
first_image_location = os.listdir(train_image_path) #.tolist()
for i in first_image_location:
    first_image_path = os.path.join(train_image_path, i)
    if(os.path.exists(first_image_path) == False):
        continue
    first_img_data = cv2.imread(first_image_path)
    if first_img_data is None: 
        continue
    first_img_data = cv2.resize(first_img_data, (48,48))
    #print(np.shape(first_img_data))
#    print('processing train data ' + str(first_img_data))
#    first_img_data = cv2.resize(first_img_data, (img_height, img_width)).astype(np.float32)/255
#    first_img_data = first_img_data.astype(np.float32)/255
    training_data_image_1.append(np.array(first_img_data))

len(training_data_image_1), type(training_data_image_1)
np.shape(training_data_image_1)

np.save('D:/Assignment/DL/Identify_the_apparel_AV/train_images_not_divided_by_255.npy', training_data_image_1)
#np.save('D:/Assignment/DL/Identify_the_apparel_AV/train_images_1_to_100.npy', training_data_image_1)

VGG_DATA_PATH = '.'
MODEL_PATH_1 = 'CNN_VGG16_identify_apparel_model_v1.hdf5'
MODEL_WEIGHTS_PATH_1 = 'CNN_VGG16_identify_apparel_weights_v1.h5'
TRAINING_LOG_FILE = 'training_labels_v1.csv'

########################################################
#TRAIN_Check = 'D:/Identify_the_apparel_AV/train_images_1_to_100.npy'
TRAIN_NPY = 'D:/Identify_the_apparel_AV/train_images_not_divided_by_255.npy'
INPUT_ID_NPY = 'D:/Identify_the_apparel_AV/train_id.npy'
INPUT_LABEL_NPY = 'D:/Identify_the_apparel_AV/train_label.npy'

train_images = np.load(TRAIN_NPY)
data_id = np.load(INPUT_ID_NPY)
data_label = np.load(INPUT_LABEL_NPY)

batch_size=32
gen = ImageDataGenerator(horizontal_flip = False,
                         rescale=1./255,
                         vertical_flip = False,
                         width_shift_range = 0.0,
                         height_shift_range = 0.0,
                         zoom_range = 0.0,
                         rotation_range = 0.0)

def gen_flow_for_two_inputs(X1, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    #genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            yield ([X1i[0]], X1i[1])

#num_classes = 10
#data_label_ohe = np_utils.to_categorical(data_label, num_classes)         

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

data_label_ohe=indices_to_one_hot(data_label,10)

#data_image_1 = np.load(INPUT_IMG_NPY)
#est_cost =  np.load(INPUT_COST_NPY)
#claim_ids =  np.load(CLAIM_IDS_NPY)
#image_ids =  np.load(IMAGE_IDS_NPY)
#image_locations =  np.load(IMAGE_LOCATION_NPY)
#severity =  np.load(SEVERITY_NPY)

#img_height, img_width = 28, 28
#train_images_reshaped = np.reshape(train_images,newshape=(-1, img_height, img_width, nb_channels))
training_data_image_1_train, training_data_image_1_test, data_label_train, data_label_test = train_test_split(train_images, data_label_ohe,test_size=0.1, random_state=1111)
gen_flow = gen_flow_for_two_inputs(training_data_image_1_train, data_label_train)
gen_flow_validation = gen_flow_for_two_inputs(training_data_image_1_test, data_label_test)

print(training_data_image_1_train.shape)  #(54000, 48, 48, 3)
print(data_label_train.shape)  #(54000, 10)
print(training_data_image_1_test.shape)  #(6000, 48, 48, 3)
print(data_label_test.shape)   #(6000, 10)

a = np.array(data_label) 
fig = plt.figure()
plt.hist(a, bins = [0,1,2,3,4,5,6,7,8,9])
plt.title("histogram-class-labels")
plt.xlabel("Class")
plt.ylabel("Frequency claims")
plt.show()
fig.savefig('D:/Assignment/DL/Identify_the_apparel_AV/histogram-class-labels.jpg')


def create_VGG16_model():
    vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = (48,48,3))
#    vgg16 = VGG16(weights = None, include_top = False, input_shape = (224,224,3)) #uncomment in next iteration  
    
    #Add the fully-connected layers
    x = Flatten(name='flatten', input_shape=vgg16.output_shape[1:])(vgg16.output)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation = 'relu', name = 'fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name = 'predictions')(x)
    #Create your own model
    model = Model(inputs=vgg16.input, outputs=x)
    
    # LOAD WEIGHTS AFTER FIRST ITERATION
#    model.load_weights(weight_to_load)
    
#     for layers in model.layers[:15]:
#         layers.trainable = False
        
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True) # uncomment in next iteration
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = sgd,
                 metrics = ['accuracy']
                 )
    model.summary()
    return model

model16 = create_VGG16_model()

#CHECKPOINT
filepath=os.path.join(VGG_DATA_PATH, MODEL_PATH_1)
weights_path = os.path.join(VGG_DATA_PATH, MODEL_WEIGHTS_PATH_1)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=5, save_best_only=True, mode='max', save_weights_only=False)
checkpoint_weights = ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True, mode='max', save_weights_only=True)
earlyStopping= EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='max')
csv_logger = CSVLogger(TRAINING_LOG_FILE)
tensorbd = TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=False)
callbacks_list = [checkpoint, checkpoint_weights, earlyStopping, csv_logger,tensorbd]

nb_epoch = 30
for e in range(nb_epoch):
    print('Training Epoch', e)
    batches = 0
    for x_batch, y_batch in gen_flow:
        print(type(x_batch))
#        x_batch = x_batch.astype(np.float32)/255
        model16.fit(x_batch, y_batch, callbacks=callbacks_list, shuffle=True, validation_data=(training_data_image_1_test, data_label_test))
        batches += 1
        if batches >= len(training_data_image_1_train) / 32:
            break







