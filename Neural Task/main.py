import cv2
import os
from random import shuffle
from tqdm import tqdm
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import testing_data
## ########## preprocesssing data set##########

Train_DIR = 'Scenes training set'
Test_DIR = ''
Image_Size = 150
LR = 0.0005

MODEL_NAME = 'Classification-cnn'


def load_images_from_folder(folder):
    images_folder = []
    for image in tqdm(os.listdir(folder)):
        path = os.path.join(folder, image)
        image_data = cv2.imread(path, 0)
        image_data = cv2.resize(image_data, (Image_Size, Image_Size))
        if image_data is not None:
            images_folder.append([np.array(image_data), create_label(image)])
    return images_folder


def create_train_data(root):
    training_data = []
    folders = [os.path.join(Train_DIR, x) for x in ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')]
    training_data = [img for folder in folders for img in load_images_from_folder(folder)]
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_label(image_name):
    data = pa.read_csv('Classes.csv')
    Class_names = data['ClassName']
    Class_Label = data['ClassLabel']
    word_label = image_name.split('.')[-3]
    if word_label == 'buildings':
        return Class_Label[0]
    elif word_label == 'forest':
        return Class_Label[1]
    elif word_label == 'glacier':
        return Class_Label[2]
    elif word_label == 'mountain':
        return Class_Label[3]
    elif word_label == 'sea':
        return Class_Label[4]
    elif word_label == 'street':
        return Class_Label[5]


# def create_test_data()

if (os.path.exists('train_data.npy')):
    train_data = np.load('train_data.npy')
else:
    train_data = create_train_data(Train_DIR)

train = train_data
X_train = np.array([i[0] for i in train]).reshape(-1, Image_Size, Image_Size, 1)
y_train = np.array([i[1] for i in train])

############## split dataset for Validation######

train_X,test_X,train_Y,test_Y = train_test_split(X_train,y_train,test_size=0.30,shuffle=True)

########### Create MODEL ######

tf.reset_default_graph()
data_input = input_data(shape=[None, Image_Size, Image_Size, 1], name='input')
conv1 = conv_2d(data_input, 32, 5, activation='relu')
pooling1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pooling1, 64, 5, activation='relu')
pooling2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pooling2, 128, 5, activation='relu')
pooling3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pooling3, 256, 5, activation='relu')
pooling4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pooling4,512, 5,activation='relu')
pooling5 = max_pool_2d(conv5, 5)

conv6 = conv_2d(pooling5, 265, 5, activation='relu')
pooling6 = max_pool_2d(conv6, 5)

conv7 = conv_2d(pooling6, 128, 5, activation='relu')
pooling7 = max_pool_2d(conv7, 5)

conv8 = conv_2d(pooling7, 64, 5, activation='relu')
pooling8 = max_pool_2d(conv8, 5)

conv9 = conv_2d(pooling8,32, 5,activation='relu')
pooling9 = max_pool_2d(conv9, 5)

fully_layer = fully_connected(pooling9, 1024, activation= 'relu')
fully_layer = dropout(fully_layer,0.5)

cnn_layer = fully_connected(fully_layer,6,'softmax')
cnn_layer = regression(cnn_layer,optimizer='adam', loss='categorical_crossentropy',learning_rate=LR  )

model = tflearn.DNN(cnn_layer,tensorboard_dir= 'log',tensorboard_verbose= 3)


######Learning MODEL#########

if(os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    model.fit({'input':train_X}, {'targets': train_Y}, n_epoch= 50,
              validation_set=({'input':test_X,'targets':test_Y }),
              snapshot_step=500, show_metric=True, run_id= MODEL_NAME)
    model.save('model.tfl')




# img = cv2.imread('20096.jpg',0)
# img_test= cv2.resize(img,(Image_Size,Image_Size))
# img_test = img_test.reshape(Image_Size,Image_Size,1)
# prediction = model.predict([img_test])[0]
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# ax.imshow(img,cmap='gray')
# print(f" buildings: {prediction[0]}, forest: {prediction[1]},glacier: {prediction[2]},mountain: {prediction[3]},sea: {prediction[4]},street: {prediction[5]}, ")
#
# plt.show()
