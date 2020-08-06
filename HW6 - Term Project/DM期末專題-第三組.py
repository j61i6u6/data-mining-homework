#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
# 指定要列出所有檔案的目錄
#mypath = "processed_img"

# 取得所有檔案與子目錄名稱
#files = listdir(mypath)

x_img_train = np.load('x_img_train_file.npy')
x_img_test = np.load('x_img_test_file.npy')
y_label_train = np.load('y_label_train_file.npy')
y_label_test = np.load('y_label_test_file.npy')

label_dict ={0:"十",1:"一",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九"}

#all_data = 17691
#train_data = 12383(70%)
#test_data = 5308(30%)

print("驗證檔案",len(x_img_train))
print("訓練檔案",len(x_img_test))
#print(x_img_train.shape)
#print(x_img_test)
#print(y_label_test)
def show_train_history(train_history,train,validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train','validation'],loc='upper left')
        plt.show()

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25 
    for i in range(0, num):
        ax=plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap = 'binary')
        title= "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx]) 
            
        ax.set_title(title, fontsize = 10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx += 1 
    plt.show()

print('x_train_image:',x_img_train.shape)
print('y_train_label:',y_label_train.shape)
print('x_img_test:',x_img_test.shape)
print('y_label_test:',y_label_test.shape)

x_Train4D=x_img_train.reshape(x_img_train.shape[0],190,190,1).astype('float32')
x_Test4D=x_img_test.reshape(x_img_test.shape[0],190,190,1).astype('float32')

#imgplot = plt.imshow(x_Train4D[0])
#plt.show()

x_train_normalize = x_Train4D / 255
x_Test_normalize = x_Test4D / 255

y_TrainOneHot = np_utils.to_categorical(y_label_train)
y_TestOneHot = np_utils.to_categorical(y_label_test)

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(5,5)
                 ,padding='same',input_shape=(190,190,1),activation='relu'))

model.add(MaxPooling2D(pool_size=(5,5)))

model.add(Conv2D(filters=36,kernel_size=(5,5)
                 ,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#try:
    #model.load_weights("SaveModel/Final.h5")
    #print("載入模型成功!繼續訓練模型")
#except :    
    #print("載入模型失敗!開始訓練一個新模型")

train_history = model.fit(x=x_train_normalize,y=y_TrainOneHot,validation_split=0.2,
                          epochs=50,batch_size=300,verbose=2)

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x_Test_normalize,y_TestOneHot)
print(scores[1])

prediction = model.predict_classes(x_Test4D)

import pandas as pd
print(label_dict)
print(prediction)
print(pd.crosstab(y_label_test,prediction,rownames=['label'],colnames=['predict']))

df = pd.DataFrame({'label':y_label_test,'predict':prediction})

x = df[(df['label']=='0') & (df['predict']==7) ].index.values.tolist()
#print(x)

path = "processed_img"
files = listdir(path)
for i in range(0,len(x)):
    print(files[(x[i])])


#model.save_weights("SaveModel/Final.h5")


