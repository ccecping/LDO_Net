import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
#matplotlib inline

pd.options.display.max_colwidth = 100

import random
import os

from numpy.random import seed
seed(42)

random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import glob
import cv2

from tensorflow.random import set_seed
set_seed(42)

import warnings
from sklearn.metrics import confusion_matrix
import metrics
warnings.filterwarnings('ignore')

IMG_SIZE = 224
BATCH = 32
SEED = 42

import getData

main_path = 'croppedImage/Both/'
data_path = 'datafiles/mobileNet/both/'
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
(x_train,y_train,labels) = getData.get_predict_data(main_path,data_path,'Both_train',t='predict')
(x_test,y_test,_) = getData.get_predict_data(main_path,data_path,'Both_test',t='test')


#Setting callbakcs

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-7,
    restore_best_weights=True,
)

plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.2,                                     
    patience = 2,                                   
    min_delt = 1e-7,                                
    cooldown = 0,                               
    verbose = 1
)


##base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,pooling='max')
##model_pretrained = keras.Sequential([
##    base_model,
##    
####    layers.Dense(32, activation='relu'),
####    layers.Dropout(0.1),
##    layers.Dense(3)
##    ])
##model_pretrained.summary()
base_model = tf.keras.applications.EfficientNetB7(include_top=False, weights="imagenet",classifier_activation="softmax",)

#base_model.trainable = False
##base_model.trainable = True
##
### Freeze all layers except for the
##for layer in base_model.layers[:-13]:
##    layer.trainable = False

model_pretrained = keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),#解决ValueError: Shapes (None, None) and (None, None, None, 10) are incompatible
    
##    layers.Dense(32, activation='relu'),
##    layers.Dropout(0.1),
    layers.Dense(units=3)
    ])
#model_pretrained.summary()
model_pretrained.build(input_shape=(224, 224, 3))  # 0.9316
model_pretrained.summary()
model_pretrained.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

checkpoint_filepath = "tmp_both/mobileNet/checkpoint"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
history = model_pretrained.fit(
          x=x_train,
          y=y_train, 
          batch_size = BATCH, epochs = 30,
          validation_split = 0.3,
          shuffle=True,
          callbacks=[early_stopping, plateau,checkpoint_callback],
          #callbacks=[checkpoint_callback],
          steps_per_epoch=(len(x_train)/BATCH),
          )
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
#model.save_weights('transfer_weights_2classes.h5')
model_pretrained.load_weights(checkpoint_filepath)
##score = model_pretrained.evaluate(ds_val, steps = len(val_df)/BATCH, verbose = 0)
##print('Val loss:', score[0])
##print('Val accuracy:', score[1])

score = model_pretrained.evaluate(x_test,y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
results = model_pretrained.predict(x_test)





