from Data_Processing import *
# #### NN
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd


# 数据标准化
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)
X_train_NN =(X_train - mean_px) / std_px
X_test_NN  = (X_test - mean_px) / std_px
X_train_NN = (X_train.values).astype('float32') # all pixel values
y_train_NN = y_train.astype('int32')
X_test_NN = (X_test.values).astype('float32') # all pixel values
# In[23]:

# 修改初始化、加归一层、加dropout、改用不同的metrics
seed = 43
np.random.seed(seed)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC


def auroc(y_true, y_pred):
    return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)


input_shape = X_train_NN.shape[1]
b_size = 1024
max_epochs = 10

import tensorflow.keras as K

init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam(lr=0.001)

model = K.models.Sequential()
model.add(K.layers.Dense(units=256, input_dim=input_shape, kernel_initializer='he_normal', activation='relu',
                         kernel_regularizer=l2(0.0001)))
model.add(K.layers.LayerNormalization())
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(units=64, kernel_initializer='he_normal', activation='relu'))
model.add(K.layers.LayerNormalization())
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(units=1, kernel_initializer='he_normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=simple_adam, metrics=['accuracy', AUC(name='auc')])

# In[24]:


model.summary()

# In[25]:


print("Starting NN training")
h = model.fit(X_train_NN, y_train_NN, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("NN training finished")

# test_loss, test_acc, test_auc = model.evaluate(X_test,  y_test, verbose=2)
# print('\nTest loss:', test_loss)
# print('\nTest accuracy:', test_acc)
# print('\nTest auc:', test_auc)
# In[30]:


pred_NN = model.predict(X_test_NN)
pred_NN = [item[0] for item in pred_NN]
print(len(pred_NN))

# In[32]:


model.save('NN_model.h5')
submission = pd.DataFrame({'id':test['loan_id'], 'isDefault':pred_NN})
submission.to_csv('submission.csv', index = None)
