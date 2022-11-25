#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import tensorflow as tf
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator


# In[2]:


train_dir = pathlib.Path("C:/Users/irush/Downloads/archive/fruits-360_dataset/fruits-360/Training")

batchSize = 100
height = 100
width = 100


# In[3]:


trainData =  tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (height, width),
    batch_size = batchSize
)


# In[4]:


validationData =tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (height, width),
    batch_size = batchSize
)


# In[5]:


class_names = trainData.class_names
print (class_names)


# In[6]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = trainData.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validationData.cache().prefetch(buffer_size=AUTOTUNE)


# In[7]:


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(height,
                                 width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# In[49]:


numberClass = 131
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape = (height, width, 3)),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
 
    
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = 'sigmoid'),
    layers.Dense(numberClass)
])

model.compile(
    optimizer = 'adam', 
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics = ['accuracy'] )


# In[50]:


history=model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,

)


# In[93]:


from PIL import Image

image = Image.open(r'C:\Users\irush\Downloads\archive\fruits-360_dataset\fruits-360\WorkExample\onion.jpg')
image.load()
image.show()

import numpy as np



image = image.resize((100, 100))
img_array = tf.keras.utils.img_to_array(image)
img_array = np.array([img_array]) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    " {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
     


# In[ ]:





# In[ ]:




