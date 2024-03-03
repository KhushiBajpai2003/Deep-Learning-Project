#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np


# In[23]:


IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 25
EPOCHS = 30
N_CLASSES = 2


# In[24]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
   "D:\Kidney Dataset",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[25]:


class_names = dataset.class_names
class_names


# In[26]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[27]:


# Create a mapping between numerical labels and class names
class_names = {0: 'Kidney - inflammation', 1: 'Kidney - normal'}

unique_labels = set()

for _, labels_batch in dataset.take(1):
    unique_labels.update(labels_batch.numpy())

print("Unique Labels/Classes in the Dataset:", unique_labels)

# Convert numerical labels to class names for visualization
class_names_batch = [class_names[label] for label in labels_batch.numpy()]


# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):  # Display examples from 3 batches
    labels_batch_np = labels_batch.numpy()  # Convert the entire labels_batch tensor to a NumPy array
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)  # 3 batches * 4 subplots per batch
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        
        # Use the NumPy array as an index in the class_names dictionary
        label_key = labels_batch_np[i]
        plt.title(class_names[label_key])
        
        plt.axis("off")

plt.show()


# In[41]:


len(dataset)


# In[42]:


train_size = 0.8
len(dataset)*train_size


# In[43]:


train_ds = dataset.take(11)
len(train_ds)


# In[44]:


test_ds = dataset.skip(11)
len(test_ds)


# In[48]:


val_size=0.1
len(dataset)*val_size


# In[49]:


val_ds = test_ds.take(1)
len(val_ds)


# In[50]:


test_ds = test_ds.skip(1)
len(test_ds)


# In[51]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[52]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[53]:


len(train_ds)


# In[54]:


len(val_ds)


# In[55]:


len(test_ds)


# In[ ]:




