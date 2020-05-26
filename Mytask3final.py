#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:



train , test = dataset


# In[4]:


X_test , y_test = test
X_train , y_train = train


# In[5]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[6]:


X_train_1d.shape


# In[7]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[8]:


from keras.utils.np_utils import to_categorical


# In[9]:


y_train_cat = to_categorical(y_train)
y_test_cat=to_categorical(y_test)


# In[10]:


y_train_cat


# In[11]:


from keras.models import Sequential


# In[12]:


from keras.layers import Dense


# In[13]:


model = Sequential()


# In[14]:


model.add(Dense(units=1000, input_dim=28*28, activation='relu'))


# In[15]:


model.summary()


# In[16]:


model.add(Dense(units=515, activation='relu'))


# In[17]:


model.add(Dense(units=250, activation='relu'))


# In[18]:


model.add(Dense(units=100, activation='relu'))


# In[19]:


model.summary()


# In[20]:


model.add(Dense(units=10, activation='softmax'))


# In[21]:


from keras.optimizers import RMSprop


# In[22]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[ ]:





# In[23]:


h = model.fit(X_train, y_train_cat,epochs=3)


# In[28]:


print(h.history['accuracy'][-1])


# In[ ]:




