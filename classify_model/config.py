
# coding: utf-8

# In[2]:


from easydict import EasyDict as edict

config = edict()

config.train = edict()
config.train.img_list = "../data/train_imglist.txt"
config.train.batch_size = 16 # use large number if you have enough memory
config.train.n_epoch = 10

config.test = edict()
config.test.batch_size = 16
config.test.img_list = "../data/test_imglist.txt"

