# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../var/folders/j2/7dy2frm975zd_vg2nfv7_9d80000gn/T'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# # 导入库

#%%
import keras 
import numpy as np
import urllib
import tarfile
import os
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing import image
from scipy.misc import imresize
from imageio import imread, imsave, mimsave
import glob

#%% [markdown]
# # 下载数据并解
#%%
url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz' 
file_name = '/Users/liuhuan/Downloads/lfw.tar'
directory = '/Users/liuhuan/Downloads/lfw'
new_dir = '/Users/liuhuan/Downloads/lfw_new_imgs'

if not os.path.isdir(new_dir):
    os.mkdir(new_dir)

    if not os.path.isdir(directory):
        if not os.path.isfile(file_name):
            urllib.request.urlretrieve(url=url, filename=file_name)
        tar = tarfile.open(file_name, 'r')
        tar.extractall(path=directory)
        tar.close()

    count = 0

    
    for dir_,_, files in os.walk(directory):
        for file_ in files:
            img = imread(os.path.join(dir_, file_))
            imsave(os.path.join(new_dir, '%d.png' %count), img)
            count += 1


#%%


