
# coding: utf-8

# In[ ]:

import os
import pandas as pd
from shutil import copyfile
from IPython.display import clear_output

TRAIN_FILENAME="./../Dataset/train.csv"
IMAGE_FOLDER="./../Dataset/train_"
TRAIN_FOLDER="./../Dataset/train_data"
VAL_FOLDER="./../Dataset/val_data"

train=pd.read_csv(TRAIN_FILENAME)
train.head()

sep=int(train.shape[0]*0.8)
train.shape[0],sep

def copy_to_target_path(filename,target,src_folder,des_folder):
    des_dir=des_folder+"/"+target
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    copyfile(src_folder+"/"+filename,des_dir+"/"+filename)

def load_data(df,src_folder,des_folder):
    df=df.reset_index()
    row_len=df.shape[0]    
    for i in range(row_len):
        copy_to_target_path(df.image_name[i],df.detected[i],src_folder,des_folder)
        clear_output(wait=True)
        print(str(i+1)+"/"+str(row_len)+" copied!")


load_data(train[:sep],IMAGE_FOLDER,TRAIN_FOLDER)

load_data(train[sep:],IMAGE_FOLDER,VAL_FOLDER)

