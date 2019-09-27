import os
import glob
import shutil
import numpy as np
import pandas as pd
import random
from PIL import Image
from config import esri_retinanet_config as config
from bs4 import BeautifulSoup
from imutils import paths
import argparse
import random
import os

#os.chdir("./retinanet/")
files = glob.glob('./dataset/images/*')
for f in files:
    os.remove(f)
 
files = glob.glob('./dataset/annotations/*')
for f in files:
    os.remove(f)
    
#this will need to go on a parser
case = "rgb"
lbls = "taxonID"
sites = ["OSBS"]

ls_f = (glob.glob('../'+case+'/*.tif'))
for f in ls_f:
    shutil.copy(f, './dataset/images/')
    
ls_f = (glob.glob('../csv_labels/*'+case+'.csv'))
for f in ls_f:
    shutil.copy(f, './dataset/annotations/')
    
# initialize the base path for the logos dataset
BASE_PATH = "dataset"

# build the path to the annotations and input images
annot_path = os.path.sep.join([BASE_PATH, 'annotations'])
images_path = os.path.sep.join([BASE_PATH, 'images'])

# define the training/testing split
train_test_split = 0.8

#  build the path to the output training and test .csv files
train_csv = os.path.sep.join([BASE_PATH, 'train.csv'])
test_csv = os.path.sep.join([BASE_PATH, 'test.csv'])

# build the path to the output classes CSV files
classes_csv = os.path.sep.join([BASE_PATH, 'classes.csv'])

# build the path to the output predictions dir
OUTPUT_DIR = os.path.sep.join([BASE_PATH, 'predictions'])

# grab all image paths then construct the training and testing split
imagePaths = list(paths.list_files(images_path))
random.seed( 30 )
random.shuffle(imagePaths)
i = int(len(imagePaths) * train_test_split)
trainImagePaths = imagePaths[:i]
testImagePaths = imagePaths[i:]

# create the list of datasets to build
dataset = [ ("train", trainImagePaths, train_csv),
            ("test", testImagePaths, test_csv)]

veg_structure = pd.read_csv("vegetation_structure.csv")
veg_structure = veg_structure[["individualID", "siteID", lbls]]
#is_sites = veg_structure.siteID.isin(list(sites))
veg_structure = veg_structure[veg_structure.siteID.isin(list(sites))]
list_labels = veg_structure[lbls].unique()
lb_code = range(len(list_labels)+1)
labelnet = {}
for i in range(len(list_labels)):
    labelnet[list_labels[i]] = lb_code[i+1]

for (dType, imagePaths, outputCSV) in dataset:
    # load the contents
    print ("[INFO] creating '{}' set...".format(dType))
    print ("[INFO] {} total images in '{}' set".format(len(imagePaths), dType))

    # open the output CSV file
    #csv = open(outputCSV, "w")

    # loop over the image paths
    labels = pd.DataFrame()
    for imagePath in imagePaths:
        fl = imagePath.split("/")[2][:-4]
        tmp = pd.read_csv('./dataset/annotations/'+fl+'.csv')
        labels = labels.append(tmp, sort=False)
        
    labels = labels.rename(columns={'label': 'individualID'})
    labels = labels.join(veg_structure.set_index('individualID'), on='individualID')
    labels=labels.replace({lbls: labelnet})
    labels = labels.rename(columns={lbls: 'label'})


    labels.to_csv("./"+outputCSV)
    # update the set of unique class labels
    #CLASSES.add(label)

    
#pd.DataFrame.from_dict(labelnet)
new_dict = {k: [v] for k, v in labelnet.items()}
df = pd.DataFrame(new_dict)
df = df.T
df.to_csv("./species_dictionary.csv")
df['index'] = list(range(len(df)))
print(df)
df.to_csv("./"+classes_csv, header=False, index = False)



