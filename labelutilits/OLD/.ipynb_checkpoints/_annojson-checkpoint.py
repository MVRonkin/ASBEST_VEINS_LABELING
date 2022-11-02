
import os
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

import numpy as np

import pandas as pd
import json
import time
import datetime
from PIL import Image

from pycocotools.coco import COCO

from ._path import *

IMAGE_EXTENTIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
IMAGE_AND_LABELS_EXTANTIONS = (*IMAGE_EXTENTIONS,'txt','csv','xml','json','txt')


#----------------------------------------------
def print_anno_info(anno_path):
    ''' 
    Print some information about json file 
    of annotations in COCO format'''
    with open(anno_path) as json_file:
        data = json.load(json_file)
    print('DATASET KEYS:' ,anno_data.keys())
    print('DATASET ANN KEYS:', anno_data['annotations'][0].keys())
    print('CATEGORIES: ',anno_data['categories'])
    print('IMAGE KEYS',anno_data['images'][0].keys())
    print('LEN IMAGE',len(anno_data['images']))
    
#----------------------------------------------
def anno_info(anno_path, image_dir_path = None):
    '''
    Dataset information from json file 
    of annotations in COCO format.
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory;   

    Returns
    -----------
    dataset_info_dict ['string':[int, string]]: dict of fileds discribed below.   
    
    Notes
    -------------
    annotation inclues: 
    * dataset name;
    * path to annotation;
    * annotation file name;
    * path to image directory;
    * length of dataset (images number);
    * anno_number (number of instances for all images);
    * class ids: identification number of each class;
    * class_names: names of classes;
    * height: image heights;
    * width: image widths;
    * COCO_obj:  pointer to specific COCO object;
    * dataset_keys: keys in dataset;
    * anno_keys: keys for each annotatnio instant;
    * image_keys: keys for each image instant;
    * image_fname_example: image file name example.

    '''
    dataset_info_dict = dict()  
    name_dataset = os.path.split(os.path.split(anno_path)[0])[1]
    dataset_info_dict['name']  = name_dataset
    dataset_info_dict['anno_path']   = anno_path
    dataset_info_dict['anno_fname']  = os.path.split(anno_path)[1]
    if image_dir_path is None:
        image_dir_path = os.path.split(anno_path)[0]
    dataset_info_dict['image_dir_path']  = image_dir_path

    coco=COCO(anno_path)
    dataset_info_dict['length']      = len(coco.imgs)
    dataset_info_dict['anno_number'] = len(coco.anns)    

    cats     = coco.loadCats(coco.getCatIds())
    class_id = [cat['id'] for cat in cats]
    class_names = [cat['name'] for cat in cats]
    dataset_info_dict['class_id']     = class_id
    dataset_info_dict['class_names']  = class_names

    df = pd.DataFrame(coco.imgs).transpose()
    dataset_info_dict['height'] = list(set(df['height']))
    dataset_info_dict['width']  = list(set(df['width']))    
    dataset_info_dict['COCO_obj']  = coco
    
    with open(anno_path) as json_file:
        anno_data = json.load(json_file)
    '''         
    dataset_info_dict['dataset_keys']  = (anno_data.keys())   
    if anno_data['annotations'] !=[]:
        dataset_info_dict['anno_keys']     = (anno_data['annotations'][0].keys())
    else:
        dataset_info_dict['anno_keys'] = None
    dataset_info_dict['image_keys']    = (anno_data['images'][0].keys())
    '''
    dataset_info_dict['image_fname_example'] = anno_data['images'][0]['file_name']

    return dataset_info_dict

#----------------------------------------------
def anno2coco(anno_path, image_dir_path=None):
    '''
    Dataset information (see anno_info) and COCO object.
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory.   

    Returns
    -----------
    dataset_info_dict ['string':[int, string]], 
      dict of fileds discribed below. 
    coco: pointer to object,
      COCO object.
    image_dir_path: strig,
      path to image directory.
    '''
    coco_anno_dict = anno_info(anno_path, image_dir_path)
    coco = coco_anno_dict['COCO_obj']
    image_dir_path = coco_anno_dict['image_dir_path']
    return coco_anno_dict,  coco, image_dir_path

#----------------------------------------------
def correct_anno_img_names(anno_path, image_dir_path=None, image_id = None):
    '''
    Correct image pathers in annotation file 
    in correspondance with real image path.
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    image_id: list[int],
      images to correct, all if None.

    Returns
    ----------
    dict[string:string],
      old and new entries in annotation.
    '''
#     coco_anno_dict,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)
    df = pd.DataFrame(columns = ['old_path', 'new_path'])
    
    with open(anno_path) as json_file:
        data = json.load(json_file)
        json_file.close()

     
    for i in range(len(data['images'])):  
        fpth = data['images'][i]['file_name'].split('/')[-1]

        if image_dir_path !=None:
            fpth = os.path.join(image_dir_path, fpth)
            
        if image_id == None or data['images'][i]['id'] in image_id:
            data['images'][i]['file_name'] = fpth
        
        dict_ = {'old_path': data['images'][i]['file_name'],                        
                 'new_path': fpth},
        
        df = pd.concat([df, pd.DataFrame(dict_)])

    with open(anno_path, 'w') as f:
        json.dump(data, f)
    
    return df



