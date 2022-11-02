
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

import matplotlib.pyplot as plt

from pycocotools.coco import COCO

IMAGE_EXTENTIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
IMAGE_AND_LABELS_EXTANTIONS = (*IMAGE_EXTENTIONS,'txt','csv','xml','json','txt')

from ._annojson import *
from ._coco_func import *
from ._path import *
from ._coco_func import (_get_coco_image_dir_pathes,
                         _get_coco_annotations,
                         _anno2bbox,
                         _anno2instant_masks,
                         _anno2semantic_mask,
                        _plot_all_anno)
    
                                     
class Annotation:
    def __init__(self, anno_path, image_dir_path = None):
        self.anno_path = anno_path
        self.image_dir_path = image_dir_path
        
        (anno_dict, 
         coco, 
         image_dir_path) = anno2coco(self.anno_path, 
                                     self.image_dir_path)
        
        self.anno_dict = anno_dict
        self.coco           = coco
        self.image_dir_path = image_dir_path
    #------------------------------------
    def get_anno_info(self):
        ''' Get Info Dict see anno_info'''
        return self.anno_dict
    
    #------------------------------------
    def get_image_pathes(self, image_id = [1,2,3]):
        ''' Get image pathes by id in annotation'''
        if 0 in image_id: print('Warning: 0 idx is prohibited!'); return
        return _get_coco_image_dir_pathes(self.image_dir_path, self.coco, image_id = image_id)
    
    #-------------------------------    
    def get_image(self, image_id = [1]):
        ''' Get one image by id in annotation'''
        image_name = self.get_image_pathes(image_id = image_id)
        return np.array(Image.open(image_name[0]))
    
    #-------------------------------  
    def get_anno_image(self, image_id = [1], cat_ids = None):
        ''' Get annotation in COCO foramt for specified image and classes.
        '''
        return _get_coco_annotations(self.anno_dict,  self.coco, image_id, cat_ids)
    
    #------------------------------- 
    def get_bbox(self, image_id = [1], cat_ids = None):
        ''' Get one image bounding boxes (bboxes)'''
        anno = self.get_anno_image(image_id, cat_ids)
        return _anno2bbox(anno)
    
    #------------------------------- 
    def get_instant_masks(self, image_id = [1], cat_ids = None, shape = None):
        ''' Get one image insances as separate channels not divided by class'''
        anno = self.get_anno_image(image_id, cat_ids)
        return _anno2instant_masks(anno, self.coco, shape = shape ) 
    
    #------------------------------- 
    def get_semantic_masks(self, image_id = [1], cat_ids = None):
        ''' Get one image semantic as one channel not divided by class'''
        anno = self.get_anno_image(image_id, cat_ids)
        return _anno2semantic_mask(anno, self.coco)  
    
    #------------------------------- 
    def plot_all_anno(self, image_id = [1], cat_ids = None, figsize = (18,12)):
        '''plot all anno in the form: image, BBox, Semantic Seg, Instant Seg'''
        image = self.get_image(image_id = image_id) 
        anno  = self.get_anno_image(image_id, cat_ids)
        return _plot_all_anno(image, anno, self.coco, figsize)