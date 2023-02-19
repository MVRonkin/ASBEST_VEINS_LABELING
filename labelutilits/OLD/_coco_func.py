
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

import matplotlib.patches as patches
import matplotlib
import matplotlib.colors as mcolors


from pycocotools.coco import COCO


from ._annojson import *
 

#----------------------------------------------
def coco_image_dir_path(anno_path, image_dir_path=None, image_id = [1,2,3]):
    '''
    Get image pathes by image id.
    
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
    list[string],
      image pathes.
    '''
    _,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)
    return _get_coco_image_dir_pathes(image_dir_path = image_dir_path, 
                                  coco           = coco, 
                                  image_id      = image_id)

def _get_coco_image_dir_pathes(image_dir_path, coco, image_id = [1,2,3]):
    '''
    Get image pathes by image id.
    '''
    fnames = [] 
    for image_id in np.atleast_1d(image_id) :
        if image_id>0:           
            rel_path = coco.imgs[image_id]['file_name']
            fname    = os.path.join(image_dir_path,rel_path)
            if os.path.isfile(fname):
                fnames +=[fname]
            else:
                print(f'file {fname} with id {image_id} does not exist') 

    return fnames


#---------------------------------------------- 
def get_image(anno_path, image_dir_path=None,  image_id = [1]):
    '''
    Get images by image id.
    
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
    ndarray[image:channels:width:height]
      images in the numpy format with dimention of array
      image x channels x width x height.
    '''
    image_name = coco_image_dir_path(anno_path, image_dir_path, image_id = image_id)
    image      = np.array(Image.open(image_name[0]))
    return image


#----------------------------------------------
def coco_annotations(anno_path, image_dir_path=None, image_id = None, cat_ids = None):
    '''
    Get annotation in COCO foramt for specified images and classes.
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    image_id: list[int],
      images to correct, all if None.
    cat_ids: list[int],
      classes to outout.

    Returns
    ----------
    list[dict[anno_key:value]],
      list of annotations.
    '''
    coco_anno_dict,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)
    return _get_coco_annotations(coco_anno_dict,  coco, image_id, cat_ids)

def _get_coco_annotations(coco_anno_dict,  coco, image_id, cat_ids):
    '''
    Get annotation in COCO foramt for specified images and classes.
    '''
    if image_id == None: image_id = range(1,coco_anno_dict['length'] + 1) #!
    if cat_ids   == None: cat_ids   = coco_anno_dict['class_id']
        
    anno_ids = coco.getAnnIds(imgIds =image_id, catIds=cat_ids, iscrowd=None)
    anno = coco.loadAnns(anno_ids) 
    return anno

#----------------------------------------------
def get_bbox(anno_path, image_dir_path=None,  image_id = [1], cat_ids = [1]):
    '''
    Get bbox for images and soecified class by annotation in COCO format.
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    image_id: list[int],
      images to correct, all if None.
    cat_ids: list[int],
      classes to outout.

    Returns
    ----------
    ndarary[ndarary[float]],
      list of bboxes in format [x0,y0,x1,y1] 
      (left upper and right bottom corners, absolute values).
    '''
    anno =coco_annotations(anno_path = anno_path, 
                           image_dir_path=image_dir_path, 
                           image_id = image_id, 
                           cat_ids = cat_ids)

    return _anno2bbox(anno)

def _anno2bbox(anno):
    '''
    Get bbox for images and soecified class by annotation in COCO format.
    '''
    bbox = np.zeros((len(anno),4))
    for i,ann in enumerate(anno):
        bbox_ = ann['bbox']
        bbox[i,:] =  [bbox_[0],bbox_[1], bbox_[2],bbox_[3]]
    
    return bbox

#----------------------------------------------
def get_instant_mask(anno_path,image_dir_path=None,  image_id = [1], cat_ids = [1]):
    '''
    Get masks for image instances separately as mutlichennal array.
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    image_id: list[int],
      images to correct, all if None.
    cat_ids: list[int],
      classes to outout.

    Returns
    ----------
    ndarary[instatnces:width:height],
      image-like array with dimentions: instances x width x height .
      
    TODO: instances are not divided by class!
    '''
    anno = coco_annotations(anno_path = anno_path, 
                           image_dir_path=image_dir_path, 
                           image_id = image_id, 
                           cat_ids = cat_ids)

    coco_anno_dict,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)
        
    height = coco_anno_dict['height'][0]
    width  = coco_anno_dict['width' ][0]
    return _anno2instant_masks(anno, coco, shape = (height, width) )


def _anno2instant_masks(anno, coco, shape = None ):
    '''
    Get masks for image instances separately as mutlichennal array.
    '''
    if shape != None:
        width  = shape[1]
        height = shape[0]
    else:
        height, width = coco.annToMask(ann[0]).shape[:2]
    
    masks = np.zeros((len(anno),height, width ))#[:,np.newaxis,np.newaxis]
 
    for i,ann in enumerate(anno):
        mask = coco.annToMask(ann)
        masks[i,:,:] =  mask
        #TODO: instances are not divided by class!
    return masks

#----------------------------------------------
def get_sem_seg_mask(anno_path,image_dir_path=None,  image_id = [1], cat_ids = [1]):
    '''
    Get masks for image each for each class.
    
    TODO: probably work ony for one class
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    image_id: list[int],
      images to correct, all if None.
    cat_ids: list[int],
      classes to outout.

    Returns
    ----------
    ndarary[instatnces:width:height],
      image-like array with dimentions: instances x width x height .
    '''        
    anno = coco_annotations(anno_path = anno_path, 
                       image_dir_path=image_dir_path, 
                       image_id = image_id, 
                       cat_ids = cat_ids)

    coco_anno_dict,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)

    return  _anno2semantic_mask(anno, coco)

def _anno2semantic_mask(anno, coco):
    '''
    Get masks for image instances separately as mutlichennal array.
    '''    
    mask = coco.annToMask(anno[0])

    for i in range(1,len(anno)):
        mask += coco.annToMask(anno[i])
    mask[mask>0]=250 #TODO: work here only for one class!

    return mask

#----------------------------------------------    
def plot_all_anno(anno_path, image_dir_path=None,  image_id = [1], cat_ids = [1], figsize = (12,12)):
    '''
    plot annotations in the form image; Bboxes; Semantic Segmentaion; Instance Segmentation .
    
    TODO: probably work ony for one class
    
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    image_id: list[int],
      images to correct, all if None.
    cat_ids: list[int],
      classes to outout.

    ''' 
    image = get_image(anno_path, image_dir_path=image_dir_path,  image_id = image_id)

    anno = coco_annotations(anno_path = anno_path, 
                   image_dir_path=image_dir_path, 
                   image_id = image_id, 
                   cat_ids = cat_ids)

    coco_anno_dict,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)

    _plot_all_anno(image, anno, coco, figsize)
        
def _plot_all_anno(image, anno, coco, figsize):
        n_col, n_row = 2,2            
        fig, ax = plt.subplots(n_col, n_row, figsize = figsize)  
        
        # IMAGE
        ax[0,0].imshow(image, 'gray')
        ax[0,0].axis('off')
        ax[0,0].set_title('Original Image')
        
        # BBOXES
        for i in range(1,len(anno)):
            box = anno[i]['bbox']
            bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=2, edgecolor="blue", facecolor="none")
            ax[0,1].add_patch(bb)  
        ax[0,1].imshow(image, 'gray')
        ax[0,1].axis('off')
        ax[0,1].set_title('Object Detection')
         
        # SEMANTIC SEGMENTATION
        mask = coco.annToMask(anno[0])
        for i in range(1,len(anno)):
            mask += coco.annToMask(anno[i])
        mask[mask>0]=250
        ax[1,0].imshow(mask, 'gray')
        ax[1,0].axis('off')
        ax[1,0].set_title('Semantic Segmentaion')    
        
        # INSTANT SEGMENTATION
        colors = list(matplotlib.colors.cnames.keys())
        for ann in anno:
            edgecolor = np.random.choice(colors, 1)[0]
            facecolor = np.random.choice(colors, 1)[0]
            mask = np.asarray(ann['segmentation']).reshape(-1,2)
            bb = patches.Polygon(mask, linewidth=2, edgecolor=edgecolor, facecolor=facecolor,alpha=0.4)
            ax[1,1].add_patch(bb)
        ax[1,1].imshow(image, cmap='gray')
        ax[1,1].axis('off')
        ax[1,1].set_title('Instance Segmentation')        
        
        plt.tight_layout()
        plt.show()
        
