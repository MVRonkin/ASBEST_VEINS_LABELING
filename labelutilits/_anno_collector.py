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


from ._annojson  import *
from ._coco_func import *
from ._path      import *
from .OLD._coco_func import _get_coco_annotations
        
#----------------------------------------------
def check_anno_labels(anno_path, image_dir_path = None, cat_ids = None):
    '''
    Check images with labels in annotation file.

    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory; 
      if none anno file and images are in the same directory. 
    cat_ids: list[int],
      classes to outout.

    Returns
    ----------
    list[int],
      list of image ids with labels.
    '''
    coco_anno_dict,  coco, image_dir_path = anno2coco(anno_path, image_dir_path)
    return _check_anno_labels(coco_anno_dict, coco, cat_ids)

def _check_anno_labels(coco_anno_dict, coco, cat_ids = None):
    ''' Check images with labels in annotation file.'''
    anno =  _get_coco_annotations(coco_anno_dict,  coco, image_id = None, cat_ids = cat_ids)
    labeld_image_ids = {annoi['image_id'] for annoi in anno}
    return sorted(list(labeld_image_ids))
#----------------------------------------------

def anno2df(anno_path, image_dir_path=None, cat_ids = None, start_image_id = 0, start_anno_id = 0):
    ''' 
    Create dataframe with existed label description,
    such to merge them into new anno_file.
        
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path iamge directory. 
      if none anno file and images are in the same directory. 
    cat_ids: list[int],
      classes to outout, all possible if None.
    start_image_id: int,
      first id number of the new image description
    start_anno_id: int,
      first id number of the new annotation description
    
    Returns
    ----------
    pd.DataFrame{},
      DataFrame with new description.  Keys of Frame:
       'new_image_id':  int, new image ID number,
       'old_image_id':  int, old image ID number,
       'old_anno_ids' : list[int], old annotation ID number,
       'new_anno_ids' : list[int], new annotation ID number,
       'old_file_name': string, expected file_name without image dir,
       'new_file_name': string, expected file_name with image dir,
       'class_id'   :   list[int], id of classes,
       'class_names':   list[string], names of classes,
       'img_desc'   :   dict, descriptor of iamge in COCO format,
       'anno'       :   list[dict], descriptor of annotations in COCO format,
        
    '''
    coco_anno_dict,  coco, image_dir_path  = anno2coco(anno_path, image_dir_path=image_dir_path)

    if cat_ids == None: cat_ids = coco_anno_dict['class_id']

    labeld_image_ids = _check_anno_labels(coco_anno_dict, coco, cat_ids = cat_ids)
    # labeld_image_ids = check_anno_labels(anno_path, cat_ids = None)

    imglist = coco.loadImgs(ids=labeld_image_ids)

    df = pd.DataFrame()

    for i,img_desc in enumerate(imglist):
        anno_ids = coco.getAnnIds(imgIds =[img_desc['id']], 
                                  catIds=cat_ids, 
                                  iscrowd=None)

        anno = coco.loadAnns(anno_ids) 

        cats     = coco.loadCats(coco.getCatIds())
        class_id = [cat['id'] for cat in cats]
        class_names = [cat['name'] for cat in cats]
        
        dict_desc = {'new_image_id':  int(i + start_image_id + 1),
                     'old_image_id':  int(img_desc['id']),
                     'old_anno_ids':  anno_ids,
                     'new_anno_ids':  [anno_id + start_anno_id for anno_id in anno_ids],
                     'old_file_name': img_desc['file_name'],
                     'new_file_name': os.path.join(image_dir_path, img_desc['file_name']),
                     'class_id': class_id,
                     'class_names': class_names}

        img_desc['id'] = dict_desc['new_image_id']
        img_desc['file_name'] = dict_desc['new_file_name']

        for i, anno_id in enumerate(dict_desc['new_anno_ids']):
            anno[i]['image_id'] = img_desc['id']
            anno[i]['id']       = anno_id

        dict_desc = {**dict_desc, 
                     'img_desc':img_desc, 
                     'anno':anno}

        df = pd.concat([df, pd.DataFrame([dict_desc])], ignore_index=True)  
    return df

#---------------------------------------------------------------------------
def _last_from_annodf(annodf):
    ''' Auxiliary for anno df: return last image id and alast annotation id.
    '''
    last_anno_id = annodf['new_anno_ids'].iloc[-1][-1]
    last_image_id = annodf['new_image_id'].iloc[-1]
    return last_anno_id,  last_image_id

def _get_anno_path(project_path, data_dir_name):
    ''' auxiliary return anno_path '''
    pt_ = os.path.join(project_path, data_dir_name)
    anno_file = list_json(pt_)[0]
    anno_path = os.path.join(pt_, anno_file)
    return anno_path

def collec_newanno(path, dir_names, image_dir_path = None, cat_ids = None):
    '''
    Merge into 1 dataframe of annotations all exiting annos.
    
    Paramters
    ---------------------
    anno_path: string, 
      path annotation file.
    image_dir_path: string,
      path image directory. 
      if none anno file and images are in the same directory. 
    cat_ids: list[int],
      classes to outout, , all possible if None.
    start_image_id: int,
      first id number of the new image description
    start_anno_id: int,
      first id number of the new annotation description
    
    Returns
    ----------
    pd.DataFrame{},
      DataFrame with new description.  Keys of Frame:
       'new_image_id':  int, new image ID number,
       'old_image_id':  int, old image ID number,
       'old_anno_ids' : list[int], old annotation ID number,
       'new_anno_ids' : list[int], new annotation ID number,
       'old_file_name': string, expected file_name without image dir,
       'new_file_name': string, expected file_name with image dir,
       'class_id'   :   list[int], id of classes,
       'class_names':   list[string], names of classes,
       'img_desc'   :   dict, descriptor of iamge in COCO format,
       'anno'       :   list[dict], descriptor of annotations in COCO format,
    
    '''
    last_anno_id,  last_image_id = 0,0
    annodf = pd.DataFrame()
    for dir_name in dir_names:
        anno_path = _get_anno_path(path, dir_name)
    
        df = correct_anno_img_names(anno_path, 
                                    image_dir_path=image_dir_path, 
                                    image_id = None)
        
        df_ = anno2df(anno_path, 
                      start_image_id = last_image_id, 
                      start_anno_id  = last_anno_id, 
                      image_dir_path = image_dir_path, 
                      cat_ids        = cat_ids)
                      
        if df_.shape[0] >0:
            anno_id_tmp,  image_id_tmp = last_anno_id,  last_image_id
            last_anno_id,  last_image_id = _last_from_annodf(df_)
            annodf = pd.concat([annodf, df_], ignore_index=True)
            print(f'images:{last_image_id-image_id_tmp}, instances:{last_anno_id-anno_id_tmp}')
        else: print('No labeled data')
    return annodf

#---------------------------------------------
def _data_desc_base():
    ''' Auxiliary, fill basic fields in the COCO json '''
    data = dict()
    data['licenses']   = [{'name' : '', 
                           'id'   : 0, 
                           'url'  : ''}]
    
    data['info']       = {'contributor' : '',
                          'date_created': '',
                          'description' : '', 
                          'url'         : '', 
                          'version'     : '', 
                          'year'        : ''}
    
    data['categories'] = list()
    return data

def _class_desc(annodf, data_desc):
    ''' Auxiliary, fill 'categories' field in the COCO json '''
    data_desc['categories'] = list()
    
    class_ids   = list(set(sum(annodf['class_id'   ].to_list(),[])))
    class_names = list(set(sum(annodf['class_names'].to_list(),[])))
    
    for class_id, class_name in zip(class_ids,class_names):
        data_desc['categories'] += [{'id': class_id, 'name': class_name, 'supercategory': ''}]
    
    return data_desc

def create_json(annodf, project_dir = None, new_anno_name = 'annotation.json', data_desc = None):
    '''
    Create new COCO annotation based on the annodf format,
    
    Paramteres
    -----------
    annodf: pd.DataFrame,
      data frame with annotation, see anno2df and collec_newanno.
    project_dir: string,
      path to project directory, where new_anno expected to be.
    new_anno_name: string,
      new annotation name, including extantion 'json'.
    data_desc: dict,
      base for COCO format descriptor, 
      base with empty fields if None.
    
    Returns
    -----------
    string: new annotation path.
    
    Notes
    -------
    annodf format:  pd.DataFrame{},
      DataFrame with new description.  Keys of Frame:
       'new_image_id':  int, new image ID number,
       'old_image_id':  int, old image ID number,
       'old_anno_ids' : list[int], old annotation ID number,
       'new_anno_ids' : list[int], new annotation ID number,
       'old_file_name': string, expected file_name without image dir,
       'new_file_name': string, expected file_name with image dir,
       'class_id'   :   list[int], id of classes,
       'class_names':   list[string], names of classes,
       'img_desc'   :   dict, descriptor of iamge in COCO format,
       'anno'       :   list[dict], descriptor of annotations in COCO format,
    
    COCO Json base description format:
    data['licenses']   = [{'name' : '', 'id'   : 0,  'url'  : ''}]
    
    data['info']       = {'contributor' : '',
                          'date_created': '',
                          'description' : '', 
                          'url'         : '', 
                          'version'     : '', 
                          'year'        : ''}
    
    data['categories'] = [{'id': '', 'name': '', 'supercategory': ''}]
    
    data['annotations'] = [{'id': 1,
                            'image_id': 1,
                            'category_id': 2,
                            'segmentation': [[ x, y, x, y,...]],
                            'area': float,
                            'bbox': [x0, y0, x1, x2],
                            'iscrowd': 0,
                            'attributes': {'occluded': False}}]
                            
    data['images'] = [{'id': 1,'width': float, 'height': float,
                       'file_name': '','license': 0,'flickr_url': '',
                       'coco_url': '','date_captured': 0}]
    '''
    if data_desc == None: data_desc = _data_desc_base()
    
    data_desc = _class_desc(annodf, data_desc)

    new_annos  = sum(annodf['anno'].tolist(),[])
    new_images = annodf['img_desc'].tolist()

    data_desc['images']      = new_images
    data_desc['annotations'] = new_annos
    
    if project_dir is None:
        project_dir = os.getcwd()
    
    new_anno_path = os.path.join(project_dir,new_anno_name)
    
    with open(new_anno_path, 'w') as f:
        json.dump(data_desc, f)
        
    return new_anno_path

#--------------------------------------------------

def _add_dir_descr(file_path):
    dir_,fname = os.path.split(file_path)            
    filename,extention = os.path.splitext(fname)
    base_dir = os.path.split(dir_)[-1]
    return os.path.join(dir_,filename +'_'+ base_dir + extention )

def copy2train(anno_path, new_img_dir  = 'train', project_path = None, copy_anno = True ):
    '''
    Create new directory for train database, 
      including images and annotation in json format.
    
    Parameters
    -----------
    anno_path: string,
      path to annotation in COCO JSON format.
    new_img_dir: string,
      name of the new directory for training data.
    project_path: string,
      whole project path.
    copy_anno: bool,
      copy annotation into new directory if true;
      if false old annotation be rewritten.
      
    Returns
    ---------
    pd.DataFrame: report of old path, new path, file name.
    string: new annotation path.
    
    '''
    # NEW DATA DIR
    if project_path is None: 
        project_path = os.getcwd()
    new_img_dir_path = os.path.join(project_path, new_img_dir)
    create_dir(new_img_dir_path)
    
    # IMAGE LIST TODO REPLACE WITH DATA FROM JSON?
    with open(anno_path) as json_file:
        data = json.load(json_file)
    
    # COPY IMAGES, NEW PATH INCLUDE DATA DIRICTORY
    df = pd.DataFrame() # REPORT
    for i,img_desc in enumerate(data['images']):
        file_path = img_desc['file_name']

        file_name = os.path.split(_add_dir_descr(file_path))[-1]
        
        new_path  = os.path.join(new_img_dir_path,file_name)

        dict_desc = {'old path':file_path, 
                     'new path':'ALREADY EXIST',
                     'copied': 'False',
                     'new file name':file_name}

        if not os.path.exists(new_path):
            shutil.copyfile(file_path, new_path)
            dict_desc['new path']   = new_path
            dict_desc['copied']   = 'True'
            data['images'][i]['file_name'] = new_path

        df = df.append(dict_desc, ignore_index=True)
    
    # COPY ANNO
    if copy_anno:
        anno_name = os.path.split(anno_path)[-1]
        new_anno_path = os.path.join(new_img_dir_path, anno_name)
        dict_desc = {'old path':anno_path, 
                     'new path':'ALREADY EXIST', 
                     'copied': 'False',
                     'new file name':anno_name}

        if not os.path.exists(new_anno_path):
            shutil.copyfile(anno_path, new_anno_path)
            dict_desc['new path'] = new_anno_path
            dict_desc['copied'] = 'True'

        df = df.append(dict_desc, ignore_index=True) 
    else:
        new_anno_path = anno_path
    
    # REWRITE
#     with open(new_anno_path, 'r+') as f:
#         data = json.load(f); f.seek(0)
#         data['images']  = imglist
#         json.dump(data, f); f.truncate()
    with open(new_anno_path, 'w') as f:
        json.dump(data, f)  
        
    return df, new_anno_path
    
    
