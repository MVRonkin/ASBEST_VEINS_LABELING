
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

IMAGE_EXTENTIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
IMAGE_AND_LABELS_EXTANTIONS = (*IMAGE_EXTENTIONS,'txt','csv','xml','json','txt')

def list_dirs(path, exclude_if_startwith = ('.','__') ):
    '''
    List of directory names in path.

    Parameters
    ----------
    path: string, 
      path of directory to check.
    exclude_if_startwith: tuple[string], 
      if dir names start with 'string' 
      do not include it into  output.  

    Returns
    -----------
    list [strings]: list of dir names
    '''    
    listdir = [dname for dname in os.listdir(path) 
                   if os.path.isdir(dname) and not 
                      dname.startswith(tuple(exclude_if_startwith))]
    return  listdir

#----------------------------------------------
def list_images(dirpath, ext = IMAGE_EXTENTIONS):
    '''
    List of file names in directory, 
    which are corresponds to the extentions of images.

    defoult:
    IMAGE_EXTENTIONS = ('.png', '.jpg', 
                      '.jpeg', '.tiff', 
                      '.bmp', '.gif')

    Parameters
    ----------
    dirpath: string, 
      path of directory to check.
    ext: string, 
      extention of files to output.  

    Returns
    -----------
    list [strings]: list of file names
    '''
    return list_ext(dirpath, ext)

#----------------------------------------------
def list_json(dirpath, ext = 'json'):
    '''
    List of file names in directory, 
    which are corresponds to the extention 'json'.

    Parameters
    ----------
    dirpath: string, 
      path of directory to check.
    ext: string, 
      extention of files to output.  

    Returns
    -----------
    list [strings]: list of file names
    '''
    return list_ext(dirpath, ext)

#----------------------------------------------
def list_ext(dirpath, ext = 'txt'):
    '''
    List of file names in directory, 
    which are corresponds to the extention 'ext'.

    Parameters
    ----------
    dirpath: string, 
      path of directory to check.
    ext: string, 
      extention of files to output.  

    Returns
    -----------
    list [strings]: list of file names
    '''
    return [fname for fname in os.listdir(dirpath) 
                        if fname.lower().endswith(ext)]

#----------------------------------------------
def list_not_ext(dirpath, ext = IMAGE_AND_LABELS_EXTANTIONS):
    '''
    List of file names in directory, 
    which are not corresponds to the extentions.

    defoult:
    IMAGE_AND_LABELS_EXTANTIONS = ('.png', '.jpg', 
                                  '.jpeg', '.tiff', 
                                  '.bmp', '.gif',
                                  'txt','csv','xml',
                                  'json','txt' )

    Parameters
    ----------
    dirpath: string, 
      path of directory to check.
    ext: string, 
      extention of files 
      to not include in the output.  

    Returns
    -----------
    list [strings]: list of file names
    '''
    return [fname for fname in os.listdir(dirpath) 
                    if not fname.lower().endswith(ext)]

#----------------------------------------------
def image_dir_info(dirpath):
    '''
    Return information of directory content:
      dirinfo = ('images','dirs','json',
                 'xml','csv',  'txt',  'other')
    Parameters
    ----------
    dirpath: string, 
      path of directory to check.

    Returns
    -----------
    dict ['string':int]: dict of file type and its count          
    '''
    
    dirinfo = {'images':0,
              'dirs':   0,
              'json':   0,
              'xml':    0,
              'csv':    0,  
              'txt':    0,  
              'other':  0} 

    for fname in os.listdir(dirpath):
        if os.path.isdir(fname): 
            dirinfo['dirs'] +=1
        elif fname.lower().endswith(IMAGE_EXTENTIONS): 
            dirinfo['images'] +=1
        elif fname.lower().endswith('.json'):        
            dirinfo['json'] +=1
        elif fname.lower().endswith('.xml'): 
            dirinfo['xml'] +=1
        elif fname.lower().endswith('.csv'):        
            dirinfo['csv'] +=1            
        elif fname.lower().endswith('.txt'):        
            dirinfo['txt'] +=1
        else:
            dirinfo['other'] +=1                
        
    return dirinfo

#----------------------------------------------
def print_dir_description(path = None):
    ''' Print description of project directory
    Parameters
    ----------
    path: string,
      project directory path, cwd if None
    '''
    if path is None: path = os.getcwd()
    print('dirs', list_dirs(path))
    print('content', image_dir_info(path))
    print('not content', list_not_ext(path))
    print('anno cvat', list_json(path))
    print('anno xml', list_ext(path,ext='xml'))
    print('cnt img content', len(list_images(path)))
    
#----------------------------------------------
def create_dir(pth_):
    if not os.path.exists(pth_):
        os.mkdir(pth_)
        print(f'Directory {pth_} created')
        return True
    else:
        print(f'Directory {pth_} exist')
        return False   
#----------------------------------------------    
def cp_2_dir(path_project,dataset_dir, newdir = 'test'):

    new_path = os.path.join(path_project, newdir)

    content = os.listdir(dataset_dir)

    for content_i in content:

        old_path_ = os.path.join(dataset_dir,content_i)
        new_path_ = os.path.join(new_path,content_i)
        if os.path.isdir(old_path_):
            copy_tree(old_path_, new_path_,)
        else:
            if os.path.exists(new_path_):
                ARN, extension = os.path.splitext(new_path_)
                new_path_ = ARN +'(1)'+ extension
            shutil.copyfile(old_path_, new_path_)

        print(f'{old_path_} copied to {new_path_}') 

#----------------------------------------------        
def _cp_file_list(general_dir, subdir, file_list):

    pth_ = os.path.join(general_dir,subdir)
    print(pth_)
    create_dir(pth_)  

    for file_path in file_list:
        file_name = os.path.split(file_path)[-1]
        new_path = os.path.join(pth_,file_name)
        if not os.path.exists(new_path):
            shutil.copyfile(file_path, new_path)
            print(f' copy from {file_path} to {new_path}')
        else:
            print(f' file {new_path} already exist')
            
#----------------------------------------------              
def get_anno_path(project_path, data_dir_name):
    ''' auxiliary function to find json and return path'''
    pt_ = os.path.join(project_path, data_dir_name)
    anno_file = list_json(pt_)[0]
    anno_path = os.path.join(pt_, anno_file)
    return anno_path