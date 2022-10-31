import os
import numpy as np
import json

from ._annotation_base import (_open,_set_cat_names,_cat_ids,
                               _filter_cat,_replace_image_dir,
                               _get_data_info,_reset_ids,
                               _most_frequent_size, _image_list)
  
class BaseAnnotation:
    ''' 
    Methods
    ---------
    open_data: Open data in json format
    set_cat_names: New class (categories) names
    filter_cat: Rest only selected category
    info: Return summaraized information from annotation
    new_image_dir: Replace image dir path
    save: Save data in json format
    data_dict:Return data in format dict[list[dict]]
    rest_ids: Reset category ids; image ids; anno_ids
    ''' 
    def __init__(self, anno_path, image_dir_path = None):
        self.anno_path = anno_path
        self.image_dir_path = image_dir_path
        if self.image_dir_path == None:
            self.image_dir_path = os.path.split(anno_path)[0]
        self.open_data(self.anno_path)

    def open_data(self,anno_path):
        ''' Open data in json format '''
        self.anno_path = anno_path
        with open(self.anno_path) as json_file:
            self.data = json.load(json_file)
        return self

    def set_cat_names(self,new_names = ['']):
        ''' New class (categories) names,
            work only if length of new class 
            list same as le of cat_ids'''
        self.data = _set_cat_names(self.data, new_names = new_names)
        return self
    
    def filter_cat(self, cat_ids = None):
        '''Rest only selected category 
           if None filter only images contains some labeling'''
        self.data = _filter_cat(self.data, cat_ids = cat_ids)
        return self
    
    def new_image_dir(self, new_dir = ''):
        ''' Replace image dir path'''
        self.data = _replace_image_dir(self.data, new_dir = new_dir)
        self.image_dir_path = new_dir
        return self
    
    def rest_ids(self):
        ''' Reset category ids; image ids; anno_ids'''
        self.data = _reset_ids(self.data)
        return self
        
    def add_image_path_2_anno(self):
        return self.new_image_dir(self.image_dir_path)
    
    def data_dict(self):
        ''' Return data in format dict[list[dict]]'''
        return self.data
    
    def info(self):
        '''
        Return summaraized information from annotation
        
        Returns
        -----------
        dict['string':[int, string]].   

        Notes
        -------------
        output inclues: 
        * dataset name;
        * path to annotation;
        * annotation file name;
        * path to image directory;
        * length of dataset (images number);
        * anno_number (number of instances for all images);
        * categories (class) ids: identification number of each class;
        * class_names: names of classes;
        * supercategory: names of supercategories;        
        * height: image heights;
        * width: image widths;
        * image_fname_example: image file name example.
        '''
        info =  _get_data_info(self.data)        
        info = {**info, 
                'anno_path':self.anno_path,
                'anno_fname':os.path.split(self.anno_path)[1]}
        info['image_dir_path']  = self.image_dir_path
        return info
    
    def save(self, new_path = None, replace_path = False):
        ''' Save data in json format,
            if path is none anno_path utilized'''
        if new_path == None: new_path = self.anno_path
        with open(new_path, 'w') as f:
            json.dump(self.data, f)
        if replace_path:
            if os.path.split(new_path)[0] == '':
                new_path = os.path.join(
                                os.path.split(self.anno_path)[0],
                                new_path)
            self.anno_path = new_path
            
        return self
    
    def get_anno_path(self) :
        ''' Return anno_path'''
        return anno_path
    

    