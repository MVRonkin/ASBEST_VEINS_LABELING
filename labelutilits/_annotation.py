
import os
import numpy as np
import json

class Annotation:
    ''' 
    Class for annotation in json coco format processing.
    
    Parameters
    ----------
    anno_path: string, 
      path for annotation file
    image_dir_path: string,
      path for image directory
    
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
        info['image_dir_path']  = self.image_dir_path
        info = {**info, 
                'anno_path':self.anno_path,
                'anno_fname':os.path.split(self.anno_path)[1]}
        return info
    
    def save(self, new_path = None, replace_path = False):
        ''' Save data in json format,
            if path is none anno_path utilized'''
        if new_path == None: new_path = self.anno_path
        with open(new_path, 'w') as f:
            json.dump(self.data, f)
        if replace_path:
            if os.path.split(new_path)[0] = '':
                new_path = os.path.join(
                                os.path.split(self.anno_path)[0],
                                new_path)
            self.anno_path = new_path
            
        return self
    
    def get_anno_path(self) :
        ''' Return anno_path'''
        return anno_path
    
    
#---------------------------------------
def _open(anno_path):
    ''' Open data in json format
    Parameters
    ----------
    anno_path: string, 
      path annotation file.
    
    Returns
    ----------
    dict[list[dict]],
      coco format dict for json save.
    '''
    with open(anno_path) as json_file:
        data = json.load(json_file)
        json_file.close()
    return data
#---------------------------------------
def _set_cat_names(data, new_names):
    '''Set categories (class) names
    
    Parameters
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
    new_names: list[string],
      new names for classes (categories).
      
    Returns
    ----------
    dict[list[dict]],
      coco format dict for json save.
    '''
    new_names = list(np.atleast_1d(new_names))
    if len(new_names) != len(data['categories']):
        raise ValueError('''len(new_names) != len(data['categories'])''')
    for i,name in enumerate(new_names):
        data['categories'][i]['name'] = name
    return data
#---------------------------------------
def _cat_ids(data, cat_ids = None):
    '''Set categories (class) ids
    
    Parameters
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
    cat_ids: list[string],
      new category id, all for classes (categories).
      
    Returns
    ----------
    list[int],
      categories
    dict[list[dict]],
      coco format dict for json save.
    '''
    if cat_ids == None: 
        cat_ids = [id_['id'] for id_ in data['categories']]
    else: 
        cat_ids = list(np.atleast_1d(cat_ids))
        data['categories'] = [x for x in data['categories'] if x['id'] in cat_ids]
    return cat_ids, data
#---------------------------------------
def _filter_cat(data, cat_ids = None):
    ''' Select only instasnce for category,
    if cat is None, select all labeld instances.
    
    Parameters
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
    cat_ids: list[string],
      new category id, all for classes (categories).
      
    Returns
    ----------
    dict[list[dict]],
      coco format dict for json save.
    '''
    cat_ids, data = _cat_ids(data = data, cat_ids = cat_ids)

    data['annotations'] =  list(filter(lambda x:x['category_id'] in cat_ids, data['annotations']))
    list_ids = list(set(map(lambda x:x['image_id'], data['annotations'])))
    data['images'] = list(filter(lambda x:x['id'] in list_ids, data['images']))
    return data

#---------------------------------------
def _replace_image_dir(data, new_dir=''):
    ''' Replace image directory in field filename.
    
    Parameters
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
    new_dir: string,
      new image directory path.
      
    Returns
    ----------
    dict[list[dict]],
      coco format dict for json save.
    '''    
    for i in range(len(data['images'])):
#         fname = os.path.basename(data['images'][i]['file_name']).split('\\')[-1]
        fname = os.path.split(data['images'][0]['file_name'])[-1]
        data['images'][i]['file_name'] = os.path.join(new_dir, fname)
    return data

#---------------------------------------
def _get_data_info(data):
    '''Information about data in json coco format.
    
    Parameters
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
      
    Returns
    ----------
    dict[int, string],
      dict for json coco annotation description.
    '''
    desc = dict()
    desc['cat_ids']   = [x['id'] for x in data['categories']]
    desc['class_names'] = [x['name'] for x in data['categories']]
    desc['supercategory'] = [x['supercategory'] for x in data['categories']]
    desc['width']  = list({x['width'] for x in data['images']})
    desc['height'] = list({x['height'] for x in data['images']})
    desc['length']      = len(data['images'])
    desc['anno_number'] = len(data['annotations'])
    desc['fname_example']  = data['images'][0]['file_name'] 
    image_dir_path = os.path.split(desc['fname_example'])[0]
    name_dataset = os.path.split(os.path.split(image_dir_path)[0])[1]
    desc['dataset_name']  = name_dataset

    return desc

#---------------------------------------
def _reset_ids(data):
    '''Reser all indexes,
      category id, image ids, annotation ids.
    
    Parameters
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
      
    Returns
    ----------
    data: dict[list[dict]], 
      coco format dict from json.
    '''
    tmp_cat_id = dict()
    for i in range(len(data['categories'])):
        tmp_cat_id.update({data['categories'][i]['id']: i+1})#  = data['categories'][i]['id']
        data['categories'][i]['id'] = i+1
    #---------------------------------

    #---------------------------------
    tmp_img_id = dict()
    for i in range(len(data['images'])):
        tmp_img_id.update({data['images'][i]['id']:i+1})
        data['images'][i]['id'] = i+1
    #---------------------------------    

    #---------------------------------
    tmp_anno_id = np.zeros(len(data['annotations']), dtype = int)

    for i in range(len(data['annotations'])):
        data['annotations'][i]['category_id'] =\
            tmp_cat_id[data['annotations'][i]['category_id']]
        data['annotations'][i]['image_id'] =\
            tmp_img_id[data['annotations'][i]['image_id']]
        data['annotations'][i]['id'] = i+1
    return data