
import os
import numpy as np
import json

__all__ = ['_open','_set_cat_names','_cat_ids','_filter_cat','_replace_image_dir',
           '_get_data_info','_reset_ids', '_most_frequent_size', '_image_list']
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
        fname = os.path.basename(data['images'][i]['file_name']).split('\\')[-1].split('/')[-1]
        # fname = os.path.split(data['images'][0]['file_name'])[-1]
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
    desc                = dict()
    desc['cat_ids']     = [x['id'] for x in data['categories']]
    desc['class_names'] = [x['name'] for x in data['categories']]
    desc['supercategory'] = [x['supercategory'] for x in data['categories']]
    desc['width']         = list({x['width'] for x in data['images']})
    desc['height']        = list({x['height'] for x in data['images']})
    desc['length']      = len(data['images'])
    desc['anno_number'] = len(data['annotations'])
    desc['fname_example']  = data['images'][0]['file_name'] 
    image_dir_path = os.path.split(desc['fname_example'])[0]
    name_dataset   = os.path.split(os.path.split(image_dir_path)[0])[1]
    desc['image_dir_path']  = image_dir_path
    desc['dataset_name']    = name_dataset

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
#-----------------------------

def _most_frequent_size(data):
    '''Return most frequent image size in form (width, height).
    Paramters
    ---------
    data: list[dict[list]],
      annotation dict in COCO JSON format.
    
    Returns
    --------
    width, height: int, int,
      image size in form (width, height).    
    '''
    widths  = np.asarray([x['width'] for x in data['images']])
    heights = np.asarray([x['height'] for x in data['images']])
    vals, counts = np.unique(widths,return_counts=True)
    width  = vals[np.argsort(counts)[-1]]
    vals, counts = np.unique(heights,return_counts=True)
    height = vals[np.argsort(counts)[-1]]
    return width, height

def _image_list(data):
    ''' List of image pathes.
    Parameters
    -----------
    data: list[dict[list]],
      annotation dict in COCO JSON format.
    
    Returns
    --------
    list[string],
      list of image pathes.
    '''
    return [x['file_name'] for x in data['images']]