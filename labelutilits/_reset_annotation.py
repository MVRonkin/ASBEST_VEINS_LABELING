import os
import numpy as np
import json
from PIL import Image
import pandas as pd

from ._annotation_base import (_data2df,_data_head, _df2anno)

#---------------------------------------
def _reset_indexes(data):
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
    anno_df = _data2df(data)
    anno_df =  anno_df.sort_values(by=['image_id'])
    anno_df['id'] = (np.arange(len(anno_df)).astype(int)+1)
    data_head, anno_keys, img_keys = _data_head(data)
    data = _df2anno(anno_df, data_head, anno_keys, img_keys)
    return data

#---------------------------------------
def _reset_images(data):
    '''
    Reset images in COCO JSON annotations,
      relative to existed in image directory.
    
    Paramters
    ---------
    data: dict[list[dict]],
      data annotation dictionary in JSON COCO format.
    
    Returns
    ----------
    dict[list[dict]]: data annotation  
      dictionary in JSON COCO format.
    DataFrame: report with removed file_names and ids. 
    
    '''
    report = [[x['id'], x['file_name']] 
                 for x in data['images'] 
                    if not os.path.exists(x['file_name'])]
    
  
    data['images'] = [x for x in data['images'] 
                         if os.path.exists(x['file_name'])]
    
    
    return data, pd.DataFrame(report, 
                              columns = ['delated_img_idx',
                                         'deleted_fnames',])

#---------------------------------------
def _reset_labels(data):
    '''
     Reset annotations (labels) in COCO JSON annotations,
      relative to existed in image directory.
    
    Paramters
    ---------
    data: dict[list[dict]],
      data annotation dictionary in JSON COCO format.
    
    Returns
    ----------
    dict[list[dict]]: data annotation  
      dictionary in JSON COCO format.
    DataFrame: report with removed image ids and anno-ids.    
    '''
    images_ids = [x['id'] for x in data['images']]

    report = [[x['image_id'],[x['id']]]  for x in data['annotations'] 
                 if x['image_id'] not in images_ids]

    report = pd.DataFrame(report, 
                          columns=['delated_img_idx', 
                                   'delete_id']).\
                            groupby('delated_img_idx', 
                                    as_index=False).sum()
    
    
    data['annotations'] = [x for x in data['annotations'] 
                              if x['image_id'] in images_ids]

   
    return data, report

#---------------------------------------
def _reset_image_sizes(data):
    '''Correct Image Size in data anno in COCO JSON format.
    Paramters
    -----------
    data: dict[list[dict]],
      annotation dictionary.
    
    Returns
    --------
    data: dict[list[dict]],
      annotation dictionary.
    '''
#     report = pd.DataFrame(columns = ['old_size', 'new_size'])
    for i in range(len(data['images'])):
        fname = data['images'][i]['file_name'] 
        report_={'fname': data['images'][i]['file_name'],
                 'old_size':(data['images'][i]['width'], 
                             data['images'][i]['height'])}

        width, height = Image.open(fname).size
        data['images'][i]['width']  = width
        data['images'][i]['height'] = height
        
#         report_.update({'new_size':(width, height)})
#         report = report.append(report_, ignore_index=True)

    return data#, report

#---------------------------------------
def reset_annotation(data):
    '''
     Reset annotation in COCO JSON annotations,
      relative to existed in image directory,
      also id of images and annotation will be renewd.
    
    Paramters
    ---------
    data: dict[list[dict]],
      data annotation dictionary in JSON COCO format.
    
    Returns
    ----------
    dict[list[dict]]: data annotation  
      dictionary in JSON COCO format.
    
    DataFrame: report with removed image ids and anno-ids.    
    '''
    data, report_ = _reset_images(data)
    data, report = _reset_labels(data)

    data = _reset_image_sizes(data)
    data = _reset_indexes(data)  

    report = pd.merge(report_,report, on="delated_img_idx") 
    return data, report