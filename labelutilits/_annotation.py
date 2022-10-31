import os
import numpy as np
import json

from ._annotation_base import (_open,_set_cat_names,_cat_ids,
                               _filter_cat,_replace_image_dir,
                               _get_data_info,_reset_ids,
                               _most_frequent_size, _image_list)

from ._base_annotation import BaseAnnotation

from ._image_base import (_resize_imgs,_imgs2gray)

  
    
class Annotation(BaseAnnotation):
    ''' 
    Class for annotation in json coco format processing.
    
    Atributs
    ----------
    anno_path: string, 
      path for annotation file
    image_dir_path: string,
      path for image directory

    '''
    def resize_images(self,size = (224,224)):
        '''Resize Image by list of pathes
        Parameters
        -----------
        size: tuple(int, int): width, height,
          image width and height. 
          If None, width, height are taken 
          as most frequent from data

        Returns
        --------
        list[string],
          list of corrected images (report).
        '''
        if size is None or len(size)<2:
            width, height = _most_frequent_size(self.data)
        else:
            width, height = size[:2]
        img_pths = _image_list(self.data)
        return _resize_imgs(img_pths, int(width), int(height))    
    
    #---------------------------------
    def images2gray(self):
        '''Convert Image to gray scale.

        Returns
        --------
        list[string],
          list of corrected images (report).
        '''        
        img_pths = _image_list(self.data)
        return _imgs2gray(img_pths)
    