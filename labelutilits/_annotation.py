import os
import numpy as np
import json
from PIL import Image
import pandas as pd
from pprint import pprint
 

from ._annotation_base import (_set_cat_names,
                               _cat_ids,
                               _filter_cat,
                               _replace_image_dir,
                               _get_data_info,
                               _most_frequent_size, 
                               _image_list,
                               _count_anno_at_images,
                              )

from ._reset_annotation import (_reset_indexes,
                                _reset_images,
                                _reset_labels,
                                _reset_image_sizes, 
                                reset_annotation)

from ._image_base import (_resize_imgs,
                          _imgs2gray)

from ._coco_base import (_ann2mask,
                         _masks2image,
                         _masks2d,
                         _image_with_bbox)


    
class Annotation():
    ''' 
    Class for annotation in json coco format processing.
    
    Atributs
    ----------
    anno_path: string, 
      path for annotation file
    image_dir_path: string,
      path for image directory

    '''
 

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
        self.counts_anno = _count_anno_at_images(self.data)
        self.report = dict()
    
    #---------------------------------
    def open_data(self, anno_path):
        ''' Open data in json format.
        
        Paramters
        ----------
        anno_path: string,
          annotation path for json coco comatible format file. 
        '''
        self.anno_path = anno_path
        with open(self.anno_path) as json_file:
            self.data = json.load(json_file)
        return self

    #---------------------------------
    def set_cat_names(self, new_names = ['']):
        ''' New class (categories) names,
            work only if length of new class 
            list same as le of cat_ids.
        
        Paramters
        ----------
        new_names: list[string],
          new class (categories) names.            
        '''
        self.data = _set_cat_names(self.data, new_names = new_names)
        return self
    
    #---------------------------------
    def filter_cat(self, cat_ids = None):
        '''Rest only selected category 
           if None filter only images contains some labeling.
        
        Paramters
        ----------
        cat_ids: string,
          category (class) indexes to rest.   
        '''
        self.data = _filter_cat(self.data, cat_ids = cat_ids)
        return self
    
    #---------------------------------
    def replace_image_dir(self, dir_path = None):
        ''' Replace image dir path in annotation
        
        Paramters
        ----------
        dir_path: string,
          new image directory path.
        '''
        if dir_path is None: dir_path = self.image_dir_path
        self.data = _replace_image_dir(self.data, new_dir = dir_path)
        self.image_dir_path = dir_path
        return self
    
    #---------------------------------
    def reset_annotation(self):
        '''
        Reset annotation in COCO JSON annotations,
          relative to existed in image directory,
          also id of images and annotation will be renewd.
        
        '''        
        self.data, report = reset_annotation(self.data)
        self.report.update({'deleted_as_unexisted':report})
        return self
    
    #---------------------------------    
    def data_dict(self):
        ''' Return data in COCO JSON 
            compatible format dict[list[dict]]'''
        return self.data
    
    #--------------------------------------
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
    
    def print_info(self):
        ''' print information about data'''
        pprint(self.info())
        return self
    #--------------------------------------
    def save_anno(self, new_path = None, replace_path = False):
        ''' Save annotation in json format,
            if path is none anno_path is utilized.
        
        Parameters
        ----------
        new_path: string,
          path to save JSON COCO compatible file;
          if None, old path is utilized.
        replace_path: bool,
          if True, new_path replace anno_path.
        '''
        if new_path == None: new_path = self.anno_path
        
        if os.path.split(new_path)[0] == '':
            new_path = os.path.join(
                            os.path.split(self.anno_path)[0],
                            new_path)
       
        with open(new_path, 'w') as f:
            json.dump(self.data, f)

        if replace_path:
            self.anno_path = new_path
            
        return self
    
    #--------------------------------------
    def get_anno_path(self) :
        ''' Return anno_path.'''
        return self.anno_path
    
    #--------------------------------------
    # IMAGE_DIRECTORY  
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
    
    # IMAGE_ID
    #-------------------------------------    
    def get_image_path(self, image_id):
        '''
        Get image pathes by image id.

        Parameters
        ----------
        image_id: int,
          images to select, start from 1.

        Returns
        ----------
        string: image pathes.
        '''
        self.__check_image_id(image_id)
        return os.path.join(self.image_dir_path ,self.data['images'][image_id - 1]['file_name'])
    
    #-------------------------------------    
    def get_image(self, image_id):
        '''
        Get image pathes by image id.

        Parameters
        ----------
        image_id: int,
          images to select, start from 1.

        Returns
        ----------
        ndarray: image.
        '''
        image_path = self.get_image_path(image_id)
        image      = np.array(Image.open(image_path))
        return image

    #-------------------------------------    

    def get_annotations(self, image_id = 1, cat_ids = None):
        '''
        Get annotations for image by id.
        
        Parameters
        ----------
        image_id: int,
          images to select, start from 1.
        cat_ids: list[int],
          categories to output, all possible if None.

        Returns
        ----------
        list[dict]: annotations for image in COCO format.
        
        '''
        self.__check_image_id(image_id)
        anns = [x for x in self.data['annotations'] if x['image_id'] == image_id]
        if cat_ids !=None:
            cat_ids = np.atleast_1d(cat_ids).astype(int)
            anns = [x for x in anns if x['category_id'] in cat_ids]
        
        return anns
    #-------------------------------------        
    def get_image_descriptor(self, image_id):
        '''
        Get image descriptor by id.
        
        Parameters
        ----------
        image_id: int,
          images to select, start from 1.

        Returns
        ----------
        dict: description for image in COCO format.
        '''        
        self.__check_image_id(image_id)
        return self.data['images'][image_id-1]
    
    #-----------------------------------------------
    def get_segmentations(self, image_id, cat_ids = None):
        '''
        Get segmentations for image in the fromat [x,y,...].
        
        Parameters
        ----------
        image_id: int,
          images to select, start from 1.
        cat_ids: list[int],
          categories to output, all possible if None.
        
        Returns
        ----------
        list[list]: annotation for instances segmentation 
          for image in format [x,y,x,y...].
        
        '''
        anns = self.get_annotations(image_id = image_id, 
                                    cat_ids = cat_ids)
        return sum([x['segmentation'] for x in anns],[])
    
    #-----------------------------------------------
    def get_bboxes(self, image_id, cat_ids = None):
        '''
        Get bounding boxes for image 
          in the fromat [x0,y0,w,h], where: 
          [x0,y0] is the left upper corner; 
          [w,h] is the right lower corner. 
        
        Parameters
        ----------
        image_id: int,
          images to select, start from 1.
        cat_ids: list[int],
          categories to output, all possible if None.
        
        Returns
        ----------
        list[list]: annotation for bounding boxes 
          for image in format [x0,y0,w,h].
        
        '''
        anns = self.get_annotations(image_id = image_id, cat_ids = cat_ids)
        return [x['bbox'] for x in anns]
    
    #----------------------------------------------- 
    def get_masks(self,image_id, cat_ids = None, mode = 'instances'):
        '''
        Get segmentation masks for image instanaces by id.
        
        Parameters
        ----------
        image_id: int,
          images to select, start from 1.
        cat_ids: list[int],
          categories to output, all possible if None.
        mode: string, 
          posible modes: 'instances', '3d array', '2d array'.
          * 'instances': output is the 3d ndarray in format instances x height x width.
          * '3d array':  output is the 3d ndarray in format height x width x channel,
             where each instances have random color, adobted for visualization.
          * '2d array':  output is the 2d ndarray in format height x width,
             where each instant have different value in range from 0 to max.
          * 'semseg':  output is the 2d ndarray in format height x width,
             where each instant have same value 1.
        Returns
        ----------
        ndarray: image like array .
        
        '''        
        anns = self.get_annotations(image_id = image_id, 
                                    cat_ids  = cat_ids)
        img_desc = self.get_image_descriptor(image_id)

        h = img_desc['height'] 
        w = img_desc['width' ]
        
        out = np.asarray([_ann2mask(ann,h,w) for ann in anns])
        if mode == '3d array': out = _masks2image(out)
        if mode == '2d array': out = _masks2d(out)
        if mode == 'semseg': out = _masks2d(out); out[out>0]=1
        return out #, dtype = np.int8)
    #---------------------------------------------- 
    def get_image_with_bbox(self, image_id = 1, cat_ids = None, color =0, thikness = 10):
        '''
        Get image with drawn bounding boxes.
        
        Parameters
        ----------
        image_id: int,
          images to select, start from 1.
        cat_ids: list[int],
          categories to output, all possible if None.
        color: int; [int,int,int],
          color for box bounds, int format for brightness,
          [int,int,int] format for color.
        thikness: int,
          thikness for box bounds.
        
        Returns
        ----------
        ndarray: image array with drawn bounding boxes.        
        '''
        img = self.get_image(image_id)
        bboxes = self.get_bboxes(image_id, cat_ids)
        if bboxes:
          return _image_with_bbox(img, bboxes, color, thikness)
        else:
          return img
    #----------------------------------------------
    def __check_image_id(self, image_id):
        if image_id<1 or image_id > len(self.data['images']):
            raise ValueError(f'image_id {image_id} image_id<1 or image_id > len(data[images])')