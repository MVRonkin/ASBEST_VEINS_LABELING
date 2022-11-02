
import os
import numpy as np
import json
from PIL import Image


#-----------------------------------

def _imgs2gray(img_pths):
    '''Convert Image to gray scale format.
    Parameters
    -----------
    img_pths: list[string],
      list of image pathes 2 check and resize.
    
    Returns
    --------
    list[string],
      list of corrected images
    '''    
    report_list = list()
    for img_pth in img_pths:
        img = Image.open(img_pth)
        if Image.open(img_pth).mode != 'L':
            img.convert("L").save(img_pth)
            report_list.append(img_pth)
    return report_list
#-----------------------------------

#----------------------------------

def _resize_imgs(img_pths, width, height):
    ''' Resize Image by list of pathes
    Parameters
    -----------
    img_pths: list[string],
      list of image pathes 2 check and resize.
    width, height: int, int,
      image width and height
    
    Returns
    --------
    list[string],
      list of corrected images
    '''
    report_list = list()
    for img_pth in img_pths:
        img = Image.open(img_pth)
        width_, height_ = img.size
        if width_ != width or height_ != height:
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(img_pth)
            report_list.append(img_pth)
    return report_list

#----------------------------------


