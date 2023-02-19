import numpy as np
from _path import list_ext, create_dir
from pathlib import Path

def yolo2coco(xc, yc, w, h, image_width, image_height):
    """
        Parameters
        ----------
        xc : float   X coord center point in Yolo format
        yc : float   Y coord center point in Yolo format
        w  : float   Weight of box in Yolo format
        h  : float   Height of box in Yolo format
        image_width  : float image width
        image_height : float image height
        Returns
        -------
        xtl : float X top left coord of box
        ytl : float Y top left coord of box
        w_abs : float absolute weight of box
        h_abs : float absolute height of box
    """
    xc, w_abs = xc*image_width,  w*image_width
    yc, h_abs = yc*image_height, h*image_height
    xtl = xc - w_abs/2
    ytl = yc - h_abs/2
    return xtl ,ytl, w_abs, h_abs

def yolo2xyxy(xc, yc, w, h):
    """
        Parameters
        ----------
        xc : float   X coord center point in Yolo format
        yc : float   Y coord center point in Yolo format
        w  : float   Weight of box in Yolo format
        h  : float   Height of box in Yolo format
        Returns
        -------
        x1 : float top left x
        y1 : float top left y
        x2 : float bottom right x
        y2 : float bottom right y
    """
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2 
    return x1, y1, x2, y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def convert_dir_yolo_to_xyxy(input_path, out_path):
    files = list_ext(input_path, 'txt')
    create_dir(out_path)
    for f_name in files:
        f_out = open(Path(out_path) / f_name, 'w') 
        f = open(Path(input_path)/f_name, 'r')
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            if len(items) == 5:
                _cls, xc, yc, w, h = items
                x1, y1, x2, y2 = yolo2xyxy(float(xc), float(yc), float(w),float(h))
                f_out.write("{} {} {} {} {}\n".format(_cls, x1, y1, x2, y2))
            else:
                _cls, xc, yc, w, h, p = items
                x1, y1, x2, y2 = yolo2xyxy(float(xc), float(yc), float(w),float(h))
                f_out.write("{} {} {} {} {} {}\n".format(_cls, x1, y1, x2, y2, float(p)))
        f_out.close()
        f.close()

def convert_dir_yolo_to_cls_coef_xyxy(input_path, out_path):
    files = list_ext(input_path, 'txt')
    create_dir(out_path)
    for f_name in files:
        f_out = open(Path(out_path) / f_name, 'w') 
        f = open(Path(input_path)/f_name, 'r')
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            _cls, xc, yc, w, h, p = items
            x1, y1, x2, y2 = yolo2xyxy(float(xc), float(yc), float(w),float(h))
            f_out.write("{} {} {} {} {} {}\n".format(_cls, float(p), x1, y1, x2, y2))
        f_out.close()
        f.close()

if __name__ == "__main__":
    convert_dir_yolo_to_cls_coef_xyxy("/home/reshetnikov/asbest/yolov8_segmentation/mAP/input/detection-results",
                                  "/home/reshetnikov/asbest/yolov8_segmentation/mAP/input/detect_convert/")