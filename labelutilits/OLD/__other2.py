from pycocotools import mask as cocoutils

def get_masks(image_anno):
    h = image_anno['height'] 
    w = image_anno['width' ]
    mask_anns = image_anno['annotations']
    mask = np.zeros((h,w, len(mask_anns)))
    for i,ann in enumerate(mask_anns):
        mask[:,:,i] = _ann2mask(ann,h,w)
    return mask

def _ann2mask(ann, h,w):
    segm = ann['segmentation']
    rles = cocoutils.frPyObjects(segm, h, w)
    rle  = cocoutils.merge(rles)
    instant_mask = cocoutils.decode(rle)  
    return instant_mask

def get_bboxes(image_anno):
    bboxes = np.zeros((len(image_anno['annotations']),4),dtype=int)#x0,y0,w,h

    for i,ann in enumerate(image_anno['annotations']):
        bboxes[i] = ann['bbox']
    bboxes[:,2] = bboxes[:,0]+bboxes[:,2] #x-max
    bboxes[:,3] = bboxes[:,1]+bboxes[:,3] #y-max
    return bboxe

from detectron2.structures import BoxMode

def balloon2COCO(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = np.asarray(Image.open(filename)).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px)-np.min(px), np.max(py)-np.min(py)],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def plot_instant(image, mask, bbox):

    fig, axs = plt.subplots(1, 4, figsize = (18,6))
    image_ = np.copy(np.asarray(image)).astype(float)/image.max()
    
    # Ground Truth    
    axs[0].imshow(np.clip(image_,0,1), 'gray')
    axs[0].axis('off'); axs[0].set_title('Original Image')
    #----------------------------

    # Semantic Segmentaion
    mask_ = np.asarray(mask).sum(2)
    mask_[mask_>0]=255
    axs[1].imshow(mask_, 'gray')
    axs[1].axis('off'); axs[1].set_title('Semantic Segmentaion')  
    #----------------------------

    # Instance Segmentation
    out  = np.zeros((*mask.shape[:2],3),dtype=float)
    for i,mask_ in enumerate(mask.transpose((2,0,1))):
        ch_  = i%3
        mask_ = mask_*(np.random.rand()*0.5 + 0.5)        
        out[:,:,ch_]  = out[:,:,ch_]*0.9 + mask_[:,:]
   
    axs[2].imshow(np.clip(out/out.max(),0,1), 'gray')
    axs[2].axis('off'); axs[2].set_title('Instance Segmentation')  
    #----------------------------

    # Object Detection BBoxes
    bbox_ = np.asarray(bbox)
    axs[3].imshow(np.clip(image_,0,1), 'gray')
    for box_ in bbox_:
        bb = patches.Rectangle(xy     = (box_[0],box_[1]), 
                               width  = box_[2]-box_[0],
                               height = box_[3]-box_[1], 
                               linewidth=2, 
                               edgecolor="blue", 
                               facecolor="none")
        axs[3].add_patch(bb)  
    axs[3].axis('off'); axs[3].set_title('Object Detection')

    plt.tight_layout()
    plt.show()