class CocoDetectrion2Datasets(CocoDatasets):    
    #-----------------------------------------------------
    def plot_detectron2_res(self, res, index):    
        fig, axs = plt.subplots(2,2, figsize=(19,19))

        boxes  = res[index]['instances'].pred_boxes.to('cpu')
        scores = res[index]['instances'].scores.to('cpu').detach().numpy()
        masks  = res[index]['instances'].pred_masks.cpu().detach().numpy()  # a tensor of shape (N, H, W)
        for box,score in zip(boxes,scores):
            box = box.detach().numpy()

            height = box[3] - box[1]
            width  = box[2] - box[0]

            up_right_corner = (box[0], box[1] )

            bb = patches.Rectangle(up_right_corner, width ,height, linewidth=2, edgecolor="black", facecolor="none")
            axs[0,1].add_patch(bb)
            axs[0,1].text(box[0],box[1], str(int(score*100)/100) , fontsize = 11, fontweight = 'black', color = 'white'  )


        axs[0,1].imshow(image, 'gray')
        axs[0,1].axis('off')
        axs[0,1].set_title('Object Detection')

        

        # Draw boxes and add label to each box
        colors = list(matplotlib.colors.BASE_COLORS.keys())

        mask_ = np.zeros(image.shape).astype(float)


        for mask in masks:
            i = np.random.randint(128,255,1)
            mask_ += mask.astype(float) *i+10

        mask_ /=np.max(mask_)

        axs[1,1].imshow(mask_,cmap = 'gray' )

        # axs[1,1].imshow(image, cmap='gray')

        axs[1,1].axis('off')
        axs[1,1].set_title('Instance Segmentation')

        axs[0,0].imshow(cv2.addWeighted(image/image.max(), 0.5, mask_, 0.8,0), cmap='gray'); 
        axs[0,0].axis('off')
        axs[0,0].set_title('Instance Segmentation')

        axs[1,0].imshow(image)
        axs[1,0].axis('off')
        axs[1,0].set_title('Original Image')        
        plt.tight_layout()
        plt.show()
        

class Coco2YoloDatasets(CocoDatasets):
    
    def set_coco_datasets_info(self, list_images, list_anno):
        self._coco_datasets_info(list_images, list_anno)
        
        for dataset_id in range(len(self.coco_list)):
            general_dir = self.get_dataset_dir(dataset_id=dataset_id)
            yolo_path = os.path.join(general_dir, 'labels')
            self.coco_list[dataset_id]['labels_path'] = yolo_path
            
            if not os.path.exists(yolo_path):
                self.create_yolo5anno(dataset_id, yolo_label_dir = 'labels', cat_id = None)

    #-------------------------    
    def get_labelsdir_path(self,dataset_id):
        return self.coco_list[dataset_id]['labels_path']
    
    #--------------------------
    def create_yolo5anno(self, dataset_id, yolo_label_dir = 'labels', cat_id = '1', cat_pseudonim = None):
        
        anno_path   = self.get_anno_path(dataset_id=dataset_id)
        image_dir_path  = self.get_imagedir_path(dataset_id=dataset_id)
        name        = self.get_name(dataset_id=dataset_id)
        general_dir = self.get_dataset_dir(dataset_id=dataset_id)

        yolo_path = os.path.join(general_dir, yolo_label_dir)
        self._create_dir(yolo_path)
        
        dataset = importer.ImportCoco(anno_path, path_to_images=image_dir_path, name=name)
        
        dataset.export.ExportToYoloV5(output_path=yolo_path, copy_images=False,)[0]
        
        if cat_id != None:
            self._filter_cat_yolo5(dataset_id, cat_id)
            if cat_pseudonim != None:
                self._rename_cat(dataset_id = dataset_id, cat_id = cat_id,  cat_pseudo = cat_pseudonim)
        return yolo_path
   
    #--------------------------
    def _filter_cat_yolo5(self, dataset_id, class_id = '1'):
        labels_dir = self.get_labelsdir_path(dataset_id = 0) 
        list_labels = os.listdir(labels_dir)

        class_id = str(class_id)
        for lablel_file in list_labels:
            pth_to_file = os.path.join(labels_dir, lablel_file)
            
            if pth_to_file.endswith('.txt'):
                with open(pth_to_file,'w+') as f:
                    lines = f.readlines()
                    f.close()
                
                lines = [line for line in lines if line[0] in np.atleast_1d(class_id)]   

                with open(pth_to_file,'w') as f:
                    f.writelines(lines)
                    f.close()
    #------------------------
    def _rename_cat(self, dataset_id, cat_id = '1',  cat_pseudo = '0'):
        labels_dir = self.get_labelsdir_path(dataset_id = dataset_id)
        
        for file_name in os.listdir(labels_dir):
            file_path = os.path.join(labels_dir,file_name)
            if file_path.endswith('.txt'):

                print(file_path)

                with open(file_path,'r') as f:  

                    lines = f.readlines()
                    
                    print(lines)
                    for id_,new_id in zip(np.atleast_1d(cat_id), np.atleast_1d(cat_pseudo)):

                        for i,line in enumerate(lines):

                            if line.startswith(id_):

                                lines[i] = str(new_id) + line[len(id_):]

                    f.close()
                with open(file_path,'w') as f:  

                    f.writelines(lines)
                    f.close()
       
    #------------------------
    def _train_val_split(self, dataset_ids, train_ratio = 0.7, seed = 42):
        np.random.seed(42)

        list_labels = np.array([],dtype=object)
        list_images = np.array([],dtype=object)
        for dataset_id in dataset_ids:
            anno_path   = self.get_anno_path(dataset_id=dataset_id)
            image_dir_path  = self.get_imagedir_path(dataset_id=dataset_id)
            labels_path = self.get_labelsdir_path(dataset_id=dataset_id)

            list_labels_ = [os.path.join(labels_path,label_name) for label_name in os.listdir(labels_path)]
            list_images_ = [os.path.join(image_dir_path,image_name) for image_name in os.listdir(image_dir_path)]

            list_labels = np.append(list_labels,list_labels_)
            list_images = np.append(list_images,list_images_)
       
           
        n_samples = len(list_labels)  
        indexes   = np.arange(n_samples)
        np.random.shuffle(indexes)

        train_indexes = indexes[:int(n_samples*train_ratio)]
        val_indexes   = indexes[int(n_samples*train_ratio):]

        return (list_images[train_indexes],
                list_labels[train_indexes],
                list_images[val_indexes],
                list_labels[val_indexes])
    
    #--------------------------
    def train_val_dir(self, dataset_ids, train_ratio = 0.7,  seed = 42):
        
        (train_images_list,
         train_labels_list,
         val_images_list,
         val_labels_list) = self._train_val_split(dataset_ids, 
                                                  train_ratio = 0.7, 
                                                  seed = 42)

        project_dir = self.get_project_dir()
        
        pth_ = os.path.join(project_dir,'train')
        self._create_dir(pth_)
        self._cp_file_list(pth_,'images',train_images_list)
        self._cp_file_list(pth_,'labels',train_labels_list)
        
        pth_ = os.path.join(project_dir,'val')
        self._create_dir(pth_)
        self._cp_file_list(pth_,'images',val_images_list)
        self._cp_file_list(pth_,'labels',val_labels_list)        
    
    #--------------------------
    def _cp_file_list(self,general_dir, subdir, file_list):
        
        pth_ = os.path.join(general_dir,subdir)
        print(pth_)
        self._create_dir(pth_)  
        
        for file_path in file_list:
            file_name = os.path.split(file_path)[-1]
            new_path = os.path.join(pth_,file_name)
            if not os.path.exists(new_path):
                shutil.copyfile(file_path, new_path)
                print(f' copy from {file_path} to {new_path}')
            else:
                print(f' file {new_path} already exist')
       
    #--------------------------
    def cp_2_dir(self, dataset_ids, newdir = 'test'):

        new_path = os.path.join(self.path_project, newdir)

        for dataset_id in dataset_ids:
            dataset_dir = self.get_dataset_dir(dataset_id=dataset_id)
            
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

    #--------------------------        
    def _create_dir(self, pth_):
        if not os.path.exists(pth_):
            os.mkdir(pth_)
            print(f'Directory {pth_} created')
            return True
        else:
            print(f'Directory {pth_} exist')
            return False 