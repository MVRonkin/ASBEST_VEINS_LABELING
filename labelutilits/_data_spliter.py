from sklearn.model_selection import KFold
import yaml
import numpy as np
from pathlib import Path
from _path import list_ext, list_images
import shutil
from labelutilits._path import _cp_file_list


def k_fold_split_yolo(path2label:str,
                      path2image:str,
                      path_save_fold:str,
                      number_fold: int = 4):

    l_labels = sorted(list_ext(path2label), key = lambda x: x.split('.')[0])
    l_images = sorted(list_images(path2image), key = lambda x: x.split('.')[0])
    assert len(l_labels) == len(l_images), "The length of arrays does not match"
    path_save_fold  = Path(path_save_fold)
    if path_save_fold.exists():
        shutil.rmtree(path_save_fold)
        path_save_fold.mkdir()
    else:
        path_save_fold.mkdir()

    kfold = KFold(number_fold, shuffle=True)
    for kf, (train_indxs, test_indxs) in enumerate(kfold.split(l_labels)):
        name = "Fold_{}".format(kf)
        path_2_fold = path_save_fold / name
        path_2_fold.mkdir()
        
        train_images = [path2image / name for name in list(map(l_images.__getitem__, train_indxs))]
        _cp_file_list(path_2_fold,"train/", train_images)
        
        test_images = [path2image / name for name in list(map(l_images.__getitem__, test_indxs))]
        _cp_file_list(path_2_fold,"test/", test_images)
        
        train_labels = [path2label / name for name in list(map(l_labels.__getitem__, train_indxs))]
        _cp_file_list(path_2_fold,"train/", train_labels)
        
        test_labels = [path2label / name for name in list(map(l_labels.__getitem__, test_indxs))]
        _cp_file_list(path_2_fold, "test/", test_labels)
        
        yaml_config = {"names": ['stone'],
                "nc": 1,
                "path": str(path_2_fold),
                "train": "./train",
                "val" : "./test",}

        with open(path_2_fold / "config.yaml", 'w') as file:
            documents = yaml.dump(yaml_config, file)
        print(kf, len(train_indxs), len(test_indxs), path_2_fold)
    return