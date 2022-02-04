import cv2 as cv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from dataset import *
from torch.utils.data import Dataset, DataLoader
def get_image(image_path,label_path):
    pass

def get_labels_info(info_path):
    info = pd.read_csv(info_path)
    # info has format: [['obstacles' 59 193 246]...]
    # info = info.to_numpy()
    class_names = np.array(info["name"])
    labels_values = np.array(info[["r","g","b"]])
    return class_names, labels_values

def convert_data(img, label, info_path):
    img = img/255.0
    class_names, labels_values = get_labels_info(info_path)
    sematic_maps = []
    for color in labels_values:
        same = np.equal(label, color)
        class_map = np.all(same,axis=-1)
        sematic_maps.append(class_map)
    semantic_map = np.array(np.stack(sematic_maps,axis=-1))
    return img, semantic_map

def save_checkpoints(state, file_name="checkpoint.pth.tar"):
    print("=========> saving checkpoint")
    torch.save(state,file_name)

def load_checkpoints(checkpoint, model):
    print("==========> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    info_path,
    num_workers=4,
    pin_memory=True,
):
    train_ds = WASR_dataset(
        image_dir=train_dir,
        label_dir = train_maskdir,
        info_path= info_path,
        transform=train_transform
    )
    val_ds = WASR_dataset(
        image_dir=val_dir,
        label_dir= val_maskdir,
        info_path = info_path,
        transform=val_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader,val_loader


if __name__ =="__main__":
    info_path = "/home/gumiho/project/WASR_seg/WASR/class_dict.csv"
    img = np.array(Image.open("/home/gumiho/project/WASR_seg/WASR/train/0001.png").convert("RGB"))
    label = np.array(Image.open("/home/gumiho/project/WASR_seg/WASR/train_labels/0001m.png").convert("RGB"))
    se = convert_data(img,label,info_path)
    print(se[0].shape)
    print(se[1].shape)