from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Subset
from sklearn.model_selection import KFold
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
import json
from config.config import get_config


def de_preprocess(tensor):
    return tensor*0.5 + 0.5


class DroneFace(Dataset):
    def __init__(self, json_path, transform) -> None:
        super().__init__()
        with open(json_path, 'rb') as f:
            self.img_list = json.load(f)
        self.json_path = json_path
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        img_path = item['img_path']
        p_id = item['p_id']
        yaw = item['yaw']
        img = Image.open(img_path)
        img = self.transform(img)
        return img, p_id, abs(yaw)

def get_loaders(conf, pose=False):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.Resize((112, 112)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    num_folds = conf.num_folds
    kfold = KFold(n_splits=num_folds, shuffle=False)
    if pose:    
        ds = DroneFace(conf.droneface_json, train_transform)
    else:
        ds = ImageFolder(conf.droneface_folder, train_transform)
    
    # dataset_size = len(ds)
    # test_size = dataset_size / num_folds
    splits = kfold.split(ds)

    loaders = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(train_idx)
        train_set = Subset(ds, train_idx)
        train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
        # train_set = Subset(ds, train_idx[:0.75*len(train_idx)])
        # val_set = Subset(ds, train_idx[0.75*len(train_idx):])
        test_set = Subset(ds, test_idx)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
        loaders.append((train_loader, test_loader))
    class_num = 8
    return loaders, class_num

# def get_train_loader(conf):
#     if conf.data_mode == 'droneface':
#         if conf.pose:
#             train_sets, test_sets, class_num = get_train_dataset(conf.droneface_json, conf.pose)
#         else:
#             train_sets, test_sets, class_num = get_train_dataset(conf.droneface_folder, conf.pose)
        
#     loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
#     return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label