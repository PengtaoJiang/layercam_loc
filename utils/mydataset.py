import os
import torch
import torchvision
import random
import json
import cv2
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class dataset(Dataset):   #########used for testing caffe model

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        #im = Image.open(img_name).convert('RGB')
        im = cv2.imread(img_name, 1)
        image = Image.fromarray(im)
        
        if self.transform is not None:
            image = self.transform(image)
            image = image * 255.0 
        
        if self.with_path:
            return img_name, image, self.label_list[idx]
        else:
            return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)

class ocdataset(Dataset):  #########used for training and testing pytorch model

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.with_path:
            return img_name, image, self.label_list[idx]
        else:
            return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))    
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)


class datasetMSF(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, scales=[0.5, 0.75, 1, 1.25, 1.5, 2], transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.scales = scales
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        ms_img_list = []
        for s in self.scales:
            target_size = (int(round(image.size[0]*s)),   
                           int(round(image.size[1]*s)))
            s_img = image.resize(target_size, resample=Image.CUBIC) 
            ms_img_list.append(s_img)

        if self.transform is not None:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        if self.with_path:
            return img_name, ms_img_list, self.label_list[idx]
        else:
            return ms_img_list, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)

class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels #np.array(img_labels, dtype=np.float32)
    
class VOCDatasetWBOX(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.ins_list = '/media/data/snap/acol/vgg_voc/exp7/voc_instance.json'
        self.image_list, self.label_list, self.rois_list = self.read_labeled_image_list(self.root_dir,
        self.datalist_file, self.ins_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        img_name =  self.image_list[idx]
        rois = self.rois_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx], rois 
        
        return image, self.label_list[idx], rois

    def read_labeled_image_list(self, data_dir, data_list, ins_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        rois_list = []

        with open(ins_list, 'r') as f:
            box_list = json.load(f)
        
        for l in range(len(lines)):
            line = lines[l]
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
            
            rois = []
            boxes = box_list[l]['boxes']
            for bb in range(len(boxes)):
                rois.append(boxes['id_{}'.format(bb)])
            rois_list.append(np.array(rois))
        
        return img_name_list, img_labels, rois_list
    

class VOCDatasetMSF(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, scales=[0.5, 1, 1.5, 2], transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.scales = scales
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        ms_img_list = []
        for s in self.scales:
            target_size = (int(round(image.size[0]*s)),   
                           int(round(image.size[1]*s)))
            s_img = image.resize(target_size, resample=Image.CUBIC) 
            ms_img_list.append(s_img)

        if self.transform is not None:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])
        
        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
        
        if self.testing:
            return img_name, msf_img_list, self.label_list[idx]
        
        return msf_img_list, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels #np.array(img_labels, dtype=np.float32)
    

def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id

class dataset_with_mask(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, mask_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        mask_name = os.path.join(self.mask_dir, get_name_id(self.image_list[idx])+'.png')
        mask = cv2.imread(mask_name)
        mask[mask==0] = 255
        mask = mask - 1
        mask[mask==254] = 255

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, mask, self.label_list[idx]
        else:
            return image, mask, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)



class dataset_caffe(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        import torchvision
        #center_crop = torchvision.transforms.CenterCrop((224, 224))
        resize = torchvision.transforms.Resize((256, 256))
        ten_crop = torchvision.transforms.TenCrop((224, 224), vertical_flip=False)

        # # ten crop
        im = Image.open(img_name).convert('RGB')
        in_list = ten_crop(resize(im))    
        in_ = np.stack(in_list)
        in_ = in_[:, :,:,::-1].copy()
        in_ = np.array(in_, dtype=np.float32)
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = in_.transpose((0, 3, 1, 2))

        #single crop
        # resize = torchvision.transforms.Resize((224, 224))
        # im = Image.open(img_name).convert('RGB')        
        # im = resize(im)
        # #im = center_crop(resize(im))
        # in_ = np.array(im, dtype=np.float32)
        # in_ = in_[:, :, ::-1].copy() 
        # in_ -= np.array((104.00699, 116.66877, 122.67892))
        # in_ = in_.transpose((2, 0, 1))


        if self.with_path:
            return img_name, torch.FloatTensor(in_), self.label_list[idx]
        else:
            return torch.FloatTensor(in_), self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)
