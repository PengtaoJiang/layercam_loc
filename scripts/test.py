import sys
import os
sys.path.append(os.getcwd())

import os
import cv2
import torch
import argparse
import numpy as np
from models import *
from tqdm import tqdm
import scipy.io as scio
from os.path import exists
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.imutils import large_rect, bb_IOU
from utils.LoadData import test_data_loader, test_data_loader_caffe



ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])

def get_arguments():
    parser = argparse.ArgumentParser(description='layercam')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--train_list", type=str, default='')
    parser.add_argument("--test_list", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--scales", type=list, default=[0.5, 0.75, 1, 1.25, 1.5, 2])
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='True')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disp_interval", type=int, default=40)
    parser.add_argument("--snapshot_dir", type=str, default='')
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()

def get_torch_model(args):
    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.restore_from))
    model = model.cuda()
    
    return  model

def get_torch_resnet_model(args):
    model = eval(args.arch).resnet50(pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.restore_from))
    model = model.cuda()
    
    return  model


#####layercam with tencrop
def layercam(args, model=None):   
    if model is None:
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
        
    annotations = scio.loadmat('./data/ILSVRC/val_box.mat')['rec'][0]
    bboxes = []
    for ia in range(len(annotations)):
        bbox = []
        for bb in range(len(annotations[ia][0][0])):
            xyxy = annotations[ia][0][0][bb][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    print('Number of processing images: ', len(bboxes)) 
    
    # # for relu5_3
    target_layers = [29, 30]
    length = 14 

    # # for relu4_3
    # target_layers = [22, 23]
    # length = 28 

    # for relu3_3
    # target_layers = [15, 16]
    # length = 56 

    # for relu2_2
    # target_layers = [8, 9]
    # length = 112

    # for relu1_2
    # target_layers = [3, 4]
    # length = 224

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        if idx > 100:
            break
        img_name, img, label_in = dat
        img = img.cuda()
        img = img.squeeze(0)
        bbox = bboxes[global_counter]
        la = label_in.long().numpy()[0]
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0] 
        logits = logits.sum(dim=0).unsqueeze(0)
        logits1 = torch.softmax(logits, dim=1)
        
        #######top1 cls
        _, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        _, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        if la in inds[0, :5]:
            cls5_counter += 1
            
        global_counter += 1
    
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
        # Obtain gradients for 30 layer
        guided_gradients = model.gradients[0]
        model.gradients = []
        # Upsmaple the gradients
        guided_gradients_upsample = F.interpolate(guided_gradients, 
                                           size=(length,length), mode='bilinear', align_corners=True)
        
        #compute cam
        pre_cam = target * guided_gradients_upsample
        pre_cam = F.relu(pre_cam) #relu
        pre_cam = torch.sum(pre_cam, dim=1)

        cam_ten_crop_upsample = torch.unsqueeze(pre_cam, 1) 
        cam_ten_crop_upsample = F.interpolate(cam_ten_crop_upsample, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze(0) #10*224*224
        
        #0 - topleft         #1 - topright         #2 - bottomleft        #3 - bottomright        #4 - center
        #5 - topleft - flip  #6 - topright - flip  #7 - bottomleft - flip #8 - bottomright - flip #9 - center - flip
        cam_ten_crop_upsample_pad = []
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[0], pad=(0, 32, 0, 32))]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[1], pad=(32, 0, 0, 32))]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[2], pad=(0, 32, 32, 0))]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[3], pad=(32, 0, 32, 0))]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[4], pad=(16, 16, 16, 16))]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[5], pad=(0, 32, 0, 32)).flip([2])] #1-vertical, 2-hori
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[6], pad=(32, 0, 0, 32)).flip([2])]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[7], pad=(0, 32, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[8], pad=(32, 0, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [F.pad(cam_ten_crop_upsample[9], pad=(16, 16, 16, 16)).flip([2])]

        cam = torch.tensor(np.zeros((256, 256), dtype=np.float32)).cuda()
        for i in range(len(cam_ten_crop_upsample_pad)):
            cam += cam_ten_crop_upsample_pad[i][0]

        cam = cam.detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # grabcut
        if target_layers[0] == 15 or target_layers[0] == 8 or target_layers[0] == 3:
           mask = np.zeros((height, width), dtype=np.uint8)
           mask[cam > 0.15*255] = 1
           mask[cam <= 0.15*255] = 2
           bgdModel = np.zeros((1,65),np.float64)
           fgdModel = np.zeros((1,65),np.float64)
           I1, _, _ = cv2.grabCut(cv_im, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
           cam1 = (np.where((I1 == 2) | (I1 == 0), 0, 1) * 255).astype(np.uint8)
           cam = cam1
        
        # # Scale between 0-255 to visualize
        #im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        #out_name = im_name + '_{}.png'.format(la)
        #cam1 = cv_im_gray * 0.2 + cam * 0.8
        #cv2.imwrite(out_name, cam)
        #plt.imsave(out_name, cam1, cmap=plt.cm.viridis)
        
        _, binary_att = cv2.threshold(cam, 0.15*255, 255, cv2.THRESH_BINARY) 
        contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rect.append([x, y, w, h])
        if rect == []:
            estimated_box = [0, 0, 1, 1]
        else:
            x, y, w, h = large_rect(rect)
            estimated_box = [x,y,x+w,y+h]

        ious = []
        recalls = []
        for ii in range(len(bbox)):
            iou, recall = bb_IOU(bbox[ii], estimated_box)
            ious.append(iou)
            recalls.append(recall)
        iou = max(ious)
        recall = recalls[ious.index(iou)]
       
        all_recalls += recall
        all_ious += iou

        if iou >= 0.5:
            gt_known += 1
            if ind == la:
                loc_counter += 1
            if la in inds[0, :5]:
                loc5_counter += 1
    
    gt_acc = gt_known / float(global_counter)
    cls_acc = cls_counter / float(global_counter)
    cls5_acc = cls5_counter / float(global_counter)
    loc_acc = loc_counter / float(global_counter)
    loc5_acc = loc5_counter / float(global_counter)
    ave_recall = all_recalls / float(global_counter)
    ave_iou = all_ious / float(global_counter)
    
    print('Top 1 classification accuracy: {}\t'
         'Top 5 classification accuracy: {}\t'
         'GT-known accuracy: {}\t'
         'top 1 localization accuracy: {}\t'
         'top 5 localization accuracy: {}\t'
         'The mean recall of top-1 class: {}\t'
        'The mean iou of top-1 class: {}\t'.format(
        cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

if __name__ == '__main__':
    args = get_arguments()
    layercam(args)
