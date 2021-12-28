#!/bin/sh
EXP=exp1 

CUDA_VISIBLE_DEVICES=0 python scripts/test.py \
    --img_dir=/media/data/dataset/localization/ILSVRC/val \
    --train_list=./data/ILSVRC/val_list1.txt \
    --test_list=./data/ILSVRC/val_list1.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=imagenet \
    --input_size=256 \
    --num_classes=1000 \
    --num_workers=24 \
    --restore_from=./models/vgg/VGG-16-caffe.pth \
    --save_dir=./runs/vgg_imagenet/${EXP}/ 
