# layercam_loc
## Dependence
python3  
opencv==3.4.0  
(For opencv, if you use a different version, cv2.findContours() function may by different.)

## Installation
1. Download the [vgg models](链接: https://pan.baidu.com/s/1e8EPSGWA08gl5KE0v-1MEw  密码: 78wp) and put them into ```models/vgg/```  
2. Download the ImageNet validation data and change ```img_dir``` in ```test_vgg_imagenet.sh```.

## Run Localization
```
sh test_vgg_imagenet.sh
```
